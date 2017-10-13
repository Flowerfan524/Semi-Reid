from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.models import spaco_model_utils as smu
from reid.dist_metric import DistanceMetric
from reid.trainers import Trainer
from reid.evaluators import *
from reid.utils.data import transforms as T
from reid.utils.data import spaco_data_process as sdp
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import copy

_FEATURE_NUM = 128
_DROPOUT = 0.3
_EPOCH = 30
_NUM_CLASSES = 751


def train(model_name,train_data,data_dir):
    model = smu.get_model_by_name(model_name,_NUM_CLASSES)
    model = nn.DataParallel(model).cuda()
    data_params = smu.get_params_by_name(model_name)
    dataloader = sdp.get_dataloader(train_data,data_dir,training=True,**data_params)
    epoch = 30
    if 'inception' in model_name:
        epoch = 50
    smu.train_model(model,dataloader,epochs=epoch)
    return model

def get_feature(model,data,data_dir,params):
    dataloader = sdp.get_dataloader(data,data_dir,**params)
    features,_ = extract_features(model,dataloader)
    return features

def predict_prob(model,data,data_dir,params):
    features = get_feature(model,data,data_dir,params)
    logits = np.array([logit.numpy() for logit in features.values()])
    predict_prob = np.exp(logits)/np.sum(np.exp(logits),axis=1).reshape((-1,1))
    assert len(logits) == len(predict_prob)
    return predict_prob


def train_predict(model_name,train_data,untrain_data,data_dir):
    model = train(model_name,train_data,data_dir,model_params)
    data_params = smu.get_params_by_name(model_name)
    pred_prob = predict_prob(model,untrain_data,data_dir,data_params)
    return pred_prob


def update_train_untrain(sel_idx,train_data,untrain_data,pred_y):
    assert len(train_data[0]) == len(untrain_data[0])
    add_data = [(untrain_data[i][0],pred_y[i],untrain_data[i][2]) for i,flag in enumerate(sel_idx) if flag]
    untrain_data = [untrain_data[i] for i,flag in enumerate(sel_idx) if not flag]
    train_data += add_data
    return train_data,untrain_data

def evaluate(model,dataset,data_dir,metric=None):
    query,gallery = dataset.query,dataset.gallery
    dataloader = sdp.get_dataloader(list(set(dataset.query)|set(dataset.gallery)),data_dir)
    metric = DistanceMetric(algorithm='euclidean')
    metric.train(model,dataloader)
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader,query,gallery,metric)


def sel_idx(score,train_data,ratio=0.5):
    y = np.array([label  for _,label,_ in train_data ])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y==c) for c in clss]
    pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * ratio)), indices.shape[0])
        add_indices[indices[idx_sort[:add_num]]] = 1
    return add_indices.astype('bool')

def cotrain(model_names,data,save_paths,iter_step=1):
    """
    cotrain model:
    params:
    model_name1: first view of co-train model
    model_name2: second view of co-train model
    data: dataset for spaco model

    return:
    trained model1, model2
    """
    assert iter_step >= 1
    assert len(model_names) == 2 and len(save_paths) == 2
    train_data,untrain_data = copy.deepcopy(data.train), copy.deepcopy(data.untrain)
    model_name1,model_name2 = model_names
    data_params1 = smu.get_params_by_name(model_name1)
    data_params2 = smu.get_params_by_name(model_name2)
    save_path1, save_path2 = save_paths
    assert pred_prob1.shape[0] == pred_prob2.shape[0]
    for step in range(iter_step):
        pred_prob = 0
        add_idx = 0
        for view in range(2):
            model = train(model_names[view],train_data,data.images_dir)
            data_params = smu.get_params_by_name(model_names[view])
            pred_probs[view] = predict_prob(model,untrain_data,data.images_dir,data_params)
            evaluate(model,data,data.images_dir)
            sel_idx[view] = sel_idx(pred_probs[view], data.train)
            if step == iter_step-1:
                save_checkpoint({'state_dict':model.module.state_dict(),
                                'epoch':1
                                },True,save_path2+'checkpoint.pth.tar')
            pred_prob += pred_probs[view]
            add_idx += sel_idx[view]


        train_data, untrain_data = update_train_untrain(add_idx,train_data,untrain_data,pred_y)





if __name__ == '__main__':
    dataset = datasets.create('market1501std','examples/data/market1501std/')
    model_names = ['resnet50', 'inception']
    save_path = ['./logs/softmax-loss/market1501-resnet50/','logs/softmax-loss/market1501-inception/']
    iter_step = 5
    cotrain(model_names,dataset,save_path,iter_step)
