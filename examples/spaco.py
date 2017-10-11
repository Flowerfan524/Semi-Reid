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
_EPOCH = 50
_NUM_CLASSES = 751


def train(model_name,train_data,data_dir,model_params):
    model = smu.get_model_by_name(model_name,_NUM_CLASSES)
    dataloader = sdp.get_dataloader(train_data,data_dir,**model_params)
    smu.train_model(model,dataloader,epochs=_EPOCH)
    return model

def get_feature(model,data,data_dir):
    dataloader = sdp.get_dataloader(data,data_dir,**params)
    features,_ = extract_features(model,dataloader)
    return features

def predict_prob(model,data,data_dir,params):
    features = get_feature(model,data,data_dir)
    logits = np.array([logit.numpy() for logit in logits.values()])
    predict_prob = np.exp(logits/np.sum(np.exp(logits,axis=1).reshape((-1,1))
    assert len(logits) == len(predict_prob)
    return predict_prob


def train_predict(model_name,train_data,untrain_data,data_dir):
    model_params = smu.get_params_by_name(model_name)
    model = train(model_name,train_data,data_dir,model_params)
    pred_prob = predict_prob(model,untrain_data,data_dir,model_params)
    return pred_prob


def update_train_untrain(sel_idx,train_data,untrain_data):
    add_data = [untrain_data[i] for i,flag in enumerate(sel_idx) if flag]
    untrain_data = [untrain_data[i] for i,flag in enumerate(sel_idx) if not flag]
    train_data += add_data
    return train_data,untrain_data

def evaluate(model,dataset,data_dir,metric=None):
    query,gallery = datset.query,dataset.gallery
    dataLoader = sdp.get_dataloader(list(set(dataset.query)|set(dataset.galler)),data_dir)
    metric = DistanceMetric(algorithm='euclidean')
    metric.train(model,train_loader,data_dir)
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader,query,gallery,metric)


def sel_idx(score,train_data,ratio=0.2):
    y = np.array([label  for _,label,_ in train_data ])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y==c) for c in clss]
    pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_socre)
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
    """"
    assert iter_step >= 1
    assert len(model_names) == 2 and len(save_paths) == 2
    train_data,untrain_data = copy.deepcopy(data.train), copy.deepcopy(data.untrain)
    model_name1,model_name2 = model_names
    save_path1, save_path2 = save_paths
    pred_prob1 = train_predict(model_name1,train_data,untrain_data,data_dir=data.images_dir)
    pred_prob2 = train_predict(model_name2,train_data,untrain_data,data_dir=data.images_dir)
    assert pred_prob1.shape[0] == pred_prob2.shape[0]
    for step in range(iter_step):
        pred_y = np.argmax(pred_prob1 + pred_prob2, axis=1)
        sel_idx1 = sel_idx(pred_prob2, data.train)
        sel_idx2 = sel_idx(pred_prob1, data.train)
        add_idx = merge(sel_idx1,sel_idx2)

        train_data, untrain_data = update_train_untrain(add_idx,train_data,untrain_data)

        if step + 1 == iter_step:
            model1 = train(model_name1,train_data,save_path1,epoch)
            evaluate(model1,dataset,data.images_dir)
            # train_save(model_name2,train_data,save_path2,epoch)
            return

        pred_prob1 = train_predict(model_name1,train_data,untrain_data,data_dir=data.images_dir)
        pred_prob2 = train_predict(model_name1,train_data,untrain_data,data_dir=data.images_dir)



if __name___ == '__main__':
    dataset = datasets.create('market1501std','examples/data/')
    model_names = ['resnet50', 'inception']
    save_path = './logs/softmax-loss/market1501-resnet50'
    iter_step = 1
    cotrain(model_names,dataset,save_path,iter_step)
