import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
from reid.utils.data import data_process as dp
import numpy as np
from collections import defaultdict


_FEATURE_NUM = 128
_DROPOUT = 0.3
_PARAMS_FACTORY = {
    'resnet':
        {'height':256,
            'width':128},
    'inception':
        {'height':128,
            'width':64},
    'inception_v3':
        {'height':299,
            'width':299},
    'densenet':
        {'height':224,
            'width':224},
    'vgg':
        {'height':224,
            'width':224}
}


def get_model_by_name(model_name,num_classes):
    """
    create model given the model_name and number of classes
    """
    if 'resnet' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3,num_classes=num_classes)
    elif 'inception' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3,num_classes=num_classes)
    elif 'densenet' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3, num_classes=num_classes)
    elif 'vgg' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3, num_classes=num_classes)
    else:
        raise ValueError('wrong model name, no such model!')
    return model


def get_params_by_name(model_name):
    """
    get model Parameters given the model_name
    """
    params = {}
    for k,v in _PARAMS_FACTORY.items():
        if k in model_name:
            params = v
    if not params:
        raise ValueError('wrong model name, no params!')
    params['batch_size'] = 64
    params['workers'] = 2
    return params


def train_model(model,dataloader,epochs=50):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        epochs: training epochs
        criterion
    """
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=0.1,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

    def adjust_lr(epoch):
        step_size = 40
        lr = 0.1 * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    criterion = nn.CrossEntropyLoss().cuda()
    trainer = Trainer(model,criterion)
    for epoch in range(epochs):
        adjust_lr(epoch)
        trainer.train(epoch, dataloader, optimizer)


def train(model_name,train_data,data_dir,num_classes,epochs=50):
    model = get_model_by_name(model_name,num_classes)
    model = nn.DataParallel(model).cuda()
    data_params = get_params_by_name(model_name)
    dataloader = dp.get_dataloader(
        train_data,data_dir,training=True,**data_params)
    train_model(model,dataloader,epochs=epochs)
    return model


def get_feature(model,data,data_dir,params):
    dataloader = dp.get_dataloader(data,data_dir,**params)
    features,_ = extract_features(model,dataloader)
    return features


def predict_prob(model,data,data_dir,params):
    features = get_feature(model,data,data_dir,params)
    logits = np.array([logit.numpy() for logit in features.values()])
    exp_logits = np.exp(logits)
    predict_prob = exp_logits / np.sum(exp_logits,axis=1).reshape((-1,1))
    assert len(logits) == len(predict_prob)
    return predict_prob


def train_predict(model_name,train_data,untrain_data,num_classes,data_dir):
    model = train(model_name,train_data,data_dir,num_classes)
    data_params = get_params_by_name(model_name)
    pred_prob = predict_prob(model,untrain_data,data_dir,data_params)
    return pred_prob


def get_clusters(model,data_loader,num_classes):
    features, labels = extract_features(model, data_loader)
    class_features = defaultdict(list)
    for k,v in labels.items():
        class_features[v].append(features[k])
    clusters = [np.mean(class_features[i],axis=0)
                for i in range(num_classes)]
    clusters = torch.from_numpy(np.array(clusters, dtype='float32'))
    return torch.autograd.Variable(clusters)


def evaluate(model,dataset,params,metric=None):
    query,gallery = dataset.query,dataset.gallery
    dataloader = dp.get_dataloader(
        list(set(dataset.query) | set(dataset.gallery)),
        dataset.images_dir,**params)
    metric = DistanceMetric(algorithm='euclidean')
    metric.train(model,dataloader)
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader,query,gallery,metric)
