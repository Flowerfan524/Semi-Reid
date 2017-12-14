import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.loss import TripletLoss
from reid.dist_metric import DistanceMetric
from reid.utils.data import data_process as dp
import numpy as np
from collections import defaultdict


def train_model(model, dataloader, config):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training configuration
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

    if config.loss_name is 'softmax':
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(param_groups, lr=config.lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay,
                                    nesterov=True)
    elif config.loss_name is 'triplet':
        criterion = TripletLoss(margin=config.margin).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                     weight_decay=config.weight_decay)
    else:
        raise ValueError('wrong loss name')

    trainer = Trainer(model, criterion)

    # schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = config.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(config.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, dataloader, optimizer)


def train(train_data, data_dir, config):
    assert config.training
    model = models.create(config.model_name,
                          num_features=config.num_features,
                          dropout=config.dropout,
                          num_classes=config.num_classes)
    model = nn.DataParallel(model).cuda()
    dataloader = dp.get_dataloader(train_data, data_dir, config)
    train_model(model, dataloader, config)
    config.training = False
    return model


def get_feature(model, data, data_dir, config):
    dataloader = dp.get_dataloader(data, data_dir, config)
    features, _ = extract_features(model, dataloader)
    return features


def predict_prob(model, data, data_dir, config):
    features = get_feature(model, data, data_dir, config)
    logits = np.array([logit.numpy() for logit in features.values()])
    exp_logits = np.exp(logits)
    predict_prob = exp_logits / np.sum(exp_logits,axis=1).reshape((-1,1))
    assert len(logits) == len(predict_prob)
    return predict_prob


def train_predict(train_data, untrain_data, data_dir, config):
    model = train(train_data, data_dir, config)
    pred_prob = predict_prob(model, untrain_data, data_dir, config)
    return pred_prob


def get_clusters(model, data_loader, num_classes):
    features, labels = extract_features(model, data_loader)
    assert np.unique(labels.values()) == num_classes
    class_features = defaultdict(list)
    for k, v in labels.items():
        class_features[v].append(features[k].numpy())
    clusters = [np.mean(class_features[i], axis=0)
                for i in range(num_classes)]
    clusters = torch.from_numpy(np.array(clusters, dtype='float32'))
    return torch.autograd.Variable(clusters)


def evaluate(model, dataset, config):
    query, gallery = dataset.query, dataset.gallery
    dataloader = dp.get_dataloader(
        list(set(dataset.query) | set(dataset.gallery)),
        dataset.images_dir, config)
    metric = DistanceMetric(algorithm=config.dist_metric)
    metric.train(model, dataloader)
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader, query, gallery, metric)
