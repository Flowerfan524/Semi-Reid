
import torch
from torch import nn
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import models
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from collections import OrderedDict

_FEATURE_NUM = 128
_DROPOUT = 0.3
_PARAMS_FACTORY={
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
                    {'height':256,
                    'width':128},
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


def train_model(model,dataloader,epochs=30):
    """
    train model given the dataloader the criterion, stop when epochs are reached
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
        lr = 0.1  * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    criterion = nn.CrossEntropyLoss().cuda()
    trainer = Trainer(model,criterion)
    for epoch in range(epochs):
        adjust_lr(epoch)
        trainer.train(epoch, dataloader, optimizer)

