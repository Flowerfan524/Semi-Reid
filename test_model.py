from __future__ import print_function, absolute_import
from reid.utils.serialization import save_checkpoint, load_checkpoint
from reid.models import model_utils as mu
from reid import models
from reid.config import Config
from reid import datasets
import torch
import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Full supervised Test')


parser.add_argument('-d', '--dataset', type=str, default='market1501std',
                    choices=datasets.names())
parser.add_argument('-a', '--arch', type=str, default='resnet50')
parser.add_argument('-h', '--height', type=int, default=256)
parser.add_argument('-w', '--width', type=int, default=128)
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--checkpoint', default='', type=str)
args = parser.parse_args()


config = Config(model_name=args.arch,img_translation=None,
                height=args.height, width=args.width)
# config1.height = 224
# config1.width = 224
# config1.epochs = 200
dataset = args.dataset
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'data',dataset)

data = datasets.create(dataset, data_dir)
# train_data,untrain_data = dp.split_dataset(data.trainval, 0.2)

if args.checkpoint is not None:

    model = models.create(config.model_name, num_features=config.num_features,
                          dropout=config.dropout, num_classes=config.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    save_pth = os.path.join(config.logs_dir, config.model_name, '%s'%(args.checkpoint))
    if os.path.exists(save_pth) is not True:
        raise ValueError('wrong model pth')
    checkpoint = load_checkpoint(save_pth)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items()
                  if k in model.state_dict().keys()}
    model.load_state_dict(state_dict)
else:
    model = mu.train(data.trainval, data.images_dir, config)

    save_checkpoint({
        'state_dict': model.state_dict(),
        'epoch': 0,
        'train_data': data.trainval}, False,
        fpath=os.path.join(config.logs_dir, config.model_name, 'full_supervised')
    )
mu.evaluate(model, data, config)

if args.evaluate is True:
    train_data = checkpoint['train_data']
    weight = [w for (_,_,_,w) in train_data]
    print(np.median(weight))
    print(np.mean(weight))
