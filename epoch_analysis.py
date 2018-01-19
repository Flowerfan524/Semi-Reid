from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.config import Config
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import datasets
from reid import models
import numpy as np
import torch
import os

import argparse

parser = argparse.ArgumentParser(description='Model Test')

parser.add_argument('-d', '--dataset', type=str, default='market1501std',
                    choices=datasets.names())

parser.add_argument('-a', '--arch', type=str, default='resnet50',
                    choices=models.names())
parser.add_argument('-m', '--model', type=str, default='spaco')
parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('--height', type=int,
                    help="input height, default: 256 for resnet*, "
                         "224 for densenet*")
parser.add_argument('--width', type=int,
                    help="input width, default: 128 for resnet*, "
                         "224 for densenet*")

args = parser.parse_args()

# prepare dataset
dataset = args.dataset
cur_path = os.getcwd()

logs_dir = os.path.join(cur_path, "logs")
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset,data_dir)
train_data = data.trainval

# model config
config = Config()
config.num_features = 512
config.width = args.width
config.height = args.height
config.set_training(False)
config.model_name = args.arch

# create model
model = models.create(config.model_name, num_features=config.num_features,
                      dropout=config.dropout, num_classes=config.num_classes)
model = torch.nn.DataParallel(model).cuda()
# model = model.cuda()

#epoch analysis
for i in range(0, 5):
    # load model weights
    save_pth = os.path.join(config.logs_dir, config.model_name, '%s.epoch%d'%(args.model, i))
    if os.path.exists(save_pth) is not True:
        continue
    checkpoint = load_checkpoint(save_pth)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in model.state_dict().keys()}
    model.load_state_dict(state_dict)
    # predict
    pred_prob = mu.predict_prob(model, train_data, data.images_dir, config)
    pred_y = np.argmax(pred_prob, axis=1)
    y = [cls for (_, cls, _, _) in train_data]
    print(np.mean(pred_y == y))
    mu.evaluate(model, data, config)
