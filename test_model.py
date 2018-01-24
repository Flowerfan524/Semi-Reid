from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.config import Config, TripletConfig
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os

config1 = Config()
config1.model_name='resnet50m'
# config1.loss_name = 'triplet'
config1.batch_size = 64
config1.num_features = 512
config1.epochs = 60
# config1.height = 224
# config1.width = 224
# config1.epochs = 200
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'examples','data',dataset)

data = datasets.create(dataset, data_dir)
# train_data,untrain_data = dp.split_dataset(data.trainval, 0.2)

model = mu.train(data.trainval, data.images_dir, config1)
mu.evaluate(model, data, config1)
