from reid.models import model_utils as mu
from reid.utils.serialization import load_checkpoint
from reid.config import Config
from reid import datasets
from reid import models
import numpy as np
import torch
import os
import argparse

parser = argparse.ArgumentParser(description='Model Test')

parser.add_argument('-d', '--dataset', type=str, default='market1501std',
                    choices=datasets.names())
parser.add_argument('-c', '--checkpoint', type=str, default='spaco.epoch4')
parser.add_argument('-b', '--batch-size', type=int, default=256)
args = parser.parse_args()


dataset = args.dataset
cur_path = os.getcwd()

logs_dir = os.path.join(cur_path, "logs")
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset,data_dir)
query_gallery = list(set(data.query) | set(data.gallery))

# model config
config1 = Config()
config2 = Config()
config1.num_features = 512
config2.num_features = 512
config2.model_name = 'densenet121'
config2.width = 224
config2.height = 224

features = []
for config in [config1, config2]:
    model = models.create(config.model_name, num_features=config.num_features,
                          dropout=config1.dropout, num_classes=config.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    save_pth = os.path.join(config.logs_dir, config.model_name, '%s'%(args.checkpoint))
    if os.path.exists(save_pth) is not True:
        raise ValueError('wrong model pth')
    checkpoint = load_checkpoint(save_pth)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items()
                  if k in model.state_dict().keys()}
    model.load_state_dict(state_dict)
    mu.evaluate(model, data, config)
    features.append(mu.predict_prob(model, query_gallery,
                                    data.images_dir, config))

features = np.sum(features, axis=0)
mu.combine_evaluate(features, data)
