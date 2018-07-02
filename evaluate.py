from reid.models import model_utils as mu
from reid.utils.serialization import load_checkpoint
from reid.config import Config
from reid import datasets
from reid import models
import numpy as np
import torch
import os
import argparse

MODEL=['resnet50', 'densenet121', 'resnet101']
parser = argparse.ArgumentParser(description='Model Test')

parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    default='market1501std',
    choices=datasets.names())
parser.add_argument('-c', '--checkpoint', type=str, default='spaco.epoch4')
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--combine', type=str, default='123')
parser.add_argument(
    '--single-eval', action='store_true', help='evaluate single view')
args = parser.parse_args()

# load data set
dataset = args.dataset
cur_path = os.getcwd()
data_dir = os.path.join(cur_path, 'data', dataset)
data = datasets.create(dataset, data_dir)
query_gallery = list(set(data.query) | set(data.gallery))

# model config
config1 = Config(batch_size=128)
config2 = Config(
    model_name='densenet121', height=224, width=224, batch_size=128)
config3 = Config(model_name='resnet101', batch_size=128)
if args.combine == '123':
    configs = [config1, config2, config3]
elif args.combine == '12':
    configs = [config1, config2]
elif args.combine == '23':
    configs = [config2, config3]
else:
    raise ValueError('wrong combination')

for epoch in range(5):
    features = []
    Accs = []
    for idx, config in enumerate(configs):
        #  if idx == 0:
            #  continue
        #  import pdb; pdb.set_trace()
        model = models.create(
            config.model_name,
            num_features=config.num_features,
            dropout=config.dropout,
            num_classes=config.num_classes)
        model = torch.nn.DataParallel(model).cuda()

        model_name = MODEL[idx]
        save_pth = os.path.join(args.checkpoint, '%s.epoch%s' % (model_name, epoch))

        if os.path.exists(save_pth) is not True:
            raise ValueError('wrong model pth %s' % save_pth)
        checkpoint = load_checkpoint(save_pth)
        state_dict = {
            k: v
            for k, v in checkpoint['state_dict'].items()
            if k in model.state_dict().keys()
        }
        model.load_state_dict(state_dict)
        if args.single_eval:
            Accs += [mu.evaluate(model, data, config)]

        features.append(
            mu.get_feature(model, query_gallery, data.images_dir, config))

    Accs += [mu.combine_evaluate(features, data)]
    print(Accs)
