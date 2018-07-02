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
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
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

def eval(save_dir):
    mAP = []
    Acc = []
    features = []
    for idx, config in enumerate(configs):
        model = models.create(
            config.model_name,
            num_features=config.num_features,
            dropout=config.dropout,
            num_classes=config.num_classes)
        model = torch.nn.DataParallel(model).cuda()
        model_name = MODEL[idx]
        feature = []
        for epoch in range(5):
            save_pth = os.path.join(save_dir, '%s.epoch%s' % (model_name, epoch))

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
                result = mu.evaluate(model, data, config)
                mAP += [result[0]]
                Acc += [result[1]]

            feature.append(
                mu.get_feature(model, query_gallery, data.images_dir, config))
        features += [feature]


    for idx in range(5):
        feas = [features[j][idx] for j in range(3)]
        result = mu.combine_evaluate(feas, data)
        mAP += [result[0]]
        Acc += [result[1]]
    return mAP, Acc

mAPs = []
Accs = []
for seed in range(1,11):
    save_dir = os.path.join(args.checkpoint, 'seed_%s' % seed)
    
    mAP, Acc = eval(save_dir)
    mAPs += [mAP]
    Accs += [Acc]

with open(args.output, 'w') as fw:
    for mAP in mAPs:
        fw.write(','.join(mAP) + '\n')
    for Acc in Accs:
        fw.write(','.join(Acc) + '\n')




    
