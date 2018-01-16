from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import datasets
from reid import models
from reid.config import Config
import torch
import numpy as np
import os
import argparse


def cotrain(configs,data,iter_step=1,train_ratio=0.2):
    """
    cotrain model:
    params:
    model_names: model configs
    data: dataset include train and untrain data
    save_paths: paths for storing models
    iter_step: maximum iteration steps
    train_ratio: labeled data ratio

    """
    assert iter_step >= 1
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    data_dir = data.images_dir
    for step in range(iter_step):
        pred_probs = []
        add_ids = []
        for view in range(2):
            configs[view].set_training(True)
            model = mu.train(train_data, data_dir, configs[view])
            pred_probs.append(mu.predict_prob(
                model,untrain_data,data_dir,configs[view]))
            add_ids.append(dp.sel_idx(pred_probs[view], train_data))
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': step + 1}, False,
                fpath=os.path.join(configs[view].logs_dir,
                                   configs[view].model_name,
                                   'cotrain.epoch%d' % (step + 1))
            )
            mu.evaluate(model, data, configs[view])

        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_id = sum(add_ids)
        train_data, untrain_data = dp.update_train_untrain(
            add_id,train_data,untrain_data,pred_y)


def main(args):

    config1 = Config()
    config2 = Config()
    config1.model_name = args.arch1
    config2.model_name = args.arch2
    config2.height = 224
    config2.width = 224
    config1.batch_size = 32
    config2.batch_size = 32
    config1.epochs = 50
    config2.epochs = 50
    config1.logs_dir = args.logs_dir
    config2.logs_dir = args.logs_dir
    #config1.checkpoint = 'logs/resnet50/spaco.epoch1'
    #config2.checkpoint = 'logs/densenet121/spaco.epoch1'
    config1.num_features = 512
    config2.num_features = 512
    dataset = args.dataset
    cur_path = os.getcwd()
    data_dir = os.path.join(cur_path,'data',dataset)
    data = datasets.create(dataset, data_dir)
    cotrain([config1, config2], data,
            iter_step=args.iter_step,
            train_ratio=args.train_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-paced cotraining Reid')
    parser.add_argument('-d', '--dataset', type=str, default='market1501std',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-a1', '--arch1', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('-a2', '--arch2', type=str, default='densenet121',
                        choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=4)
    parser.add_argument('-r', '--train_ratio', type=float, default=0.2)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
