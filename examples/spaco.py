from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import datasets
from reid import models
from reid.config improt Config
import numpy as np
import torch
import argparse
import os


def spaco(configs,data,iter_step=1,gamma=0.3,train_ratio=0.2):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    configs: model configs for spaco
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    num_view = len(configs)
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    data_dir = data.images_dir
    ###########
    # initiate classifier to get preidctions
    ###########

    add_ratio = 0.5
    pred_probs = []
    add_ids = []
    start_step = 0
    for view in range(num_view):
        if configs[view].checkpoint is None:
            model = mu.train(train_data, data_dir, configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0}, False,
                fpath=os.path.join(configs[view].logs_dir,
                                   configs[view].model_name, 'spaco.epoch0')
            )
        else:
            model = models.create(configs[view].model_name,
                                  num_features=configs[view].num_features,
                                  dropout=configs[view].dropout,
                                  num_classes=configs[view].num_classes)
            checkpoint = load_checkpoint(configs[view].checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            start_step = checkpoint['epoch']
            configs[view].set_training(False)
            mu.evaluate(model, data, configs[view])
            configs[view].set_training(True)
        pred_probs.append(mu.predict_prob(model, untrain_data, data_dir, configs[view]))
        add_ids.append(dp.sel_idx(pred_probs[view], train_data, add_ratio))
    pred_y = np.argmax(sum(pred_probs), axis=1)
    for step in range(start_step, iter_step):
        for view in range(num_view):
            # update v_view
            ov = add_ids[1 - view]
            pred_probs[view][ov,pred_y[ov]] += gamma
            add_id = dp.sel_idx(pred_probs[view],train_data, add_ratio)

            # update w_view
            new_train_data,_ = dp.update_train_untrain(add_id,train_data,untrain_data,pred_y)
            configs[view].set_training(True)
            model = mu.train(new_train_data, data_dir, configs[view])

            # update y
            pred_probs[view] = mu.predict_prob(model,untrain_data,data_dir, configs[view])
            pred_y = np.argmax(sum(pred_probs),axis=1)

            # udpate v_view for next view
            add_ratio += 0.5
            add_ids[view] = dp.sel_idx(pred_probs[view], train_data,add_ratio)

#             evaluation current model and save it
            mu.evaluate(model,data,configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': step +1}, False,
                fpath = os.path.join(configs[view].logs_dir, configs[view].model_name, 'spaco.epoch%d' % (step + 1))
            )

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

    spaco([config1,config2], data,
          iter_step=args.iter_step,
          gamma=args.gamma,
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
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-r', '--train_ratio', type=float, default=0.2)
    working_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
