from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid import datasets
from reid import models
import torch
import numpy as np
import os
import argparse


def cotrain(model_names,data,save_paths,iter_step=1,train_ratio=0.2):
    """
    cotrain model:
    params:
    model_names: model names such as ['resnet50','densenet121']
    data: dataset include train and untrain data
    save_paths: paths for storing models
    iter_step: maximum iteration steps
    train_ratio: labeled data ratio

    """
    assert iter_step >= 1
    assert len(model_names) == 2 and len(save_paths) == 2
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    data_dir = data.images_dir
    for step in range(iter_step):
        pred_probs = []
        add_ids = []
        for view in range(2):
            model = mu.train(model_names[view],train_data,
                             data_dir,data.num_trainval_ids,epochs=50)
            data_params = mu.get_params_by_name(model_names[view])
            pred_probs.append(mu.predict_prob(
                model,untrain_data,data_dir,data_params))
            add_ids.append(dp.sel_idx(pred_probs[view], data.train))
            torch.save(model.state_dict(),save_paths[view] +
                       '.cotrain.epoch%d' % (step + 1))
            mu.evaluate(model,data,params=data_params)

        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_id = sum(add_ids)
        train_data, untrain_data = dp.update_train_untrain(
            add_id,train_data,untrain_data,pred_y)


def main(args):
    print(args.iter_step)
    assert args.iter_step >= 1
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, dataset_dir)
    model_names = [args.arch1, args.arch2]
    save_paths = [os.path.join(args.logs_dir, args.arch1),
                  os.path.join(args.logs_dir, args.arch2)]
    cotrain(model_names,dataset,save_paths,args.iter_step,args.train_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-paced cotraining Reid')
    parser.add_argument('-d', '--dataset', type=str, default='market1501std',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-a1', '--arch1', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('-a2', '--arch2', type=str, default='densenet121',
                        choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-r', '--train_ratio', type=float, default=0.2)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
