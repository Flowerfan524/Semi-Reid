from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os


def spaco(model_names,data,save_paths,iter_step=1,gamma=0.3,train_ratio=0.2):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    assert iter_step >= 1
    assert len(model_names) == 2 and len(save_paths) == 2
    num_view = len(model_names)
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    data_dir = data.images_dir
    num_classes = data.num_trainval_ids
    ###########
    # initiate classifier to get preidctions
    ###########

    add_ratio = 0.5
    pred_probs = []
    add_ids = []
    for view in range(num_view):
        pred_probs.append(mu.train_predict(
            model_names[view],train_data,untrain_data,num_classes,data_dir))
        add_ids.append(dp.sel_idx(pred_probs[view], train_data, add_ratio))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    for step in range(iter_step):
        for view in range(num_view):
            # update v_view
            ov = add_ids[1 - view]
            pred_probs[view][ov,pred_y[ov]] += gamma
            add_id = dp.sel_idx(pred_probs[view],train_data, add_ratio)

            # update w_view
            new_train_data,_ = dp.update_train_untrain(
                add_id,train_data,untrain_data,pred_y)
            model = mu.train(model_names[view],new_train_data,
                             data_dir,num_classes)

            # update y
            data_params = mu.get_params_by_name(model_names[view])
            pred_probs[view] = mu.predict_prob(
                model,untrain_data,data_dir,data_params)
            pred_y = np.argmax(sum(pred_probs),axis=1)

            # udpate v_view for next view
            add_ratio += 0.5
            add_ids[view] = dp.sel_idx(pred_probs[view], train_data,add_ratio)

            # evaluation current model and save it
            mu.evaluate(model,data,data_params)
            torch.save(model.state_dict(),save_paths[view] +
                       '.spaco.epoch%d' % (step + 1))


def main(args):
    assert args.iter_step >= 1
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, dataset_dir)
    model_names = [args.arch1, args.arch2]
    save_paths = [os.path.join(args.logs_dir, args.arch1),
                  os.path.join(args.logs_dir, args.arch2)]
    spaco(model_names,dataset,save_paths,args.iter_step,
          args.gamma,args.train_ratio)


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
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-r', '--train_ratio', type=float, default=0.2)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
