from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid import datasets
import copy
import numpy as np
import torch
import argparse


def spaco(model_names,data,save_paths,iter_step=1,gamma=0.3):
    """
    cotrain model:
    params:
    model_name1: first view of co-train model
    model_name2: second view of co-train model
    data: dataset for spaco model

    return:
    trained model1, model2
    """
    assert iter_step >= 1
    assert len(model_names) == 2 and len(save_paths) == 2
    num_view = len(model_names)
    train_data = copy.deepcopy(data.train)
    untrain_data = copy.deepcopy(data.untrain)
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
        add_ids.append(dp.sel_idx(pred_probs[view], data.train, add_ratio))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    for step in range(iter_step):
        for view in range(num_view):
            # update v_view
            ov = add_ids[1 - view]
            pred_probs[view][ov,pred_y[ov]] += gamma
            add_id = dp.sel_idx(pred_probs[view],data.train, add_ratio)

            # update w_view
            train_data,_ = dp.update_train_untrain(
                add_id,data.train,untrain_data,pred_y)
            model = mu.train(model_names[view],train_data,data_dir,num_classes)

            # update y
            data_params = mu.get_params_by_name(model_names[view])
            pred_probs[view] = mu.predict_prob(
                model,untrain_data,data_dir,data_params)
            pred_y = np.argmax(sum(pred_probs),axis=1)

            # udpate v_view for next view
            add_ratio += 0.5
            add_ids[view] = dp.sel_idx(pred_probs[view], data.train,add_ratio)

            # evaluation current model and save it
            mu.evaluate(model,data,data_params)
            torch.save(model.state_dict(),save_paths[view] + '.epoch%d' % (step + 1))

def main(args):
    dataset_dir = os.path.join(args.
    dataset = datasets.create(args.dataset,'examples/data/market1501std/')
    model_names = ['resnet50', 'densenet121']
    save_path = ['./logs/softmax-loss/market1501/resnet50.spaco',
                 './logs/softmax-loss/market1501/densenet121.spaco']
    iter_step = 5
    spaco(model_names,dataset,save_path,iter_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-paced cotraining Reid')
    parser.add_argument('-d', '--dataset', type=str, default='market1501std', choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-a1', '--arch1', type=str, default='resnet50', choices=models.names())
    parser.add_argument('-a2', '--arch2', type=str, default='densenet121', choices=models.names())
    working_dir = os.path.direname(os.path(__file__))
    parser.add_argument('--data_dir', type=str, metavar='PATH', default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs_dir', type=str, metavar='PATH', default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
    
