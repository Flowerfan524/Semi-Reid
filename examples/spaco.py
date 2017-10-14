from __future__ import print_function, absolute_import
from reid.models import spaco_model_utils as smu
from reid.utils.data import spaco_data_process as sdp
from reid import datasets
import copy
import numpy as np
import torch


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
    ###########
    # initiate classifier to get preidctions
    ###########

    add_ratio = 0.5
    pred_probs = []
    add_ids = []
    for view in range(num_view):
        pred_probs.append(smu.train_predict(
            model_names[view],train_data,untrain_data,data_dir))
        add_ids.append(sdp.sel_idx(pred_probs[view], data.train, add_ratio))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    for step in range(iter_step):
        for view in range(num_view):
            # update v_view
            ov = add_ids[3 - view]
            pred_probs[view][ov,pred_y[ov]] += gamma
            add_id = sdp.sel_idx(pred_probs[view],data.train, add_ratio)

            # update w_view
            train_data,_ = sdp.update_train_untrain(
                add_id,train_data,untrain_data,pred_y)
            model = smu.train(model_names[view],train_data,data_dir)

            # update y
            data_params = smu.get_params_by_name(model_names[view])
            pred_probs[view] = smu.predict_prob(
                model,untrain_data,data_dir,data_params)
            pred_y = np.argmax(sum(pred_probs),axis=1)

            # udpate v_view for next view
            add_ratio += 0.5
            add_ids[view] = sdp.sel_idx(pred_probs[view], data.train,add_ratio)

            # evaluation current model and save it
            smu.evaluate(model,data)
            torch.save(model.state_dict(),save_paths[view] + str(step + 1))


if __name__ == '__main__':
    dataset = datasets.create('market1501std','examples/data/market1501std/')
    model_names = ['resnet50', 'inception']
    save_path = ['./logs/softmax-loss/market1501-resnet50/resnet_spaco',
                 './logs/softmax-loss/market1501-inception/inception_spaco']
    iter_step = 5
    spaco(model_names,dataset,save_path,iter_step)
