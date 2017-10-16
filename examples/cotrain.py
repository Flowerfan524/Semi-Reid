from __future__ import print_function, absolute_import
from reid.models import spaco_model_utils as smu
from reid.utils.data import spaco_data_process as sdp
from reid import datasets
import copy
import torch
import numpy as np


def cotrain(model_names,data,save_paths,iter_step=1):
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
    train_data = copy.deepcopy(data.train)
    untrain_data = copy.deepcopy(data.untrain)
    data_dir = data.images_dir
    for step in range(iter_step):
        pred_probs = []
        add_ids = []
        for view in range(2):
            model = smu.train(model_names[view],train_data,
                              data.images_dir,data.num_trainval_ids)
            data_params = smu.get_params_by_name(model_names[view])
            pred_probs.append(smu.predict_prob(
                model,untrain_data,data_dir,data_params))
            add_ids.append(sdp.sel_idx(pred_probs[view], data.train))
            torch.save(model.state_dict(),save_paths[view] +
                       '.epoch%d'%(step + 1))
            smu.evaluate(model,data)

        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_id = sum(add_ids)
        train_data, untrain_data = sdp.update_train_untrain(
            add_id,train_data,untrain_data,pred_y)


if __name__ == '__main__':
    dataset = datasets.create('market1501std','examples/data/market1501std/')
    model_names = ['densenet121', 'resnet50']
    save_path = ['logs/softmax-loss/market1501/densenet121', './logs/softmax-loss/market1501/resnet50']
                 
    iter_step = 5
    cotrain(model_names,dataset,save_path,iter_step)
