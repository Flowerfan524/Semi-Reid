from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.utils.serialization import save_checkpoint
from reid import datasets
from reid import models
from reid.config import Config
import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Cotrain args')
parser.add_argument('-s', '--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


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
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio, args.seed)
    data_dir = data.images_dir

    new_train_data = train_data
    for step in range(iter_step):
        pred_probs = []
        add_ids = []
        for view in range(2):
            configs[view].set_training(True)
            model = mu.train(new_train_data, data_dir, configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': step + 1,
                'train_data': new_train_data}, False,
                fpath = os.path.join(configs[view].logs_dir, configs[view].model_name, 'cotrain.epoch%d' % step)
            )
            if len(untrain_data) == 0:
                continue
            pred_probs.append(mu.predict_prob(
                model,untrain_data,data_dir,configs[view]))
            add_ids.append(dp.sel_idx(pred_probs[view], train_data))

            # calculate predict probility on all data
            p_b = mu.predict_prob(model, data.trainval, data_dir, configs[view])
            p_y = np.argmax(p_b, axis=1)
            t_y = [c for (_,c,_,_) in data.trainval]
            print(np.mean(t_y == p_y))

        if len(untrain_data) == 0:
            break

        # update training data
        pred_y = np.argmax(sum(pred_probs), axis=1)
        add_id = sum(add_ids)
        new_train_data, untrain_data = dp.update_train_untrain(
            add_id,new_train_data,untrain_data,pred_y)



config1 = Config()
config2 = Config(model_name='densenet121', height=224, width=224)
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset, data_dir)


cotrain([config1,config2], data, 5)
