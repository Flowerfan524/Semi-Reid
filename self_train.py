from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.utils.serialization import save_checkpoint
from reid import datasets
from reid import models
from reid.config import Config
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Cotrain args')
parser.add_argument('-s', '--seed', type=int, default=0)
args = parser.parse_args()


def self_train(configs, data, iter_step=1, train_ratio=0.2):
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
    train_data, untrain_data = dp.split_dataset(
        data.trainval, train_ratio, args.seed)
    data_dir = data.images_dir

    for view in range(len(configs)):
        add_ratio = 0.5
        new_train_data = train_data
        new_untrain_data = untrain_data
        for step in range(iter_step):
            configs[view].set_training(True)
            model = mu.train(new_train_data, data_dir, configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': step + 1,
                'train_data': train_data}, False,
                fpath=os.path.join(
                    configs[view].logs_dir, configs[view].model_name, 'self_train.epoch%d' % step)
            )
            # calculate predict probility on all data
            p_b = mu.predict_prob(model, data.trainval, data_dir, configs[view])
            p_y = np.argmax(p_b, axis=1)
            t_y = [c for (_,c,_,_) in data.trainval]
            print(np.mean(t_y == p_y))

            if len(new_untrain_data) == 0:
                break
            pred_prob = mu.predict_prob(
                model, new_untrain_data, data_dir, configs[view])
            pred_y = np.argmax(pred_prob, axis=1)
            add_id = dp.sel_idx(pred_prob, new_train_data, add_ratio)
            new_train_data, new_untrain_data = dp.update_train_untrain(
                add_id, new_train_data, new_untrain_data, pred_y)


config1 = Config()
config2 = Config(model_name='densenet121', height=224, width=224)
config3 = Config(model_name='renset101', img_translation=2)
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path, 'data', dataset)
data = datasets.create(dataset, data_dir)


self_train([config1, config2, config3], data, 5)
