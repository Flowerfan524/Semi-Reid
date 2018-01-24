from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.config import Config
from reid import datasets
import os

config = Config(model_name='resnet50m',img_translation=4)
# config1.height = 224
# config1.width = 224
# config1.epochs = 200
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'examples','data',dataset)

data = datasets.create(dataset, data_dir)
# train_data,untrain_data = dp.split_dataset(data.trainval, 0.2)

model = mu.train(data.trainval, data.images_dir, config)
mu.evaluate(model, data, config)
