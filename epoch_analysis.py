from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid.config import Config, TripletConfig
from reid.utils.osutils import mkdir_if_missing
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import datasets
from reid import models
import numpy as np
import torch
import os
import pdb


# prepare dataset
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, "logs")
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset,data_dir)
train_data = data.trainval

# model config
config = Config()
config.num_features = 512
config.width = 224
config.height = 224
config.set_training(False)
config.model_name = 'densenet121'
model = models.create(config.model_name, num_features=config.num_features,
                      dropout=config.dropout, num_classes=config.num_classes)
model = torch.nn.DataParallel(model).cuda()
# model = model.cuda()

#load model weights
#save_dir = 'logs'
#save_pth = './logs/resnet50/spaco.epoch1'
#checkpoint = load_checkpoint(save_pth)
#state_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in model.state_dict().keys()}
#model.load_state_dict(state_dict)

#epoch analysis
for i in range(0, 5):
    # load model weights
    save_pth = os.path.join(config.logs_dir, config.model_name, 'spaco.epoch%d'%i)
    checkpoint = load_checkpoint(save_pth)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in model.state_dict().keys()}
    model.load_state_dict(state_dict)
    # predict 
    pred_prob = mu.predict_prob(model, train_data, data.images_dir, config)
    pred_y = np.argmax(pred_prob, axis = 1)
    y = [cls for (_, cls, _, _) in train_data]
    print(np.mean(pred_y == y))
    mu.evaluate(model, data, config)



