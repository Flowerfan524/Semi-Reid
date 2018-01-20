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


def spaco(configs,data,iter_step=1,gamma=0.3,train_ratio=0.2):
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
    num_view = len(configs)
    train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    data_dir = data.images_dir
    num_classes = data.num_trainval_ids
    ###########
    # initiate classifier to get preidctions
    ###########

    add_ratio = 0.5
    pred_probs = []
    start_step = 0
    for view in range(num_view):
        if configs[view].checkpoint is None:
            model = mu.train(train_data, data_dir, configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0}, False,
                fpath = os.path.join(configs[view].logs_dir, configs[view].model_name, 'soft_spaco.epoch0')
            )
        else:
            model = models.create(configs[view].model_name,
                                  num_features=configs[view].num_features,
                                  dropout=configs[view].dropout,
                                  num_classes=configs[view].num_classes)
            model = torch.nn.DataParallel(model).cuda()
            checkpoint = load_checkpoint(configs[view].checkpoint)
            model.load_state_dict(checkpoint['state_dict'])
            start_step = checkpoint['epoch']
            add_ratio += start_step * 0.5
            configs[view].set_training(False)
            #mu.evaluate(model, data, configs[view])
            configs[view].set_training(True)
        pred_probs.append(mu.predict_prob(model, untrain_data, data_dir, configs[view]))
    pred_y = np.argmax(sum(pred_probs), axis=1)

    #### set lambdas and weights for unlabled samples
    lambdas = [dp.get_lambda_class(pred_prob, pred_y, train_data, add_ratio)
               for pred_prob in pred_probs]
    weights = [[(pred_probs[v][i, l] - lambdas[v][l]) / gamma
                for i,l in enumerate(pred_y)]
               for v in range(num_view)]
    weights = np.array(weights, dtype='float32')
    for view in range(num_view):
        weights[view][weights[view] > 1] = 1
        weights[view][weights[view] < 0] = 0
    for step in range(start_step, iter_step):
        for view in range(num_view):
            # update v_view
            weight = weights[1 - view] + weights[view]
            weight[weight > 1] = 1
            weight[weight < 0] = 0

            # update w_view
            sel_idx = weight > 0
            new_train_data,_ = dp.update_train_untrain(sel_idx,train_data,untrain_data,pred_y, weight)
            configs[view].set_training(True)
            model = mu.train(new_train_data, data_dir, configs[view])

            # update y
            pred_probs[view] = mu.predict_prob( model,untrain_data,data_dir, configs[view])
            pred_y = np.argmax(sum(pred_probs),axis=1)

            # udpate v_view for next view
            lambdas[view] = dp.get_lambda_class(pred_probs[view], pred_y, train_data, add_ratio)
            weights[view] = [(pred_probs[view][i,l] - lambdas[view][l]) / gamma
                             for i,l in enumerate(pred_y)]
            weights[view][weights[view] > 1] = 1
            weights[view][weights[view] < 0] = 0
            add_ratio += 0.5


            # calculate predict probility on all data
            p_b = mu.predict_prob(model, data.trainval, data_dir, configs[view])
            p_y = np.argmax(p_b, axis=1)
            t_y = [c for (_,c,_,_) in data.trainval]
            print(np.mean(t_y == p_y))
#             evaluation current model and save it
            # mu.evaluate(model,data,configs[view])
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': step +1}, False,
                fpath = os.path.join(configs[view].logs_dir, configs[view].model_name, 'soft_spaco.epoch%d' % (step + 1))
            )
            # mkdir_if_missing(logs_pth)
            # torch.save(model.state_dict(), logs_pth +
            #           '/spaco.epoch%d' % (step + 1))

config1 = Config()
config2 = Config()
config1.loss_name = 'weight_softmax'
config2.loss_name = 'weight_softmax'
config2.model_name = 'densenet121'
config2.height = 224
config2.width = 224
config1.batch_size = 32
config2.batch_size = 32
config1.epochs = 50
config2.epochs = 50
config1.checkpoint = 'logs/resnet50/soft_spaco.epoch4'
config2.checkpoint = 'logs/densenet121/soft_spaco.epoch4'
config1.num_features = 512
config2.num_features = 512
dataset = 'market1501std'
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path,'data',dataset)
data = datasets.create(dataset, data_dir)

spaco([config1,config2], data, 7)
