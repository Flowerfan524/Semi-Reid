from reid.utils.data.sampler import RandomIdentitySampler

class Config(object):

    model_name = 'resnet50'
    loss_name = 'softmax'
    logs_dir = 'logs'
    num_classes = 751

    # resize height and width
    height = 256
    width = 128

    # add noise
    img_translation = None


    #model training parameters
    batch_size = 64
    epochs = 50
    workers = 4
    num_features = 128
    dropout = 0.5
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    sampler = None

    #resume dir
    checkpoint = None
    print_freq = 20

    #training flag
    training = True
    shuffle = True
    evaluate = False

    # distance metric
    dist_metric = 'euclidean'

    def set_training(self, state):
        self.training = state
        self.shuffle = state




class TripletConfig(Config):

    loss_name = 'triplet'
    # quantity of each identity in one training batch
    num_instances = 4
    num_classes = 128
    num_features = 1024
    epochs = 150
    dropout = 0

    # margin of triplet loss
    margin = 0.5
    shuffle = False

    lr = 0.0002


    if Config.training:
        sampler = RandomIdentitySampler

    def set_training(self, state):
        self.training = state
