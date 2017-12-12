
class Config(object):

    model_name = 'Resnet50'
    loss_name = 'Softmax'
    data_name = 'market1501std'
    data_dir = './examples/data'
    logs_dir = './logs'
    num_classes = 751
    height = 256
    width = 128
    batch_size = 64
    epochs = 50
    workers = 4
    num_features = 128
    dropout = 0.5
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4

    checkpoints = None

    evaluate = False

    seed = 1

    dist_metric = 'euclidean'


class TripletConfig(Config):

    # quantity of each identity in one training batch
    num_instances = 4

    # margin of triplet loss
    margin = 0.5
