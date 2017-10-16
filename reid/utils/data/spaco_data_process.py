import numpy as np
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor


def get_dataloader(dataset,data_dir,
                   training=False, height=256,
                   width=128, batch_size=64, workers=1):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    if training:
        transformer = T.Compose([
            T.RandomSizedRectCrop(height, width),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
        ])
    else:
        transformer = T.Compose([
            T.RectScale(height, width),
            T.ToTensor(),
            normalizer,
        ])
    data_loader = DataLoader(
        Preprocessor(dataset, root=data_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=training, pin_memory=True, drop_last=training)
    return data_loader


def update_train_untrain(sel_idx,train_data,untrain_data,pred_y):
    assert len(train_data[0]) == len(untrain_data[0])
    add_data = [(untrain_data[i][0],int(pred_y[i]),untrain_data[i][2])
                for i,flag in enumerate(sel_idx) if flag]
    data1 = [untrain_data[i]
             for i,flag in enumerate(sel_idx) if not flag]
    data2 = train_data + add_data
    return data2, data1


def sel_idx(score,train_data,ratio=0.5):
    y = np.array([label for _,label,_ in train_data])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score)
        add_num = min(int(np.ceil(count_per_class[cls] * ratio)),
                      indices.shape[0])
        add_indices[indices[idx_sort[:add_num]]] = 1
    return add_indices.astype('bool')
