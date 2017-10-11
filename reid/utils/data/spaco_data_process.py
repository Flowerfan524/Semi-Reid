import numpy as np
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor


def get_dataloader(dataset,data_dir,training=False, height=256, width=128, batch_size=64, workers=1):
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
