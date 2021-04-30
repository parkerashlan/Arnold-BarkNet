import numpy as np
import pandas as pd
import itertools

import DatasetMaker
from torchvision.datasets.folder import pil_loader

from torch.utils.data import Dataset
from torchvision import transforms


class BarkDataset(Dataset):

    def __init__(self, path_to_data, data,
                 transform=transforms.Compose([
                     transforms.Resize((2000, 912)),
                     transforms.RandomCrop(224),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]),
                 val=False, test=False):

        if not val and not test:
            self.images = data['train_imgs'].values
            self.labels = data['train_labels'].values
        elif val and not test:
            self.images = data['val_imgs'].values
            self.labels = data['val_labels'].values
        elif test:
            self.images = data['fold 1'].values
            self.labels = data['fold 1 labels'].values

        self.path = path_to_data
        self.transform = transform


    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        X = pil_loader(self.path + img_name)

        y = self.labels[index]

        if self.transform:
            X = self.transform(X)

        return X, y, img_name


def make_train_validation_set(ds):
    """
    Make a list of k train and validation sets where k is the number of folds specified when
    making the dataset.

    :return:
        train_sets (list): A list of DataFrames which have the images in the train_imgs column and labels in the
        train_labels column.

        val_sets (list): A list of DataFrames which have the images in the val_imgs column and labels in the
        val_labels column.
    """

    folds = ds[1]
    data = ds[0]

    train_sets = []
    val_sets = []
    j = 0
    for i in range(0, folds):

        train_df = pd.DataFrame(columns=['train_imgs', 'train_labels'])
        val_df = pd.DataFrame(columns=['val_imgs', 'val_labels'])

        val = data.iloc[:, j:j+2]

        val_names = val.columns

        train = data.drop(val_names, axis=1)

        train_df['train_imgs'] = train.values[:, ::2].reshape(1, -1)[0]
        train_df['train_labels'] = train.values[:, 1::2].reshape(1, -1)[0]

        val_df['val_imgs'] = val.values[:, 0]
        val_df['val_labels'] = val.values[:, 1]

        train_sets.append(train_df)
        val_sets.append(val_df)

        j += 2

    return train_sets, val_sets

if __name__ == "__main__":
    PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/train_1_2/'

    dg = DatasetMaker.DatasetGenerator(path=PATH)

    k_folds = dg.stratified_kfold_cv()

    train_sets, val_sets = make_train_validation_set(ds=k_folds)

    print(len(train_sets[0]), len(val_sets[0]))
    train_dataset = BarkDataset(path_to_data=PATH, data=train_sets[0])

    print(train_dataset)