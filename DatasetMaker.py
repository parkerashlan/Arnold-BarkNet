# Other Libs
import numpy as np
import os
import copy
import pandas as pd

from sklearn.model_selection import StratifiedKFold


class DatasetGenerator:

    def __init__(self, path):
        self.path = path
        self.m = None

    def get_labels_and_names(self):
        """
        Get the labels and image names for all of the images in the dataset.

        :return:
            labels (list): List of all of the labels for the images in the order of files in the directory.

            image_names (list): List of all of the image names in the directory.
        """
        image_names = os.listdir(self.path)

        labels = [classes.split('_')[0] for classes in image_names]

        self.m = len(labels)

        return labels, image_names

    def stratified_kfold_cv(self, folds=5, seed=None):
        """
        Split the dataset into folds where each one represents the distribution of the dataset. Creates
        a DataFrame with the image names and labels where each fold is one column of the DataFrame and the
        corresponding labels for that fold is the next column. Can be loaded into Tensorflow using .flow_from_dataframe
        method.

        :param:
            folds (int): How many folds to make.
            seed (int): Seed for random shuffling.

        :return:
            folds_array (Pandas DataFrame object): A DataFrame with the folds and the corresponding labels.
        """
        # Get arrays of image names and labels
        image_names = np.asarray(self.get_labels_and_names()[1]).reshape(self.m, 1)
        labels = np.asarray(self.get_labels_and_names()[0], dtype=np.int).reshape(self.m, 1)

        # Set seed for shuffle if you want
        np.random.seed(seed=seed)
        np.random.shuffle(image_names)

        # Get an array of all the different classes
        classes = np.unique(labels)

        class_dist = np.zeros((len(classes), 1))  # the index corresponds to the class number

        # Create an array of the class distributions. Each element is the percentage of images which are a part of that
        # class from the dataset.
        for i in classes:

            class_dist[i-1] = sum(labels == i)/self.m

        # Get the number of images in each fold and round down to avoid out of index errors.
        num_images_in_fold = np.floor(self.m/folds)

        # Get the number of images which will be in each class for each fold and round down to avoid index errors.
        stratified_image_count = np.floor(num_images_in_fold * class_dist)

        # Create the column labels for the DataFrame
        fold_labels = []
        for i in range(folds):
            fold_labels.append(f'fold {i+1}')
            fold_labels.append(f'fold {i+1} labels')

        fold_array = pd.DataFrame(np.zeros((int(num_images_in_fold), folds*2)), columns=fold_labels)

        for fold in range(1, folds+1):

            class_image_counts = [np.floor(sum(labels == i)/folds) for i in range(1, len(classes)+1)]
            class_image_counts = np.hstack(class_image_counts)

            df_idx = 0
            image_name_idx = 0

            # This will stop the loop once all entries are 0.
            while class_image_counts.any():

                class_num = int(image_names[image_name_idx][0].split('_')[0])

                # Skip any images that have been used (All used ones replaced by 0)
                if class_num == 0:
                    image_name_idx += 1
                    continue

                # Check if the number of images for that class is filled
                if class_image_counts[class_num-1] != 0:

                    fold_array.loc[df_idx, f'fold {fold}'] = image_names[image_name_idx]
                    fold_array.loc[df_idx, f'fold {fold} labels'] = class_num

                    class_image_counts[class_num-1] -= 1
                    image_names[image_name_idx] = 0
                    df_idx += 1
                    image_name_idx += 1

                else:

                    image_name_idx += 1
                    continue

        return fold_array, folds

    def create_k_folds(self, folds=5, seed=None, shuffle=True):

        image_names = np.asarray(self.get_labels_and_names()[1]).reshape(self.m, 1)
        labels = np.asarray(self.get_labels_and_names()[0], dtype=np.int).reshape(self.m, 1)

        skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=seed)

        X_train_sets = []
        y_train_labels = []
        X_test_sets = []
        y_test_labels = []
        for train_index, test_index in skf.split(image_names, labels):

            X_train, X_test = image_names[train_index], image_names[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            X_train_sets.append(X_train)
            y_train_labels.append(y_train)
            X_test_sets.append(X_test)
            y_test_labels.append(y_test)



        return X_train_sets, y_train_labels, X_test_sets, y_test_labels

if __name__ == '__main__':

    PATH = '/home/aparker/PycharmProjects/local_objects_PersonalBarkNet/data/train_1_2'

    dg = DatasetGenerator(path=PATH)

    k_folds = dg.stratified_kfold_cv()

    k_folds_sk = dg.create_k_folds()
    train = k_folds_sk[0]
    train_labels = k_folds_sk[1]

    print(len(k_folds_sk[0]))

    # print(k_folds[0].iloc[:, 2:5:2])