# Other Libs
import numpy as np
import os
import copy
import pandas as pd


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

            class_dist[i-1] = len(labels[labels == i])/self.m

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

            # Create an array for the amount of images for each class, need to make a copy because it has to be
            # reset for each fold. Need to reshape to make it a row vector for element-wise mult with the arange.
            class_image_counts = np.multiply(copy.deepcopy(stratified_image_count.reshape(1, len(classes))),
                                             np.arange(1, len(classes)+1))

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
                if class_image_counts[0][class_num-1] != 0:

                    fold_array.loc[df_idx, f'fold {fold}'] = image_names[image_name_idx]
                    fold_array.loc[df_idx, f'fold {fold} labels'] = class_num

                    class_image_counts[0][class_num-1] -= class_num

                    image_names[image_name_idx] = 0
                    df_idx += 1
                    image_name_idx += 1

                else:

                    image_name_idx += 1
                    continue

        return fold_array, folds
