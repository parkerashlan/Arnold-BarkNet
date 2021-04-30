import pandas as pd
import os
import subprocess


def data_labeller(path_to_labels, path_to_imgs, start_tree_numbers):
    """ Automatically labels images in the directory specified. Only based on the the current four classes
    with images.

    :param
        path_to_labels (str): The path to the text file with the labels and number of pics in it.
        path_to_imgs (str): The path to the directory with the images which need to be labelled.
        start_tree_numbers (list of ints): A list of the starting tree number for each class.

    :return:
        Doesn't return anything but modifies the image names in the directory.
    """
    label_df = pd.read_csv(path_to_labels, header=None, names=['class_names', 'num_pics'])

    # need to index to -1 to cut off last entry, an artifact of splitting on the newlines
    img_names = subprocess.check_output(['ls', path_to_imgs], universal_newlines=True).split('\n')[0:-1]

    tree_nums = {'PON': start_tree_numbers[0], 'CAL': start_tree_numbers[1],
                 'WFIR': start_tree_numbers[2], 'VOAK': start_tree_numbers[3]}
    class_num = {'PON': 1, 'CAL': 2, 'WFIR': 3, 'VOAK': 4}

    start = 0
    fin = label_df['num_pics'][0]

    for i in range(len(label_df)):

        images_set = img_names[start:fin]
        class_name = label_df['class_names'][i]

        for j in range(len(images_set)):

            class_number = class_num[class_name]
            tree_numbers = tree_nums[class_name]

            os.rename(path_to_imgs + '/' + images_set[j],
                      path_to_imgs + '/' + f'{class_number}_0_{class_name}_OnePlus7_{tree_numbers:04d}_{j:04d}.jpg')

        tree_nums[class_name] += 1

        if i != len(label_df)-1:  # avoid index error
            start = fin
            fin = fin + label_df['num_pics'][i+1]


if __name__ == "__main__":
    path_labels = '../local_objects_PersonalBarkNet/raw_data/12_10_2020_labels.txt'
    path_imgs = '../local_objects_PersonalBarkNet/raw_data/test_imgs'
    start_tree_nums = [154, 41, 13, 1]  # in order of class numbers

    data_labeller(path_labels, path_imgs, start_tree_nums)
