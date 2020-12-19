import matplotlib.pyplot as plt
import _pickle as pickle
import os
import numpy as np


def plot_accuracy(results_path, num_folds, graph_loss=False, graph_train_accuracy=False,
                  graph_val_accuracy=False, graph_accuracy=True):

    results_names = os.listdir(results_path)
    results = []
    for i in range(0, num_folds):

        results_file = open(results_path + '/' + results_names[i], 'rb')

        results.append(pickle.load(results_file))

        results_file.close()

    if graph_accuracy:
        for j in range(num_folds):

            fig, ax = plt.subplots()

            x = np.arange(0, len(results[j]['val_categorical_accuracy']))

            val_accuracy, = ax.plot(x, results[j]['val_categorical_accuracy'], linestyle='--', marker='o')
            train_accuracy, = ax.plot(x, results[j]['categorical_accuracy'], linestyle='--', marker='o')

            plt.title(f'Fold {j+1} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            ax.legend((val_accuracy, train_accuracy), ('Validation Set Accuracy', 'Training Set Accuracy'))
            plt.savefig(f'./results_graphs/fold_{j+1}_results.png')
            plt.show()

    if graph_loss:
        for i in range(num_folds):

            fig, ax = plt.subplots()

            x = np.arange(0, len(results[j]['loss']))

            val_accuracy, = ax.plot(x, results[j]['val_loss'])
            train_accuracy, = ax.plot(x, results[j]['loss'])

            plt.title(f'Fold {j + 1} Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            ax.legend((val_accuracy, train_accuracy), ('Training Set Loss', 'Validation Set Loss'))
            plt.savefig(f'./results_graphs/fold_{j + 1}_loss.png')
            plt.show()

    if graph_train_accuracy:
        for j in range(num_folds):

            fig, ax = plt.subplots()

            x = np.arange(0, len(results[j]['categorical_accuracy']))

            train_accuracy, = ax.plot(x, results[j]['categorical_accuracy'])

            plt.title(f'Fold {j+1} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            ax.legend(train_accuracy, 'Training Set Accuracy')
            plt.savefig(f'./results_graphs/fold_{j+1}_train_results.png')
            plt.show()

    if graph_val_accuracy:
        for j in range(num_folds):

            fig, ax = plt.subplots()

            x = np.arange(0, len(results[j]['val_categorical_accuracy']))

            val_accuracy, = ax.plot(x, results[j]['val_categorical_accuracy'])

            plt.title(f'Fold {j+1} Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            ax.legend(val_accuracy, 'Validation Set Accuracy')
            plt.savefig(f'./results_graphs/fold_{j+1}_val_results.png')
            plt.show()



if __name__ == '__main__':

    plot_accuracy('./results', 4)

