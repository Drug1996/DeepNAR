import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from preprocess import ACTION_TYPE as act
from preprocess import CORRECTNESS as cor
from data_optimization import dimension_reduce, normalization

FONTSIZE = 12


def data_visual(dataset_name='dataset_lin.pkl'):
    pkl_file = open(dataset_name, 'rb')
    dataset = pickle.load(pkl_file)
    show_image(dataset[0:80:20])
    print('debug')
    pkl_file.close()


def show_image(data):
    plt.figure(figsize=(14, 12))
    figure_num = len(data)
    X = []
    Y = []

    for i in range(figure_num):
        X.append(data[i][0][:, 0])
        Y.append(data[i][0][:, 1:])

    # Y = normalization(Y)

    for i in range(figure_num):
        x = X[i]
        y = Y[i]
        x = dimension_reduce(x, 4)
        y = dimension_reduce(y, 4)
        action_type = data[i][1]
        action_type = list(act.keys())[list(act.values()).index(action_type)]
        correctness = data[i][2]
        correctness = list(cor.keys())[list(cor.values()).index(correctness)]
        # print(action_type, correctness)
        # plt.subplot(4, (figure_num/4), i+1)
        plt.subplot(2, 2, i + 1)
        plt.xlabel('time/sec')
        plt.plot(x/1000.0, y)
        plt.title(action_type+' '+correctness)
    show()


def variance_and_bias_analysis(training_losseslist, test_accuracieslist):
    fig = plt.figure(figsize=(8, 6))
    trials_num = len(training_losseslist)
    # print(training_losseslist)
    # print(test_accuracieslist)
    for i in range(trials_num):
        x1 = list(range(1, len(training_losseslist[i])+1))
        y1 = training_losseslist[i]
        x2 = list(range(1, len(test_accuracieslist[i]) + 1))
        y2 = test_accuracieslist[i]
        # ax1 = fig.add_subplot(2, int(np.ceil(trials_num/2.0)), i+1)
        ax1 = fig.add_subplot(111)
        ax1.plot(x1, y1, marker='*', label='training loss')
        ax1.set_xlabel(r'$batches\ /\ times$')
        ax1.set_ylabel(r'$Loss$')
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(x2, y2, 'r', marker='*', label='test accuracy')
        ax2.set_ylim(0, 1)
        ax2.set_yticks(np.arange(0, 1.1, 0.1))
        ax2.set_ylabel(r'$Accuracy$')
        ax2.legend(loc='upper right')
        plt.title('Losses Analysis')
        plt.grid()


def plot_confusion_matrix(y_truelist, y_predlist, labels_name):
    plt.figure(figsize=(12, 10))
    trials_num = len(y_truelist)
    # print(y_truelist)
    # print(y_predlist)
    for i in range(trials_num):
        # plt.subplot(2, int(np.ceil(trials_num/2.0)), i+1)
        plt.subplot(1, 1, 1)
        y_true = np.array(y_truelist[i])
        y_pred = np.array(y_predlist[i])
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        # print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
        x_locations = np.array(range(len(labels_name)))
        plt.xticks(x_locations, labels_name, fontsize=FONTSIZE)
        plt.yticks(x_locations, labels_name, fontsize=FONTSIZE)
        plt.colorbar()
        plt.ylabel('True label', fontsize=FONTSIZE)
        plt.xlabel('Predicted label', fontsize=FONTSIZE)
        plt.title('Confusion Matrix', fontsize=FONTSIZE)


def show():
    plt.show()


def save(name):
    plt.savefig(name)


if __name__ == '__main__':
    y_true = [[0,0,0,0,1,1,1,1,2,2,2,2]]
    y_pred = [[2,0,0,0,1,1,1,1,2,2,2,2]]
    name = ['None', 'Right turning', 'Wrong turning']
    plot_confusion_matrix(y_true, y_pred, name)
    show()
