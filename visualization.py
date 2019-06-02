import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from data_optimization import dimension_reduce, normalization
from data_augmentation import jitter, crop

FONTSIZE = 12


def data_visual(sample):
    datafile = open(sample)
    data_matrix = []
    datafile.readline() # omit yje first line
    data = datafile.readlines()
    for line in data:
        data_matrix.append(list(map(eval, line.split()[1:])))
    datafile.close()
    data_matrix = np.array(data_matrix)
    data = []
    x, y = data_matrix[:, 0], data_matrix[:, 1:]
    # y = normalization([y])[0]
    data.append((x, y))

    # # test for data augmentation
    # data.append((x, jitter(y)))
    y, _ = crop([y], [0])
    print(_)
    for i in range(len(y)):
        data.append((x[0:len(y[i])], y[i]))
    show_image(data)


def show_image(data):
    for (x, y) in data:
        plt.figure()
        plt.plot(x/1000.0, y)
        plt.xlabel('time/sec')
    show()


def variance_and_bias_analysis(training_losseslist, test_accuracieslist):
    fig = plt.figure(figsize=(8, 6))
    test_num = len(training_losseslist)
    # print(training_losseslist)
    # print(test_accuracieslist)
    for i in range(test_num):
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
    plt.figure(figsize=(10, 8))
    test_num = len(y_truelist)
    for i in range(test_num):
        # plt.subplot(2, int(np.ceil(trials_num/2.0)), i+1)
        plt.subplot(1, 1, 1)
        y_true = np.array(y_truelist[i])
        y_pred = np.array(y_predlist[i])
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        # print(cm)
        plt.title('Confusion Matrix', fontsize=FONTSIZE)
        ax = sns.heatmap(cm, annot=True, cmap='Blues',
                         xticklabels=labels_name, yticklabels=labels_name)
        ax.set_ylabel('True label', fontsize=FONTSIZE)
        ax.set_xlabel('Predicted label', fontsize=FONTSIZE)


def show():
    plt.show()


def save(name):
    plt.savefig(name)


if __name__ == '__main__':
    # y_true = [[0,0,0,0,1,1,1,1,2,2,2,2]]
    # y_pred = [[2,0,0,0,1,1,1,1,2,2,2,2]]
    # name = ['None', 'Right turning', 'Wrong turning']
    # plot_confusion_matrix(y_true, y_pred, name)
    # show()

    # sample adjustment
    lin_samples = ['standing_right_2_2019_04_20_06_02_46.txt',
                   'turning_right_2_2019_04_20_06_23_51.txt',
                   'continuous_rightright_1_2019_04_20_06_40_43.txt']
    for sample in lin_samples:
        os.chdir('lin')
        data_visual(sample)
        os.chdir('..')
    # zhong_samples = ['standing_right_1_2019_04_20_06_55_20.txt']
    # for sample in zhong_samples:
    #     os.chdir('zhong')
    #     data_visual(sample)
    #     os.chdir('..')

