import pickle
import matplotlib.pyplot as plt
from preprocess import ACTION_TYPE as act
from preprocess import CORRECTNESS as cor


def data_visual(dataset_name='dataset_lin.pkl'):
    pkl_file = open(dataset_name, 'rb')
    dataset = pickle.load(pkl_file)
    show_image(dataset[0:12])
    print('debug')
    pkl_file.close()


def show_image(data):
    figure_num = len(data)
    for i in range(figure_num):
        x = data[i][0][:, 0]
        y = data[i][0][:, 1:]
        action_type = data[i][1]
        action_type = list(act.keys())[list(act.values()).index(action_type)]
        correctness = data[i][2]
        correctness = list(cor.keys())[list(cor.values()).index(correctness)]
        # print(action_type, correctness)
        plt.subplot(4, (figure_num/4), i+1)
        plt.plot(x, y)
        plt.title(action_type+' '+correctness, fontsize='x-small')
    show()


def variance_and_bias_analysis(training_losseslist, test_accuracieslist):
    plt.figure(figsize=(10, 5))
    trials_num = len(training_losseslist)
    for i in range(trials_num):
        plt.subplot(1, trials_num, i+1)
        x1 = list(range(1, len(training_losseslist[i])+1))
        y1 = training_losseslist[i]
        x2 = list(range(1, len(test_accuracieslist[i]) + 1))
        y2 = test_accuracieslist[i]
        plt.plot(x1, y1, marker='*', label='training loss')
        plt.plot(x2, y2, marker='*', label='test accuracy')
        plt.xlabel(r'$batches\ /\ times$')
        plt.ylabel(r'$loss$')
        plt.legend()
        plt.grid()
        plt.title('Losses Analysis')


def show():
    plt.show()


def save(name):

    plt.savefig(name)
