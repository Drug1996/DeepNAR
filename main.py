import pickle
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from network import RNN
from network import BiRNN
from visualization import variance_and_bias_analysis, plot_confusion_matrix, show, save, data_visual
from data_optimization import dimension_reduce, normalization

DATASET_NAME = 'dataset.pkl'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 0  # this parameter will be renewed in 'dataloader' function
input_size = 36  # including 6 channels of 6 IMU sensors, totally 36 channels
hidden_size = 64  # parameters for LSTM (Long Short Term Memory)
num_layers = 2  # the depth of Deep-RNNs
num_classes = 4
batch_size = 48
num_epochs = 10
learning_rate = 0.1
training_dev_test_ratio = [0.6, 0.5]
dimension_interval = 1

# Test parameters
TEST_NUM = 1
RANDOM_SEED_NUM = 0
TRIALS_NUM = 1
LABELS_NAME = {'standing': ['None', 'Right standing', 'Wrong standing'],
               'turning': ['None', 'Right turning', 'Wrong turning']}
LABELS = ['right_standing', 'wrong_standing', 'right_turning', 'wrong_turning']


def data_load():

    dataset_name0 = 'lin'
    dataset_name1 = 'zhong'
    actions = LABELS
    pkl_file = open(DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    dataset0 = dataset[dataset_name0]
    dataset1 = dataset[dataset_name1]

    max_t = max(dataset0['length_range'][1], dataset1['length_range'][1])
    global sequence_length
    sequence_length = max_t  # the longest length of sample

    # divide the dataset into train set, dev set and test set
    x_train, y_train, x_dev, y_dev, x_test, y_test = [], [], [], [], [], []
    for label, action in enumerate(actions):

        # lin
        labels = [label] * len(dataset0[action])
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(dataset0[action], labels, test_size=0.4)
        x_train += temp_train_x
        y_train += temp_train_y
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(temp_test_x, temp_test_y, test_size=0.5)
        x_dev += temp_train_x
        y_dev += temp_train_y
        x_test += temp_test_x
        y_test +=temp_test_y

        # zhong
        labels = [label] * len(dataset1[action])
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(dataset1[action], labels, test_size=0.4)
        x_train += temp_train_x
        y_train += temp_train_y
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(temp_test_x, temp_test_y, test_size=0.5)
        x_dev += temp_train_x
        y_dev += temp_train_y
        x_test += temp_test_x
        y_test += temp_test_y

    # pad the data samples
    for i in range(len(x_train)):
        x_train[i] = \
            np.pad(x_train[i][:, 1:], ((0, max_t - x_train[i].shape[0]), (0, 0)), 'constant', constant_values=0)
    for i in range(len(x_dev)):
        x_dev[i] = \
            np.pad(x_dev[i][:, 1:], ((0, max_t - x_dev[i].shape[0]), (0, 0)), 'constant', constant_values=0)
    for i in range(len(x_test)):
        x_test[i] = \
            np.pad(x_test[i][:, 1:], ((0, max_t - x_test[i].shape[0]), (0, 0)), 'constant', constant_values=0)

    # shuffle
    x_train, y_train = shuffle(x_train, y_train)
    x_dev, y_dev = shuffle(x_dev, y_dev)
    x_test, y_test = shuffle(x_test, y_test)

    # change dataset's data format
    x_train, y_train = \
        torch.from_numpy(np.array(x_train)).type(torch.FloatTensor),\
        torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    x_dev, y_dev = \
        torch.from_numpy(np.array(x_dev)).type(torch.FloatTensor),\
        torch.from_numpy(np.array(y_dev)).type(torch.LongTensor)
    x_test, y_test = \
        torch.from_numpy(np.array(x_test)).type(torch.FloatTensor),\
        torch.from_numpy(np.array(y_test)).type(torch.LongTensor)

    return x_train, y_train, x_dev, y_dev, x_test, y_test


# Load the data of sub-network
def dataloader(data_type='standing', training_dev_test_ratio=[0.6, 0.5]):
    # fix the random seed
    random.seed(RANDOM_SEED_NUM)

    X = []
    Y = []
    pkl_file = open(DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    dataset_size = len(dataset)

    MAX_T = 0
    for i in range(dataset_size):
        dataset[i][0] = dataset[i][0][:, 1:]
        MAX_T = max(MAX_T, dataset[i][0].shape[0])

    for i in range(dataset_size):
        X.append(np.pad(dataset[i][0], ((0, MAX_T-dataset[i][0].shape[0]), (0, 0)), 'constant', constant_values=0))
        if data_type == 'standing':
            if dataset[i][1] != 1:
                Y.append(0)
            else:
                if dataset[i][2] == 0:
                    Y.append(1)
                else:
                    Y.append(2)
        elif data_type == 'turning':
            if dataset[i][1] != 2:
                Y.append(0)
            else:
                if dataset[i][2] == 0:
                    Y.append(1)
                else:
                    Y.append(2)

    # divide the whole dataset as training dataset and test dataset
    index = list(range(dataset_size))

    # Get training index
    index_0 = [i for i in index if Y[i] == 0]
    index_0 = random.sample(index_0, int(len(index_0) * 0.5))  # reduce the number of class 0
    index_1 = [i for i in index if Y[i] == 1]
    index_2 = [i for i in index if Y[i] == 2]
    index_0_size, index_1_size, index_2_size = len(index_0), len(index_1), len(index_2)
    training_index = []
    training_index += random.sample(index_0, int(index_0_size*training_dev_test_ratio[0]))
    training_index += random.sample(index_1, int(index_1_size*training_dev_test_ratio[0]))
    training_index += random.sample(index_2, int(index_2_size*training_dev_test_ratio[0]))
    random.shuffle(training_index)  # shuffle the training data

    # Get dev index
    index_0 = [num for num in index_0 if num not in training_index]
    index_1 = [num for num in index_1 if num not in training_index]
    index_2 = [num for num in index_2 if num not in training_index]
    index_0_size, index_1_size, index_2_size = len(index_0), len(index_1), len(index_2)
    dev_index = []
    dev_index += random.sample(index_0, int(index_0_size * training_dev_test_ratio[1]))
    dev_index += random.sample(index_1, int(index_1_size * training_dev_test_ratio[1]))
    dev_index += random.sample(index_2, int(index_2_size * training_dev_test_ratio[1]))
    random.shuffle(dev_index)

    # Get test index
    index_0 = [num for num in index_0 if num not in dev_index]
    index_1 = [num for num in index_1 if num not in dev_index]
    index_2 = [num for num in index_2 if num not in dev_index]
    test_index = index_0 + index_1 + index_2
    random.shuffle(test_index)

    # Data selection and dimension reduction
    X_training = [dimension_reduce(X[i], dimension_interval) for i in training_index]
    Y_training = [Y[i] for i in training_index]
    X_dev = [dimension_reduce(X[i], dimension_interval) for i in dev_index]
    Y_dev = [Y[i] for i in dev_index]
    X_test = [dimension_reduce(X[i], dimension_interval) for i in test_index]
    Y_test = [Y[i] for i in test_index]

    # update the value of max sequence length after dimension reduction
    MAX_T = X_training[0].shape[0]
    global sequence_length
    sequence_length = MAX_T

    # Data normalization
    # X_training = normalization(X_training)
    # X_dev = normalization(X_dev)
    # X_test = normalization(X_test)

    # change the raw data form to pytorch data form

    X_training, Y_training = np.array(X_training), np.array(Y_training)
    X_dev, Y_dev = np.array(X_dev), np.array(Y_dev)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # # check the dimensions
    # print(X_training.shape)
    # print(Y_training.shape)
    # print(X_dev.shape)
    # print(Y_dev.shape)
    # print(X_test.shape)
    # print(Y_test.shape)

    X_training, Y_training = torch.from_numpy(X_training), torch.from_numpy(Y_training)
    X_dev, Y_dev = torch.from_numpy(X_dev), torch.from_numpy(Y_dev)
    X_test, Y_test = torch.from_numpy(X_test), torch.from_numpy(Y_test)

    return X_training.type(torch.FloatTensor), Y_training.type(torch.LongTensor),\
           X_dev.type(torch.FloatTensor), Y_dev.type(torch.LongTensor),\
           X_test.type(torch.FloatTensor), Y_test.type(torch.LongTensor)


def training_model(num, type='standing'):
    # torch.manual_seed(SEED)

    # Load the dataset
    # X_training, Y_training, X_dev, Y_dev, X_test, Y_test = \
    #     dataloader(data_type=type, training_dev_test_ratio=training_dev_test_ratio)
    X_training, Y_training, X_dev, Y_dev, X_test, Y_test = data_load()

    # print(X_training.shape)
    # print(X_dev.shape)
    # print(X_test.shape)
    # print(Y_training.data)
    # print(Y_dev.data)
    # print(Y_test.data)

    modellist = []
    training_losseslist = []
    test_accuracieslist = []
    training_losses = []
    test_accuracies = []
    y_truelist = []
    y_predlist = []
    y_true = []
    y_pred = []

    for t in range(TEST_NUM):
        try:
            # Create the model
            model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            training_losses = []
            test_accuracies = []

            # Train the model
            X_training = X_training.reshape(-1, batch_size, sequence_length, input_size)
            Y_training = Y_training.reshape(-1, batch_size)
            X_dev = X_dev.reshape(-1, 1, sequence_length, input_size)
            Y_dev = Y_dev.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1, sequence_length, input_size)
            Y_test = Y_test.reshape(-1, 1)

            # print('Y_dev', Y_dev)
            # print('Y_test', Y_test)

            total_step = len(X_training)  # how many batches for one epoch
            for epoch in range(num_epochs):
                for i in range(total_step):
                    inputs = X_training[i]
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    labels = Y_training[i]
                    labels = labels.reshape(-1)
                    labels = labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    training_loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    training_loss.backward()
                    optimizer.step()

                if (epoch + 1) % 5 == 0:
                    print('Trials [{}/{}], Epoch [{}/{}], Loss: {:.4f}'
                          .format(t + 1, TEST_NUM, epoch + 1, num_epochs, training_loss.item()))

                    # Get the value of loss
                    training_losses.append(training_loss.item())

                    # Test the model on dev set
                    with torch.no_grad():
                        y_true = []
                        y_pred = []
                        correct = 0
                        total = 0
                        for j in range(len(X_dev)):
                            inputs = X_dev[j]
                            inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                            labels = Y_dev[j]
                            labels = labels.reshape(-1)
                            labels = labels.to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            y_true.append(labels.item())
                            y_pred.append(predicted.item())
                        test_accuracies.append(correct/total)

        except KeyboardInterrupt:
            print('Stop!')

        # print('dev true:', y_true)
        # print('dev pred:', y_pred)
        modellist.append(model)
        training_losseslist.append(training_losses)
        test_accuracieslist.append(test_accuracies)
        y_truelist.append(y_true)
        y_predlist.append(y_pred)

    # Print accuracy of the model
    accuracy = []
    for item in test_accuracieslist:
        accuracy.append(item[-1] * 100)
    max_accuracy = max(accuracy)
    print('Dev accuracy of the No.{} model on dev action samples: {} %'.format(num+1, max_accuracy))

    # Show or save the graph of variance and bias analysis, and confusion matrix graph
    variance_and_bias_analysis(training_losseslist, test_accuracieslist)
    save('trials' + str(num) + '_loss_accuracy' + '.png')
    # plot_confusion_matrix(y_truelist, y_predlist, LABELS_NAME[type])
    plot_confusion_matrix(y_truelist, y_predlist, LABELS)
    save('trials_' + str(num) + '_confusion_matrix' + '.png')

    return max_accuracy, modellist[accuracy.index(max_accuracy)], X_test, Y_test


def test_model(model, X_test, Y_test, type='standing'):
    # Test the model on test set
    with torch.no_grad():
        y_true = []
        y_pred = []
        correct = 0
        total = 0
        for j in range(len(X_test)):
            inputs = X_test[j]
            inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
            labels = Y_test[j]
            labels = labels.reshape(-1)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.append(labels.item())
            y_pred.append(predicted.item())

        # print('test y_true', y_true)
        # print('test y_pred', y_pred)
        print('Final accuracy is {} %'.format((correct/total)*100))
        # plot_confusion_matrix([y_true], [y_pred], LABELS_NAME[type])
        plot_confusion_matrix([y_true], [y_pred], LABELS)
        save('test_confusion_matrix' + '.png')


def main(num, type):
    accuracylist = []
    modellist = []
    X_test = np.array([])
    Y_test = np.array([])
    for i in range(num):
        accuracy, model, X_test, Y_test = training_model(i, type)
        accuracylist.append(accuracy)
        modellist.append(model)
    index = accuracylist.index(max(accuracylist))
    print('Choose No.{} model as test model.'.format(index+1))
    best_model = modellist[index]
    test_model(best_model, X_test, Y_test, type)


if __name__ == '__main__':
    main(num=TRIALS_NUM, type='turning')
    # data_visual()
    # data_load()
