import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from network import RNN
from network import BiRNN
from preprocess import ACTION_TYPE as act
from preprocess import CORRECTNESS as cor

DATASET_NAME = 'dataset_lin.pkl'

# Device configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 0  # this parameter will be renewed in 'dataloader' function
input_size = 36  # including 6 channels of 6 IMU sensors, totally 36 channels
hidden_size = 64  # parameters for LSTM (Long Short Term Memory)
num_layers = 2  # the depth of Deep-RNNs
num_classes = 3
batch_size = 45
num_epochs = 120
learning_rate = 0.1
training_test_ratio = 0.75

# Test parameters
TEST_NUM = 1


def visialized():
    pkl_file = open(DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    show_image(dataset[0:12])
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
    plt.show()


def loss_analysis(losseslist):
    plt.figure(figsize=(10, 5))
    trials_num = len(losseslist)
    for i in range(trials_num):
        plt.subplot(1, trials_num, i+1)
        x = list(range(1, len(losseslist[i])+1))
        y = losseslist[i]
        plt.plot(x, y, marker='*')
        plt.xlabel(r'$batches\ /\ times$')
        plt.ylabel(r'$loss$')
        plt.title('Losses Analysis for Trial ' + str(i))


def variance_and_bias_analysis():
    pass


# Load the data sub-network
def dataloader(data_type = 'standing', training_test_ratio = 0.75):
    X = []
    Y = []
    pkl_file = open(DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    dataset_size = len(dataset)
    # print(dataset_size)

    MAX_T = 0
    for i in range(dataset_size):
        dataset[i][0] = dataset[i][0][:, 1:]
        MAX_T = max(MAX_T, dataset[i][0].shape[0])

    # renew the parameter of sequence length
    global sequence_length
    sequence_length = MAX_T

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
    index = [i for i in range(len(X))]
    index_0_ori = [i for i in index if Y[i] == 0]
    index_0 = random.sample(index_0_ori, int(len(index_0_ori) * 0.5))
    index_1 = [i for i in index if Y[i] == 1]
    index_2 = [i for i in index if Y[i] == 2]

    # check the length of index_0 - index_2
    index_0_size, index_1_size, index_2_size = len(index_0), len(index_1), len(index_2)
    # print(index_0_size, index_1_size, index_2_size)

    training_index = []
    training_index += random.sample(index_0, int(index_0_size*training_test_ratio))
    training_index += random.sample(index_1, int(index_1_size*training_test_ratio))
    training_index += random.sample(index_2, int(index_2_size*training_test_ratio))
    test_index = []
    for num in index_0 + index_1 + index_2:
        if num not in training_index:
            test_index.append(num)
    X_training = [X[i] for i in training_index]
    Y_training = [Y[i] for i in training_index]
    X_test = [X[i] for i in test_index]
    Y_test = [Y[i] for i in test_index]

    # change the raw data form to pytorch data form
    X_training, Y_training, X_test, Y_test = np.array(X_training), np.array(Y_training), np.array(X_test), np.array(Y_test)

    # check the dimensions
    # print(X_training.shape)
    # print(Y_training.shape)
    # print(X_test.shape)
    # print(Y_test.shape)

    X_training, Y_training, X_test, Y_test = torch.from_numpy(X_training), torch.from_numpy(Y_training), torch.from_numpy(X_test), torch.from_numpy(Y_test)
    return X_training.type(torch.FloatTensor), Y_training.type(torch.LongTensor), X_test.type(torch.FloatTensor), Y_test.type(torch.LongTensor)


def main():
    accuracy = 0
    losseslist = []

    for t in range(TEST_NUM):
        losses = []
        X_training, Y_training, X_test, Y_test = dataloader(data_type='turning', training_test_ratio=training_test_ratio)

        model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        X_training = X_training.reshape(-1, batch_size, sequence_length, input_size)
        Y_training = Y_training.reshape(-1, batch_size)
        X_test = X_test.reshape(-1, 1, sequence_length, input_size)
        Y_test = Y_test.reshape(-1, 1)
        total_step = len(X_training)
        for epoch in range(num_epochs):
            for i in range(total_step):
                inputs = X_training[i]
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                labels = Y_training[i]
                labels = labels.reshape(-1)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                #print(outputs.shape, labels.data)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0:
                    print('Trials [{}/{}], Epoch [{}/{}]], Loss: {:.4f}'
                        .format(t + 1, TEST_NUM, epoch + 1, num_epochs, loss.item()))

                    losses.append(loss.item())

        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for i in range(len(X_test)):
                inputs = X_test[i]
                inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                labels = Y_test[i]
                labels = labels.reshape(-1)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy += 100 * correct / total
        losseslist.append(losses)

    print('Test Accuracy of the model on the 20 action samples: {} %'.format(accuracy/TEST_NUM))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

    loss_analysis(losseslist=losseslist)
    variance_and_bias_analysis()
    plt.show()


if __name__ == '__main__':
    # visialized()
    main()
