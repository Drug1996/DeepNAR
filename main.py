import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from network import RNN
from network import BiRNN
from preprocess import ACTION_TYPE as act
from preprocess import CORRECTNESS as cor

# Device configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 280
input_size = 36
hidden_size = 32
num_layers = 2
num_classes = 3
batch_size = 1
num_epochs = 4
learning_rate = 0.002

def visialized():
    pkl_file = open('dataset_lin.pkl', 'rb')
    dataset = pickle.load(pkl_file)
    showimage(dataset[0:12])
    pkl_file.close()

def showimage(data):
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
        plt.plot(x,y)
        plt.title(action_type+' '+correctness, fontsize='x-small')
    plt.show()

def standing_dataloader():
    X = []
    Y = []
    pkl_file = open('dataset_lin.pkl', 'rb')
    dataset = pickle.load(pkl_file)

    MAX_T = 0
    for i in range(len(dataset)):
        dataset[i][0] = dataset[i][0][:, 1:]
        MAX_T = max(MAX_T, dataset[i][0].shape[0])
    for i in range(len(dataset)):
        X.append(np.pad(dataset[i][0], ((0, MAX_T-dataset[i][0].shape[0]), (0, 0)), 'constant', constant_values=0))
        if dataset[i][1] != 1:
            Y.append(0)
        else:
            if dataset[i][2] == 0:
                Y.append(1)
            else:
                Y.append(2)
    index = [i for i in range(len(X))]
    training_index = random.sample(index, 60)
    test_index = []
    for num in index:
        if num not in training_index:
            test_index.append(num)
    X_training = [X[i] for i in training_index]
    Y_training = [Y[i] for i in training_index]
    X_test = [X[i] for i in test_index]
    Y_test = [Y[i] for i in test_index]
    X_training, Y_training, X_test, Y_test = np.array(X_training), np.array(Y_training), np.array(X_test), np.array(Y_test)

    # check the dimensions
    # print(X_training.shape)
    # print(Y_training.shape)
    # print(X_test.shape)
    # print(Y_test.shape)

    X_training, Y_training, X_test, Y_test = torch.from_numpy(X_training), torch.from_numpy(Y_training), torch.from_numpy(X_test), torch.from_numpy(Y_test)
    return X_training.type(torch.FloatTensor), Y_training.type(torch.LongTensor), X_test.type(torch.FloatTensor), Y_test.type(torch.LongTensor)


def main():
    X_training, Y_training, X_test, Y_test = standing_dataloader()

    model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(X_training)
    for epoch in range(num_epochs):
        for i in range(len(X_training)):
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

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

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

        print('Test Accuracy of the model on the 20 actions: {} %'.format(100 * correct / total))

        # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__ == '__main__':
    # test()
    # visialized()
    main()
