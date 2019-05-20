import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from network import BiRNN, RNN
from visualization import variance_and_bias_analysis, plot_confusion_matrix, show, save, data_visual
from data_optimization import dimension_reduce, normalization

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 0  # this parameter will be renewed in 'dataloader' function
input_size = 36  # including 6 channels of 6 IMU sensors, totally 36 channels
hidden_size = 256  # parameters for LSTM (Long Short Term Memory)
num_layers = 2  # the depth of Deep-RNNs
num_classes = 4
batch_size = 48
num_epochs = 100
learning_rate = 0.01
dimension_interval = 1

# Test parameters
TEST_NUM = 1
RANDOM_SEED_NUM = 0
TRIALS_NUM = 5
LABELS = ['right_standing', 'wrong_standing', 'right_turning', 'wrong_turning']
DATASET_NAME = 'dataset.pkl'


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
            train_test_split(dataset0[action], labels, test_size=0.4, random_state=RANDOM_SEED_NUM)
        x_train += temp_train_x
        y_train += temp_train_y
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(temp_test_x, temp_test_y, test_size=0.5, random_state=RANDOM_SEED_NUM)
        x_dev += temp_train_x
        y_dev += temp_train_y
        x_test += temp_test_x
        y_test +=temp_test_y

        # zhong
        labels = [label] * len(dataset1[action])
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(dataset1[action], labels, test_size=0.4, random_state=RANDOM_SEED_NUM)
        x_train += temp_train_x
        y_train += temp_train_y
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(temp_test_x, temp_test_y, test_size=0.5, random_state=RANDOM_SEED_NUM)
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
    x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED_NUM+1)
    x_dev, y_dev = shuffle(x_dev, y_dev, random_state=RANDOM_SEED_NUM+2)
    x_test, y_test = shuffle(x_test, y_test, random_state=RANDOM_SEED_NUM+3)

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


def training_model(num):
    # torch.manual_seed(RANDOM_SEED_NUM)

    # Load the dataset
    X_training, Y_training, X_dev, Y_dev, X_test, Y_test = data_load()

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
            train_dataset = TensorDataset(X_training, Y_training)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dev_dataset = TensorDataset(X_dev, Y_dev)
            dev_loader = DataLoader(dev_dataset)
            test_dataset = TensorDataset(X_test, Y_test)
            test_loader = DataLoader(test_dataset)

            # total_step = len(train_loader)  # how many batches for one epoch
            for epoch in range(num_epochs):
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                    labels = labels.reshape(-1).to(device)

                    # Forward pass
                    outputs = model(inputs)
                    training_loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    training_loss.backward()
                    optimizer.step()

                if (epoch + 1) % 5 == 0:
                    print('Test [{}/{}], Epoch [{}/{}], Loss: {:.4f}'
                          .format(t + 1, TEST_NUM, epoch + 1, num_epochs, training_loss.item()))

                    # Get the value of loss
                    training_losses.append(training_loss.item())

                    # Test the model on dev set
                    with torch.no_grad():
                        y_true = []
                        y_pred = []
                        correct = 0
                        total = 0
                        for j, (inputs, labels) in enumerate(dev_loader):
                            inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
                            labels = labels.reshape(-1).to(device)
                            outputs = model(inputs)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()
                            y_true.append(labels.item())
                            y_pred.append(predicted.item())
                        test_accuracies.append(correct/total)

        except KeyboardInterrupt:
            print('Stop!')

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
    save('trials' + str(num+1) + '_loss_accuracy' + '.png')
    plot_confusion_matrix(y_truelist, y_predlist, LABELS)
    save('trials_' + str(num+1) + '_confusion_matrix' + '.png')

    return max_accuracy, modellist[accuracy.index(max_accuracy)], test_loader


def test_model(model, test_loader):
    # Test the model on test set
    with torch.no_grad():
        y_true = []
        y_pred = []
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.reshape(-1).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.append(labels.item())
            y_pred.append(predicted.item())

        print('Final accuracy is {} %'.format((correct/total)*100))
        plot_confusion_matrix([y_true], [y_pred], LABELS)
        save('test_confusion_matrix' + '.png')

    # Save the modal checkpoint
    torch.save(model.state_dict(), 'DeepNAR_model.ckpt')


def main(num):
    accuracylist = []
    modellist = []
    for i in range(num):
        accuracy, model, test_loader = training_model(i)
        accuracylist.append(accuracy)
        modellist.append(model)
    index = accuracylist.index(max(accuracylist))
    print('Choose No.{} model as test model with {} % Dev accuracy.'.format(index+1, max(accuracylist)))
    best_model = modellist[index]
    test_model(best_model, test_loader)


if __name__ == '__main__':
    main(num=TRIALS_NUM)