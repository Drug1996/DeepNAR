import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Recurrent neural network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size= hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Bidirectional recurrent neural network
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(36, 16, 5)
        # self.batchNorm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 16, 5)
        self.batchNorm2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, 5)
        self.batchNorm3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, 5)
        self.batchNorm4 = nn.BatchNorm1d(32)
        self.conv5 = nn.Conv1d(32, 64, 5)
        self.batchNorm5 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(64 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.batchNorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.batchNorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        # x = self.batchNorm3(x)
        x = self.pool(F.relu(self.conv4(x)))
        # x = self.batchNorm4(x)
        x = self.pool(F.relu(self.conv5(x)))
        # x = self.batchNorm5(x)
        x = x.view(-1, 64 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x