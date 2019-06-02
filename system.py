import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import model
from network import BiRNN

class sliding_windows:
    def __init__(self, model, threshold=0.8, dim=0, size=[], step=10):
        self.model = model
        self.threshold = threshold
        self.dim = dim
        self.size = size
        self.step = step

    def init_marker(self, length):
        self.marker = []
        for i in range(length):
            self.marker.append([])

    def generate_windows_series(self, data):
        self.init_marker(data.shape[0])
        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.windows_series = []
        for size in self.size:
            self.windows_series.append(self.data.unfold(self.dim, size, self.step))
        return self.windows_series

    def one_sliding(self, windows):
        num = windows.shape[0]
        size = windows.shape[2]
        with torch.no_grad():
            for i in range(num):
                inputs = windows[i]
                inputs = inputs.permute(1,0)
                inputs = inputs.reshape(1, size, -1).to(model.device)
                outputs = self.model(inputs)
                probability , predicted = torch.max(outputs.data, 1)
                if probability.item() > self.threshold:
                    start = i * self.step
                    end = start + size
                    for j in range(start, end):
                        self.marker[j].append(predicted.item())

    def get_results(self):
        for i in range(len(self.marker)):
            c = Counter(self.marker[i])
            if 0 in c or 1 in c:
                if c[0] >= c[1]:
                    self.marker[i] = 0
                elif c[1] > c[0]:
                    self.marker[i] = 1
            elif 2 in c or 3 in c:
                if c[2] >= c[3]:
                    self.marker[i] = 2
                elif c[3] > c[2]:
                    self.marker[i] = 3
            else:
                self.marker[i] = -1
        c = Counter(self.marker)
        for i in range(3):
            if i not in c:
                c[i] = 0
        results = []
        if c[0] >= c[1]:
            results.append(0)
        elif c[1] > c[0]:
            results.append(1)
        if c[2] >= c[3]:
            results.append(2)
        elif c[3] > c[2]:
            results.append(3)

        return results

    def __call__(self, data):
        self.generate_windows_series(data)
        for windows in self.windows_series:
            self.one_sliding(windows)
        return self.get_results()


def model_load():
    system_model = BiRNN(model.input_size, model.hidden_size, model.num_layers, model.num_classes).to(model.device)
    system_model.load_state_dict(torch.load('DeepNAR_model.ckpt'))
    # print("Model's state_dict:")
    # for param_tensor in system_model.state_dict():
    #     print(param_tensor, '\t', system_model.state_dict()[param_tensor])
    return system_model


def data_load_threshold():
    pkl_file = open(model.DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    dataset0 = dataset['lin']
    dataset1 = dataset['zhong']
    test_x, test_t = [], []
    for label, action in enumerate(model.LABELS):
        test_x += dataset0[action]
        test_x += dataset1[action]
        test_t += [label] * len(dataset0[action])
        test_t += [label] * len(dataset1[action])
    for i in range(len(test_x)):
        test_x[i] = test_x[i][:, 1:]

    return test_x, test_t


def data_load():
    LABELS = ['rightright_continuous',
               'rightwrong_continuous',
               'wrongright_continuous',
               'wrongwrong_continuous']
    LABELS_VALUES = [[0,2], [0,3], [1,2], [1,3]]
    pkl_file = open(model.DATASET_NAME, 'rb')
    dataset = pickle.load(pkl_file)
    pkl_file.close()
    dataset0 = dataset['lin']
    dataset1 = dataset['zhong']
    test_x, test_t = [], []
    for label, action in zip(LABELS_VALUES, LABELS):
        test_x += dataset0[action]
        test_t += [label] * len(dataset0[action])
        test_x += dataset1[action]
        test_t += [label] * len(dataset1[action])
    for i in range(len(test_x)):
        test_x[i] = test_x[i][:, 1:]

    return test_x, test_t


def threshold():
    system_model = model_load()
    test_x, test_t = data_load_threshold()

    pred_prob = []
    pred_y = []
    with torch.no_grad():
        for inputs in test_x:
            inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
            inputs = inputs.reshape(1, -1, model.input_size).to(model.device)
            outputs = system_model(inputs)
            outputs = F.softmax(outputs,dim=1)
            probability, predicted = torch.max(outputs.data, 1)
            pred_prob.append(probability.item())
            pred_y.append(predicted.item())
    count = 0
    point = []
    threshold = 0.8
    for i in range(len(pred_y)):
        if pred_y[i] == test_t[i]:
            count += 1
        else:
            point.append(i)
    print('accuracy: ', count/len(pred_y))
    _x = list(range(len(pred_prob)))
    plt.bar(point, [pred_prob[i] for i in point], color='r', label='wrong prediction')
    x = [i for i in _x if i not in point]
    plt.bar(x, [pred_prob[i] for i in x], color='orange', label='right prediction')
    plt.plot([threshold]*len(pred_prob), label='threshold', linestyle='-')
    plt.xlim(0,160)
    plt.ylim(0,1.4)
    plt.xlabel('Sample No')
    plt.ylabel('Probability')
    plt.legend(loc='best')
    plt.show()


def main():
    system_model = model_load()
    sw = sliding_windows(model=system_model, threshold=0.8, dim=0, size=[60, 80, 100, 120], step=10)
    test_x, test_t = data_load()
    count = 0
    for i in range(len(test_x)):
        data = test_x[i]
        results = sw(data)
        if results[0] == test_t[i][0]:
            count += 1
        if results[1] == test_t[i][1]:
            count += 1
        print('results:', results, 'labels:', test_t[i])

    print('Accuracy: ', count/(len(test_t)*2) )


if __name__ == '__main__':
    main()