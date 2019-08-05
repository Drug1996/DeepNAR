from system import data_load # load continuous action data
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle


class filter:
    def __init__(self):
        pass

    def lowPass(self, data):
        b, a = signal.butter(5, 0.2, 'lowpass')
        return signal.filtfilt(b, a, data)

def show_image(data):
    # x's unit is ms, y is data signal with the size of (time length, 36 channels)
    for (x, y) in data:
        plt.figure()
        # for i in range(y.shape[1]):
        #     plt.plot(x/1000.0, y[:,i], label=str(i))
        plt.plot(x / 1000.0, y)
        plt.scatter(x / 1000.0, y, marker='*')
        plt.plot(x / 1000.0, np.array([np.max(y)*0.55]*len(y)))
        plt.plot(np.array([getDivPoint(x,y)/1000.0]*2), np.array([np.min(y), np.max(y)]))
        plt.xlabel('time/sec')
        # plt.legend()
    plt.show()

def func(x, A, B):
    return A*x + B

def getDivPoint(x, y):
    t = np.max(y)*0.55 # threshold
    flag = 0
    r = 5 # window range
    p = 0
    for i in range(len(y)):
        if y[i] > t:
            flag = 1
        elif flag and y[i] < t:
            A, B = curve_fit(func, x[i:i+r], y[i:i+r])[0]
            if A > 0:
                p = x[i]
                break
    return p

if __name__ == '__main__':
    test_x, test_t = data_load()
    f = filter()
    data_x = []
    data_y = []
    for i in range(len(test_x)):
        y = test_x[i]
        y_ = f.lowPass(y[:,16])
        x = np.array([20 * i for i in range(len(y))])
        p = getDivPoint(x, y_)
        # plt.figure()
        # plt.plot(x / 1000.0, y)
        # plt.plot(np.array([p / 1000.0] * 2), np.array([np.min(y), np.max(y)]), color='k', linewidth=2, linestyle=':')
        # plt.ylim(np.min(y), np.max(y))
        # plt.figure()
        # plt.subplot(121)
        # plt.plot(x[0:int(p/20+1)], y[0:int(p/20+1)])
        data_x.append(y[0:int(p/20+1)])
        data_y.append(test_t[i][0])
        # plt.subplot(122)
        # plt.plot(x[int(p/20+1):len(x)], y[int(p/20+1):len(x)])
        data_x.append(y[int(p/20+1):len(x)])
        data_y.append(test_t[i][1])
        # plt.show()
    output = open('divider.pkl', 'wb')
    pickle.dump([data_x, data_y], output)
    output.close()



