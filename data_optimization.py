import numpy as np


def dimension_reduce(x, interval = 2):
    return x[0:len(x):interval]


def normalization(X):
    max_value = X[0].max(axis=0)
    min_value = X[0].min(axis=0)
    for x in X:
        max_value = np.maximum(max_value, x.max(axis=0))
        min_value = np.minimum(min_value, x.min(axis=0))
    for i in range(len(X)):
        X[i] = (X[i] - min_value)/(max_value - min_value)
    return X

