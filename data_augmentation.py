import torch
import numpy as np
from scipy.interpolate import CubicSpline # for warping
from transforms3d.axangles import axangle2mat # for rotation

def crop(x, y):
    new_x, new_y = [], []
    for i in range(len(x)):
        length = x[i].shape[0]
        size = length - 20
        step = 10
        samples = torch.from_numpy(x[i]).type(torch.FloatTensor)
        samples = samples.unfold(0, size, step).permute(0,2,1).numpy()
        for sample in samples:
            new_x.append(sample)
            new_y.append(y[i])

    return new_x, new_y


def jitter(x, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


def scaling(x, sigma=0.1):
    pass


def magnitude_warping(x, sigma=0.2, knot=4):
    pass


def time_warping(x, sigma=0.2):
    pass


def rotation(x):
    pass
