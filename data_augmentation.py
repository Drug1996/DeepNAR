import numpy as np
from scipy.interpolate import CubicSpline # for warping
from transforms3d.axangles import axangle2mat # for rotation


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
