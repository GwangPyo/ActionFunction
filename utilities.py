import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    summation = np.sum(exp_x)
    y = exp_x / summation
    return y
