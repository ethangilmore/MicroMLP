import numpy as np

def mean_squared_error(y_pred, y_true):
    y = np.mean((y_pred - y_true) ** 2)
    dy = 2 * (y_pred - y_true) / y_pred.size
    return y, dy

def cross_entropy(y_pred, y_true):
    y = -sum(y_true * np.log(y_pred))
    dy = -y_true / y_pred
    return y, dy