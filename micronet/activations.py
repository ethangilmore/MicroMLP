import numpy as np

def relu(x):
    y = np.maximum(0, x)
    def backward(dy):
        dx = dy * (y > 0)
        return dx
    return y, backward

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    def backward(dy):
        dx = dy * y * (1 - y)
        return dx
    return y, backward

def softmax(x):
    y = np.exp(x) / sum(np.exp(x))
    def backward(dy):
        J = -np.outer(y, y)
        np.fill_diagonal(J, y * (1 - y))
        return dy @ J
    return y, backward