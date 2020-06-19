import numpy as np

def Softmax(X):
    _, n_classes = X.shape
    return np.exp(X) / np.repeat(np.sum(np.exp(X), 1).reshape((-1, 1)), n_classes, axis=1)

def dSoftmax(X):
    raise NotImplementedError

def derivate(fn):
    functions = {
        Relu: dRelu,
        Softmax: dSoftmax,
        LeakyRelu: dLeakyRelu,
        np.tanh: dTanh
    }
    return functions[fn]

def Sigmoid(X):
    return 1 / (1 + np.exp(-X))


def dSigmoid(X):
    s = Sigmoid(X)
    return s * (1 - s)

def Relu(X):
    X[X < 0] = 0
    return X

def dRelu(X):
    X[X >= 0] = 1
    X[X < 0] = 0
    return X


def LeakyRelu(X, slope=0.01):
    X[X < 0] *= slope
    return X


def dLeakyRelu(X, slope=0.01):
    X[X >= 0] = 1
    X[X < 0] = slope
    return X

def dTanh(X):
    return 1 - np.tanh(X) ** 2

    