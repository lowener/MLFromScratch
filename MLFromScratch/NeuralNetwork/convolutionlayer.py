import numpy as np
from scipy.signal import fftconvolve
from MLFromScratch.Tools import (
    ScoreMulticlass,
    Softmax,
    Relu,
    LeakyRelu,
    cross_entropy,
    cross_entropy,
    derivate,
)
from .neurallayer import BaseLayer


class ConvolutionLayer(BaseLayer):
    def __init__(
        self,
        n_filters,
        kernel_size,
        dimensions=1,
        strides=1,
        padding="valid",
        activation=Relu,
        bias=True,
        derivative=None,
        l1_ratio=0.0,
        l2_ratio=0.0,
    ):
        if type(kernel_size) == int:
            kernel_size = [n_filters] + [kernel_size] * dimensions
        self.W = np.random.uniform(size=kernel_size, high=0.1, low=-0.1)
        self.bias = bias
        if self.bias:
            self.B = np.zeros((1, kernel_size[-1]))
        self.activation = activation
        if derivative is None:
            self.derivative = derivate(activation)
        else:
            self.derivative = derivative
        self.n_filters = n_filters
        self.strides = strides
        self.padding = padding
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.BGrad = np.zeros_like(self.B)
        self.WGrad = np.zeros_like(self.W)

    def forward(self, X):
        channel_size = X.shape[0]
        self.ins = X
        preds = X.dot(self.W)
        if self.bias:
            preds += self.B
        self.preds = preds
        return self.activation(preds)

    def backward(self, error):
        d = self.derivative(self.preds)
        error *= d

        l1_W = self.l1_ratio * np.sign(self.W)
        l2_W = self.l2_ratio * self.W

        if self.bias:
            # l1_B = self.l1_ratio * np.sign(self.B)
            # l2_B = self.l2_ratio * self.B
            error = np.mean(error, 0).reshape((1, -1))
            self.BGrad += error  # + l1_B + l2_B

        grad = self.ins.T.dot(error)
        self.WGrad += grad + l1_W + l2_W

        next_grad = error.dot(self.W.T)
        return next_grad

    def updateGrad(self, lr):
        self.B -= lr * self.BGrad
        self.W -= lr * self.WGrad
        self.BGrad = np.zeros_like(self.B)
        self.WGrad = np.zeros_like(self.W)
