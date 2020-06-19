import numpy as np
from MLFromScratch.Tools import ScoreMulticlass, Softmax, Relu, LeakyRelu, cross_entropy, cross_entropy, derivate


class BaseLayer():
    def __init__(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError
    
    def backward(self, error):
        raise NotImplementedError
    
    def updateGrad(self, lr):
        raise NotImplementedError

class NeuralLayer(BaseLayer):
    def __init__(self, size_in, size_out, activation=Relu, bias=True, derivative=None,
                l1_ratio=0.0, l2_ratio=0.0):
        self.W = np.random.uniform(size=(size_in, size_out), high=0.1, low=-0.1)
        self.bias = bias
        if self.bias:
            self.B = np.zeros((1, size_out))
        self.activation = activation
        if derivative is None:
            self.derivative = derivate(activation)
        else:
            self.derivative = derivative
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.BGrad = np.zeros_like(self.B)
        self.WGrad = np.zeros_like(self.W)


    def forward(self, X):
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
            #l1_B = self.l1_ratio * np.sign(self.B)
            #l2_B = self.l2_ratio * self.B
            error = np.mean(error, 0).reshape((1, -1))
            self.BGrad += error #+ l1_B + l2_B
            
        grad = self.ins.T.dot(error)
        self.WGrad += grad + l1_W + l2_W
        
        next_grad = error.dot(self.W.T)
        return next_grad

    def updateGrad(self, lr):
        self.B -= lr * self.BGrad
        self.W -= lr * self.WGrad
        self.BGrad = np.zeros_like(self.B)
        self.WGrad = np.zeros_like(self.W)