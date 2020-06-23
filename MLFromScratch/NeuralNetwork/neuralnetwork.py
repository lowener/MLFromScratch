import numpy as np
from MLFromScratch.Tools import ScoreMulticlass, Softmax, Relu, LeakyRelu, cross_entropy, scale
from MLFromScratch.Tests import testIris, testDigits
from MLFromScratch.Base import AlgorithmMixin
from layers import NeuralLayer

np.random.seed(0)

class NeuralNetwork(AlgorithmMixin):
    def __init__(self, layerSizes=[100,], activation=Relu, finalActivation=Softmax, lr=0.01,
                 l1_ratio=0.0, l2_ratio=0.0, n_iters=0, scale=True, bias=True):
        self.layerSizes = layerSizes
        self.activation = activation
        self.finalActivation = finalActivation
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.n_iters = n_iters
        self.scale = scale
        self.bias = bias
        

    def fit(self, X, y):
        EPS = 1e-10
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        
        self.layers = []
        nbIn = n_features
        for nbOut in self.layerSizes:
            self.layers.append(NeuralLayer(nbIn, nbOut, self.activation, self.bias, l1_ratio=self.l1_ratio, l2_ratio=self.l2_ratio))
            nbIn = nbOut
        derivFinal = lambda x: np.ones_like(x)
        self.layers.append(NeuralLayer(nbIn, n_classes, self.finalActivation, self.bias, derivative=derivFinal, l1_ratio=self.l1_ratio, l2_ratio=self.l2_ratio))
        
        
        self.history = []
        for i in range(self.n_iters):
            for j in range(n_samples):
                preds = self.forward(X[j,None])
                error = preds - y[j,None]
                self.backward(error)
            self.updateGrad(self.lr / n_samples)
            ce = cross_entropy(y, self.forward(X))
            self.history.append(ce)



    def forward(self, X):
        preds = X
        for layer in self.layers:
            preds = layer.forward(preds)
        return preds


    def backward(self, errors):
        for layer in reversed(self.layers):
            errors = layer.backward(errors)
        pass


    def updateGrad(self, lr):
        for layer in self.layers:
            layer.updateGrad(lr)



    def predict(self, X):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        preds = X
        for layer in self.layers:
            preds = layer.forward(preds)
        return preds


    def score(self, X, y):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)


if __name__ == "__main__":
    testIris(NeuralNetwork(layerSizes=[20,10], n_iters=100, lr=1, activation=LeakyRelu, l1_ratio=1e-3, l2_ratio=1e-3))
    testDigits(NeuralNetwork(layerSizes=[60, 20], n_iters=60, lr=1, activation=LeakyRelu))