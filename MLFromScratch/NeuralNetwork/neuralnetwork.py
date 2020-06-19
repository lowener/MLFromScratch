import numpy as np
from MLFromScratch.Tools import ScoreMulticlass, Softmax, Relu, LeakyRelu, cross_entropy, cross_entropy, derivate
from MLFromScratch.Tests import testIris
from MLFromScratch.Base import AlgorithmMixin

class Layer():
    def __init__(self, size_in, size_out, activation=Relu, bias=True, derivative=None):
        self.W = np.random.uniform(size=(size_in, size_out), high=0.01, low=-0.01)
        self.bias = bias
        if self.bias:
            self.B = np.zeros((1, size_out))
        self.activation = activation
        if derivative is None:
            self.derivative = derivate(activation)
        else:
            self.derivative = derivative


    def forward(self, X):
        self.ins = X
        preds = X.dot(self.W)
        if self.bias:
            preds += self.B
        self.preds = preds
        return self.activation(preds)


    def backward(self, error, lr):
        d = self.derivative(self.preds)
        error *= d
        grad = self.ins.T.dot(error)
        next_grad = error.dot(self.W.T)

        if self.bias:
            self.B -= lr * np.mean(error, 0).reshape((1, -1))
        self.W -= lr * grad
        return next_grad


class NeuralNetwork(AlgorithmMixin):
    def __init__(self, layerSizes=[100,], activation=Relu, finalActivation=Softmax, solver="adam", lr=0.01,
                 l1_ratio=0.0, l2_ratio=0.0, n_iters=0, scale=True, bias=True):
        self.layerSizes = layerSizes
        self.activation = activation
        self.finalActivation = finalActivation
        self.solver = solver
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.n_iters = n_iters
        self.scale = scale
        self.bias = bias
        

    def fit(self, X, y):
        if self.scale:
            X = np.array(X, dtype=np.float32)
            self.X_offset = np.average(X, axis=0)
            X -= self.X_offset
            self.X_scale = np.max(X, axis=0)
            X /= self.X_scale
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        
        self.layers = []
        nbIn = n_features
        for nbOut in self.layerSizes:
            self.layers.append(Layer(nbIn, nbOut, self.activation, self.bias))
            nbIn = nbOut
        derivFinal = lambda x: np.ones_like(x)
        self.layers.append(Layer(nbIn, n_classes, self.finalActivation, self.bias, derivative=derivFinal))
        

        self.history = []
        for i in range(self.n_iters):
            for j in range(n_samples):
                preds = self.forward(X[j,None])
                error = preds - y[j,None]
                self.backward(error)
            self.history.append(cross_entropy(y, self.forward(X)))



    def forward(self, X):
        preds = X
        for layer in self.layers:
            preds = layer.forward(preds)
        return preds


    def backward(self, errors):
        for layer in reversed(self.layers):
            errors = layer.backward(errors, self.lr)
        pass



    def predict(self, X):
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / self.X_scale
        preds = X
        for layer in self.layers:
            preds = layer.forward(preds)
        return preds


    def score(self, X, y):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)


if __name__ == "__main__":
    testIris(NeuralNetwork(layerSizes=[90,20], n_iters=40, lr=0.1))