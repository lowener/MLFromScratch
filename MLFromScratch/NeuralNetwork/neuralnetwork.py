import numpy as np
from MLFromScratch.Tools import ScoreMulticlass, Softmax, Relu, LeakyRelu, cross_entropy, cross_entropy, derivate
from MLFromScratch.Tests import testIris, testDigits
from MLFromScratch.Base import AlgorithmMixin

np.random.seed(0)

class Layer():
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
            l1_B = self.l1_ratio * np.sign(self.B)
            l2_B = self.l2_ratio * self.B
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
            X = np.array(X, dtype=np.float32)
            self.X_offset = np.average(X, axis=0)
            X -= self.X_offset
            self.X_scale = np.max(np.abs(X), axis=0)
            X /= (self.X_scale + EPS)
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        
        self.layers = []
        nbIn = n_features
        for nbOut in self.layerSizes:
            self.layers.append(Layer(nbIn, nbOut, self.activation, self.bias, l1_ratio=self.l1_ratio, l2_ratio=self.l2_ratio))
            nbIn = nbOut
        derivFinal = lambda x: np.ones_like(x)
        self.layers.append(Layer(nbIn, n_classes, self.finalActivation, self.bias, derivative=derivFinal, l1_ratio=self.l1_ratio, l2_ratio=self.l2_ratio))
        
        
        self.history = []
        for i in range(self.n_iters):
            for j in range(n_samples):
                preds = self.forward(X[j,None])
                error = preds - y[j,None]
                self.backward(error)
            self.updateGrad(self.lr / n_samples)
            ce = cross_entropy(y, self.forward(X))
            self.history.append(ce)
            print(ce)



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