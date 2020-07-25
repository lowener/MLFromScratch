import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import cross_entropy, binary_cross_entropy, ScoreMulticlass, scale, Softmax
from MLFromScratch.Tests import testIris
from MLFromScratch.Optimizer import LinearGradientDescent


class LinearSVC(AlgorithmMixin):
    '''
    References:
        https://scikit-learn.org/stable/modules/svm.html#linearsvc
        https://en.wikipedia.org/wiki/Hinge_loss
        https://cs231n.github.io/optimization-1/#analytic
    '''
    def __init__(self, C=1.0, l1_ratio=0.0, l2_ratio=1e-1, multi_class='multi',
                fit_intercept=True, n_iters=1000, lr=1e-3, scale=True):
        self.C = C
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.n_iters = n_iters
        self.lr = lr
        self.scale = scale


    def dcrammer_singer_loss(self, X: np.array, y: np.array, W: np.array) -> np.array:
        n_samples, n_classes = y.shape
        preds = X.dot(W)
        dW = np.zeros(W.shape)
        for i in range(n_samples):
            yi = y[i].argmax()
            for j in range(n_classes):
                if j == yi:
                    continue
                if preds[i, j] - preds[i, yi] + 1 > 0:
                    dW[:, j] += X[i]
                    dW[:, yi] += -X[i]
        dW = self.C * (1/self.n_samples) * dW
        return dW


    def fit(self, X: np.array, y: np.array):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
        
        self.n_samples, n_features = X.shape
        
        if self.multi_class == 'ovr':
            raise NotImplementedError()
            
        elif self.multi_class == 'multi':
            from MLFromScratch.Tools import Sigmoid
            predFn = lambda X, W: Softmax(X.dot(W))
            W = np.random.rand(n_features, n_classes)
            self.W, self.history = LinearGradientDescent(self.dcrammer_singer_loss, X, y, W, self.n_iters, 
                                            self.lr, l1_ratio=self.l1_ratio, l2_ratio=self.l2_ratio,
                                            metric=cross_entropy, predFn=predFn)
        else:
            raise NotImplementedError


    def predict(self, X: np.array) -> np.array:
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        n_samples, _ = X.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
        
        if self.multi_class == 'ovr':
            raise NotImplementedError
        elif self.multi_class == 'multi':
            preds = X.dot(self.W)
        else:
            raise NotImplementedError

        return preds


    def score(self, X: np.array, y: np.array) -> ScoreMulticlass:
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)


if __name__ == '__main__':
    testIris(LinearSVC(lr=0.1, fit_intercept=True, multi_class='multi'))
    testIris(LinearSVC(lr=0.1, fit_intercept=False, multi_class='multi'))