import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, scale
from MLFromScratch.Optimizer import LinearGradientDescent
from MLFromScratch.Tests import testHousing


class LinearSVR(AlgorithmMixin):
    def __init__(self, C=1.0, epsilon=1e-1, l2_ratio=1e-1, n_iters=10000, lr=1e-3, scale=True):
        self.C = C
        self.epsilon = epsilon
        self.l2_ratio = l2_ratio
        self.n_iters = n_iters
        self.lr = lr
        self.scale = scale


    def epsilon_loss(self, X, y, W):
        preds = X.dot(W)
        epsilon_loss = preds - y
        epsilon_loss[np.abs(epsilon_loss) < self.epsilon] = 0
        dMSE = self.C * (1/self.n_samples) * X.T.dot(epsilon_loss)
        return dMSE


    def fit(self, X: np.array, y: np.array):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
            y, self.y_offset, self.y_scale = scale(y)
        n_samples, n_features = X.shape
        # fit_intercept:
        ones = np.ones((n_samples, 1))
        X = np.concatenate((ones, X), axis=1)
        self.n_samples, n_features = X.shape

        W = np.random.rand((n_features))
        predFn = lambda X, W: X.dot(W)
        self.W, self.history = LinearGradientDescent(self.epsilon_loss, X, y, W, self.n_iters, 
                                                    self.lr, l1_ratio=0, l2_ratio=self.l2_ratio,
                                                    metric=mse, predFn=predFn)


    def predict(self, X: np.array) -> np.array:
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        n_samples, _ = X.shape
        ones = np.ones((n_samples, 1))
        X = np.concatenate((ones, X), axis=1)
        preds = np.dot(X, self.W)
        if self.scale:
            preds = preds * self.y_scale + self.y_offset
        return preds


    def score(self, X, y):
        preds = self.predict(X)
        return mse(y, preds)


def dummyTest(algo: AlgorithmMixin):
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    algo.fit(X, y)
    preds = algo.predict(np.array([[3, 5]]))
    print("Dummy MSE: " + str(mse(preds, 18)))


if __name__ == '__main__':
    dummyTest(LinearSVR())
    testHousing(LinearSVR())