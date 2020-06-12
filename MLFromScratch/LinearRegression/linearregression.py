import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, Score

class LinearRegression(AlgorithmMixin):
    def __init__(self, fit_intercept=True, gradient_descent=True, n_iters=1000, lr=1e-3):
        self.fit_intercept = fit_intercept
        self.gradient_descent = gradient_descent
        self.n_iters = n_iters
        self.lr = lr


    def invert(self, X, y):
        '''
        Y = Wx
        MSE(Y, y) = (1/m) * (Y - y)² = (1/m) * (Wx - y).T * (Wx - y) = (1/m) * W.T*x.T*x - 2Wxy + y²
        dMSE_w = 2Wx² - 2xy = 0
               = X.T * X * w - X.T y
        W = (x²)^(-1)* y * x
        '''
        n_samples, n_features = X.shape
        w = np.zeros((n_features))
        X2 = X.T.dot(X)
        W = np.linalg.pinv(X2).dot(X.T).dot(y)
        self.W = W


    def fit(self, X: np.array, y: np.array):
        n_samples, n_features = X.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
            n_samples, n_features = X.shape

        if not self.gradient_descent:
            self.invert(X, y)
        else:
            W = np.random.rand((n_features))
            self.history = []
            for _ in range(self.n_iters):
                preds = X.dot(W)
                dMSE = (1/n_samples) * X.T.dot(preds - y)
                W = W - self.lr * dMSE
                self.history.append(mse(X.dot(W), y))
            self.W = W



    def predict(self, X: np.array):
        if self.fit_intercept:
            n_samples, n_features = X.shape
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
        return np.dot(X, self.W)

    def score(self, X, y):
        preds = self.predict(X)
        return Score(y, preds)

if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    lr = LinearRegression()
    lr.fit(X, y)
    preds = lr.predict(np.array([[3, 5]]))
    pass