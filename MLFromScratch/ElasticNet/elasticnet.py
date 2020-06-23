import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, Score, scale

class ElasticNet(AlgorithmMixin):
    def __init__(self, l1_ratio=0.5, l2_ratio=1.0, fit_intercept=True,
                gradient_descent=True, n_iters=1000, lr=1e-3, scale=True):
        self.fit_intercept = fit_intercept
        self.gradient_descent = gradient_descent
        self.n_iters = n_iters
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.scale = scale


    def fit(self, X: np.array, y: np.array):
        '''
        1 / (2 * n_samples) * ||y - Xw||^2_2
        + l1_ratio * ||w||_1
        + 0.5 * l2_ratio * ||w||^2_2
        '''
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
            y, self.y_offset, self.y_scale = scale(y)
        n_samples, n_features = X.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
            n_samples, n_features = X.shape

        W = np.random.rand((n_features))
        self.history = []
        for _ in range(self.n_iters):
            preds = X.dot(W)
            dMSE = (1/n_samples) * X.T.dot(preds - y)
            dl1 = np.sign(W)
            dl2 = 2*W
            W = W - self.lr * (dMSE + self.l1_ratio * dl1 + 0.5 * self.l2_ratio * dl2)
            self.history.append(mse(X.dot(W), y))
        self.W = W



    def predict(self, X: np.array):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        if self.fit_intercept:
            n_samples, n_features = X.shape
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
        preds = np.dot(X, self.W)
        if self.scale:
            preds = preds * self.y_scale + self.y_offset
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return Score(y, preds)


def dummyTest():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    en = ElasticNet()
    en.fit(X, y)
    preds = en.predict(np.array([[3, 5]]))
    print(preds)


def housingTest():
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    en=ElasticNet(lr=0.1, l1_ratio=0.001, l2_ratio=0.01, n_iters=3000)
    en.fit(X_train, y_train)
    preds=en.predict(X_test)
    res = mse(y_test, preds)
    # Result is approximately 1.8
    print(res)

if __name__ == '__main__':
    dummyTest()
    housingTest()
    pass