import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, cross_entropy,ScoreMulticlass, scale

'''
References:
Pattern Recognition and Machine Learning, Section 4.3.4, Page 209
https://en.wikipedia.org/wiki/Logistic_regression#Model_fitting
'''
class LogisticRegression(AlgorithmMixin):
    def __init__(self, lr=0.001, l1_ratio=0.0, l2_ratio=0.0, fit_intercept=True, n_iters=100, scale=True):
        self.lr = lr
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.fit_intercept = fit_intercept
        self.n_iters = n_iters
        self.scale = scale

    def fit(self, X, y):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
            n_samples, n_features = X.shape

        W = np.zeros((n_features, n_classes))
        self.history = []
        for _ in range(self.n_iters):
            preds = X.dot(W)
            preds = self.softmax(preds)
            dCE = np.zeros((n_features, n_classes))
            for j in range(n_classes):
                ydiff = preds[:, j] - y[:, j]
                dCE[:, j] = (1/n_samples) * X.T.dot(ydiff)

            #W = W - self.lr * dCE
            dl1 = np.sign(W)
            dl2 = 2*W
            W = W - self.lr * (dCE + self.l1_ratio * dl1 + 0.5 * self.l2_ratio * dl2)
            self.history.append(cross_entropy(y, preds))
        self.W = W
        pass


    def softmax(self, alpha):
        _, n_classes = alpha.shape
        return np.exp(alpha) / np.repeat(np.sum(np.exp(alpha), 1).reshape((-1, 1)), n_classes, axis=1)


    def predict(self, X):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        if self.fit_intercept:
            n_samples, n_features = X.shape
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)
        preds = np.dot(X, self.W)
        preds = self.softmax(preds)
        return preds

    def score(self, X, y):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)



def testIris():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    X, y = load_iris(return_X_y=True)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_train2 = enc.fit_transform(y_train.reshape(-1, 1))
    

    lr = LogisticRegression(n_iters=4000, lr=0.1)
    lr.fit(X_train, y_train2)
    score = lr.score(X_test, y_test)
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())


    

if __name__ == '__main__':
    testIris()