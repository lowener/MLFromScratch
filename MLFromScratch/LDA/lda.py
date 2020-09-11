import numpy as np
from MLFromScratch.Tests import testRecoIris, testRecoWine, testBreast, testIris
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import mse, scale, ScoreMulticlass


class LDA(AlgorithmBase):
    """
    LDA Algorithm (Fisher)
    References:

        Pattern Recognition and Machine Learning, Section 4.1.4, Page 168

    """

    def __init__(self, dims: int = None, solver="svd", scale: bool = True):
        self.dims = dims
        self.scale = scale
        self.solver = solver

    def fit(self, X: np.array, y: np.array):
        EPS = 1e-10
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, self.n_classes = y.shape

        self.p_classes = np.zeros(self.n_classes)
        self.mean_class = np.zeros((self.n_classes, n_features))
        self.cov_class = np.zeros((n_features, n_features))
        # within

        for i in range(self.n_classes):
            self.p_classes[i] = sum(y[:, i]) / n_samples
            Xi = X[y[:, i] == 1]
            self.mean_class[i] = Xi.mean(0)
            self.cov_class += np.cov(Xi.T)

        self.S_within = self.cov_class
        # between
        self.S_between = (self.mean_class - X.mean(0)).T @ (self.mean_class - X.mean(0))
        pinv = np.linalg.pinv(self.S_within)
        if self.solver == "svd":
            u, s, v = np.linalg.svd(pinv @ self.S_between)
            self.proj = v.T
            # self.bias = TODO
        elif self.solver == "eig":
            xc = pinv @ self.S_between
            evalue, evector = np.linalg.eigh(xc)
            self.proj = evector[::-1].T
            # self.bias = TODO
        else:
            raise NotImplementedError

    def predict(self, X: np.array):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        n_samples, n_features = X.shape

        res = X.dot(self.proj)[:, : self.n_classes]
        return res

    def transform(self, X):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)

        if self.dims is None:
            dims = self.n_classes - 1
        else:
            dims = np.min([self.dims, self.n_classes - 1])

        res = X.dot(self.proj)[:, :dims]
        return res

    def score(self, X, y=None):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)


if __name__ == "__main__":
    testBreast(LDA(2, solver="eig"))
    testIris(LDA(2, solver="eig"))
    testBreast(LDA(2, solver="svd"))
    testIris(LDA(2, solver="svd"))
    testRecoIris(LDA(2, solver="eig"), display2D=True)
    testRecoWine(LDA(2, solver="eig"), display=True)
