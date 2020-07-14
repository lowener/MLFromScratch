import numpy as np
from MLFromScratch.Tests import testRecoIris, testRecoWine
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, scale



class LDA(AlgorithmMixin):
    '''
    LDA Algorithm (Fisher)
    References:

        Deep Learning, Ian Goodfellow, Section 5.8.1, Page 146
        Pattern Recognition and Machine Learning, Section 12.1, Page 561

    '''
    def __init__(self, dims: int = None, solver='svd', scale: bool = True):
        self.dims = dims
        self.scale = scale
        self.solver = solver


    def fit(self, X: np.array, y: np.array):
        EPS=1e-10
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, self.n_classes = y.shape

        if self.dims is None:
            dims = self.n_classes - 1
        else:
            dims = np.min([self.dims, self.n_classes - 1])

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
        if self.solver == 'svd':
            u, s, v = np.linalg.svd(pinv @ self.S_between)
            self.proj = v.T[:, :dims]
        elif self.solver == 'eig':
            xc = np.cov((pinv @ self.S_between).T)
            evalue, evector = np.linalg.eigh(xc)
            self.proj = evector[-dims:].T
        else:
            raise NotImplementedError

    
    def predict(self, X: np.array):
        EPS=1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        n_samples, n_features = X.shape
        y = np.zeros((n_samples, self.n_classes))
        # TODO:  continue
        raise NotImplementedError


    def transform(self, X):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        res = (X-self.mean_class.mean(0)).dot(self.proj)
        return res


    def inverse_transform(self, X):
        res = X.dot(self.proj.T)
        if self.scale:
            res = np.array(res, dtype=np.float32)
            res = (res * self.X_scale) + self.X_offset
        return res
    

    def score(self, X, y=None):
        preds = self.inverse_transform(self.transform(X))
        return mse(X, preds)


if __name__ == '__main__':
    testRecoIris(LDA(2, solver="svd"))#, display2D=True)
    testRecoWine(LDA(2, solver="eig"))#, display=True)