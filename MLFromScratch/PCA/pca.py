import numpy as np
from MLFromScratch.Tests import testBlobs, testIris
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, scale



class PCA(AlgorithmMixin):
    '''
    K-Means Algorithm
    References:
        Deep Learning, Ian Goodfellow, Section 5.8.2, Page 149
        Pattern Recognition and Machine Learning, Section 9.1, Page 424

    '''
    def __init__(self, dims, scale=True, solver='eig'):
        self.dims = dims
        self.scale = scale
        self.solver = solver


    def fit(self, X : np.array, y=None):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        
        if self.solver == 'eig':
            xc = np.cov(X.T)
            evalue, evector = np.linalg.eigh(xc)
            #evalue = evalue[-self.dims::-1] * np.eye(dims)
            self.proj = evector[-self.dims:].T
        elif self.solver == 'svd':
            u, s, v = np.linalg.svd(X)
            self.proj = v[:self.dims].T
        else:
            raise NotImplementedError

    
    def transform(self, X):
        EPS = 1e-10
        n_samples, n_features = X.shape
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)

        return X.dot(self.proj)


    def inverse_transform(self, X):
        n_samples, n_features = X.shape
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X * self.X_scale) + self.X_offset)

        return X.dot(self.proj.T)
    

    def score(self, X, y=None):
        preds = self.inverse_transform(self.transform(X))
        return mse(X, preds)


def testReco():
    from sklearn.datasets import load_wine
    raise NotImplementedError



if __name__ == '__main__':
    raise NotImplementedError