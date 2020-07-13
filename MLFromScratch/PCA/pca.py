import numpy as np
from MLFromScratch.Tests import testRecoIris
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse, scale



class PCA(AlgorithmMixin):
    '''
    PCA Algorithm
    References:
        Deep Learning, Ian Goodfellow, Section 5.8.1, Page 146
        Pattern Recognition and Machine Learning, Section 12.1, Page 561

    '''
    def __init__(self, dims, scale=True, solver='eig'):
        # Solver = 'eig' or 'svd'
        self.dims = dims
        self.scale = scale
        self.solver = solver


    def fit(self, X : np.array, y=None):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        
        if self.solver == 'eig':
            xc = np.cov(X.T)
            evalue, evector = np.linalg.eigh(xc)
            self.proj = evector[-self.dims:].T
        elif self.solver == 'svd':
            u, s, v = np.linalg.svd(X)
            self.proj = v[:self.dims].T
        else:
            raise NotImplementedError

    
    def transform(self, X):
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)

        return X.dot(self.proj)


    def inverse_transform(self, X):
        n_samples, n_features = X.shape
        res = X.dot(self.proj.T)
        if self.scale:
            res = np.array(res, dtype=np.float32)
            res = (res * self.X_scale) + self.X_offset

        return res
    

    def score(self, X, y=None):
        preds = self.inverse_transform(self.transform(X))
        return mse(X, preds)


if __name__ == '__main__':
    testRecoIris(PCA(3, solver="eig"))