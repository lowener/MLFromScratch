import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tests import testIris, testDigits, testHousing
from MLFromScratch.Tools import ScoreMulticlass, Score, scale

class KNNClassifier(AlgorithmMixin):
    def __init__(self, N, weights='uniform', scale=True):
        self.scale = scale
        self.N = N
        self.weights = weights
        pass


    def fit(self, X: np.array, y: np.array):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        self.X_params = X
        self.Y_params = y
        self.n_classes = y.shape[1]
        pass

    
    def predict(self, X: np.array):
        n_samples, _ = X.shape
        Y = np.zeros((n_samples, self.n_classes))
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        
        for n in range(n_samples):
            distance = np.sqrt(np.sum((X[n] - self.X_params)**2, axis=1))
            distance_ind = np.argsort(distance)
            distance_ind = distance_ind[:self.N]
            nearestY = self.Y_params[distance_ind]
            if self.weights == 'distance':
                nearestY = ((1/distance[distance_ind]).T * nearestY.T).T # weight points by the inverse of their distance
            Y[n] = np.mean(nearestY, 0)
            Y[n] /= Y[n].sum()
                
        return Y

    
    
    def score(self, X, y):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)



class KNNRegressor(AlgorithmMixin):
    def __init__(self, N, weights='uniform', scale=True):
        self.scale = scale
        self.N = N
        self.weights = weights
        pass


    def fit(self, X: np.array, y: np.array):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
            y, self.Y_offset, self.Y_scale = scale(y)
        self.X_params = X
        self.Y_params = y
        pass

    
    def predict(self, X: np.array):
        n_samples, _ = X.shape
        Y = np.zeros((n_samples, 1))
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        
        for n in range(n_samples):
            distance = np.sqrt(np.sum((X[n] - self.X_params)**2, axis=1))
            distance_ind = np.argsort(distance)
            distance_ind = distance_ind[:self.N]
            nearestY = self.Y_params[distance_ind]
            if self.weights == 'distance':
                weight = (1/distance[distance_ind]).T
                weight /= weight.mean()
                nearestY = (weight * nearestY.T).T # weight points by the inverse of their distance
            Y[n] = np.mean(nearestY, 0)
                
        return (Y * self.Y_scale) + self.Y_offset

    
    
    def score(self, X, y):
        preds = self.predict(X)
        return Score(y, preds)


if __name__ == '__main__':
    testIris(KNNClassifier(5))
    testDigits(KNNClassifier(5, 'distance'))
    testHousing(KNNRegressor(5))
    pass