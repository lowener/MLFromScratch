import numpy as np
from MLFromScratch.Tests import testBlobs, testIris
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import ScoreMulticlass, scale



class KMeans(AlgorithmMixin):
    '''
    K-Means Algorithm
    References:
        Deep Learning, Ian Goodfellow, Section 5.8.2, Page 149
        Pattern Recognition and Machine Learning, Section 9.1, Page 424

    '''
    def __init__(self, K=3, max_iters=1000, scale=False, tol=1e-5):
        self.K = K
        self.scale = scale
        self.max_iters = max_iters
        self.tol = tol
        pass


    def fit(self, X, y=None):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        
        self.centers = np.zeros((self.K, n_features))

        # Initialize centers
        for f in range(n_features):
            self.centers[:, f] = np.linspace(np.quantile(X[:, f], 1/self.K), np.quantile(X[:, f], 1-1/self.K), self.K)

        self.history = []
        for _ in range(self.max_iters):
            currentPoints = np.zeros((n_samples, self.K))
            # Expectation
            for s in range(n_samples):
                closestCenter = ((self.centers - X[s])**2).sum(1)
                closestCenter = closestCenter.argmin()
                currentPoints[s, closestCenter] = 1

            # Maximization
            points = (currentPoints == 1).argmax(1)
            for k in range(self.K):
                if sum(points == k) > 0:
                    self.centers[k] = X[points == k].mean()

            # Logging
            objective = 0
            for s in range(n_samples):
                objective += (currentPoints[s] * ((self.centers - X[s])**2).sum(1)).sum()

            self.history.append(objective)
            if len(self.history) > 2:
                if np.abs(self.history[-2] - objective) < self.tol:
                    break

    
    def predict(self, X):
        EPS = 1e-10
        n_samples, n_features = X.shape
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)

        currentPoints = np.zeros((n_samples, self.K))
        for s in range(n_samples):
            closestCenter = ((self.centers - X[s])**2).sum(1).argmin()
            currentPoints[s, closestCenter] = 1
        return currentPoints.argmax(1)


    

    def score(self, X, y):
        preds = self.predict(X)
        return ScoreMulticlass(y, preds)


if __name__ == '__main__':
    testBlobs(KMeans(3))