import numpy as np
from MLFromScratch.Tools import Score
from MLFromScratch.Tests import testBreast

class Stump():
    def __init__(self, stumpNb, stumpThr, stumpErr, alphaT=0):
        self.stumpNb = stumpNb
        self.stumpThr = stumpThr
        self.stumpErr = stumpErr
        self.alphaT = alphaT

    def getAttributes(self):
        return self.stumpNb, self.stumpThr, self.stumpErr, self.alphaT

    def predict(self, X):
        return np.array((X[:, self.stumpNb] < self.stumpThr) * 2 - 1, dtype=np.int)* self.alphaT
        
        
class Adaboost():
    def __init__(self, n_estimator):
        super().__init__()
        self.n_estimator = n_estimator


    def findBestStump(self, XColumn, y, distrib_example):
        thresholds = np.unique(XColumn)
        bestStump = thresholds[0]
        globalError = np.inf
        for thr in thresholds:
            thisPredictionError = np.array(XColumn < thr, dtype=np.int) != y.flatten()
            thisError = np.sum(thisPredictionError * distrib_example)
            if thisError < globalError:
                globalError = thisError
                bestStump = thr
        return globalError, bestStump


    def weakLearn(self, X, y, distrib_example):
        n_samples, n_features = X.shape
        featuresStump = np.zeros((n_features, 2))
        for featNb in range(n_features):
            XColumn = X[:, featNb]
            stumpErr, stumpThr = self.findBestStump(XColumn, y, distrib_example)
            featuresStump[featNb][0] = stumpThr
            featuresStump[featNb][1] = stumpErr
        bestStumpNb = featuresStump.argmin(0)[1]
        bestStump = Stump(bestStumpNb, featuresStump[bestStumpNb][0], featuresStump[bestStumpNb][1])
        return bestStump


    def fit(self, X, y):
        # Y = {0, 1}
        # y.shape = [n_samples, 1]
        # x.shape = [n_samples, n_features]
        n_samples, n_features = X.shape
        distrib_example = np.ones((n_samples)) / y.shape[0]
        bestEstimators = []
        for t in range(self.n_estimator):
            # Train weak learner using distribution
            bestStump = self.weakLearn(X, y, distrib_example)
            stumpNb, stumpThr, stumpErr, _ = bestStump.getAttributes()
            # Choose alphaT
            EPS = 1e-10
            alphaT = 0.5 * np.log((1-stumpErr + EPS) / (stumpErr+ EPS))
            bestStump.alphaT = alphaT
            # Update distribution
            stumpPred = bestStump.predict(X)
            distrib_example *= np.exp(- stumpPred * (y.flatten() * 2 - 1))
            distrib_example /= np.sum(distrib_example)
            #register this weak learner
            bestEstimators.append(bestStump)
        self.bestEstimators = bestEstimators


    def predict(self, X):
        predictions = np.array(list(map(lambda est: est.predict(X), self.bestEstimators))).T

        return np.array(predictions.sum(axis=1) > 0, dtype=np.int)

    
    def score(self, X, y):
        preds = self.predict(X)
        return Score(y, preds)


if __name__ == '__main__':
    testBreast(Adaboost(9))