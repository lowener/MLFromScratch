import numpy as np

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
        return getTpr(y, preds)


def customTest():
    a = Adaboost(12)
    x= np.array([
        [1,1,205   ],
        [-1,1,180   ],
        [1,-1,210   ],
        [1,1,167    ],
        [-1,1,156   ],
        [-1,1,125   ],
        [1,-1,168   ],
        [1,1,172    ],
    ])
    y=np.array([
        [1],
        [1],
        [1],
        [1],
        [0],
        [0],
        [0],
        [0],
    ])
    a.fit(x,y)
    preds = a.predict(x[-2:])


def getTpr(labels, preds):
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    
    TP = (preds[labels == 1] == 1).sum()
    FN = (preds[labels == 1] == 0).sum()
    FP = (preds[labels == 0] == 1).sum()
    TN = (preds[labels == 0] == 0).sum()
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = TN / (TN + FP)
    FNR = FN / (FN + TP)

    ACCURACY = (TP + TN) / (P + N)
    return TPR, FPR, TNR, FNR, ACCURACY


def breastTest():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    a=Adaboost(9) 
    a.fit(X_train, y_train)
    preds=a.predict(X_test)
    tpr = getTpr(y_test, preds)

    from sklearn.ensemble import AdaBoostClassifier
    ABC = AdaBoostClassifier(n_estimators=9)
    ABC.fit(X_train, y_train)
    preds2=ABC.predict(X_test)
    tpr2 = getTpr(y_test, preds2)
    

if __name__ == '__main__':
    breastTest()