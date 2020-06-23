import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import mse


def testHousing(algorithm: AlgorithmMixin):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    algorithm.fit(X_train, y_train)
    preds = algorithm.predict(X_test)
    res = mse(y_test, preds)
    print("MSE: " + str(res))