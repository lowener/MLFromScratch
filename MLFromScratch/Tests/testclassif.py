import numpy as np
from MLFromScratch.Base import AlgorithmMixin

def testIris(algorithm: AlgorithmMixin):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    print("Start Iris")
    X, y = load_iris(return_X_y=True)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_train2 = enc.fit_transform(y_train.reshape(-1, 1))
    
    algorithm.fit(X_train, y_train2)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")


def testDigits(algorithm: AlgorithmMixin):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    print("Start Digits")
    X, y = load_digits(return_X_y=True)
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_train2 = enc.fit_transform(y_train.reshape(-1, 1))
    
    algorithm.fit(X_train, y_train2)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")


def breastTest(algorithm: AlgorithmMixin):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    print("Start Breast")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    algorithm.fit(X_train, y_train)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")
    