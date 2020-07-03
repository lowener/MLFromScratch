import numpy as np
from MLFromScratch.Base import AlgorithmMixin

def testIris(algorithm: AlgorithmMixin, oneHot=True):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    print("Start Iris")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if oneHot:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))

    algorithm.fit(X_train, y_train)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")


def testDigits(algorithm: AlgorithmMixin, oneHot=True):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    print("Start Digits")
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if oneHot:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))

    algorithm.fit(X_train, y_train)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")


def testBreast(algorithm: AlgorithmMixin, oneHot=True):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    print("Start Breast")
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    if oneHot:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))
    algorithm.fit(X_train, y_train)
    score = algorithm.score(X_test, y_test)
    print("F1 Score: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")
    

def testBlobs(algo: AlgorithmMixin, display=False):
    from sklearn.datasets import make_blobs

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    algo.fit(X)
    y_pred = algo.predict(X)
    score = algo.score(X, y)
    
    print("F1 Score Classical: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")

    if display:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        plt.title("Classical Blobs")

    
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    algo.fit(X_aniso)
    y_pred = algo.predict(X_aniso)
    score = algo.score(X_aniso, y)
    print("F1 Score Anisotropic: ")
    print(score.F1Score)
    print((score.F1Score * score.P).sum() / score.P.sum())
    print("------------------------------------")


    if display:
        plt.subplot(122)
        plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
        plt.title("Anisotropic Blobs")
        plt.show()