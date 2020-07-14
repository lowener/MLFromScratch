import numpy as np
from MLFromScratch.Base import AlgorithmMixin

def testRecoIris(algo: AlgorithmMixin, oneHot: bool = True, display2D: bool = False, display3D: bool = False):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    if display3D or display2D:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if oneHot:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))

    algo.fit(X_train, y_train)
    score = algo.score(X_test, y_test)
    
    print("MSE: " + str(score))

    if display2D:
        X_reduced = algo.transform(X_test)
        fig = plt.figure(1, figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_test,
                cmap=plt.cm.get_cmap('Set1'), edgecolor='k', s=40)
        plt.show()


    if display3D:
        X_reduced = algo.transform(X_test)
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_test,
                cmap=plt.cm.get_cmap('Set1'), edgecolor='k', s=40)
        ax.set_title("First three directions")
        ax.set_xlabel("1st")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd")
        ax.w_zaxis.set_ticklabels([])

        plt.show()

def testRecoWine(algo: AlgorithmMixin, oneHot: bool = True, display: bool = False):
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    if display:
        import matplotlib.pyplot as plt

    X, y = datasets.load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    if oneHot:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_train = enc.fit_transform(y_train.reshape(-1, 1))

    algo.fit(X_train, y_train)
    score = algo.score(X_test, y_test)
    
    print("MSE: " + str(score))

    if display:
        X_reduced = algo.transform(X_test)
        fig = plt.figure(1, figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_test,
                cmap=plt.cm.get_cmap('Set1'), edgecolor='k', s=40)
        plt.show()