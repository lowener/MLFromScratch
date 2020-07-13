import numpy as np
from MLFromScratch.Base import AlgorithmMixin

def testRecoIris(algo: AlgorithmMixin, display: bool = False):
    from sklearn import datasets
    if display:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

    X, y = datasets.load_iris(return_X_y=True)

    algo.fit(X)
    X_reduced = algo.transform(X)
    score = algo.score(X)
    print("MSE: " + str(score))

    if display:
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                cmap=plt.cm.get_cmap('Set1'), edgecolor='k', s=40)
        ax.set_title("First three directions")
        ax.set_xlabel("1st")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd")
        ax.w_zaxis.set_ticklabels([])

        plt.show()

