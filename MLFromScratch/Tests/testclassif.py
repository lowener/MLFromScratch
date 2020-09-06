import numpy as np
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import colors

def testBase(dataset, nameTest='test'):
    def testCode(algorithm: AlgorithmMixin, oneHot=True):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        print("Start %s" % nameTest)
        # Load dataset
        X, y = dataset(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        if oneHot:
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            y_train = enc.fit_transform(y_train.reshape(-1, 1))

        # Test the algorithm
        algorithm.fit(X_train, y_train)
        score = algorithm.score(X_test, y_test)
        score_avg_multi = (score.F1Score * score.P).sum() / score.P.sum()
        print(f"F1 Score: {score.F1Score}")
        # Check that we have 90% average accuracy
        if score_avg_multi > 0.9: 
            print("F1 Averaged score: {}{:.2f}".format(colors.OKGREEN, score_avg_multi))
            print(f'SUCCESS{colors.ENDC}')
        else:
            print("F1 Averaged score: {}{:.2f}".format(colors.FAIL, score_avg_multi))
            print(f'FAIL{colors.ENDC}')
        print("------------------------------------")
    return testCode


try:
    from sklearn import datasets
    testIris = testBase(datasets.load_iris, nameTest='Iris')
    testDigits = testBase(datasets.load_digits, nameTest='Digits')
    testOlivetti = testBase(datasets.fetch_olivetti_faces, nameTest='Olivetti')
    testBreast = testBase(datasets.load_breast_cancer, nameTest='Breast Cancer')
except ImportError:
    print("SKLearn not found: No tests")


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