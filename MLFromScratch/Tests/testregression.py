import numpy as np
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import mse


def testBase(dataset):
    def testBase2(algorithm: AlgorithmBase):
        from sklearn.model_selection import train_test_split

        X, y = dataset(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        algorithm.fit(X_train, y_train)
        preds = algorithm.predict(X_test)
        res = mse(y_test, preds)
        print("MSE: " + str(res))

    return testBase2


try:
    from sklearn import datasets

    testHousing = testBase(datasets.fetch_california_housing)
    testDiabetes = testBase(datasets.load_diabetes)
except ImportError:
    print("SKLearn not found: No tests")


def loadFaithful():
    import pandas as pd

    data = pd.read_csv("MLFromScratch/Tests/data/old_faithful.csv")
    return data[["eruptions","waiting"]]


def testFaithful(algorithm: AlgorithmBase):
    raise NotImplementedError("TODO")
    from sklearn.model_selection import train_test_split

    X = loadFaithful()
    X_train, X_test = train_test_split(X, random_state=0)

    algorithm.fit(X_train)
    preds = algorithm.predict(X_test)


def load_mauna_loa_atmospheric_co2():
    from sklearn.datasets import fetch_openml

    ml_data: dict = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    return months, avg_ppmvs


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    from matplotlib import cm
    import matplotlib.pyplot as plt

    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label="Mean")
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls="--", label=f"Sample {i+1}")
    if X_train is not None:
        plt.plot(X_train, Y_train, "rx")
    plt.legend()
    plt.show()


def testMauna(algorithm: AlgorithmBase, display: bool = False):
    X, y = load_mauna_loa_atmospheric_co2()
    algorithm.fit(X, y)
    X_test = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
    y_mu, y_cov = algorithm.predict(X_test, returnCov=True)
    print("MSE: " + str(algorithm.score(X, y)))
    if display:
        samples = np.random.multivariate_normal(y_mu, y_cov, 5)
        plot_gp(y_mu, y_cov, X_test, X, y, samples)
