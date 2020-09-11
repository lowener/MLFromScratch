import numpy as np
from MLFromScratch.Tests import testMauna
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import scale, mse


class GaussianProcessRegressor(AlgorithmBase):
    """
    References:
        https://krasserm.github.io/2018/03/19/gaussian-processes/
        Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams
        https://www.youtube.com/watch?v=R-NUdqxKjos

    """

    def __init__(self, kernel, alpha=1e-8, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y

    def fit(self, X, y):
        n_sample, n_features = X.shape
        if self.normalize_y:
            y, self.y_offset, self.y_scale = scale(y)
        self.X_dataset = X
        self.y_dataset = y
        noise = self.alpha * np.eye(n_sample)
        # Prior
        w_cov = self.kernel(X, X) + noise
        self.K = w_cov
        self.w_cov_i = np.linalg.pinv(w_cov)

    def predict(self, X, returnCov=False):
        n_sample, _ = X.shape
        noise = self.alpha * np.eye(n_sample)
        Ks = self.kernel(self.X_dataset, X)
        Kss = self.kernel(X, X) + noise

        # Posterior
        mu_s = Ks.T.dot(self.w_cov_i).dot(self.y_dataset)
        if self.normalize_y:  # Undo Normalisation
            mu_s = mu_s * self.y_scale + self.y_offset

        if not returnCov:
            return mu_s
        else:
            cov_s = Kss - Ks.T.dot(self.w_cov_i).dot(Ks)
            if self.normalize_y:  # Undo Normalisation
                cov_s = cov_s * self.y_scale + self.y_offset
            return mu_s, cov_s

    def score(self, X, y):
        preds, _ = self.predict(X, True)
        return mse(preds, y)


def easyTest():
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

    X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
    # Noise free training data
    X_train = np.array([-4, -3, -2, -1, 1], dtype=np.float).reshape(-1, 1)
    Y_train = np.sin(X_train)
    algo = GaussianProcessRegressor(kernel=KernelRBF(), alpha=1e-8, normalize_y=False)
    algo.fit(X_train, Y_train)
    mu_s, cov_s = algo.predict(X_test, returnCov=True)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X_test, X_train=X_train, Y_train=Y_train, samples=samples)


if __name__ == "__main__":
    from MLFromScratch.Kernels import (
        KernelRBF,
        KernelExpSineSquared,
        KernelWhite,
        KernelRationalQuadratic,
    )

    easyTest()
    k1 = 50.0 ** 2 * KernelRBF(length=50.0)
    k2 = (
        2.0 ** 2
        * KernelRBF(length=100.0)
        * KernelExpSineSquared(length=1.0, periodicity=1.0)
    )
    k3 = 0.5 ** 2 * KernelRationalQuadratic(length=1.0, alpha=1.0)
    k4 = 0.1 ** 2 * KernelRBF(length=0.1) + KernelWhite(noise=0.1 ** 2)
    kernel = k1 + k2 + k3 + k4
    algo = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False)
    testMauna(algo)
