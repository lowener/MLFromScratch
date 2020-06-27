import numpy as np
from MLFromScratch.Tests import testMauna
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import scale, mse
from MLFromScratch.Kernels import KernelRBF, KernelExpSineSquared, KernelWhite, KernelRationalQuadratic



class GaussianProcessRegressor(AlgorithmMixin):
    '''
    References:
        https://krasserm.github.io/2018/03/19/gaussian-processes/
        Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams
        https://www.youtube.com/watch?v=R-NUdqxKjos

    '''
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
        noise = self.alpha**2 * np.eye(n_sample)
        #prior
        w_cov = self.kernel(X, X) + noise
        self.K = w_cov
        self.w_cov_i = np.linalg.pinv(w_cov)


    def predict(self, X, returnParams=False):
        n_sample, _ = X.shape
        noise = self.alpha * np.eye(n_sample)
        Ks = self.kernel(self.X_dataset, X)
        Kss = self.kernel(X, X) + noise

        #Posterior
        mu_s = Ks.T.dot(self.w_cov_i).dot(self.y_dataset)
        cov_s = Kss - Ks.T.dot(self.w_cov_i).dot(Ks)


        preds = np.random.multivariate_normal(mu_s.ravel(), cov_s, n_sample)
        if self.normalize_y:
            preds = preds * self.y_scale + self.y_offset
        if returnParams:
            return preds, mu_s, cov_s
        return preds
        


    def score(self, X, y):
        preds = self.predict(X)
        return mse(preds, y)



def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()
    plt.show()


def easyTest():
    X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
    # Noise free training data
    X_train = np.array([-4, -3, -2, -1, 1], dtype=np.float).reshape(-1, 1)
    Y_train = np.sin(X_train)
    algo = GaussianProcessRegressor(kernel=KernelRBF(), alpha=1e-8, normalize_y=False)
    algo.fit(X_train, Y_train)
    Y_test, mu_s, cov_s = algo.predict(X_test, returnParams=True)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X_test, X_train=X_train, Y_train=Y_train, samples=samples)
    pass

if __name__ == "__main__":
    #easyTest()
    # Kernel with parameters given in GPML book
    k1 = 66.0**2 * KernelRBF(length=67.0)  # long term smooth rising trend
    k2 = 2.4**2 * KernelRBF(length=90.0) \
        * KernelExpSineSquared(length=1.3, periodicity=1.0)  # seasonal component
    # medium term irregularity
    k3 = 0.66**2 \
        * KernelRationalQuadratic(length=1.2, alpha=0.78)
    k4 = 0.18**2 * KernelRBF(length=0.134) \
        + KernelWhite(noise=0.19**2)  # noise terms
    kernel_gpml = k1 + k2 + k3 + k4

    algo = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, normalize_y=True)
    testMauna(algo)