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
        noise = self.alpha * np.eye(n_sample)
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

        
        if self.normalize_y: # Undo Normalisation
            mu_s = mu_s * self.y_scale + self.y_offset
            cov_s = cov_s * self.y_scale + self.y_offset


        preds = np.random.multivariate_normal(mu_s.ravel(), cov_s, 1)
        if returnParams:
            return preds, mu_s, cov_s
        return preds
        


    def score(self, X, y):
        preds = self.predict(X)
        return mse(preds, y)



def easyTest():
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
    easyTest()
    k1 = 50.0**2 * KernelRBF(length=50.0)
    k2 = 2.0**2 * KernelRBF(length=100.0) \
        * KernelExpSineSquared(length=1.0, periodicity=1.0)
    k3 = 0.5**2 * KernelRationalQuadratic(length=1.0, alpha=1.0)
    k4 = 0.1**2 * KernelRBF(length=0.1) \
        + KernelWhite(noise=0.1**2)
    kernel = k1 + k2 + k3 + k4
    algo = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False)
    testMauna(algo)