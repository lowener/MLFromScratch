import numpy as np
from MLFromScratch.Tests import testMauna
from MLFromScratch.Base import AlgorithmMixin
from MLFromScratch.Tools import Score, ScoreMulticlass, Sigmoid, Softmax



class GaussianProcessClassificator(AlgorithmMixin):
    '''
    References:
        https://krasserm.github.io/2018/03/19/gaussian-processes/
        Gaussian Processes for Machine Learning (GPML) by Rasmussen and Williams
        https://www.youtube.com/watch?v=R-NUdqxKjos

    '''
    def __init__(self, kernel, alpha=1e-8, normalize_y=False):
        self.kernel = kernel
        self.alpha = alpha
        raise NotImplementedError()


    def fit(self, X, y):
        n_sample, n_features = X.shape
        _, n_classes = y.shape
        self.X_dataset = X
        self.y_dataset = y
        noise = self.alpha * np.eye(n_sample)
        #prior
        w_cov = self.kernel(X, X) + noise
        self.K = w_cov
        self.w_cov_i = np.linalg.pinv(w_cov)
        raise NotImplementedError()


    def predict(self, X, returnCov=False):
        raise NotImplementedError()
        n_sample, _ = X.shape
        noise = self.alpha * np.eye(n_sample)
        Ks = self.kernel(self.X_dataset, X)
        Kss = self.kernel(X, X) + noise

        #Posterior
        mu_s = Ks.T.dot(self.w_cov_i).dot(self.y_dataset)


        if not returnCov:
            return mu_s
        else:
            cov_s = Kss - Ks.T.dot(self.w_cov_i).dot(Ks)
            return mu_s, cov_s
        


    def score(self, X, y):
        preds, _ = self.predict(X, True)
        return ScoreMulticlass(y, preds)



def easyTest():
    pass


if __name__ == "__main__":
    from MLFromScratch.Kernels import KernelRBF, KernelExpSineSquared, KernelWhite, KernelRationalQuadratic
    
    algo = GaussianProcessClassificator(kernel=kernel, alpha=0, normalize_y=False)