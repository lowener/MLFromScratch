import numpy as np
from MLFromScratch.Tests import testFaithful
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import scale, ScoreMulticlass
from MLFromScratch.KMeans import KMeans
from scipy import stats


class GMM(AlgorithmBase):
    """
    Gaussian Mixture Model
    References:
        Pattern Recognition and Machine Learning, Section 9.2.2, Page 435
    """

    def __init__(self, n_components, tol=1e-3, max_iter=100, initialization="random"):
        """
        @param n_components: Number of Gaussians
        @param tol: Tolerance for convergence
        @param max_iter: maximum iteration of the EM algorithm
        @param initialization: {'random', 'kmeans'}
                Initializatioin strategy
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        if initialization not in {"random", "kmeans"}:
            raise NotImplementedError
        self.initialization = initialization
        self.init_params()

    def init_params(self, means=None, covs=None, coefs=None):
        self.means_init = means
        self.covs_init = covs
        self.coefs_init = coefs

    def init_kmeans(self, X, max_iters=1000, scale=False, tol=1e-5):
        km = KMeans(self.n_components, max_iters, scale, tol)
        km.fit(X)
        self.means_init = km.centers

    def fit(self, X, y=None):
        n_sample, n_features = X.shape

        # Initialize means, covariance and mixing coefficients
        if self.means_init is None:
            if self.initialization == "kmeans":
                self.init_kmeans(X)
            else:
                self.means_init = np.random.uniform(size=(self.n_components, n_features))
        if self.covs_init is None:
            self.covs_init = np.random.uniform(size=(self.n_components, n_features))
        if self.coefs_init is None:
            self.coefs_init = np.random.uniform(size=(self.n_components))

        self.means = self.means_init
        self.covs = self.covs_init
        self.coefs = self.coefs_init

        # Evaluate initial log likelihood
        llh = self.loglikelihood(X)
        prev_llh = 0

        self.history = [llh]
        n_iters = 0
        # Check convergence
        while np.abs(prev_llh - llh) > self.tol and n_iters < self.max_iter:
            # Classical Expectation-Maximization (EM) algorithm
            gamma_z = self.e_step(X)
            means, covs, coefs = self.m_step(X, gamma_z)

            # Update parameters
            self.means = means
            self.covs = covs
            self.coefs = coefs
            # Update log likelihood
            prev_llh = llh
            llh = self.loglikelihood(X)
            self.history.append(llh)
            n_iters += 1

    def e_step(self, X):
        """
        E step: Evaluate the responsibilities using the current parameter values
        """
        n_sample, n_features = X.shape
        components = np.zeros((n_sample, self.n_components))
        for k in range(self.n_components):
            pdf = stats.multivariate_normal.pdf(
                X, self.means[k], self.covs[k], allow_singular=True
            )
            components[:, k] = self.coefs[k] * pdf
        sum_components = np.sum(components, axis=1)
        # Contribution of each component to the llh
        # Gamma_z is the responsibility that a component takes for explaining an observation
        gamma_z = np.zeros((n_sample, self.n_components))
        for k in range(self.n_components):
            gamma_z[:, k] = components[:, k] / sum_components
        return gamma_z

    def m_step(self, X, gamma_z):
        """
        M step: Re-estimate the parameters (means, covs, coefs) using the current responsibilities
        """
        EPS = 1e-8
        n_sample, n_features = X.shape
        nks = np.sum(gamma_z, axis=(0))
        means = np.zeros((self.n_components, n_features))
        covs = np.zeros((self.n_components, n_features))
        coefs = nks / n_sample
        for k in range(self.n_components):
            for n in range(n_sample):
                means[k] += gamma_z[n, k] * X[n]
            means[k] /= nks[k] + EPS
            for n in range(n_sample):
                covs[k] += gamma_z[n, k] * ((X[n] - means[k]) ** 2)
            covs[k] /= nks[k] + EPS
        return means, covs, coefs

    def loglikelihood(self, X):
        """
        Compute Log-Likelihood
        """
        n_sample, n_features = X.shape
        llh = 0
        for n in range(n_sample):
            local_llh = np.zeros(n_features)
            for k in range(self.n_components):
                # Compute the probability density function
                pdf = stats.multivariate_normal.pdf(
                    X[n], self.means[k], self.covs[k], allow_singular=True
                )
                local_llh += self.coefs[k] * pdf
            llh += np.log(local_llh)
        llh = np.sum(llh)
        return llh

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def predict_proba(self, X):
        return self.e_step(X)

    def score(self, X, y):
        preds = self.predict(X)
        return ScoreMulticlass(y, preds)


def testGMM(algo):
    # Taken from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import StratifiedKFold

    colors = ["navy", "turquoise", "darkorange"]

    def make_ellipses(gmm, ax):
        for n, color in enumerate(colors):
            covariances = np.diag(gmm.covs[n][:2])
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                gmm.means[n, :2], v[0], v[1], 180 + angle, color=color, fill=False
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")

    iris = datasets.load_iris()

    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    X_test = iris.data[test_index]
    y_test = iris.target[test_index]

    n_classes = len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    estimators = {
        cov_type: GMM(n_components=n_classes, max_iter=20) for cov_type in ["full"]
    }

    n_estimators = len(estimators)
    plt.figure(figsize=(6, 4))
    plt.subplots_adjust(
        bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
    )
    for index, (name, estimator) in enumerate(estimators.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.init_params(
            np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])
        )

        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        h = plt.subplot(2, 1, index + 1)
        make_ellipses(estimator, h)

        for n, color in enumerate(colors):
            data = iris.data[iris.target == n]
            plt.scatter(
                data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n]
            )
        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

        y_train_pred = estimator.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(
            0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes
        )

        y_test_pred = estimator.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(
            0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes
        )

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))

    plt.show()


if __name__ == "__main__":
    testGMM(GMM(3))
    # testFaithful(GMM(2))
