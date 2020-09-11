import numpy as np
from MLFromScratch.Tests import testIris, testBreast, testDigits
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import ScoreMulticlass


class NaiveBayes(AlgorithmBase):
    """
    Gaussian Naive Bayes
    References:
        https://en.wikipedia.org/wiki/Naive_Bayes_classifier

    """

    def __init__(
        self,
    ):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        _, self.n_classes = y.shape

        self.p_classes = np.zeros(self.n_classes)
        self.mean_class = np.zeros((self.n_classes, n_features))
        self.variance_class = np.zeros((self.n_classes, n_features))

        for i in range(self.n_classes):
            self.p_classes[i] = sum(y[:, i]) / n_samples
            Xi = X[y[:, i] == 1]
            for j in range(n_features):
                self.mean_class[i, j] = Xi[:, j].mean()
                self.variance_class[i, j] = Xi[:, j].var()

    def predict(self, X):
        EPS = 1e-10
        n_samples, n_features = X.shape
        y = np.zeros((n_samples, self.n_classes))
        for i in range(self.n_classes):
            y[:, i] = self.p_classes[i]
            for j in range(n_features):
                y[:, i] *= 1 / (EPS + np.sqrt(2 * np.pi * self.variance_class[i, j]))
                y[:, i] *= np.exp(
                    -((X[:, j] - self.mean_class[i, j]) ** 2)
                    / (EPS + 2 * self.variance_class[i, j])
                )
        return y

    def score(self, X, y):
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)


if __name__ == "__main__":
    testIris(NaiveBayes())
    testBreast(NaiveBayes())
    testDigits(NaiveBayes())
