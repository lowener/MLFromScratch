import numpy as np
from MLFromScratch.Base import AlgorithmBase
from MLFromScratch.Tools import cross_entropy, ScoreMulticlass, scale, Softmax
from MLFromScratch.Tests import testIris
from MLFromScratch.Optimizer import LinearGradientDescent


class LinearSVC(AlgorithmBase):
    """
    References:
        https://scikit-learn.org/stable/modules/svm.html#linearsvc
        https://en.wikipedia.org/wiki/Hinge_loss
        https://cs231n.github.io/optimization-1/#analytic
    """

    def __init__(
        self,
        C=1.0,
        l1_ratio=0.0,
        l2_ratio=1e-1,
        multi_class="ovr",
        fit_intercept=True,
        n_iters=10000,
        lr=1e-3,
        scale=True,
    ):
        """
        @param C: Regularization parameter
        @param l1_ratio: Ratio of L1 loss
        @param l2_ratio: Ratio of L2 loss
        @param multi_class: {'ovr', 'multi'}
                 Multi-class strategy to adopt (One Versus Rest with Hinge Loss, Multi class with Crammer and Singer Loss)
        @param fit_intercept: Use a bias?
        @param n_iters: Number of iterations
        @param scale: Scale the data before using it?
        """
        self.C = C
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.n_iters = n_iters
        self.lr = lr
        self.scale = scale


    def fit(self, X: np.array, y: np.array):
        if self.scale:
            X, self.X_offset, self.X_scale = scale(X)
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)

        self.n_samples, n_features = X.shape

        if self.multi_class == "ovr":
            predFn = lambda X, W: (X.dot(W))
            self.W = []
            self.history = []
            for i in range(n_classes):
                W = np.random.rand(n_features)
                this_y = np.zeros((n_samples))
                this_y[y[:, i] == 1] = 2
                this_y -= 1
                W, history = LinearGradientDescent(
                    self.dhinge_loss,
                    X,
                    this_y,
                    W,
                    self.n_iters,
                    self.lr,
                    l1_ratio=self.l1_ratio,
                    l2_ratio=self.l2_ratio,
                    metric=self.hinge_loss,
                    predFn=predFn,
                )
                self.W.append(W)
                self.history.append(history)
        elif self.multi_class == "multi":
            predFn = lambda X, W: Softmax(X.dot(W))  # Softmax for cross_entropy metric
            W = np.random.rand(n_features, n_classes)
            self.W, self.history = LinearGradientDescent(
                self.dcrammer_singer_loss,
                X,
                y,
                W,
                self.n_iters,
                self.lr,
                l1_ratio=self.l1_ratio,
                l2_ratio=self.l2_ratio,
                metric=cross_entropy,
                predFn=predFn,
            )
        else:
            raise NotImplementedError

    def predict(self, X: np.array) -> np.array:
        EPS = 1e-10
        if self.scale:
            X = np.array(X, dtype=np.float32)
            X = (X - self.X_offset) / (self.X_scale + EPS)
        n_samples, _ = X.shape
        if self.fit_intercept:
            ones = np.ones((n_samples, 1))
            X = np.concatenate((ones, X), axis=1)

        if self.multi_class == "ovr":
            preds = []
            for i in range(len(self.W)):
                preds.append(X.dot(self.W[i]))
            preds = np.array(preds).T
        elif self.multi_class == "multi":
            preds = X.dot(self.W)
        else:
            raise NotImplementedError

        return preds

    def score(self, X: np.array, y: np.array) -> ScoreMulticlass:
        preds = self.predict(X).argmax(1)
        return ScoreMulticlass(y, preds)

    def hinge_loss(self, target: np.array, preds: np.array) -> np.float32:
        hinge_loss = 1 - target * preds
        hinge_loss[hinge_loss < 0] = 0
        return np.sum(hinge_loss)

    def dhinge_loss(self, X: np.array, y: np.array, W: np.array) -> np.array:
        preds = X.dot(W)
        dHL = np.zeros(np.shape(preds))
        ty = y * preds
        dHL[ty < 1] = -y[ty < 1]
        dHL = self.C * (1 / self.n_samples) * X.T.dot(dHL)
        return dHL

    def dcrammer_singer_loss(
        self, X: np.array, y: np.array, W: np.array, vectorized: bool = True
    ) -> np.array:
        n_samples, n_classes = y.shape
        preds = X.dot(W)
        if vectorized:
            dW = np.zeros(preds.shape)
            yis = np.where(y == 1)
            error = preds - preds[yis].reshape((-1, 1)).repeat(3, axis=1) + 1
            error[yis] = 0
            dW[error > 0] = 1
            where_e = (error > 0).sum(1)
            dW[yis] -= where_e
            dW = self.C * (1 / self.n_samples) * X.T.dot(dW)
        else:
            dW = np.zeros(W.shape)
            for i in range(n_samples):
                yi = y[i].argmax()
                for j in range(n_classes):
                    if j == yi:
                        continue
                    if preds[i, j] - preds[i, yi] + 1 > 0:
                        dW[:, j] += X[i]
                        dW[:, yi] += -X[i]
            dW = self.C * (1 / self.n_samples) * dW
        return dW


if __name__ == "__main__":
    testIris(
        LinearSVC(lr=0.1, l2_ratio=1e-3, fit_intercept=True, multi_class="ovr")
    )  # F1 0.92
    testIris(
        LinearSVC(lr=0.1, l2_ratio=1e-2, fit_intercept=False, multi_class="ovr")
    )  # F1 0.84
    testIris(LinearSVC(lr=0.1, fit_intercept=True, multi_class="multi"))  # F1 0.92
    testIris(LinearSVC(lr=0.1, fit_intercept=False, multi_class="multi"))  # F1 0.72
