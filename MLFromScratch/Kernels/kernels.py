import numpy as np

from scipy.spatial.distance import cdist

class Kernel:
    def __init__(self):
        raise NotImplementedError

    def __call__(self, X, Y):
        raise NotImplementedError

    def __add__(self, other):
        if not isinstance(other, Kernel):
            return KernelSum(self, KernelConstant(other))
        return KernelSum(self, other)

    def __radd__(self, other):
        if not isinstance(other, Kernel):
            return KernelSum(KernelConstant(other), self)
        return KernelSum(other, self)

    def __mul__(self, other):
        if not isinstance(other, Kernel):
            return KernelProduct(self, KernelConstant(other))
        return KernelProduct(self, other)

    def __rmul__(self, other):
        if not isinstance(other, Kernel):
            return KernelProduct(KernelConstant(other), self)
        return KernelProduct(other, self)



class KernelRBF(Kernel):
    def __init__(self, length=1.0):
        self.length = length

    def __call__(self, X, Y):
        #dist = euclideanDist(X,Y)
        dist = cdist(X / self.length, Y / self.length,
                          metric='sqeuclidean')
        return np.exp(-0.5 * dist)


class KernelProduct(Kernel):
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2
        pass

    def __call__(self, X, Y):
        return self.K1(X,Y) * self.K2(X,Y)


class KernelSum(Kernel):
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2
        pass

    def __call__(self, X, Y):
        return self.K1(X,Y) + self.K2(X,Y)


class KernelExpSineSquared(Kernel):
    def __init__(self, length=1.0, periodicity=1.0):
        self.length = length
        self.periodicity = periodicity

    def __call__(self, X, Y):
        dist = cdist(X,Y)
        sin = np.sin(np.pi * dist / self.periodicity)
        return np.exp(-2 * (sin/self.length)**2)


class KernelRationalQuadratic(Kernel):
    def __init__(self, length=1.0, alpha=1.0):
        self.length = length
        self.alpha = alpha

    def __call__(self, X, Y):
        dist = cdist(X,Y)
        k = 1 + (dist**2) / (2* self.alpha * (self.length**2))
        return k**(-self.alpha)


class KernelWhite(Kernel):
    def __init__(self, noise=1.0):
        self.noise = noise

    def __call__(self, X, Y=None):
        if Y is None:
            return self.noise * np.eye(X.shape[0], X.shape[1])
        return np.zeros((len(X), len(Y)))


class KernelConstant(Kernel):
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return np.full((X.shape[0], Y.shape[0]), self.value)
