import numpy as np

def euclidianDist(A,B):
    return np.sqrt(A**2 - B**2)

class Kernel:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def __add__(self, other):
        return KernelSum(self, other)

    def __mul__(self, other):
        return KernelProduct(self, other)



class RBF(Kernel):
    def __init__(self, length=1):
        self.length = length

    def __call__(self, X, Y):
        X /= self.length
        Y /= self.length
        dist = euclidianDist(X,Y)
        return np.exp(-0.5 * dist)


class KernelProduct(Kernel):
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2
        pass

    def __call__(self, X, Y):
        return K1(X,Y) * K2(X,Y)


class KernelSum(Kernel):
    def __init__(self, K1, K2):
        self.K1 = K1
        self.K2 = K2
        pass

    def __call__(self, X, Y):
        return K1(X,Y) + K2(X,Y)


class ExpSineSquared(Kernel):
    def __init__(self, length=1.0, periodicity=1.0):
        self.length = length
        self.periodicity = periodicity

    def __call__(self, X, Y):
        dist = euclidianDist(X,Y)
        sin = np.sin(np.pi * dist / self.periodicity)
        return np.exp(-2 * (sin/self.length)**2)


class RationalQuadratic(Kernel):
    def __init__(self, length=1.0, alpha=1.0):
        self.length = length
        self.alpha = alpha

    def __call__(self, X, Y):
        dist = euclidianDist(X,Y)
        k = 1 + (dist**2) / (2* self.alpha * (self.length**2))
        return k**(-self.alpha)


class WhiteKernel(Kernel):
    def __init__(self, noise=1.0):
        self.noise = noise

    def __call__(self, X, Y=None):
        if Y is None or X == Y:
            return self.noise * np.eye(X.shape[0], X.shape[1])
        return np.zeros_like(X)
