import numpy as np

def mse(target:np.array, preds:np.array):
    return ((target - preds) ** 2).mean()

def cross_entropy(target:np.array, preds:np.array):
    res=0
    
    EPS = 1e-10
    for c in range(target.shape[1]):
        tmp = np.array(target.argmax(1) == c, dtype=np.int)
        tmp2 = np.log(preds[:, c] + EPS)
        res += np.sum(tmp * tmp2)

    return -res


def binary_cross_entropy(target:np.array, preds:np.array):
    EPS = 1e-10
    target = target / 2 + 0.5
    res = target * np.log(preds + EPS) + (1 - target) * np.log(1 - preds + EPS)

    return -res.sum()