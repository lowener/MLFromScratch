import numpy as np

def scale(X):
    EPS = 1e-10
    X = np.array(X, dtype=np.float32)
    X_offset = np.average(X, axis=0)
    X -= X_offset
    X_scale = np.max(np.abs(X), axis=0)
    X /= X_scale + EPS
    return X, X_offset, X_scale