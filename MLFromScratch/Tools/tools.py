import numpy as np

def scale(X):
    '''
    Returns: X, X_offset, X_scale
    '''
    EPS = 1e-10
    X = np.array(X, dtype=np.float32)
    X_offset = np.mean(X, axis=0)
    X -= X_offset
    X_scale = np.max(np.abs(X), axis=0)
    X /= X_scale + EPS
    return X, X_offset, X_scale