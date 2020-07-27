import numpy as np

def LinearGradientDescent(fun, X, y, W, n_iters, lr=0.01, l1_ratio=0.0, l2_ratio=0.0, metric=None, predFn=None):
    history = []
    for i in range(n_iters):
        deriv = fun(X, y, W)
        
        dl1 = np.sign(W)
        dl2 = 2*W
        W = W - lr * (deriv + l1_ratio * dl1 + 0.5 * l2_ratio * dl2)
        if np.isnan(W).sum() > 0:
            raise ArithmeticError("Nan encountered in W at epoch " + str(i))
        if metric is not None:
            history.append(metric(y, predFn(X, W)))
    return W, history

def LinearStochasticGradientDescent(fun, X, y, W, n_iters, lr=0.01, l1_ratio=0.0, l2_ratio=0.0, metric=None, predFn=None, batch_size=4):
    history = []
    for i in range(n_iters):
        shuffle_idx = np.arange(len(X))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        for j in range(len(X) // batch_size):
            Xb = X[j*batch_size:(j+1)*batch_size]
            yb = y[j*batch_size:(j+1)*batch_size]
            deriv = fun(Xb, yb, W)
            
            dl1 = np.sign(W) * (len(Xb) / len(X))
            dl2 = 2*W * (len(Xb) / len(X))
            W = W - lr * (deriv + l1_ratio * dl1 + 0.5 * l2_ratio * dl2)
            if np.isnan(W).sum() > 0:
                raise ArithmeticError("Nan encountered in W at epoch " + str(j) + " iteration " + str(i))
        if metric is not None:
            history.append(metric(y, predFn(X, W)))
    return W, history