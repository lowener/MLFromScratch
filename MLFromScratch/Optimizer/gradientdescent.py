import numpy as np

def LinearGradientDescent(fun, X, y, W, n_iters, lr=0.01, l1_ratio=0.0, l2_ratio=0.0, metric=None, predFn=None):
    history = []
    for i in range(n_iters):
        deriv = fun(X, y, W)
        
        dl1 = np.sign(W)
        dl2 = 2*W
        W = W - lr * (deriv + l1_ratio * dl1 + 0.5 * l2_ratio * dl2)
        if np.isnan(W).sum() > 0:
            raise ArithmeticError("Nan encountered in W at iteration " + str(i))
        if metric is not None:
            history.append(metric(y, predFn(X, W)))
    return W, history