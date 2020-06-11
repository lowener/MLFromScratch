import numpy as np

def mse(y:np.array, y2:np.array):
    return ((y - y2) ** 2).mean()