import numpy as np
import math


def evaluate(x):
    NP, D = x.shape
    fit = np.zeros(NP)
    for ind in range(NP):
        # fit[ind] = x[ind, 0]**2 + x[ind, 1]**2
        fit[ind] = -(1 - (math.sin(math.sqrt(x[ind, 0] ** 2 + x[ind, 1] ** 2))) ** 2) / (
                    1 + 0.001 * (x[ind, 0] ** 2 + x[ind, 1] ** 2))
    return fit
