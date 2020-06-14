import numpy as np


class Fitness:
    def __init__(self, dimension, weight_limit):
        self.values = np.random.randint(100, size=dimension)
        self.weights = np.random.randint(10, size=dimension)
        self.weight_limit = weight_limit

    def evaluate(self, x):
        (ro, col) = x.shape
        for ind1 in range(ro):
            fit = 0
            weight = 0
            for ind2 in range(col - 1):
                weight = weight + self.weights[ind2] * x[ind1, ind2]
                if weight > self.weight_limit:
                    x[ind1, ind2] = 0
                else:
                    fit = fit + self.values[ind2] * x[ind1, ind2]

            x[ind1, col - 1] = fit

        return x
