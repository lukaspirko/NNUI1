import numpy as np
from Fitness import Fitness


class Genetic_Algorithm:
    def __init__(self, dimension, weight_limit, number_ind, number_players, crossover_point, prob_mutation):
        self.fitness = Fitness(dimension, weight_limit)
        self.x = self.initialise(number_ind, dimension)
        self.number_players = number_players
        self.crossover_point = crossover_point
        self.prob_mutation = prob_mutation

    def initialise(self, number_ind, dimension):
        return np.random.randint(2, size=(number_ind, dimension + 1))

    def select(self):
        (ro, col) = self.x.shape
        x_new = np.zeros((ro, col))
        for ind1 in range(ro):
            indices = np.random.choice(ro, self.number_players, replace=False)
            fitness_local = np.zeros(self.number_players)
            for ind2 in range(self.number_players):
                fitness_local[ind2] = self.x[indices[ind2], col - 1]

            ind_win = np.argmax(fitness_local)
            x_new[ind1, :] = self.x[indices[ind_win], :]

        return x_new

    def crossover(self):
        (ro, col) = self.x.shape
        x_new = self.x
        for ind in range(0, ro - 1, 2):
            x_new[ind, self.crossover_point:col - 2] = self.x[ind + 1, self.crossover_point:col - 2]
            x_new[ind + 1, self.crossover_point:col - 2] = self.x[ind, self.crossover_point:col - 2]

        return x_new

    def mutate(self):
        (ro, col) = self.x.shape
        x_new = self.x
        for ind1 in range(ro):
            for ind2 in range(col - 1):
                if np.random.rand() < self.prob_mutation:
                    if x_new[ind1, ind2] == 1:
                        x_new[ind1, ind2] = 0
                    else:
                        x_new[ind1, ind2] = 1
        return x_new

    def evaluate(self):
        return self.fitness.evaluate(self.x)

    def get_fitness(self):
        (ro, col) = self.x.shape
        fitness = np.zeros(ro)
        for ind in range(ro):
            fitness[ind] = self.x[ind, col - 1]

        fitness_max = np.max(fitness)
        fitness_mean = np.mean(fitness)

        return fitness_max, fitness_mean

    def iteration(self):
        self.x = self.evaluate()
        self.x = self.select()
        self.x = self.crossover()
        self.x = self.mutate()
