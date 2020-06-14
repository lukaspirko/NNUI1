from Genetic_Algorithm import Genetic_Algorithm
import numpy as np
import matplotlib.pyplot as plt

dimension = 500
weight_limit = 20
number_ind = 50
number_players = 2
crossover_point = 250
prob_mutation = 0.05

ga = Genetic_Algorithm(dimension, weight_limit, number_ind, number_players, crossover_point, prob_mutation)

gens = 100

fitness_max_course = np.zeros(gens + 1)
fitness_mean_course = np.zeros(gens + 1)

for ind in range(gens):
    fitness_max_course[ind], fitness_mean_course[ind] = ga.get_fitness()
    ga.iteration()
    print(ind)

fitness_max_course[gens], fitness_mean_course[gens] = ga.get_fitness()

plt.plot(fitness_max_course, label='Max')
plt.plot(fitness_mean_course, label='Mean')

plt.xlabel('Generace')
plt.ylabel('Fitness')
plt.legend()
plt.show()
