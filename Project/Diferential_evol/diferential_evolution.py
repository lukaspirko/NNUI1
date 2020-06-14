import numpy as np
from Fitness import evaluate


class DiferentialEvolution:
    def __init__(self, F, CR, NP, D):
        self.F = F
        self.CR = CR
        self.NP = NP
        self.D = D
        self.x = self.initialize()

    def initialize(self):
        # inicializace
        return 30 * np.random.rand(self.NP, self.D) - 15

    def generate_noise_vector(self):
        # mutace
        noise_vector = np.zeros(self.x.shape)
        for ind in range(self.NP):
            r = np.random.randint(0, self.NP, size=3)
            noise_vector[ind, :] = self.x[r[0], :] + self.F * (self.x[r[1], :] - self.x[r[2], :])

        return noise_vector

    def generate_trial_vector(self, noise_vector):
        # krizeni
        trial_vector = np.zeros(self.x.shape)
        for ind1 in range(self.NP):
            for ind2 in range(self.D):
                if np.random.rand() < self.CR:
                    trial_vector[ind1, ind2] = noise_vector[ind1, ind2]
                else:
                    trial_vector[ind1, ind2] = self.x[ind1, ind2]

        return trial_vector

    def generate_offspring(self, trial_vector):
        # selekce
        offspring = np.zeros(self.x.shape)
        fitness_x = evaluate(self.x)
        fitness_trial = evaluate(trial_vector)
        for ind in range(self.NP):
            if fitness_trial[ind] < fitness_x[ind]:
                offspring[ind, :] = trial_vector[ind, :]
            else:
                offspring[ind, :] = self.x[ind, :]

        return offspring

    def iterate(self):
        noise_vector = self.generate_noise_vector()
        trial_vector = self.generate_trial_vector(noise_vector)
        self.x = self.generate_offspring(trial_vector)
