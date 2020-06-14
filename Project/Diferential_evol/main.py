from diferential_evolution import DiferentialEvolution
import matplotlib.pyplot as plt

G = 1000
F = 1
CR = 0.5
D = 2
NP = 100

DE = DiferentialEvolution(F, CR, NP, D)
for ind in range(G):
    DE.iterate()
    print(ind)
    plt.clf()
    plt.scatter(DE.x[:, 0], DE.x[:, 1], marker='o')
    plt.xlim((-15, 15))
    plt.ylim((-15, 15))
    plt.pause(0.05)

plt.show()
