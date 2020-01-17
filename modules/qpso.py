import numpy as np
import random
from modules import ann

a = 0.2
b = 0.95
initial_fitness = -100000

class Swarm:
    def __init__(self, Np, MaxIter, N, input_nodes, hidden_nodes, train_data, y_true_train, C):
        self.X = []
        self.Np = Np
        self.D = input_nodes * hidden_nodes
        self.p = []
        self.pg = []
        self.MaxIter = MaxIter
        self.iteration = 0
        self.a = -1
        self.phi = -1
        self.mu = -1
        self.mBest = []
        self.N = N

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.train_data = train_data
        self.y_true_train = y_true_train
        self.C = C

    def init_swarm(self):
        for i in range(self.D):
            self.X.append([])
            self.p.append([])
            for j in range(self.Np):
                aux = random.random()
                self.X[-1].append(aux)
                self.p[-1].append(aux)

            self.pg.append(initial_fitness)

        self.phi = random.random()
        self.mu = random.random()

        while self.iteration < self.MaxIter:
            self.iteration += 1
            self.mbest()
            self.alpha()

            self.local_atractor()

            for i in range(self.D):
                for j in range(self.Np):
                    self.new_swarm()
                self.test()

    def new_x(self, i, j):
        if random.random() > 0.5:
            self.X[i][j] = self.p[i][j] + self.a * abs(self.mbest[j] - self.X[i][j]) * np.log(1/self.mu)
        else:
            self.X[i][j] = self.p[i][j] - self.a * abs(self.mbest[j] - self.X[i][j]) * np.log(1 / self.mu)

    def new_swarm(self):
        for i in range(self.Np):
            for j in range(self.D):
                self.new_x(i, j)

    def local_atractor(self):
        for i in range(self.Np):
            for j in range(self.D):
                self.p[i][j] = self.phi * self.p[i][j] + (1 - self.phi) * self.pg[j]

    def mbest(self):
        self.mBest = []
        for j in range(self.D):
            self.mBest.append((1 / self.Np) * np.sum(self.p[:][j]))

    def alpha(self):
        self.a = (b - a) * ((self.MaxIter - self.iteration) / self.MaxIter) + a

    def test(self):
        weight_matrix = np.reshape(self.pg, self.input_nodes, self.hidden_nodes)
        return ann.test_ann(weight_matrix, self.train_data, self.y_true_data, self.C)
