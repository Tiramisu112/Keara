import numpy as np
import random
from modules import ann


class Swarm:
    def __init__(self, Np, maxIter, a, b, input_nodes, hidden_nodes, train_data, y_true_train, C):
        self.Np = Np
        self.D = input_nodes * hidden_nodes
        self.error_pbest = np.full(Np, np.inf)
        self.error_gbest = np.inf
        self.pbest = []
        self.gbest = []
        self.maxIter = maxIter
        self.iteration = 0
        self.alpha = -1
        self.mBest = []
        self.a = a
        self.b = b

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.train_data = train_data
        self.y_true_train = y_true_train
        self.C = C

    def init_swarm(self):
        X = np.random.random((self.Np, self.D))
        self.pbest = X.copy()

        while self.iteration < self.maxIter:
            self.mbest()
            self.calculate_alpha()

            for i in range(self.Np):
                actual_error = self.test(X[i, :])
                #print(str(actual_error))

                if actual_error < self.error_pbest[i]:
                    self.pbest[i, :] = X[i, :]
                    self.error_pbest[i] = actual_error

                if actual_error < self.error_gbest:
                    self.gbest = X[i, :]
                    self.error_gbest = actual_error

                for j in range(self.D):
                    phi = random.random()
                    p = self.local_atractor(phi, i, j)
                    mu = random.random()

                    if random.random() > 0.5:
                        X[i, j] = p + self.alpha * abs(self.mBest[j] - X[i, j]) * np.log(1 / mu)
                    else:
                        X[i, j] = p - self.alpha * abs(self.mBest[j] - X[i, j]) * np.log(1 / mu)

            self.iteration += 1
            print()
            print('Iteration: ' + str(self.iteration))
            print('MSE: ' + str(self.error_gbest))
            print()
            print('---------------------------------')

    def local_atractor(self, phi, i, j):
        return phi * self.pbest[i, j] + (1 - phi) * self.gbest[j]

    def mbest(self):
        self.mBest = []
        for j in range(self.D):
            self.mBest.append(np.sum(self.pbest[:, j]) / self.Np)

    def calculate_alpha(self):
        self.alpha = (self.b - self.a) * ((self.maxIter - self.iteration) / self.maxIter) + self.a

    def test(self, weights_matrix):
        weights_matrix = weights_matrix.reshape(self.input_nodes, self.hidden_nodes)
        return ann.test_mse(weights_matrix, self.train_data, self.y_true_train, self.C)

    def get_gbest_weights(self):
        return self.gbest.reshape(self.input_nodes, self.hidden_nodes)