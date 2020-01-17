import sys
import numpy as np
from modules import data_arrangement as data_arr
from modules import qpso
from modules import performance_evaluate as perf_ev
from modules import ann

root_path = sys.executable[:-23]

# Define network topology
input_nodes = 42                                # Number of input nodes
hidden_nodes = 42
# hidden_nodes = np.sqrt(input_nodes + 1) + 5     # Number of hidden nodes
N = 5000                                        # Data sample

train_data, y_true_train = data_arr.pre_process_data(root_path + 'data/KDDTrain_20Percent.txt', N)
test_data, y_true_test = data_arr.pre_process_data(root_path + 'data/KDDTest.txt', N)

# Define QPSO topology
P = 20                                          # Number of particles
maxIter = 100

# QPSO
swarm = qpso.Swarm(P, maxIter, N, input_nodes, hidden_nodes, train_data, y_true_train, 2)
swarm.init_swarm()