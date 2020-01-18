import sys
from modules import data_arrangement as data_arr
from modules import qpso
from modules import ann

from modules import performance_evaluate

#root_path = sys.executable[:-23]

# Define network topology
input_nodes = 41                                # Number of input nodes
hidden_nodes = 41
# hidden_nodes = np.sqrt(input_nodes + 1) + 5     # Number of hidden nodes
N = int(sys.argv[1])                                       # Data sample
M = 3000

train_data, y_true_train = data_arr.pre_process_data(
    'data/KDDTrain_20Percent.txt', N)
test_data, y_true_test = data_arr.pre_process_data('data/KDDTest.txt', M)


# Define QPSO topology
P = int(sys.argv[2])                                          # Number of particles
maxIter = int(sys.argv[3])

C = int(sys.argv[4])

a = 0.2
b = 0.95

# QPSO
swarm = qpso.Swarm(P, maxIter, a, b, input_nodes, hidden_nodes, train_data, y_true_train, C)
swarm.init_swarm()

weights = swarm.get_gbest_weights()
ann.test(weights, test_data, y_true_test, C)

B = ann.B_vector(ann.get_H(train_data, weights), y_true_train, hidden_nodes, C)


x = '-1'
while x != 'exit':
    if x != 'exit':
        x = str(input('Ingrese datos a predecir: (ingrese exit para salir)'))
        x = x.strip().split(',')

        prediction = ann.predict(data_arr.norma(data_arr.numericalise_data_x(x)), weights, B)
        print()
        print(prediction)
        if prediction > 0:
            print('Normal')
        else:
            print('Amenaza')

        print()
