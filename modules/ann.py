import numpy as np
import math
from modules import performance_evaluate


# Get z
def get_z(x_vector, w_vector):
    z = np.subtract(x_vector, w_vector)
    z = np.linalg.norm(z)
    return z


# Get h_j
def activation_function(z):
    h = z * math.exp((-0.5) * math.pow(z, 2))
    return h


# Get y_n
def y_out(B, H):
    return np.dot(B, H)


# Calculate Beta
def B_vector(H, T_vector, N_h, C):
    return np.dot(np.linalg.pinv(np.dot(H, np.transpose(H)) + np.identity(N_h)/C), np.dot(H, np.transpose(T_vector)))


# Calculate MSE
def mse(T_vector, y_vector):
    return np.square(T_vector - y_vector).mean()


# Get the h_j vector values of hidden nodes
def get_h_j(x_vector, w_matrix):
    h_j = []
    for i in range(len(w_matrix[0])):
        h_j.append(activation_function(get_z(x_vector, w_matrix[:, i])))
    return h_j


# Get the H matrix values of hidden nodes
def get_H(train_data, w_matrix):
    H = []
    for x_vector in train_data:
        H.append(get_h_j(x_vector, w_matrix))
    return np.transpose(H)


# Predict whether attack or not for a given features vector
def predict(x_vector, w_matrix, B):
    return np.dot(B, get_h_j(x_vector, w_matrix))


# Test ann for QPSO
def test_mse(weight_matrix, train_data, y_true_train, C):
    H = get_H(train_data, weight_matrix)
    H = np.array(H)
    B = B_vector(H, y_true_train, len(weight_matrix[0, :]), C)

    y_vector = y_out(B, H)

    return mse(y_true_train, y_vector)


def test(weight_matrix, test_data, y_true_test, C):
    H = get_H(test_data, weight_matrix)
    H = np.array(H)
    B = B_vector(H, y_true_test, len(weight_matrix[0, :]), C)

    y_vector = y_out(B, H)

    for i in range(len(y_vector)):
        if y_vector[i] >= 0:
            y_vector[i] = 1
        else:
            y_vector[i] = -1

    tp, tn, fp, fn = performance_evaluate.get_values(y_true_test, y_vector)

    p, r, f_s, a = performance_evaluate.get_metrics(tp, tn, fp, fn)

    confusion_matrix = performance_evaluate.get_confusion_matrix(y_true_test, y_vector)
    print(confusion_matrix)

    print(performance_evaluate.get_report(y_true_test, y_vector))

    print('---------------------------------')
    print('MSE: ', mse(y_true_test, y_vector))
    print()
    print('Precision: ', str(p))
    print('Recall: ', str(r))
    print('F-score: ', str(f_s))
    print('Accuracy: ', str(a))
    print('---------------------------------')
