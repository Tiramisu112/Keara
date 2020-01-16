import numpy as np
import math


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
def mse(T_vector, y_vector, N):
    return np.square(T_vector - y_vector).mean()/N


# Get the h_j vector values of hidden nodes
def get_h_j(x_vector, w_matrix):
    h_j = []
    for i in range(len(x_vector)):
        h_j.append(activation_function(get_z(x_vector, w_matrix[:, i])))
    return h_j


# Get the H matrix values of hidden nodes
def get_H(train_data, w_matrix):
    H = []
    for x_vector in train_data:
        H.append(get_h_j(x_vector, w_matrix))
    return H


# Predict whether attack or not for a given features vector
def predict(x_vector, w_matrix, B):
    return np.dot(B, get_h_j(x_vector, w_matrix))
