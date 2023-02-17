import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A

def relu(Z):
    A = np.maximum(0, Z)
    return A

def initialize_parameters_layers(layers_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layers_dims)
    for i in range(1, L):
        parameters["W"+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) / np.sqrt(layers_dims[i-1])
        parameters["b"+str(i)] = np.zeros((layers_dims[i], 1))
    return parameters


def linear_fwd(A, W, b):
    Z = (np.dot(W, A)) + b
    return Z

def linear_activation_fwd(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = linear_fwd(A_prev, W, b)
        A = sigmoid(Z)
    elif activation == "relu":
        Z = linear_fwd(A_prev, W, b)
        A = relu(Z)
    return A

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * ((np.dot(Y, (np.log(AL)).T)) + (np.dot((1-Y), (np.log(1-AL)).T)))
    cost = np.squeeze(cost)
    return cost

def update_parameters(parameters, grads, learningRate):
    L = len(parameters) // 2 
    for i in range (L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - (learningRate*grads["dW" + str(i+1)])
        parameters["b" + str(i+1)] = parameters["b" + str(i+1)] - (learningRate*grads["db" + str(i+1)])
    return parameters

def predict(X, Y, parameters, layers_dims):
    Afn = {}
    Afn["A1"] = X
    m = X.shape[1]
    L = len(layers_dims)
    for l in range(1, L-1):
        Afn["A"+str(l+1)] = linear_activation_fwd(Afn["A"+str(l)], parameters["W"+str(l)], parameters["b"+str(l)], "relu")
    Afn["A"+str(L)] = linear_activation_fwd(Afn["A"+str(L-1)], parameters["W"+str(L-1)], parameters["b"+str(L-1)], "sigmoid")
    pred = np.zeros((1, m))
    pred = (Afn["A"+str(L)] > 0.5) * 1.0
    print("Accuracy : " + str(np.sum((pred == Y)/m)))
    return pred