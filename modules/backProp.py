import numpy as np
from modules.activationFunctions import sigmoid_gradient, relu_gradient


def backprop(dA, Z, W, A_prev, activation, lambd=0):
    """
    Derivative formulas.
    """
    m = A_prev.shape[1]

    if activation == "relu":
        dZ = relu_gradient(dA, Z)
        dW = (np.dot(dZ, A_prev.T) / m) + (lambd * W / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

    elif activation == "sigmoid":
        dZ = sigmoid_gradient(dA, Z)
        dW = (np.dot(dZ, A_prev.T) / m) + (lambd * W / m)
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

    assert dW.shape == W.shape
    assert dA_prev.shape == A_prev.shape

    return dA_prev, dW, db


def L_model_backprop(AL, Y, caches, lambd=0):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing values from forwar propagation step for every Layer
            Should contain: {'Z', 'A', 'W'} for every layer.

    Returns:
    grads -- A dictionary with the gradients
    """

    grads = {}
    L = len([key for key in caches.keys() if 'W' in key]) # the number of layers
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Cost Function's partial derivative in respect to AL
    # EPSILON ADDED FOR NUMERICAL STABILITY (not to divide by 0)
    epsilon = 1e-5
    dAL = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
    
    # Lth layer (SIGMOID -> LINEAR) gradients
    grads["dA" + str(L-1)], \
    grads["dW" + str(L)], \
    grads["db" + str(L)] = backprop(dAL, \
                                caches['Z'+str(L)], \
                                caches['W'+str(L)], \
                                caches['A'+str(L-1)], \
                                "sigmoid", lambd)

    for l in reversed(range(1, L)):
        # lth layer: (RELU -> LINEAR) gradients
        dA_prev_temp, dW_temp, db_temp = backprop(grads["dA" + str(l)], \
            caches['Z'+str(l)], caches['W'+str(l)], caches['A'+str(l-1)], "relu", lambd)
        grads["dA" + str(l-1)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads
