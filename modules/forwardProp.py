import numpy as np
from modules.activationFunctions import sigmoid, relu


def activation(Aprev, W, b, activation):

    if activation == "sigmoid":
    
        Z = np.dot(W, Aprev) + b
        A = sigmoid(Z)
    

    elif activation == "relu":
        
        Z = np.dot(W, Aprev) + b
        A = relu(Z)
    
    return A, Z


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation.
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_random() or initialize_parameters_he() 
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches with values from every Layer needed for BackwardProp
    """

    # NEED TO COLLECT CACHE FOR BACWARD PROP!
    # List of Dictionaries in form of {Zidx: value, Widx: value, Aidx: value}
    cache = {}
    A = X
    L = len(parameters) // 2 # number of layers in the neural network (using floored division)

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, Z = activation(A_prev, W, b, "relu")
        #print("Z{} : {}".format(l,Z.shape))
        cache['A' + str(l-1)] = A_prev
        cache['Z' + str(l)] = Z
        cache['W' + str(l)] = W
        
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, Z = activation(A, W, b, "sigmoid")
    cache['A' + str(L-1)] = A
    cache['Z' + str(L)] = Z
    cache['W' + str(L)] = W
    cache['A' + str(L)] = AL
    
    # Sanity check of AL dimensions 
    # (i.e. has activation value for every training example)
    assert(AL.shape == (1,X.shape[1]))

    return AL, cache
