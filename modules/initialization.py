import numpy as np

def initialize_parameters_random(architecture):
    """
    architecture -- list with activation units in every layer (including input).

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL"
    
    Values or W -- randomly chosen using Normal distribution in range of apprx. -0.03 : 0.03
    """
    
    parameters = {}
    L = len(architecture) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(architecture[l], architecture[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros([architecture[l], 1])
        
    return parameters


def initialize_parameters_he(architecture):
    """
    architecture -- list with activation units in every layer (except input).
    
    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
    
    Values of W -- randomly values using He initialization.
    Shows better performance with deep network (many layers) and ReLU activation function
    to partially overcome Vanishing/Exploding weight problem.
    """
    
    parameters = {}
    L = len(architecture) # number of layers in the network
     
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(architecture[l], architecture[l-1]) * np.sqrt(2 / architecture[l-1])
        parameters['b' + str(l)] = np.zeros([architecture[l], 1])
        
    return parameters
