import numpy as np

def relu(Z):
    """
    Implement the RELU function.
    """
    return np.maximum(0, Z)


def relu_gradient(dA, Z):
    """
    Implement the backward propagation for a single RELU unit.
    """
    
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)
    return dZ


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy.
    """
    return 1/(1+np.exp(-Z))


def sigmoid_gradient(dA, Z):
    """
    Implement the backward propagation for a single SIGMOID unit.
    """
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    return dZ
