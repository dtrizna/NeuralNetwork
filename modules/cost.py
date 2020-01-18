import numpy as np

def compute_cost(AL, Y):
    """
    cross-entropy cost
    """
    
    m = Y.shape[1]
    cost = (np.dot(Y, np.log(AL).T) + np.dot((1 - Y), np.log(1 - AL).T)) / -m
    
    cost = np.squeeze(cost) # e.g. this turns [[17]] into 17
    assert(cost.shape == ())
    return cost

def compute_cost_with_L2regularization(AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization.
    Returns:    
    cost - value of the regularized loss function.
    """
    m = Y.shape[1]
    L = len([key for key in parameters.keys() if 'W' in key])
    
    cross_entropy_cost = compute_cost(AL, Y) # This gives the cross-entropy part of the cost
    
    L2_regularization_cost = 0
    for l in range(1, L+1):
        L2_regularization_cost += np.sum(np.square(parameters["W" + str(l)])) * lambd / (2 * m) 
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
