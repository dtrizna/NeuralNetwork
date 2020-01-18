import numpy as np
from modules.forwardProp import L_model_forward
from modules.cost import compute_cost

def dictionary_to_vector(d):
    shape_cache = {} 
    key_cache = []
    for i, key in enumerate(d.keys()):
        vector = np.reshape(d[key], (-1,1)) # reshapes any type of array into vector

        if i == 0:
            theta = vector
        else:
            theta = np.concatenate((theta, vector), axis=0)
        
        shape_cache[key] = d[key].shape
        key_cache = key_cache + [key]*vector.shape[0]
        
    return theta, key_cache, shape_cache

def vector_to_dictionary(theta, key_cache, shape_cache):

    dictionary = {}

    for i, key in enumerate(shape_cache.keys()):

        if i == 0:
            first = 0
        else:
            first = last
        
        # Fiding last element of this parameter
        last = len(key_cache) - key_cache[::-1].index(key)

        # Restoring parameter
        vector = theta[first:last].reshape(shape_cache[key])

        # Saving
        dictionary[key] = vector

    return dictionary

def gradient_check(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward propagation logic is implemented correctly.
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", .. "WL", "bL";
    grad --  gradients of the cost with respect to the parameters 
        (can take grads dict from L_model_backProp());
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient
    
    Returns:
    difference -- difference between the approximated gradient and the backward propagation gradient
    """
    
    # Correlate grads and parameters keys
    gradsToCheck = {}
    for key in parameters.keys():
        gradsToCheck['d' + key] = gradients['d' + key]
        
    # Set-up variables
    parameters_values, param_key_cache, param_shape_cache = dictionary_to_vector(parameters)
    grads, _, _ = dictionary_to_vector(gradsToCheck)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        print('Checking gradient of {}/{}'.format(i, num_parameters), end='\r')
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon # [0] needed to take float out of np.array
        parameters = vector_to_dictionary(thetaplus, param_key_cache, param_shape_cache)

        # need to compute:
        AL, _ = L_model_forward(X, parameters)
        J_plus[i] = compute_cost(AL, Y)
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        parameters = vector_to_dictionary(thetaminus, param_key_cache, param_shape_cache)
        AL, _ = L_model_forward(X, parameters)
        J_minus[i] = compute_cost(AL, Y)
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        
    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(grads) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
    
    if difference > 2e-7:
        print("There is a mistake in the backward propagation! difference = {}".\
            format(difference))
    else:
        print("Your backward propagation works perfectly fine! difference = {}".\
            format(difference))
    
    return difference