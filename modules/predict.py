import numpy as np
from modules.forwardProp import L_model_forward

def predict(X, Y, parameters):
    """
    This function is used to predict the results of a L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1, m))
    
    # Forward propagation
    probabilities, _ = L_model_forward(X, parameters)
    
    # convert probabilities to 0/1 predictions
    for i in range(0, probabilities.shape[1]):
        if probabilities[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    
    print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
    return p
