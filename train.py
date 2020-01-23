import numpy as np
import matplotlib.pyplot as plt
from modules.forwardProp import L_model_forward
from modules.cost import compute_cost, compute_cost_with_L2regularization
from modules.backProp import L_model_backprop
from modules.update import update_parameters

def train(X, Y, parameters, iterations = 1000, learning_rate = 0.0075, lambd = None, keep_prob = None, print_cost = True):
    """
    This function is used to train L-layer neural network (NN).
    
    Arguments:
    X -- data set of examples you would like to label
    Y -- marked classification of X data set, shape: (1, X.shape[1])
    parameters -- parameters of the trained model (represent NN architecture)
    
    Hyperparameter (optional):
    iterations -- number of Gradient Descent steps
    learning_rate -- Gradient Descent update rate
    lambd -- if specified, NN will be trained using L2 Regularization
    keep_prob -- if specificed, NN will be trained using DropOut Regularization

    Returns:
    weigths -- trained parameters of NN after Nr. of iterations
    """
    
    costs = []
    if keep_prob and lambd:
        print("Please specify either 'keep_prob' or 'lambd', not both!")
        return 1

    for i in range(0, iterations):

        if keep_prob:
            #AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob)
            pass
        else:
            AL, caches = L_model_forward(X, parameters)
        
        if lambd:
            cost = compute_cost_with_L2regularization(AL, Y, parameters, lambd)
        else:
            cost = compute_cost(AL, Y)

        if not lambd and not keep_prob:
            grads = L_model_backprop(AL, Y, caches)
        elif lambd:
            grads = L_model_backprop(AL, Y, caches, lambd)
        elif keep_prob:
            #grads = backward_propagation_with_dropout(X, Y, caches, keep_prob)
            pass

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    if print_cost:
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    return parameters