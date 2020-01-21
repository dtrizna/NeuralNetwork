import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from modules.forwardProp import L_model_forward
from modules.initialization import initialize_parameters_random, initialize_parameters_he
from train import train
from modules.predict import predict

def load_2D_dataset(visualize=False):
    data = scipy.io.loadmat('datasets/binary_classification_2D/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    if visualize:
        plt.scatter(train_X[0, :], train_X[1, :], s=40, \
            c=train_Y.squeeze(), cmap=plt.cm.get_cmap("PiYG"))
        plt.show()

    return train_X, train_Y, test_X, test_Y

def predict_dec(parameters, X):
    AL, _ = L_model_forward(X, parameters)
    predictions = (AL > 0.5)
    return predictions

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.get_cmap("PiYG"))
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.squeeze(), cmap=plt.cm.get_cmap("PiYG"))
    ax = plt.gca()
    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.75])
    plt.show()

def main():
    train_X, train_Y, test_X, test_Y = load_2D_dataset(False)

    np.random.seed(1)
    parameters = initialize_parameters_random([train_X.shape[0], 15, 10, 1])
    print('Training without Regularization..')
    trained_weights = train(train_X, train_Y, parameters, iterations=30000, learning_rate=0.5)
    predict(train_X, train_Y, trained_weights)
    predict(test_X, test_Y, trained_weights)
    plot_decision_boundary(lambda x: predict_dec(trained_weights, x.T), train_X, train_Y)

    for l in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
        print('\nTraining with L2 Regularization (lambda = {})'.format(l))
        parameters = initialize_parameters_random([train_X.shape[0], 15, 10, 1])
        trained_weights = train(train_X, train_Y, parameters, \
            iterations=30000, learning_rate=0.3, lambd=l, print_cost=False)
        predict(train_X, train_Y, trained_weights)
        predict(test_X, test_Y, trained_weights)
        #plot_decision_boundary(lambda x: predict_dec(trained_weights, x.T), train_X, train_Y)



if __name__ == '__main__':
    main()
