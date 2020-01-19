import matplotlib.pyplot as plt
import scipy.io

def load_2D_dataset(visualize=False):
    data = scipy.io.loadmat('datasets/binary_classification_2D/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    if visualize:
        plt.scatter(train_X[0, :], train_X[1, :], s=40, c=train_Y.squeeze(), cmap=plt.cm.get_cmap("Spectral"))
        plt.show()
    
    return train_X, train_Y, test_X, test_Y

#load_2D_dataset(True)
