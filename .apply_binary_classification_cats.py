import numpy as np
import h5py
import matplotlib.pyplot as plt
from modules.initialization import initialize_parameters_random
from modules.predict import print_results
from modules.gradCheck import gradient_check
from modules.forwardProp import L_model_forward
from modules.backProp import L_model_backprop
from train import train

def load_data():
    train_dataset = h5py.File('datasets/binary_classification_cats/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    dev_dataset = h5py.File('datasets/binary_classification_cats/test_catvnoncat.h5', "r")
    dev_set_x_orig = np.array(dev_dataset["test_set_x"][:]) # your dev set features
    dev_set_y_orig = np.array(dev_dataset["test_set_y"][:]) # your dev set labels

    classes = np.array(dev_dataset["list_classes"][:]) # the list of classes
   
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    dev_set_y_orig = dev_set_y_orig.reshape((1, dev_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, dev_set_x_orig, dev_set_y_orig, classes


def gradient_check_cats(train_x, train_y):
    parameters = initialize_parameters_random([train_x[:10, :3].shape[0], 3, 1])
    AL, caches = L_model_forward(train_x[:10, :3], parameters)
    grads = L_model_backprop(AL, train_y[:10, :3], caches)
    gradient_check(parameters, grads, train_x[:10, :3], train_y[:10, :3])


def see_cat(idx, train_x_orig, train_y, classes):
    plt.imshow(train_x_orig[idx])
    print ("y = " + str(train_y[0, idx]) + ". It's a " + \
        classes[train_y[0, idx]].decode("utf-8") +  " picture.")
    plt.show()


def main(to_see_cat=False):
    train_x_orig, train_y, dev_x_orig, dev_y, classes = load_data()

    if to_see_cat:
        while True:
            idx = input("Please enter index [0 - {}] of image you want \
to see it OR anything else to start training: ".format(len(train_x_orig)-1))
            try:
                idx = int(idx)
                see_cat(idx, train_x_orig, train_y, classes)
            except ValueError:
                break
            except IndexError:
                break

    # NORMALIZE FEATURES
    # Reshape the training and dev examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
    # The "-1" makes reshape flatten the remaining dimensions
    dev_x_flatten = dev_x_orig.reshape(dev_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    dev_x = dev_x_flatten/255.
    # END NORMALIZE

    # CAN BE ENABLED FOR BACKPROP VERIFICATION
    # gradient_check_cats(train_x, train_y)

    # TO HAVE CONSISTENT RESULTS
    np.random.seed(3)
    #parameters = initialize_parameters_random([train_x.shape[0], 20, 7, 5, 1])
    parameters = initialize_parameters_random([train_x.shape[0], 7, 1])

    # WITHOUT REGULARIZATION
    trained_weights = train(train_x, train_y, parameters, 3501, learning_rate=0.0075)
    print_results(train_x, train_y, trained_weights, 'Train')
    print_results(dev_x, dev_y, trained_weights, 'Dev')

    # WITH L2 REGULARIZATION - SEEKING BEST LAMBDA
    for l in [0.01, 0.03, 0.1, 0.3, 1, 3, 10]:
    # ZOOMING RANGE
    #for l in [0.01, 0.02, 0.03, 0.06, 0.1, 0.2, 0.3]:
        trained_weights = train(train_x, train_y, parameters, \
                            iterations=3501, learning_rate=0.0075, \
                            lambd=l, print_cost=False)
        print("Lambda value: {}".format(l))
        print_results(dev_x, dev_y, trained_weights, 'Dev')

    # WITH REGULARIZATION AND BEST LAMBDA
    #trained_weights = train(train_x, train_y, parameters, 2501, learning_rate=0.0075, lambd=0.1)
    #print_results(dev_x, dev_y, trained_weights, 'Dev')

if __name__ == "__main__":
    main(False)
