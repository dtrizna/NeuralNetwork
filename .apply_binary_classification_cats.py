import numpy as np
import h5py
import matplotlib.pyplot as plt
from modules.initialization import initialize_parameters_random
from train import train
from modules.predict import predict


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


def gradient_check(train_x, train_y):
    from modules.gradCheck import gradient_check
    from modules.forwardProp import L_model_forward
    from modules.backProp import L_model_backprop
    parameters = initialize_parameters_random([train_x[:10, :3].shape[0], 3, 1])
    AL, caches = L_model_forward(train_x[:10, :3], parameters)
    grads = L_model_backprop(AL, train_y[:10, :3], caches)
    gradient_check(parameters, grads, train_x[:10, :3], train_y[:10, :3])


def see_cat(idx, train_x_orig, train_y, classes):
    plt.imshow(train_x_orig[idx])
    print ("y = " + str(train_y[0, idx]) + ". It's a " + \
        classes[train_y[0, idx]].decode("utf-8") +  " picture.")
    plt.show()
    

def main(to_see_cat = False):
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
    # gradient_check(train_x, train_y)

    parameters = initialize_parameters_random([train_x.shape[0], 7, 1])

    # WITHOUT REGULARIZATION
    #trained_weights = train(train_x, train_y, parameters, 1501, learning_rate=0.02)

    # WITH L2 REGULARIZATION - SEEKING BEST LAMBDA
    for l in [0.01, 0.1, 0.7, 1, 1.5, 3, 10]:
        trained_weights = train(train_x, train_y, parameters, 1501, learning_rate=0.02, lambd=l, print_cost=False)
        print("Lambda value: {}".format(l))
        #print("Prediction on Train set:")
        #predict(train_x, train_y, trained_weights)
        print("Prediction on Dev set:")
        predict(dev_x, dev_y, trained_weights)


if __name__ == "__main__":
    main(False)