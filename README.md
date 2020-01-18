
# Usage Notes

Neural Network (NN) interface is localed at train.train()  
To train network with your dataset:  
- initialize NN (defines NN architecture):  
  * using `initialization.initialize_parameters_random()`  
  * in case of Deep Networks (5+ hidden layers), you may use `initializaiton.initialize_parameters_he()` in order to overcome Vanishing/Exploding weight problem.  
  Example:  
    initialization of Binary Classification NN with 2 hidden layers 5 activation units each:  
        `parameters = initialization.initialize_parameters_random([X.shape[0], 5, 5, 1])`

# Binary Classification Results

## Dataset "binary_classification_cats"

### Training without Regularization

Cost after iteration 1500: 0.02472818901125718  
Prediction on Train set:  
Accuracy: 0.9952153110047844  
Prediction on Test set:  
Accuracy: 0.6600000000000001  

### Training with L2 Regularization (lambda = 0.7)

Cost after iteration 1500: 0.05416693578711332  
Prediction on Train set:  
Accuracy: 0.9999999999999998  
Prediction on Test set:  
Accuracy: 0.78  
