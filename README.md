
# Usage Notes

To train network with your dataset:  
1. Initialize Neural Network (NN), defines architecture:  

  * using `initialization.initialize_parameters_random()`  
  * in case of Deep Networks (5+ hidden layers), you may use `initializaiton.initialize_parameters_he()` in order to overcome Vanishing/Exploding weight problem.  

   *Example*

   Initialization of Binary Classification NN with 2 hidden layers 5 Activation Units (AU) each:  
   `parameters = initialization.initialize_parameters_random([X.shape[0], 5, 5, 1])`

2. Normalize features of Train/Dev/Test sets using `modules.featureNormalization()`
3. Call `train.train()` by feeding into Train data set (X) and it's labeled data (Y)  
  
   *Usage Notes!*

  * find correct `learning_rate` by using `print_cost=True`. If:
    - CF value is increasing:
      + choose lower `learning_rate`
    - CF value is decreasing, but slowly:
      + choose higher `learning_rate`
      + verify if features are correctly normalized
  * if you supply `lambd` value to function, it will implement L2 Regularization.
  * if you supply `keep_prob` value to function, it will implement DropOut Regularization.

4. Tune Parameters/Hyperparameters on Dev set.
5. Analyze algorithm performance on Test set.
    - Use F1 Score  

## TODO:
- F1 Score
- Adam optimization
- DropOut Regularization
- Batch Normalization

# Binary Classification Example

## Dataset "binary_classification_cats"

Code: `.apply_binary_classification_cats.py`  

### Training without Regularization 

*learning_rate = 0.0075, 1 hidden layer with 7 AU:*

```
Cost after iteration 2500: 0.033421158465249526
Prediction on Train set:
Accuracy: 0.9999999999999998
Prediction on Dev set:
Accuracy: 0.68
```

*Optimization Rationale:*

Algorithm does well on Training set, thus there's no big Avoidable Bias (assuming that Human and Bayer errors are close to 0%). Therefore, Bias reduction tecnhiques as:  
- training more Deep/Complex network
- improving quality of pictures
- longer training
- Gradient Descent optimizations (e.g. Adam)  
... won't help, assuming that Train and Dev sets come from same distribution (which indeed are).  
  
There's huge gat between Dev and Test set results, which is due to Variance. So:  
- regularization
- more training data (which gives better generalization)
.. should help.

### Training with L2 Regularization

Improved of prediction up to 76% using L2 Regularization with lambda = 0.1:
```
Prediction on Dev set:

Lambda value: 0.01
Accuracy: 0.72

Lambda value: 0.03
Accuracy: 0.76

Lambda value: 0.1
Accuracy: 0.76

Lambda value: 0.3
Accuracy: 0.74
```
