
# Usage Notes

![equation](http://latex.codecogs.com/gif.latex?%24%24%2d%5c%66%72%61%63%7b%31%7d%7b%6d%7d%5c%73%75%6d%5c%6c%69%6d%69%74%73%5f%7b%69%20%3d%20%31%7d%5e%7b%6d%7d%28%79%5e%7b%28%69%29%7d%5c%6c%6f%67%5c%6c%65%66%74%28%61%5e%7b%5b%4c%5d%28%69%29%7d%5c%72%69%67%68%74%29%20%2b%20%28%31%2d%79%5e%7b%28%69%29%7d%29%5c%6c%6f%67%5c%6c%65%66%74%28%31%2d%20%61%5e%7b%5b%4c%5d%28%69%29%7d%5c%72%69%67%68%74%29%29%a0%5c%74%61%67%7b%37%7d%24%24)

To train network with your dataset:  
1. Initialize Neural Network (NN), defines architecture using `initialization.initialize_parameters_random()`
  
   *Example:*

   Initialization of Binary Classification NN with 2 hidden layers and 5 Activation Units (AU) each:  
   `parameters = initialization.initialize_parameters_random([X.shape[0], 5, 5, 1])`

2. Normalize features of Train/Dev/Test sets using `modules.featureNormalization()`
3. Call `train.train()` by feeding into Train data set (X) and it's labeled data (Y)  
  * to find correct `learning_rate`, use `print_cost=True`
  * if you supply `lambd` value to function, it will implement L2 Regularization
  * if you supply `keep_prob` value to function, it will implement DropOut Regularization

4. Tune Parameters/Hyperparameters on Dev set.
5. Analyze algorithm performance on Test set.
    - Use F1 Score  

## TODO:
- F1 Score
- Adam optimization
- DropOut Regularization
- Batch Normalization

# Optimization Notes

1. In case of Deep Networks (5+ hidden layers), consider to use `initializaiton.initialize_parameters_he()` in order to overcome Vanishing/Exploding weight problem.  

2. If Cost Function (CF) is not decreasing over epochs:
  - try to decrease `learning_rate`

3. If training process is slow (little decrease of CF over epochs, long time to iterate over epoch), then:
  - try to increase `learning_rate`
  - verify whether features are Normalized correctly
  - try to use Mini-Batch Gradient Descent (GD)
  - try to use GD optimization algorithms (momentum, RMSProp, Adam)

4. If algorithm does poorly predicting even Training set (you should compare this result with Human level error / Bayes error), then Bias reduction techniques should improve performance:
  - training more Complex Network (more AU in level, more levels)
  - longer training
  - adding more features (more information about dataset, e.g. improving quality of pictures)

5. If algorithm does well on Training set, but prediction on Dev set is poor, then Variance reduction tecnhiques should help:
  - Regularization (L2, Dropout)
  - more training data (gives better generalization)

6. If prediction on Train and Dev sets is good, but algorithm fails to predict Test set or Real world data:
  - be sure Dev and Test set come from same distribution (same data divided randomly)
  - understand whether Dev set represents Real world requiremenents well
  - analyze your evaluation metrics

# Binary Classification Example

## Dataset "binary_classification_cats"

Number of training examples: 209
Number of dev examples: 
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
Assuming that Human and Bayer errors are close to 0%, this realization has High Variance problem. Regularization or more training data should improve results.

### Training with L2 Regularization

Improved of prediction up to 76% using L2 Regularization with lambda = 0.1 or 0.03:
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
