## Training Logistic Regression via Stochastic Gradient Ascent

### Batch learning pseudo code:

```
initiate feature_weight (aka feature parameter value), initiate counter

while not converged:
    for each feature_weight:
        partial_derivative_of_this_feature_weight = 
                  sum(for each data_point, calculate residual_in_terms_of_probability * feature_weight)
        feature_weight_new = feature_weight + learning_rate * partial_derivative_of_this_feature_weight
    counter = counter + 1   
```

### Online learning pseudo code:
```
initiate feature_weight (aka feature parameter value), initiate counter

while not converged:
    for each data_point:
        for each feature_weight:
            partial_derivative_of_this_feature_weight = 
                      using this data_point, calculate residual_in_terms_of_probability * feature_weight
            feature_weight_new = feature_weight + learning_rate * partial_derivative_of_this_feature_weight
        counter = counter + 1
```
