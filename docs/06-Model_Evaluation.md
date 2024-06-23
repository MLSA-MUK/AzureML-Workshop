# Model Evaluation

## Overview

Model evaluation is the process of assessing the performance of a machine learning model to ensure it generalizes well to new, unseen data. This step involves using various metrics and techniques to understand how well the model is performing and to identify any potential areas for improvement.

## Objectives

By the end of this section, you will be able to:
- Understand the importance of model evaluation.
- Use different evaluation metrics for classification and regression models.
- Implement cross-validation to assess model performance.
- Evaluate model performance using Azure Machine Learning.

## 1. Importance of Model Evaluation

Model evaluation is crucial because it helps in:
- Measuring how well the model will perform on new, unseen data.
- Comparing different models and selecting the best one.
- Identifying any overfitting or underfitting issues.
- Understanding the model's strengths and weaknesses.

## 2. Evaluation Metrics

### Classification Metrics

For classification problems, common evaluation metrics include:

- **Accuracy**: The proportion of correctly predicted instances out of the total instances.
- **Precision**: The proportion of true positive predictions out of the total positive predictions.
- **Recall (Sensitivity)**: The proportion of true positive predictions out of the actual positive instances.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC AUC**: The area under the Receiver Operating Characteristic curve.

Example:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC: {roc_auc}')
```

### Regression Metrics

For regression problems, common evaluation metrics include:

- **Mean Absolute Error (MAE)**: The average absolute difference between predicted and actual values.
- **Mean Squared Error (MSE)**: The average squared difference between predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The square root of the mean squared error.
- **R-squared (R2)**: The proportion of the variance in the dependent variable that is predictable from the independent variables.

Example:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')
```

## 3. Cross-Validation

Cross-validation is a technique for assessing how the results of a statistical analysis will generalize to an independent data set. It is mainly used to estimate the skill of a model on unseen data. The most common form of cross-validation is k-fold cross-validation.

Example:

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')
print(f'Standard deviation of CV score: {cv_scores.std()}')
```

## 4. Model Evaluation with Azure ML

Azure Machine Learning provides tools to evaluate and compare models efficiently. The `Run` object in Azure ML allows you to log metrics and retrieve them later for comparison.

Example:

```python
from azureml.core import Run

# Get the current run
run = Run.get_context()

# Log metrics
run.log('Accuracy', accuracy)
run.log('Precision', precision)
run.log('Recall', recall)
run.log('F1 Score', f1)
run.log('ROC AUC', roc_auc)

# Complete the run
run.complete()
```

## Conclusion

Model evaluation is a critical step in the machine learning workflow that ensures your models perform well on new, unseen data. By using appropriate evaluation metrics and techniques, you can better understand your model's performance and make necessary adjustments. In the next section, we will cover model deployment and how to make your trained models available for use in production.

Proceed to the [Model Deployment](Model_Deployment.md) section to continue.
