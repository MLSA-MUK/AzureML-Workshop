# Feature Engineering

## Overview

Feature engineering is the process of using domain knowledge to extract features (characteristics, properties, or attributes) from raw data that help machine learning models learn better. This step is critical as it directly impacts the performance of your machine learning models. In this section, we will cover various techniques for feature engineering and how to implement them using Python and Azure Machine Learning.

## Objectives

By the end of this section, you will be able to:
- Understand the importance of feature engineering.
- Apply various feature engineering techniques.
- Use Azure Machine Learning for automated feature engineering.

## 1. Understanding Feature Engineering

### Why Feature Engineering?

Feature engineering helps in:
- Enhancing the predictive power of machine learning models.
- Reducing the complexity of models by creating simpler features.
- Making the data more understandable and interpretable.

## 2. Common Feature Engineering Techniques

### Creating New Features

Creating new features involves deriving new attributes from existing data. This can include:
- Mathematical transformations (e.g., log, square root)
- Aggregations (e.g., sum, mean, count)
- Date and time transformations (e.g., extracting day, month, year)

Example:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/data.csv')

# Create new features
df['log_feature'] = df['numerical_feature'].apply(lambda x: np.log(x + 1))
df['day_of_week'] = pd.to_datetime(df['date_column']).dt.dayofweek
```

### Handling Categorical Variables

Categorical variables need to be converted into numerical values for machine learning models. Common techniques include:
- One-Hot Encoding: Converts categorical variables into a series of binary columns.
- Label Encoding: Assigns a unique integer to each category.

Example:

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['categorical_feature']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['categorical_feature']))
df = df.join(encoded_df).drop('categorical_feature', axis=1)

# Label Encoding
label_encoder = LabelEncoder()
df['label_encoded_feature'] = label_encoder.fit_transform(df['categorical_feature'])
```

### Feature Scaling

Feature scaling ensures that numerical features are on the same scale. Common techniques include:
- Standardization: Centers the data to have a mean of 0 and a standard deviation of 1.
- Normalization: Scales the data to a range of [0, 1].

Example:

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler = StandardScaler()
df['standardized_feature'] = scaler.fit_transform(df[['numerical_feature']])

# Normalization
normalizer = MinMaxScaler()
df['normalized_feature'] = normalizer.fit_transform(df[['numerical_feature']])
```

### Feature Interaction

Feature interaction involves creating new features by combining existing features. This can include:
- Polynomial features: Creating interaction terms (e.g., feature1 * feature2).
- Ratios: Creating ratio features (e.g., feature1 / feature2).

Example:

```python
from sklearn.preprocessing import PolynomialFeatures

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
interaction_features = poly.fit_transform(df[['feature1', 'feature2']])
interaction_df = pd.DataFrame(interaction_features, columns=['feature1', 'feature2', 'feature1_feature2'])
df = df.join(interaction_df)
```

## 3. Automated Feature Engineering with Azure ML

Azure Machine Learning provides automated feature engineering capabilities using the Azure ML SDK and AutoML. AutoML can automatically generate new features and select the most relevant ones for your model.

Example:

```python
from azureml.core import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.experiment import Experiment

# Connect to the Azure ML workspace
ws = Workspace.from_config()

# Load the dataset
df = pd.read_csv('data/data.csv')

# Prepare the dataset for AutoML
target_column = 'target'
features = df.drop(columns=[target_column])
labels = df[target_column]

# Define AutoML configuration
automl_config = AutoMLConfig(
    task='classification',
    primary_metric='accuracy',
    training_data=df,
    label_column_name=target_column,
    featurization='auto',
    iterations=10
)

# Create and run the AutoML experiment
experiment = Experiment(ws, 'automl_feature_engineering')
run = experiment.submit(automl_config)
run.wait_for_completion(show_output=True)

# Retrieve the best model
best_model = run.get_output()
```

## Conclusion

Feature engineering is a vital step in the machine learning pipeline that can significantly improve model performance. By applying the techniques covered in this section, you can create more informative and useful features for your models. In the next section, we will discuss how to train and evaluate your machine learning models.

Proceed to the [Model Training](Model_Training.md) section to continue.
Certainly! Here's the content for `docs/Model_Training.md`:

### `docs/Model_Training.md`

```markdown
# Model Training

## Overview

Model training is the process of feeding data to a machine learning algorithm to help it learn how to make predictions or decisions. In this section, we will cover the steps involved in training machine learning models using Azure Machine Learning, including data splitting, selecting algorithms, training models, and tuning hyperparameters.

## Objectives

By the end of this section, you will be able to:
- Split your dataset into training and testing sets.
- Choose appropriate machine learning algorithms.
- Train models using Azure Machine Learning.
- Tune hyperparameters to improve model performance.

## 1. Splitting the Data

Before training a model, it's essential to split the data into training and testing sets. The training set is used to train the model, while the testing set is used to evaluate its performance.

Example:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data/data.csv')

# Define features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 2. Choosing a Machine Learning Algorithm

The choice of algorithm depends on the type of problem (classification, regression, clustering, etc.) and the characteristics of the data. Common algorithms include:

- **Classification**: Logistic Regression, Decision Trees, Random Forest, Support Vector Machines, Neural Networks.
- **Regression**: Linear Regression, Ridge Regression, Lasso Regression, Decision Trees, Random Forest.
- **Clustering**: K-Means, Hierarchical Clustering, DBSCAN.

## 3. Training a Model

### Using Scikit-Learn

Scikit-learn is a popular machine learning library in Python. Below is an example of training a Random Forest classifier:

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)
```

### Using Azure ML

You can also use Azure Machine Learning to train models. This provides benefits such as distributed training, automatic logging, and version control.

Example:

```python
from azureml.core import Workspace, Experiment
from azureml.train.sklearn import SKLearn
from azureml.train.estimator import Estimator

# Connect to the Azure ML workspace
ws = Workspace.from_config()

# Create an experiment
experiment = Experiment(workspace=ws, name='model-training')

# Define the estimator
estimator = Estimator(
    source_directory='.',
    entry_script='train.py',
    compute_target='cpu-cluster',
    conda_packages=['scikit-learn', 'pandas']
)

# Submit the experiment
run = experiment.submit(estimator)
run.wait_for_completion(show_output=True)
```

`train.py` example:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('data/data.csv')

# Define features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

## 4. Hyperparameter Tuning

Hyperparameter tuning involves adjusting the parameters of the machine learning algorithm to improve its performance. This can be done using techniques like grid search and random search.

### Using Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f'Best parameters: {grid_search.best_params_}')
```

### Using Azure ML HyperDrive

Azure Machine Learning provides HyperDrive for hyperparameter tuning.

Example:

```python
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

# Define the parameter space
param_space = {
    'n_estimators': choice(50, 100, 200),
    'max_depth': choice(None, 10, 20, 30),
    'min_samples_split': choice(2, 5, 10)
}

# Define the sampling method
param_sampling = GridParameterSampling(param_space)

# Define the HyperDrive configuration
hyperdrive_config = HyperDriveConfig(
    run_config=estimator,
    hyperparameter_sampling=param_sampling,
    primary_metric_name='accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=20
)

# Submit the HyperDrive run
hyperdrive_run = experiment.submit(hyperdrive_config)
hyperdrive_run.wait_for_completion(show_output=True)

# Get the best run
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
print(f'Best run metrics: {best_run_metrics}')
```

## Conclusion

Training a machine learning model is a critical step that involves selecting the right algorithm, preparing the data, training the model, and tuning its parameters. With the tools and techniques covered in this section, you can build robust machine learning models using Azure Machine Learning. In the next section, we will explore how to evaluate and validate your trained models.

Proceed to the [Model Evaluation](Model_Evaluation.md) section to continue.