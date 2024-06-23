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
