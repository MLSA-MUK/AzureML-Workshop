# Data Preparation and Exploration

## Overview

Data preparation is a crucial step in the machine learning workflow. It involves cleaning, transforming, and organizing raw data into a suitable format for analysis and modeling. This process ensures that the data quality is high and that the machine learning models built on this data will perform well.

In this section, we will cover the following topics:

1. Importing Data into Azure Machine Learning
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)

## 1. Importing Data into Azure Machine Learning

### Uploading Data to the Azure ML Workspace

1. **Using the Azure ML Studio**
   - Go to the Azure Machine Learning Studio.
   - Select your workspace and navigate to the "Datasets" section.
   - Click on "Create dataset" and choose the data source (local files, datastore, web URL, etc.).
   - Follow the instructions to upload your dataset.

2. **Using the Azure ML SDK**
   - You can also upload data programmatically using the Azure ML SDK. Below is an example of how to upload a CSV file:

   ```python
   from azureml.core import Workspace, Dataset

   # Connect to the Azure ML workspace
   ws = Workspace.from_config()

   # Define the path to the local file
   local_file_path = 'path/to/your/data.csv'

   # Upload the file to the datastore
   datastore = ws.get_default_datastore()
   datastore.upload_files([local_file_path], target_path='data/', overwrite=True, show_progress=True)

   # Create a dataset from the uploaded file
   dataset = Dataset.Tabular.from_delimited_files(path=(datastore, 'data/data.csv'))

   # Register the dataset
   dataset = dataset.register(workspace=ws, name='my_dataset', create_new_version=True)
   ```

## 2. Data Cleaning and Preprocessing

### Handling Missing Values

Missing data can significantly impact the performance of machine learning models. Common strategies to handle missing values include:

- **Removing missing values**: If the dataset is large and the number of missing values is small, you can remove rows or columns with missing values.
- **Imputation**: Replace missing values with statistical measures such as mean, median, or mode.

Example of handling missing values using pandas:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/data.csv')

# Remove rows with missing values
df.dropna(inplace=True)

# Impute missing values with mean
df.fillna(df.mean(), inplace=True)
```

### Data Transformation

Data transformation includes operations such as scaling, normalization, encoding categorical variables, and feature engineering.

- **Scaling and Normalization**: Standardize numerical features to a common scale.
- **Encoding Categorical Variables**: Convert categorical variables into numerical values using techniques like one-hot encoding.

Example of data transformation:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Standardize numerical features
scaler = StandardScaler()
df['numerical_feature'] = scaler.fit_transform(df[['numerical_feature']])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['categorical_feature']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['categorical_feature']))
df = df.join(encoded_df).drop('categorical_feature', axis=1)
```

## 3. Exploratory Data Analysis (EDA)

### Understanding the Data

EDA involves summarizing the main characteristics of the dataset, often using visual methods. This step helps in understanding the patterns, relationships, and anomalies in the data.

### Common EDA Techniques

- **Descriptive Statistics**: Calculate summary statistics such as mean, median, and standard deviation.
- **Data Visualization**: Use plots and charts to visualize data distributions and relationships.

Example of EDA using pandas and matplotlib:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/data.csv')

# Display descriptive statistics
print(df.describe())

# Visualize the distribution of a numerical feature
plt.figure(figsize=(10, 6))
sns.histplot(df['numerical_feature'], bins=30, kde=True)
plt.title('Distribution of Numerical Feature')
plt.show()

# Visualize the relationship between two features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='feature1', y='feature2', data=df)
plt.title('Relationship between Feature1 and Feature2')
plt.show()
```

## Conclusion

Data preparation and exploration are essential steps in the machine learning workflow. Properly prepared data can significantly improve the performance of machine learning models. In the next section, we will cover feature engineering techniques to further enhance the data for modeling.

Proceed to the [Feature Engineering](Feature_Engineering.md) section to continue.
