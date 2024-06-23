# Automated Machine Learning (AutoML)

## Overview

Automated Machine Learning (AutoML) simplifies the process of building machine learning models by automating the selection, training, and tuning of models. Azure Machine Learning provides powerful AutoML capabilities that allow users to quickly develop models without extensive machine learning expertise.

## Objectives

By the end of this section, you will be able to:

- Understand the concept of AutoML and its benefits.
- Use Azure AutoML to automatically train and select the best models.
- Configure and run AutoML experiments.
- Analyze and deploy the best-performing AutoML models.

## 1. Introduction to AutoML

AutoML automates the end-to-end process of applying machine learning to real-world problems. It includes:

- **Data Preprocessing**: Cleaning and transforming data.
- **Model Selection**: Choosing the best model architecture.
- **Hyperparameter Tuning**: Optimizing model parameters.
- **Model Evaluation**: Assessing model performance.

### Benefits of AutoML

- **Efficiency**: Saves time and resources by automating repetitive tasks.
- **Accessibility**: Makes machine learning accessible to non-experts.
- **Performance**: Often results in high-performing models due to comprehensive search and optimization.

## 2. Setting Up AutoML in Azure

To use AutoML in Azure, you need to set up your environment, including the workspace and compute resources.

### Step 1: Initialize the Workspace

```python
from azureml.core import Workspace

# Connect to the Azure ML workspace
ws = Workspace.from_config()
```

### Step 2: Create a Compute Cluster

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Specify the configuration for the compute cluster
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2', max_nodes=4)

# Create the compute cluster
compute_cluster = ComputeTarget.create(ws, 'cpu-cluster', compute_config)
compute_cluster.wait_for_completion(show_output=True)
```

## 3. Running an AutoML Experiment

### Step 1: Prepare the Data

```python
import pandas as pd
from azureml.core.dataset import Dataset

# Load the dataset
data = pd.read_csv('data/data.csv')

# Convert the dataframe to an Azure ML dataset
dataset = Dataset.Tabular.register_pandas_dataframe(data, ws, 'my-dataset')
```

### Step 2: Configure the AutoML Experiment

```python
from azureml.train.automl import AutoMLConfig

# Define the AutoML configuration
automl_config = AutoMLConfig(
    task='classification',
    training_data=dataset,
    label_column_name='target',
    primary_metric='accuracy',
    n_cross_validations=5,
    enable_early_stopping=True,
    experiment_timeout_minutes=60,
    max_concurrent_iterations=4,
    compute_target=compute_cluster
)
```

### Step 3: Submit the Experiment

```python
from azureml.core.experiment import Experiment

# Create an experiment
experiment = Experiment(ws, 'automl-classification')

# Submit the AutoML experiment
automl_run = experiment.submit(automl_config, show_output=True)
```

## 4. Analyzing AutoML Results

After the experiment completes, you can analyze the results to find the best-performing model.

### Example: Retrieving the Best Model

```python
from azureml.train.automl.run import AutoMLRun

# Get the best model
best_run, fitted_model = automl_run.get_output()

# Print the best run details
print(best_run)
print(fitted_model)
```

### Example: Evaluating the Best Model

```python
# Make predictions on the test set
X_test = data.drop(columns=['target'])
y_test = data['target']
y_pred = fitted_model.predict(X_test)

# Calculate evaluation metrics
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

## 5. Deploying the Best AutoML Model

You can deploy the best model as a web service for real-time predictions.

### Step 1: Create an Inference Configuration

```python
from azureml.core.model import InferenceConfig

# Define the inference configuration
inference_config = InferenceConfig(entry_script='score.py', environment=env)
```

### Step 2: Deploy the Model

```python
from azureml.core.webservice import AciWebservice, Webservice

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(workspace=ws,
                       name='automl-service',
                       models=[best_run.register_model()],
                       inference_config=inference_config,
                       deployment_config=deployment_config)
service.wait_for_deployment(show_output=True)

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
```

### Step 3: Create the Scoring Script (`score.py`)

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    # Load the model from the registered model
    model_path = Model.get_model_path("automl_best_model")
    model = joblib.load(model_path)

def run(data):
    try:
        # Parse the input data
        input_data = json.loads(data)["data"]
        # Convert to numpy array
        input_array = np.array(input_data)
        # Make prediction
        result = model.predict(input_array)
        # Return the prediction result
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

## Conclusion

AutoML simplifies the machine learning workflow by automating data preprocessing, model selection, and hyperparameter tuning. With Azure Machine Learning, you can efficiently train, evaluate, and deploy high-performing models. In the next section, we will explore model management and versioning.

Proceed to the [Model Management and Versioning](Model_Management_Versioning.md) section to continue.
