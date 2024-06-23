# Monitoring Models

## Overview

Monitoring deployed machine learning models is crucial to ensure they perform as expected over time. It involves tracking various metrics to detect performance degradation, data drift, and other issues. Azure Machine Learning provides robust tools for monitoring models, allowing you to maintain their reliability and effectiveness.

## Objectives

By the end of this section, you will be able to:

- Understand the importance of model monitoring.
- Set up monitoring for deployed models using Azure ML.
- Track performance metrics and detect data drift.
- Implement alerts and automated responses for model performance issues.

## 1. Importance of Model Monitoring

Monitoring models helps to:

- **Detect Performance Degradation**: Identify when a model's accuracy or other performance metrics decline.
- **Identify Data Drift**: Detect changes in the input data distribution that may affect model performance.
- **Ensure Compliance**: Maintain adherence to regulatory and business standards.
- **Trigger Maintenance**: Initiate retraining or other maintenance tasks when necessary.

## 2. Setting Up Model Monitoring

### Example: Initializing the Workspace and Model

```python
from azureml.core import Workspace, Model

# Connect to the Azure ML workspace
ws = Workspace.from_config()

# Load the model
model = Model(ws, 'your_model_name')
```

### Example: Deploying a Model with Monitoring

#### Step 1: Create an Inference Configuration

```python
from azureml.core.model import InferenceConfig

# Define the inference configuration
inference_config = InferenceConfig(entry_script='score.py', environment=env)
```

#### Step 2: Deploy the Model

```python
from azureml.core.webservice import AciWebservice, Webservice

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, enable_app_insights=True)

# Deploy the model
service = Model.deploy(workspace=ws,
                       name='monitoring-service',
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)
service.wait_for_deployment(show_output=True)

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
```

## 3. Tracking Performance Metrics

### Example: Logging Metrics

You can log custom metrics during the scoring process to track model performance over time.

#### Step 1: Modify the Scoring Script (`score.py`)

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model
from applicationinsights import TelemetryClient

def init():
    global model
    global telemetry_client
    # Load the model
    model_path = Model.get_model_path("your_model_name")
    model = joblib.load(model_path)
    # Initialize telemetry client
    telemetry_client = TelemetryClient(instrumentation_key='your_instrumentation_key')

def run(data):
    try:
        # Parse the input data
        input_data = json.loads(data)["data"]
        input_array = np.array(input_data)
        # Make predictions
        result = model.predict(input_array)
        # Log custom metrics
        telemetry_client.track_metric("PredictionCount", len(result))
        telemetry_client.track_event("ModelPrediction", {"result": str(result)})
        telemetry_client.flush()
        # Return the prediction result
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```

## 4. Detecting Data Drift

Data drift occurs when the statistical properties of the input data change over time, which can impact model performance.

### Example: Setting Up Data Drift Monitoring

```python
from azureml.datadrift import DataDriftDetector, DataDriftProfile

# Define the baseline and target datasets
baseline_data = dataset.take(1000)
target_data = dataset.take(1000, offset=1000)

# Set up the Data Drift Detector
drift_detector = DataDriftDetector.create_from_datasets(ws, 'drift-detector', baseline_data, target_data)

# Configure data drift parameters
drift_detector.set_monitor(interval=1, feature_list=['feature1', 'feature2'])

# Run the data drift profile
profile = drift_detector.run()
print(profile)
```

## 5. Implementing Alerts and Automated Responses

### Example: Setting Up Alerts

You can set up alerts to notify you when certain conditions are met, such as performance degradation or data drift.

```python
from azureml.core.webservice import AciWebservice, Webservice

# Get the deployed service
service = Webservice(ws, 'monitoring-service')

# Set up application insights alerts
alert_rules = [{
    'name': 'HighErrorRate',
    'condition': {
        'type': 'Metric',
        'name': 'RequestsFailed',
        'threshold': 10,
        'operator': 'GreaterThanOrEqual'
    },
    'actions': [{
        'type': 'Email',
        'email': 'your_email@example.com'
    }]
}]

# Apply the alert rules
service.update(alert_rules=alert_rules)
```

## Conclusion

Monitoring machine learning models is essential for maintaining their performance and reliability in production. By leveraging Azure Machine Learning's monitoring tools, you can track important metrics, detect data drift, and implement alerts and automated responses to ensure your models continue to operate effectively. This proactive approach helps in maintaining high standards of model performance and compliance.
