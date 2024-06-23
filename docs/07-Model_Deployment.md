# Model Deployment

## Overview

Model deployment is the process of making your trained machine learning model available for use in production. This involves setting up an environment where the model can receive input, process it, and return predictions. In this section, we will cover the steps involved in deploying machine learning models using Azure Machine Learning, including creating and managing inference environments, deploying models as web services, and monitoring deployed models.

## Objectives

By the end of this section, you will be able to:
- Understand the model deployment process.
- Create and manage inference environments.
- Deploy models as web services using Azure Machine Learning.
- Monitor and manage deployed models.

## 1. Understanding Model Deployment

Model deployment ensures that your machine learning models are accessible for making predictions on new data. It involves:

- **Packaging**: Preparing the model and its dependencies for deployment.
- **Hosting**: Making the model available via a web service or API.
- **Scaling**: Ensuring the service can handle the required load.
- **Monitoring**: Keeping track of the model’s performance and usage.

## 2. Creating and Managing Inference Environments

An inference environment includes the runtime and dependencies needed to run your model. Azure ML allows you to create environments using Docker and Conda.

### Example: Creating an Environment with Conda

```python
from azureml.core import Environment

# Create a Python environment for the inference
env = Environment(name="inference-env")
env.python.conda_dependencies.add_pip_package("scikit-learn")
env.python.conda_dependencies.add_pip_package("pandas")
env.python.conda_dependencies.add_pip_package("numpy")

# Register the environment
env.register(workspace=ws)
```

## 3. Deploying Models as Web Services

Deploying a model as a web service involves creating a Docker image with your model and exposing an API endpoint.

### Example: Deploying a Model

#### Step 1: Register the Model

```python
from azureml.core import Model

# Register the model
model = Model.register(workspace=ws,
                       model_path="path/to/your/model.pkl",
                       model_name="your_model_name")
```

#### Step 2: Create an Inference Configuration

```python
from azureml.core.model import InferenceConfig

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)
```

#### Step 3: Deploy the Model

```python
from azureml.core.webservice import AciWebservice, Webservice

# Define the deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(workspace=ws,
                       name="your-service-name",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)
service.wait_for_deployment(show_output=True)

print(f"Service state: {service.state}")
print(f"Scoring URI: {service.scoring_uri}")
```

#### Step 4: Create the Scoring Script (`score.py`)

```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

def init():
    global model
    # Load the model from the registered model
    model_path = Model.get_model_path("your_model_name")
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

## 4. Monitoring and Managing Deployed Models

Monitoring the performance of deployed models is crucial to ensure they continue to perform well in production. Azure ML provides tools to monitor logs, update models, and scale services.

### Example: Monitoring Logs

```python
# Get logs from the deployed service
logs = service.get_logs()
for line in logs.split('\n'):
    print(line)
```

### Example: Updating a Deployed Model

If you need to update the model, you can deploy a new version and replace the existing service.

```python
# Register the new model
new_model = Model.register(workspace=ws,
                           model_path="path/to/your/new_model.pkl",
                           model_name="your_model_name")

# Deploy the new model
new_service = Model.deploy(workspace=ws,
                           name="your-service-name",
                           models=[new_model],
                           inference_config=inference_config,
                           deployment_config=deployment_config)
new_service.wait_for_deployment(show_output=True)

print(f"New service state: {new_service.state}")
print(f"New scoring URI: {new_service.scoring_uri}")
```

### Example: Scaling a Deployed Service

You can scale your service to handle more requests by adjusting the CPU and memory.

```python
from azureml.core.webservice import AksWebservice

# Define the new deployment configuration with more resources
scaled_deployment_config = AksWebservice.deploy_configuration(cpu_cores=2, memory_gb=4)

# Update the existing service
service.update(deployment_config=scaled_deployment_config)
service.wait_for_deployment(show_output=True)
```

## Conclusion

Model deployment is a critical step in bringing your machine learning solutions to production. By following the steps outlined in this section, you can package, deploy, and monitor your models effectively using Azure Machine Learning. This ensures that your models are not only accessible but also performant and scalable.

You have now completed the core concepts of Azure Machine Learning. Continue exploring advanced topics and best practices to enhance your machine learning projects.

## Next Steps

- Explore advanced topics in machine learning.
- Continue learning about model management and versioning.
- Experiment with different deployment strategies and monitor their performance.
