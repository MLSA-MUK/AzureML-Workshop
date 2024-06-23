# Setting Up Your Azure Machine Learning Environment

## Prerequisites

Before you begin, ensure you have the following:

- An [Azure account](https://azure.microsoft.com/en-us/free/). If you don't have one, you can create a free account.
- Basic knowledge of Python programming.
- Internet access to download necessary tools and libraries.

## Step 1: Create an Azure Machine Learning Workspace

1. **Sign in to the Azure Portal**
   - Go to the [Azure Portal](https://portal.azure.com/) and sign in with your Azure account.

2. **Create a new resource**
   - Click on the "+ Create a resource" button on the left-hand menu.

3. **Search for Azure Machine Learning**
   - In the search bar, type "Machine Learning" and select "Machine Learning" from the list of services.

4. **Create the workspace**
   - Click "Create" and fill in the required information:
     - **Subscription**: Select your Azure subscription.
     - **Resource group**: Create a new resource group or select an existing one.
     - **Workspace name**: Provide a unique name for your workspace.
     - **Region**: Select the region closest to you.
     - **Workspace edition**: Choose the edition that best suits your needs (Basic or Enterprise).

5. **Review and create**
   - Review your configurations and click "Create".

## Step 2: Install Azure ML SDK

To interact with Azure Machine Learning services, you need to install the Azure ML SDK. You can do this using pip:

```bash
pip install azureml-sdk
```

## Step 3: Clone the Repository

Clone the workshop repository to your local machine:

```bash
git clone https://github.com/your-username/AzureML-Workshop.git
cd AzureML-Workshop
```

## Step 4: Configure Your Development Environment

1. **Install required Python packages**
   - Navigate to the root directory of the repository and install the required packages using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Jupyter Notebook**
   - If you prefer to use Jupyter Notebook for the hands-on exercises, ensure it is installed:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

## Step 5: Verify the Installation

To verify that everything is set up correctly, run a simple Azure ML script to check the installation:

```python
import azureml.core
print("Azure ML SDK Version:", azureml.core.VERSION)
```

If the installation is successful, you should see the Azure ML SDK version printed out.

You are now ready to start the workshop. Proceed to the next section to begin with data preparation and exploration.
