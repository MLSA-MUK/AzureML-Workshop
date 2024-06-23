#!/bin/bash

# Define the root directory
ROOT_DIR="AzureML-Workshop"

# Create the root directory
mkdir -p $ROOT_DIR

# Create the main files
touch $ROOT_DIR/README.md
touch $ROOT_DIR/LICENSE
touch $ROOT_DIR/CONTRIBUTING.md

# Create the docs directory and files
mkdir -p $ROOT_DIR/docs
touch $ROOT_DIR/docs/Introduction.md
touch $ROOT_DIR/docs/Setup.md
touch $ROOT_DIR/docs/Data_Preparation.md
touch $ROOT_DIR/docs/Feature_Engineering.md
touch $ROOT_DIR/docs/Model_Training.md
touch $ROOT_DIR/docs/Model_Evaluation.md
touch $ROOT_DIR/docs/Model_Deployment.md
touch $ROOT_DIR/docs/AutoML.md
touch $ROOT_DIR/docs/Responsible_AI.md
touch $ROOT_DIR/docs/Monitoring_Models.md
touch $ROOT_DIR/docs/Additional_Resources.md

# Function to create demo directories and files
create_demo() {
    mkdir -p $ROOT_DIR/demos/$1/{data,scripts}
    touch $ROOT_DIR/demos/$1/notebook.ipynb
    touch $ROOT_DIR/demos/$1/scripts/${2:-script}.py
}

# Create demos directories and files
create_demo "data_preparation" "clean_data.py"
create_demo "feature_engineering" "feature_selection.py"
create_demo "model_training" "train_model.py"
create_demo "model_evaluation" "evaluate_model.py"
create_demo "model_deployment" "deploy_model.py"
create_demo "automl" "automl_training.py"
create_demo "responsible_ai" "interpretability.py"
create_demo "monitoring" "monitor_model.py"

# Create exercises directories and files
for i in {1..3}; do
    mkdir -p $ROOT_DIR/exercises/exercise$i/data
    touch $ROOT_DIR/exercises/exercise$i/notebook.ipynb
    touch $ROOT_DIR/exercises/exercise$i/data/dataset1.csv
    touch $ROOT_DIR/exercises/exercise$i/data/dataset2.csv
done

# Create resources directory and files
mkdir -p $ROOT_DIR/resources
touch $ROOT_DIR/resources/books.md
touch $ROOT_DIR/resources/articles.md
touch $ROOT_DIR/resources/online_courses.md
touch $ROOT_DIR/resources/tools.md

echo "File tree for AzureML-Workshop has been created successfully."
