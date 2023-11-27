import pytest
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
# TODO: add necessary import

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    # add description for the first test
    """
    # Your code here
    project_path = '/home/lylewilliams/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/'
    data_path = os.path.join(project_path, "data", "census.csv")
    print('this is the path', data_path)
    df = pd.read_csv(data_path, header=0)

    # Assuming you have a variable `df` that represents your dataframe
    expected_columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    # Check if all expected columns are present in the dataframe
    assert all(col in df.columns for col in expected_columns), "Missing columns in the dataframe"
    

# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    # Assuming you have X_train and y_train as your training data
    project_path = '/home/lylewilliams/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/'
    data_path = os.path.join(project_path, "data", "census.csv")
    print('this is the path', data_path)
    data = pd.read_csv(data_path, header=0)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # DO NOT MODIFY
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    X_train, y_train, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=None,
        lb=None,
    )
    
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,  # Pass the encoder object used during training
        lb=lb,
    )
    
    model = train_model(X_train, y_train)

    # Check if the returned model is of the expected type
    assert isinstance(model, RandomForestClassifier), "Unexpected model type"


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    project_path = '/home/lylewilliams/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/'
    with open(project_path + 'slice_output.txt', 'r') as file:
        file_content = file.read()
    
    # Check if the file contains data
    assert len(file_content) > 0, "Output file is empty"
