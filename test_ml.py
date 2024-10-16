import pytest
# TODO: add necessary import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

df = pd.read_csv("data/census.csv")

cat_features = [
    "workclass", "education", "marital-status", "occupation", 
    "relationship", "race", "sex", "native-country"
]

# TODO: implement the first test. Change the function name and input as needed
def test_process_data():
    """
    # add description for the first test
    """
    # Your code here
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label='salary', training=True)
    assert X.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


# TODO: implement the second test. Change the function name and input as needed
def test_train_model():
    """
    # add description for the second test
    """
    # Your code here
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label='salary', training=True)
    model = train_model(X,y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    # add description for the third test
    """
    # Your code here
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 1]
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert precision == 2 / 3
    assert recall == 1
    assert f1 == 0.8

def test_inference():
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label='salary', training=True)
    model = train_model(X,y)
    preds = inference(model, X)
    assert len(preds) == len(y)
    assert isinstance(preds, np.ndarray)
