import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import (
    compute_model_metrics,
    train_model,
    inference
)
# Sample data
X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([[0, 1], [1, 0]])
y_test = np.array([0, 1])

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that the train_model function returns a RandomForestClassifier.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be an instance of RandomForestClassifier."


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test that the compute_model_metrics function returns the expected precision, recall, and F1 score.
    """
    preds = inference(train_model(X_train, y_train), X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float), "Precision should be a float."
    assert isinstance(recall, float), "Recall should be a float."
    assert isinstance(fbeta, float), "F1 Score should be a float."
    
    # Optionally check for known values if you have a small sample where you know the metrics
    assert precision >= 0 and precision <= 1, "Precision should be between 0 and 1."
    assert recall >= 0 and recall <= 1, "Recall should be between 0 and 1."
    assert fbeta >= 0 and fbeta <= 1, "F1 Score should be between 0 and 1."


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that the inference function returns predictions of the expected shape.
    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0], "Predictions should match the number of input samples."
