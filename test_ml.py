import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from ml.model import train_model, compute_model_metrics

# Implement the first test. Change the function name and input as needed
def test_model_type():
    """
    # Test that train_model returns a RandomForestClassifier instance when trained on a simple dataset.
    """
    X = pd.DataFrame({
        "feature1": [0, 1] * 10,
        "feature2": [1, 0] * 10
    })
    y = pd.Series([0, 1] * 10)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


# Implement the second test. Change the function name and input as needed
def test_compute_model_metrics_values():
    """
    # Test that compute_model_metrics returns expected metric values (approximate).
    """
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert np.isclose(precision, 0.99, atol=1.0), "Precision out of expected range"
    assert np.isclose(recall, 0.66, atol=1.0), "Recall out of expected range"
    assert np.isclose(f1, 0.66, atol=1.0), "F1 out of expected range"
    


# Implement the third test. Change the function name and input as needed
def test_training_input_shape():
    """
    # Test that train_model handles correct input shape and type.
    """
    X = pd.DataFrame({
        "feature1": [0, 1] * 10,
        "feature2": [1, 0] * 10
    })
    y = pd.Series([0, 1] * 10)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict"), "Trained model does not have predict method"
