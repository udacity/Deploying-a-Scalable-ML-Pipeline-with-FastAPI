import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# Sample minimal data for tests
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "workclass": ["Private", "Self-emp"],
        "education": ["Bachelors", "Masters"],
        "marital-status": ["Never-married", "Married"],
        "occupation": ["Tech-support", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "India"],
        "salary": [">50K", "<=50K"]
    })
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return data, cat_features


def test_model_training_returns_random_forest(sample_data):
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    data, cat_features = sample_data
    X, y, _, _ = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "train_model should return a RandomForestClassifier"


def test_compute_metrics_expected_values():
    """
    Test compute_model_metrics with known predictions and labels.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    assert np.isclose(precision, 1.0), "Precision mismatch"
    assert np.isclose(recall, 2.0 / 3), "Recall mismatch"
    assert np.isclose(f1, 0.8, atol=1e-3), "F1 mismatch"



def test_data_processing_output_shape(sample_data):
    """
    Test that process_data returns features and labels of matching length.
    """
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert X.shape[0] == y.shape[0], "Mismatch in processed data samples and labels"

