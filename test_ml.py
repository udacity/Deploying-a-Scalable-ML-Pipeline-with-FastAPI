import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model

def test_load_data():
    """
    This test verifies that the data is loaded correctly and has the expected shape.
    """
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    assert X.shape == (100, 8), f"Incorrect shape for X: {X.shape}"
    assert y.shape == (100,), f"Incorrect shape for y: {y.shape}"


def test_model_type():
    """
    This test verifies that model returned is correct: RandomForestClassifier.
    """
    X, y = make_classification(n_samples=100, n_features=8, random_state=42)
    model = train_model(X, y)
    assert type(model) == RandomForestClassifier, f"Incorrect model type: {type(model)}"



def test_preprocessing_scaler():
    """
    This test ensures that the scaling of the data is done properly, an important step in preprocessing.
    """
    scaler = StandardScaler()
    X, _ = make_classification(n_samples=100, n_features=8, random_state=42)
    X_scaled = scaler.fit_transform(X)
    
    # Check that the mean and standard deviation are close to expected values
    assert np.isclose(np.mean(X_scaled), 0), f"Incorrect mean: {np.mean(X_scaled)}"
    assert np.isclose(np.std(X_scaled), 1), f"Incorrect std dev: {np.std(X_scaled)}"    