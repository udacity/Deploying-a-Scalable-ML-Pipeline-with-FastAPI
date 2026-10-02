import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model

@pytest.fixture
def trained_model():
    X = np.array([[1,1],[2,2],[3,3],[4,4]])
    y = np.array([0,1,0,1])

    return train_model(X,y)

def test_one(trained_model):
    """
    Test if the model is a RandomForestClassifier.
    """
    assert isinstance(trained_model, RandomForestClassifier)

def test_two(trained_model):
    """
    Check if inference returns the correct type and one prediction 
    for each observation.
    """
    X_test = np.array([[1,2],[3,4]])
    preds = inference(trained_model, X_test)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)

def test_three():
    """
    Check if the compute_model_metrics calculates the correct values.
    """
    y = np.array([0,1,0,1])
    preds = np.array([0,1,0,1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

def test_four(tmp_path, trained_model):
    """
    Check if save_model and load_model preserve the trained model.
    """
    model_path = tmp_path / 'model.pkl'
    save_model(trained_model, model_path)
    loaded_model = load_model(model_path)

    X_test = np.array([[1,2],[3,4]])
    original_preds = trained_model.predict(X_test)
    loaded_preds = loaded_model.predict(X_test)
    assert np.array_equal(original_preds, loaded_preds)