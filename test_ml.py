import pytest
import numpy as np
from ml.model import train_model, inference, save_model, load_model
from sklearn.ensemble import RandomForestClassifier

def test_train_model_returns_fitted_model():
    """
    Ensure that train_model returns a fitted scikit-learn model
    when given simple training data. 
    """
    X_train = np.array([[0,1],[1,0],[1,1],[0,0],[1,2],[2,1]])
    y_train = np.array([0,1,1,0,1,0])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_inference_produces_expected_shape():
    """
    # Verify that inference returns predictions of the correct shape.
    """
    X_train = np.array([[0,1],[1,0],[1,1],[0,0],[1,2],[2,1]])
    y_train = np.array([0,1,1,0,1,0])
    model = train_model(X_train, y_train)
    X_test = np.array([[1, 1], [0, 0]])
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]


def test_save_and_load_model_roundtrip(tmp_path):
    """
    # Confirm that saving and loading a model preserves its behavior.
    """
    X_train = np.array([[0,1],[1,0],[1,1],[0,0],[1,2],[2,1]])
    y_train = np.array([0,1,1,0,1,0])
    model = train_model(X_train, y_train)

    model_path = tmp_path / "model.pkl"
    save_model(model, model_path)

    loaded_model = load_model(model_path)

    X_test = np.array([[1,1],[0,0],[0,1],[0,1],[2,1],[0,2]])
    preds_original = inference(model, X_test)
    preds_loaded = inference(loaded_model, X_test)

    assert np.array_equal(preds_original, preds_loaded)
