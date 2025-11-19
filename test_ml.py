import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics
)


# ---------------------------------------------------------
# Test 1: process_data returns correct structure
# ---------------------------------------------------------
def test_process_data_shapes():
    """
    Ensures that process_data correctly encodes categorical features
    and splits labels. Checks correct output lengths and array types.
    """
    sample = {
        "age": [25, 40, 60],
        "workclass": ["Private", "Self-emp", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"]
    }

    import pandas as pd
    df = pd.DataFrame(sample)

    cat_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    # X must be 2D
    assert len(X.shape) == 2
    # y must be 1D
    assert len(y.shape) == 1
    # lengths must match
    assert X.shape[0] == y.shape[0]
    # encoder should not be None
    assert encoder is not None
    assert lb is not None


# ---------------------------------------------------------
# Test 2: Model trains and inference returns correct shape
# ---------------------------------------------------------
def test_model_training_and_inference():
    """
    Ensures that train_model returns a sklearn model and inference()
    produces predictions of correct shape.
    """
    sample = {
        "age": [22, 45, 37, 52],
        "workclass": ["Private", "Self-emp", "Private", "State-gov"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    }

    import pandas as pd
    df = pd.DataFrame(sample)

    cat_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X)

    assert len(preds) == len(y)
    assert preds.shape == y.shape


# ---------------------------------------------------------
# Test 3: compute_model_metrics outputs valid values
# ---------------------------------------------------------
def test_compute_model_metrics_values():
    """
    Ensures compute_model_metrics returns precision, recall, and fbeta 
    scores that fall between 0 and 1.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1
