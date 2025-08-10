import os
import pandas as pd
import pytest

# Load dataset once for all tests
PROJECT_PATH = "../Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
data = pd.read_csv(DATA_PATH)
MODEL_PATH = os.path.join(PROJECT_PATH, "model", "model.pkl")
ENCODER_PATH = os.path.join(PROJECT_PATH, "model", "encoder.pkl")

def test_pickles_exist():
    """
    Test that encoder.pkl and model.pkl files were created.
    """
    assert os.path.exists(MODEL_PATH), "model.pkl does not exist"
    assert os.path.exists(ENCODER_PATH), "encoder.pkl does not exist"

def test_age_range():
    """
    Test that all ages are between 0 and 100.
    """
    assert data["age"].between(0, 100).all()

def test_hours_per_week_range():
    """
    Test that hours worked per week are between 1 and 99.
    """
    assert data["hours-per-week"].between(1, 99).all()

