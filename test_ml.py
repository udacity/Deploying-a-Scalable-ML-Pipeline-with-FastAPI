import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
import pandas as pd
from sklearn.model_selection import train_test_split

# Create more comprehensive sample data
data = pd.DataFrame({
    'age': [25, 32, 47, 51, 61],
    'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Local-gov', 'Private'],
    'education': ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'HS-grad'],
    'occupation': ['Sales', 'Exec-managerial', 'Prof-specialty', 'Adm-clerical', 'Craft-repair'],
    'race': ['White', 'White', 'White', 'White', 'Black'],
    'sex': ['Male', 'Male', 'Female', 'Female', 'Male'],
    'hours-per-week': [40, 50, 40, 45, 58],
    'target': [0, 1, 0, 1, 0]
})

# Define categorical features
cat_features = ['workclass', 'education', 'occupation', 'race', 'sex']

train, test = train_test_split(data, test_size=0.5)
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label='target', training=True)


def test_one():
    """
    Test if the train_model function returns a RandomForestClassifier instance
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test if the trained model is of type RandomForestClassifier
    """
    model = train_model(X_train, y_train)
    assert type(model).__name__ == 'RandomForestClassifier'

# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test if compute_model_metrics function returns precision, recall, and fbeta values within the expected range
    """
    # Process the test data using the captured encoder and lb
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label='target', training=False, encoder=encoder, lb=lb)
    
    # Train a model for making predictions
    model = train_model(X_train, y_train)

    # Make predictions using the trained model
    preds = model.predict(X_test)
    
    # Compute the metrics
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    
    # Check that the metrics are within the expected range
    assert isinstance(precision, float) and isinstance(recall, float) and isinstance(fbeta, float)
    assert 0 <= precision <= 1 and 0 <= recall <= 1 and 0 <= fbeta <= 1
