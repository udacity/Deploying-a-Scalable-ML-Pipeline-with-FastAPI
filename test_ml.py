import pytest
# TODO: add necessary import
from ml.model import train_and_save_final_model
from ml.model import inference
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import tempfile



# TODO: implement the first test. Change the function name and input as needed
#sample data
@pytest.fixture
def fake_sample_data():
    return pd.DataFrame({
        'feature1': ['A', 'B','C'],
        'feature2': ["X", "Y", 'Z'],
        'label': [1, 0, 1]
    })


def test_train_and_save_final_model(fake_sample_data):
    
    """
    # add description for the first test
    test if the pipeline runs.
    """
    # Your code here
    with tempfile.TemporaryDirectory()as tmp_dir:
        model, encoder, lb = train_and_save_final_model(
            data=fake_sample_data,
            categorical_features=["feature1", "feature2"],
            label='label',
            model_dir=tmp_dir
        )
        #check output is not none
        assert model is not None
        assert encoder is not None
        assert lb is not None
    pass

#Effective Python Testing With pytest, Dane Hillard,Dec 08, 2024, https://realpython.com/pytest-python-testing/


# TODO: implement the second test. Change the function name and input as needed
def test_function_inference():
    """
    # add description for the second test
    test function def inference(model, X):
    """
    # Your code here
    
    #simple model to train, look for prediction 
    model = RandomForestClassifier().fit(
        pd.DataFrame({"a": [0,1], "b": [1, 0]}),
        [0,1]
    )
    
    preds = model.predict(pd.DataFrame({"a": [1], "b": [0]}))
    
    #assert what should be true
    assert preds[0] in [0, 1]
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_data_split_shape():
    """
    # add description for the third test
    checks the shape of training and test datasets
    """
    # Your code here
    #using the same sample dataset in test1
    data = pd.DataFrame({
        'feature1': ['A', 'B','C'],
        'feature2': ["X", "Y", 'Z'],
        'label': [1, 0, 1]
    })
    
    #split and label
    X=data[['feature1', 'feature2']]
    y=data['label']

    #train test split 1-train 2- split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)
    
    #check shapes
    assert X_train.shape ==(2, 2)
    assert X_test.shape ==(1, 2)
    assert y_train.shape[0] == 2
    assert y_test.shape[0] == 1
    pass
