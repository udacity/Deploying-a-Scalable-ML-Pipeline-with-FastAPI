import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")
try:
    encoder_path = os.path.join(os.getcwd(), "model", "encoder.pkl") # TODO: enter the path for the saved encoder 
    encoder = load_model(encoder_path)
    print("Encoder loaded successfully.")

    model_path = os.path.join(os.getcwd(), "model","model.pkl") # TODO: enter the path for the saved model 
    model = load_model(model_path)
    print("Model loaded successfully.")


except Exception as e:
    print(f"Error loading model, encoder, or label binarizer: {e}")
    raise HTTPException(status_code=500,  detail=f"Error loading model, encoder, or label binarizer: {str(e)}")

# Create a RESTful API using FastAPI
app = FastAPI() # your code here

# Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """ Say hello!"""
    # your code here
    return {"message": "Hello from the API!"}


# Create a POST on a different path that does model inference
@app.post("/predict")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        training=False,# your code here
        encoder=encoder
        # use data as data input
        # use training = False
        # do not need to pass lb as input
    )
    _inference = model.predict(data_processed)
    result = apply_label(_inference)
    return {"result": result}
