import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

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

project_path = "D:\\Learning\\Udacity\\Code\\Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
encoder = load_model(encoder_path)

model_path = os.path.join(project_path, "model", "model.pkl")
model = load_model(model_path)


app = FastAPI()

@app.get("/")
async def get_root():
    """ Say hello!"""
    return {"message": "Welcome to the Census Prediction API"}

@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.model_dump()
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
        categorical_features= cat_features,
        label= None,
        training= False,
        encoder= encoder,
        lb= None
    )
    _inference = inference(model, data_processed) # your code here to predict the result using data_processed
    return {"result": apply_label(_inference)}
