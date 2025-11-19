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


# ---- load saved artifacts -------------------------------------------------
project_path = "."
model_dir = os.path.join(project_path, "model")

encoder_path = os.path.join(model_dir, "encoder.pkl")
lb_path = os.path.join(model_dir, "lb.pkl")
model_path = os.path.join(model_dir, "model.pkl")

encoder = load_model(encoder_path)
lb = load_model(lb_path)
model = load_model(model_path)

# ---- create FastAPI app ---------------------------------------------------
app = FastAPI(title="Income Classification API")


# ---- root endpoint --------------------------------------------------------
@app.get("/")
async def get_root():
    """Simple welcome endpoint."""
    return {"message": "Hello from the income classification API!"}


# ---- prediction endpoint --------------------------------------------------
@app.post("/predict/")
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

    # process data for inference
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # make prediction
    preds = inference(model, data_processed)
    _inference = preds[0]

    # convert 0/1 to <=50K />50K label
    # NOTE: apply_label expects an indexable input, so wrap in a list
    return {"result": apply_label([_inference])}
