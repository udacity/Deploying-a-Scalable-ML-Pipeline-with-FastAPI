import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# ---------- Request schema (Pydantic v2 style) ----------
class Data(BaseModel):
    # allow aliases (hyphenated names) and show a full example in /docs
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 178356,
                "education": "HS-grad",
                "education-num": 10,
                "marital-status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


# ---------- Load artifacts ----------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENCODER_PATH = os.path.join(_PROJECT_ROOT, "model", "encoder.pkl")
MODEL_PATH = os.path.join(_PROJECT_ROOT, "model", "model.pkl")

if not os.path.exists(ENCODER_PATH):
    raise RuntimeError(
        f"Encoder not found at {ENCODER_PATH}. Did you run train_model.py?"
    )
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Did you run train_model.py?")

encoder = load_model(ENCODER_PATH)
model = load_model(MODEL_PATH)


# ---------- FastAPI app ----------
app = FastAPI(title="Census Income Inference API")


@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census Income Inference API"}


@app.post("/data/")
async def post_inference(data: Data):
    try:
        # Use aliases and normalize keys to hyphenated for the pipeline
        data_dict = data.model_dump(by_alias=True)
        df = pd.DataFrame([{k.replace("_", "-"): v for k, v in data_dict.items()}])

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

        X, _, _, _ = process_data(
            df,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=None,
        )
        preds = inference(model, X)
        return {"result": apply_label(preds)}
    except Exception as e:
        # Keep stacktrace in server logs but surface a clear client error
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
