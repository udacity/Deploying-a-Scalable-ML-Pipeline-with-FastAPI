"""scripts/train_tiny_model.py

Helper to train a tiny model for local testing. When running the script
directly we need to ensure the repository root is on sys.path so the
`ml` package (in the repo) can be imported.
"""
from pathlib import Path
import pandas as pd

from ml.data import process_data
from ml.model import train_model, save_model

OUT = Path("model")
OUT.mkdir(parents=True, exist_ok=True)

# Columns the app expects
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
label = "salary"

# Build a tiny, but schema-correct, dataset
df = pd.DataFrame(
    [
        # a few realistic-looking rows (mix of <=50K / >50K)
        {
            "age": 52,
            "workclass": "Private",
            "fnlgt": 209642,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States",
            "salary": ">50K",
        },
        {
            "age": 28,
            "workclass": "Private",
            "fnlgt": 338409,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Sales",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "salary": "<=50K",
        },
        {
            "age": 44,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 160187,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 7688,
            "capital-loss": 0,
            "hours-per-week": 50,
            "native-country": "United-States",
            "salary": ">50K",
        },
        {
            "age": 37,
            "workclass": "Private",
            "fnlgt": 215646,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Divorced",
            "occupation": "Craft-repair",
            "relationship": "Unmarried",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "salary": "<=50K",
        },
        {
            "age": 60,
            "workclass": "Private",
            "fnlgt": 140359,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 55,
            "native-country": "United-States",
            "salary": ">50K",
        },
        {
            "age": 23,
            "workclass": "Private",
            "fnlgt": 122272,
            "education": "Some-college",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Own-child",
            "race": "Asian-Pac-Islander",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 30,
            "native-country": "United-States",
            "salary": "<=50K",
        },
        {
            "age": 47,
            "workclass": "Private",
            "fnlgt": 178356,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Tech-support",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 15024,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
            "salary": ">50K",
        },
        {
            "age": 33,
            "workclass": "Private",
            "fnlgt": 201490,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Separated",
            "occupation": "Handlers-cleaners",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 35,
            "native-country": "United-States",
            "salary": "<=50K",
        },
    ]
)

# Fit encoder & label binarizer on the tiny data
X, y, encoder, lb = process_data(
    df,
    categorical_features=categorical_features,
    label=label,
    training=True,
)

# Train a tiny model
model = train_model(X, y)

# Save artifacts with the names the app expects
save_model(model, OUT / "model.pkl")
save_model(encoder, OUT / "encoder.pkl")
save_model(lb, OUT / "lb.pkl")

print("Artifacts saved to:", OUT.resolve())
