import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import inference, train_model

CATS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def _prep():
    df = pd.read_csv("data/census.csv")
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df[LABEL]
    )
    Xtr, ytr, enc, lb = process_data(train, CATS, label=LABEL, training=True)
    Xte, yte, _, _ = process_data(
        test, CATS, label=LABEL, training=False, encoder=enc, lb=lb
    )
    return Xtr, ytr, Xte, yte


def test_train_and_infer_shapes():
    Xtr, ytr, Xte, yte = _prep()
    m = train_model(Xtr, ytr)
    yhat = inference(m, Xte)
    assert yhat.shape[0] == yte.shape[0]


def test_model_type():
    Xtr, ytr, _, _ = _prep()
    m = train_model(Xtr, ytr)
    from sklearn.ensemble import RandomForestClassifier

    assert isinstance(m, RandomForestClassifier)


def test_metrics_reasonable():
    Xtr, ytr, Xte, yte = _prep()
    m = train_model(Xtr, ytr)
    from sklearn.metrics import f1_score

    yhat = inference(m, Xte)
    f1 = f1_score(yte, yhat)
    assert 0.3 <= f1 <= 1.0
