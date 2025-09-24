# tests/test_api.py
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


def test_predict_payload_1():
    payload = {
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
    }
    r = client.post("/data/", json=payload)
    assert r.status_code == 200
    assert r.json()["result"] in [">50K", "<=50K"]
