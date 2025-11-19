import json
import requests

BASE_URL = "http://127.0.0.1:8000"

# ---- GET request ----------------------------------------------------------
r = requests.get(f"{BASE_URL}/")

# print status code and welcome message
print("GET / status:", r.status_code)
print("GET / response:", r.json())


# ---- POST request ---------------------------------------------------------
data = {
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

r = requests.post(f"{BASE_URL}/predict/", json=data)

print("POST /predict/ status:", r.status_code)
print("POST /predict/ response:", r.json())
