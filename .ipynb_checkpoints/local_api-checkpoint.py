import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
# Send a GET request
r = requests.get("http://127.0.0.1:8000")

# Print the status code
print("Status Code:", r.status_code)

# Print the welcome message
print("Welcome Message:", r.json())

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

# Send a POST request
#r = requests.post("http://127.0.0.1:8000/data", json=data)
r = "http://127.0.0.1:8000/data"
response = requests.post(r, json=data)
#print(data)

# Print the status code
print("Status Code:", response.status_code)

# Print the result
print("Result:", response.json())