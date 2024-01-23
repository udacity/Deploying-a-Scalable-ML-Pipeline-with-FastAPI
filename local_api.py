import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
get_url = "http://127.0.0.1:8000"
get_response = requests.get(get_url)

# Print the status code and the welcome message for the GET request
print("GET Request:")
print("Status Code:", get_response.status_code)
print("Response:", get_response.json())



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

# Send a POST request using the data
post_url = "http://127.0.0.1:8000/data/"
post_response = requests.post(post_url, json=data)

# Print the status code and the result for the POST request
print("\nPOST Request:")
print("Status Code:", post_response.status_code)
print("Response:", post_response.json())