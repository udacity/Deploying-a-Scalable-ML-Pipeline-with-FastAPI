import json

import requests

# Send a GET using the URL http://127.0.0.1:8000
url="http://127.0.0.1:8000/predict" # Your code here

r = requests.get("http://127.0.0.1:8000")
# TODO: print the status code
print("Status Code:", r.status_code)
# TODO: print the welcome message
print("Results:", r.json())



input_data = {
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
    "native-country": "United-States"
}


# Send a POST using the data above
response = requests.post(url, json=input_data) # Your code here

# Print the status code
print("Status Code: {response.status_code}")

print("Response text:", response.text)
# Print the result
if response.status_code == 200:
    print("Result:", response.json())
else:
    print(f"Failed with status code {response.status_code}. Response text: {response.text}")
