import requests

# Correct POST URL to /data/
post_url = 'http://127.0.0.1:8000/data/'

# Sample data to send in the POST request
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

# Send POST request to /data/ route with JSON data
r = requests.post(post_url, json=data)

# Print the response
print(f"POST request status code: {r.status_code}")
print(f"POST result: {r.text}")
