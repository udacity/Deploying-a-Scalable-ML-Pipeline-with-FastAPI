import requests

URL = "http://127.0.0.1:8000"


def safe_request(method, url, **kwargs):
    try:
        r = requests.request(method, url, timeout=5, **kwargs)
        try:
            # Try to parse as JSON
            print(f"{method} {url}:", r.status_code, r.json())
        except ValueError:
            # Fallback to raw text if not JSON (e.g. error pages, 500s)
            print(f"{method} {url}:", r.status_code, r.text)
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {url}. Is the server running?")
    except requests.exceptions.Timeout:
        print(f"⏱️ Request to {url} timed out.")
    except Exception as e:
        print(f"⚠️ Unexpected error calling {url}: {e}")


# GET request
safe_request("GET", URL)

# First POST payload
payload1 = {
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
safe_request("POST", f"{URL}/data/", json=payload1)

# Second POST payload
payload2 = {
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
safe_request("POST", f"{URL}/data/", json=payload2)
