import requests
import ast
import json

def add_data(password, index, docs, base_url="http://embeddings.cs.upb.de:1337"):
    url = f"{base_url}/add"
    payload = {
        "password": password,
        "index": index,
        "docs": docs
    }
    response = requests.post(url, json=payload)
    b = ast.literal_eval(response.text)
    print(json.dumps(b, indent=2))
    return response
