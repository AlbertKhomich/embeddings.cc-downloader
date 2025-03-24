import requests

def create_index(password, index, dimensions, shards=5, base_url="http://embeddings.cs.upb.de:1337"):
    endpoint = f"{base_url}/create_index"
    params = {
        "password": password,
        "index": index,
        "dimensions": dimensions,
        "shards": shards
    }
    response = requests.post(endpoint, params=params)
    return response
