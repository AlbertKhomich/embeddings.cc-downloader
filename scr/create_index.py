import requests
from config import API_URL

def create_index(password, index, dimensions, shards=5, base_url=API_URL):
    endpoint = f"{base_url}/create_index"
    params = {
        "password": password,
        "index": index,
        "dimensions": dimensions,
        "shards": shards
    }
    response = requests.post(endpoint, params=params)
    return response
