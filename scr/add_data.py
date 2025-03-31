import requests
import json
import time
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=10, min=10, max=80),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def add_data(password, index, docs, base_url="http://embeddings.cs.upb.de:1337"):
    url = f"{base_url}/add"
    payload = {
        "password": password,
        "index": index,
        "docs": docs
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
