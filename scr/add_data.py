import requests
import json
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

exp_base = (3600 / 10) ** (1 / 3)

@retry(
    wait=wait_exponential(multiplier=10, exp_base=exp_base, max=3600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logger, logging.INFO)
)
def add_data(password, index, docs, base_url="http://131.234.26.202:1337"):
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
