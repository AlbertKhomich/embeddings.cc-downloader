import requests
import time
import json
import logging
from config import API_URL
from tenacity import (
    retry, 
    wait_exponential,
    stop_after_attempt, 
    retry_if_exception_type
)

logger = logging.getLogger(__name__)

exp_base = (3600 / 10) ** (1 / 3)

def ping(base_url=API_URL, timeout=5):
    try:
        r = requests.get(f"{base_url}/ping", timeout=timeout)
        r.raise_for_status()
        return True
    except Exception:
        logging.error("❌ Embedding service unreachable at %s", base_url, exc_info=True)
        return False

def _before_retry_callback(retry_state):
    err = retry_state.outcome.exception()
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed with {err!r}, "
        f"next retry in {retry_state.next_action.sleep} seconds"
    )
    healthy = ping()
    logger.error(f"Service health after failure: {'UP' if healthy else 'DOWN'}")

@retry(
    wait=wait_exponential(multiplier=10, exp_base=exp_base, max=3600),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=_before_retry_callback
)
def add_data(password, index, docs, base_url=API_URL):
    url = f"{base_url}/add"
    payload = {
        "password": password,
        "index": index,
        "docs": docs
    }

    try:
        # start = time.monotonic()
        response = requests.post(url, json=payload, timeout=(5, 30))
        # duration = time.monotonic() - start
        # logging.info(f"Request took {duration:.2f}s")
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
