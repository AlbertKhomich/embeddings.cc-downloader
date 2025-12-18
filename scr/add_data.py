import requests
import logging
from config import API_URL
from tenacity import (
    retry, 
    wait_exponential,
    stop_never, 
    retry_if_exception
)

logger = logging.getLogger(__name__)

exp_base = (3600 / 10) ** (1 / 3)

def _is_retryable_exception(exc: BaseException) -> bool:
    if not isinstance(exc, requests.exceptions.RequestException):
        return False

    if isinstance(exc, requests.exceptions.HTTPError) and getattr(exc, "response", None) is not None:
        status = exc.response.status_code
        if 400 <= status < 500 and status != 429:
            return False

    return True

def _before_retry_callback(retry_state):
    err = retry_state.outcome.exception()
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed with {err!r}, "
        f"next retry in {retry_state.next_action.sleep} seconds"
    )

@retry(
    wait=wait_exponential(multiplier=10, exp_base=exp_base, max=3600),
    stop=stop_never,
    retry=retry_if_exception(_is_retryable_exception),
    before_sleep=_before_retry_callback,
    reraise=True,
)
def add_data(password, index, docs, base_url=API_URL):
    url = f"{base_url}/add"
    payload = {
        "password": password,
        "index": index,
        "docs": docs
    }

    try:
        response = requests.post(url, json=payload, timeout=(5, 300))
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise
