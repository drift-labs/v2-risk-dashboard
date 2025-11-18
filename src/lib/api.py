import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BACKEND_URL")
STORAGE_PREFIX = os.getenv("STORAGE_PREFIX")


def fetch_api_data(
    section: str,
    path: str,
    params: Optional[dict] = None,
    retry: bool = False,
    max_wait_time: int = 45,
) -> dict:
    """
    Makes direct API calls to the backend service with optional retry logic for "miss" results.

    Args:
        section (str): API section (maps to filename in backend/api/)
        path (str): API endpoint (maps to function name)
        params (Optional[dict]): Query parameters to include in request
        retry (bool): Whether to retry on "miss" results with exponential backoff
        max_wait_time (int): Maximum total wait time in seconds before giving up

    Returns:
        dict: JSON response data
    """
    if not retry:
        if params:
            response = requests.get(f"{BASE_URL}/api/{section}/{path}", params=params)
        else:
            response = requests.get(f"{BASE_URL}/api/{section}/{path}")

        result = response.json()
        if response.status_code == 202 or (
            isinstance(result, dict) and result.get("result") == "processing"
        ):
            return result
    else:
        base_delay = 0.5
        total_wait_time = 0
        attempt = 0

        while total_wait_time < max_wait_time:
            if params:
                response = requests.get(
                    f"{BASE_URL}/api/{section}/{path}", params=params
                )
            else:
                response = requests.get(f"{BASE_URL}/api/{section}/{path}")

            result = response.json()
            if not ("result" in result and result["result"] in ("miss", "processing")):
                break

            delay = min(base_delay * (2**attempt), max_wait_time - total_wait_time)
            if delay <= 0:
                break

            time.sleep(delay)
            total_wait_time += delay
            attempt += 1
        else:
            print(
                f"Fetching {section}/{path} did not succeed after {attempt} retries (waited {total_wait_time:.1f}s)"
            )
            return None
    return response.json()


def fetch_cached_data(url: str, _params: Optional[dict] = None, key: str = "") -> dict:
    """
    Fetches cached data from storage with Streamlit caching.

    Args:
        url (str): API endpoint path
        _params (Optional[dict]): Query parameters for the request

    Returns:
        dict: Cached response content
    """
    cache_key = f"GET/api/{url}".replace("/", "_")

    if _params:
        print(f"Params: {_params}")
        query_parts = []
        for k, v in _params.items():
            if isinstance(v, str):
                v = v.replace(" ", "%2B")
            query_parts.append(f"{k}-{v}")
        query_str = "_".join(query_parts)
        cache_key = f"{cache_key}__{query_str}"

    use_storage = os.getenv("USE_STORAGE", "false").lower() == "true"

    if STORAGE_PREFIX and use_storage:
        storage_url = f"{STORAGE_PREFIX}/{cache_key}.json"
    else:
        storage_url = f"{BASE_URL}/api/ucache/{cache_key}.json"

    response = requests.get(storage_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch from storage: {response.status_code}")

    response_data = response.json()
    return response_data["content"]
