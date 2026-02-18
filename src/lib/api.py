import os
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BACKEND_URL")


def fetch_api_data(
    section: str,
    path: str = "",
    params: Optional[dict] = None,
) -> dict:
    """
    Makes direct API calls to the backend service.

    Args:
        section (str): API section (maps to filename in backend/api/)
        path (str): API endpoint path (maps to function name)
        params (Optional[dict]): Query parameters to include in request

    Returns:
        dict: JSON response data
    """
    url = f"{BASE_URL}/api/{section}/{path}" if path else f"{BASE_URL}/api/{section}"
    response = requests.get(url, params=params)
    return response.json()
