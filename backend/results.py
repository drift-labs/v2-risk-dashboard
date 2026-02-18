import json
import logging
import os

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"


def make_result_key(endpoint: str, params: dict) -> str:
    """Build a human-readable filename key from endpoint and params.

    Must match the key format used by backend/scripts/generate.py.
    """
    parts = [endpoint.replace("/", "_")]
    for k, v in sorted(params.items()):
        parts.append(f"{k}-{v}")
    return "__".join(parts)


def read_result(endpoint: str, params: dict):
    """Read a pre-generated result file.

    Returns the parsed JSON content, or None if not found.
    """
    key = make_result_key(endpoint, params)
    path = os.path.join(RESULTS_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading result file {path}: {e}")
        return None
