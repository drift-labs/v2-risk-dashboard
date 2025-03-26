import logging
from fastapi import APIRouter

from backend.state import BackendRequest

router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/")
def backend_health_check(request: BackendRequest):
    """
    Backend server health check endpoint that always returns 200 OK.
    This endpoint is used to verify the backend server is running and can handle requests.
    It bypasses the caching mechanism and is accessible even when the system is not fully ready.
    
    Returns:
        dict: A dictionary containing backend server status information:
        - status (str): Always "ok" if the server is running
        - ready (bool): Whether the backend state is fully initialized
        - pickle_path (str): The current pickle path being used
    """
    return {
        "status": "ok",
        "ready": request.state.backend_state.ready,
        "pickle_path": request.state.backend_state.current_pickle_path,
    }
