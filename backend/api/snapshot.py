from fastapi import APIRouter, BackgroundTasks
import logging

from backend.state import BackendRequest, BackendState

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/pickle")
async def pickle(request: BackendRequest, background_tasks: BackgroundTasks):
    """
    Trigger an asynchronous background task to take a new pickle snapshot.
    """
    backend_state: BackendState = request.state.backend_state
    logger.info("[pickle] Endpoint called: adding background task to take_pickle_snapshot.")
    background_tasks.add_task(backend_state.take_pickle_snapshot)
    return {"result": "background task added"}