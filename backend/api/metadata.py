from fastapi import APIRouter, BackgroundTasks
from typing import List
import os
import glob
import logging

from backend.state import BackendRequest, BackendState

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
def get_metadata(request: BackendRequest):
    """
    Return metadata about the current backend state, including the active pickle path and last oracle slot.
    """
    backend_state: BackendState = request.state.backend_state
    return {
        "pickle_file": backend_state.current_pickle_path,
        "last_oracle_slot": backend_state.vat.register_oracle_slot,
    }

@router.get("/list_pickles")
def list_pickles(request: BackendRequest) -> List[str]:
    """
    Lists all available pickle snapshots in the /pickles directory.
    This helps troubleshoot differences between local and production by comparing available snapshots.
    """
    pickles_dir = os.path.join("pickles")
    if not os.path.exists(pickles_dir):
        return []

    # We return the directories inside 'pickles' sorted by modification time
    dirs = [d for d in glob.glob(os.path.join(pickles_dir, "*")) if os.path.isdir(d)]
    # Sort by modified time descending
    dirs.sort(key=os.path.getmtime, reverse=True)
    return dirs

@router.get("/force_refresh")
async def force_refresh(request: BackendRequest, background_tasks: BackgroundTasks):
    """
    Force the backend to create a new pickle snapshot and immediately reload it.
    This can help ensure the newest data is used in production if the environment
    fell behind or is loading an outdated snapshot.
    """
    backend_state: BackendState = request.state.backend_state

    logger.info("[force_refresh] Triggering a new pickle snapshot from /force_refresh endpoint.")
    background_tasks.add_task(backend_state.take_pickle_snapshot)

    return {"detail": "A new pickle snapshot is being created and will be loaded once complete."}