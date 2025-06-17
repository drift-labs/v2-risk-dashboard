# This API endpoint serves a static, pre-computed user retention summary from a local JSON file.

import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Define the path for the static data file
STATIC_DATA_PATH = "shared/user_retention_summary.json"

# ──────────────────────── 1A. Pydantic Models ──────────────────────── #

class RetentionSummaryItem(BaseModel):
    market: str
    category: List[str]
    new_traders: int
    new_traders_list: Optional[List[str]] = None
    retained_users_14d: Optional[int] = None
    retention_ratio_14d: Optional[float] = None
    retained_users_14d_list: Optional[List[str]] = None
    retained_users_28d: Optional[int] = None
    retention_ratio_28d: Optional[float] = None
    retained_users_28d_list: Optional[List[str]] = None

    class Config:
        orm_mode = True

# ──────────────────────── 2. API Endpoint ───────────────────────── #

@router.get("/summary", response_model=List[RetentionSummaryItem])
async def get_user_retention_summary():
    """
    Provides a pre-computed summary of user retention for "hype" markets
    by reading from a static JSON file.
    """
    logger.info(f"Received request for /summary endpoint. Reading from {STATIC_DATA_PATH}.")
    
    try:
        with open(STATIC_DATA_PATH, 'r') as f:
            data = json.load(f)
        
        # Validate data with Pydantic model
        validated_results = [RetentionSummaryItem(**item) for item in data]
        logger.info(f"Successfully loaded and validated {len(validated_results)} summary items from static file.")
        return validated_results

    except FileNotFoundError:
        logger.error(f"Static data file not found at: {STATIC_DATA_PATH}")
        raise HTTPException(
            status_code=404, 
            detail=(
                "The user retention summary file was not found. "
                "Please ensure it has been generated and placed in the 'shared' directory."
            )
        )
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {STATIC_DATA_PATH}")
        raise HTTPException(status_code=500, detail="Failed to parse the summary data file.")
    except Exception as e:
        logger.error(f"Unhandled error in /summary endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")