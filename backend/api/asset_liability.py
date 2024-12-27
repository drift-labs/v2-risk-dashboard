from driftpy.pickle.vat import Vat
from fastapi import APIRouter
import logging

from backend.state import BackendRequest
from backend.utils.matrix import get_matrix

router = APIRouter()
logger = logging.getLogger(__name__)


async def _get_asset_liability_matrix(
    slot: int,
    vat: Vat,
    mode: int,
    perp_market_index: int,
) -> dict:
    logger.info("==> Starting asset liability matrix calculation...")
    logger.info(f"Mode: {mode}, Perp Market Index: {perp_market_index}")
    
    logger.info("==> Processing user data and calculating metrics...")
    df = await get_matrix(vat, mode, perp_market_index)
    
    logger.info("==> Converting DataFrame to dictionary...")
    df_dict = df.to_dict()
    
    logger.info("==> Asset liability matrix calculation complete")

    return {
        "slot": slot,
        "df": df_dict,
    }


@router.get("/matrix")
async def get_asset_liability_matrix(
    request: BackendRequest, mode: int, perp_market_index: int
):
    return await _get_asset_liability_matrix(
        request.state.backend_state.last_oracle_slot,
        request.state.backend_state.vat,
        mode,
        perp_market_index,
    )
