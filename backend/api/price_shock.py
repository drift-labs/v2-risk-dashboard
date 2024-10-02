from typing import Optional

from backend.state import BackendRequest
from backend.state import BackendState
from backend.utils.matrix import get_usermap_df
from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat
from fastapi import APIRouter


router = APIRouter()


@router.get("/usermap")
async def get_price_shock(
    request: BackendRequest,
    oracle_distortion: float = 0.1,
    asset_group: Optional[str] = None,
    n_scenarios: int = 5,
):
    backend_state: BackendState = request.state.backend_state
    vat: Vat = backend_state.vat
    drift_client: DriftClient = backend_state.dc

    result = await get_usermap_df(
        drift_client,
        vat.users,
        "oracles",
        oracle_distortion,
        asset_group,
        n_scenarios,
    )

    print(result)

    return {}