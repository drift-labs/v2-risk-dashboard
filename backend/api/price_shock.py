from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat
from fastapi import APIRouter

from backend.state import BackendRequest
from backend.utils.price_shock import PriceShockAssetGroup, get_price_shock_df

router = APIRouter()


async def _get_price_shock(
    slot: int,
    vat: Vat,
    drift_client: DriftClient,
    oracle_distortion: float = 0.1,
    asset_group: str = PriceShockAssetGroup.IGNORE_STABLES.value,
    n_scenarios: int = 5,
    pool_id: int = None,
) -> dict:
    asset_group = asset_group.replace("+", " ")
    price_shock_asset_group = PriceShockAssetGroup(asset_group)

    return get_price_shock_df(
        slot=slot,
        drift_client=drift_client,
        vat=vat,
        oracle_distortion=oracle_distortion,
        asset_group=price_shock_asset_group,
        n_scenarios=n_scenarios,
        pool_id=pool_id,
    )


@router.get("/usermap")
async def get_price_shock(
    request: BackendRequest,
    oracle_distortion: float = 0.1,
    asset_group: str = PriceShockAssetGroup.IGNORE_STABLES.value,
    n_scenarios: int = 5,
    pool_id: int = None,
):
    return await _get_price_shock(
        request.state.backend_state.last_oracle_slot,
        request.state.backend_state.vat,
        request.state.backend_state.dc,
        oracle_distortion,
        asset_group,
        n_scenarios,
        pool_id,
    )
