from driftpy.pickle.vat import Vat
from fastapi import APIRouter

from backend.state import BackendRequest
from backend.utils.matrix import get_matrix
from backend.utils.user_metrics import get_user_metrics_maintenance

router = APIRouter()


async def _get_asset_liability_matrix(
    slot: int,
    vat: Vat,
    mode: int,
    perp_market_index: int,
) -> dict:
    print("==> Getting asset liability matrix...")
    df = await get_matrix(vat, mode, perp_market_index)
    df_dict = df.to_dict()
    print("==> Asset liability matrix fetched")

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


async def _get_liquidation_simulation(
    slot: int,
    vat: Vat,
    spot_market_index: int,
    new_maintenance_asset_weight: float,
) -> dict:
    spot_market = None
    for market in vat.spot_markets.values():
        if market.data.market_index == spot_market_index:
            spot_market = market
            break

    if spot_market is None:
        return {
            "error": f"Spot market {spot_market_index} not found",
            "total_users": 0,
            "liquidated_users": 0,
            "liquidated_user_keys": [],
        }

    original_maint_weight = spot_market.data.maintenance_asset_weight

    res_original = get_user_metrics_maintenance(vat.users)
    metrics_original = res_original["metrics_maintenance"]
    user_keys_original = res_original["user_keys"]

    already_liquidated = set()
    healthy_users = {}  # user_key -> health for users with health > 0

    for i, metrics in enumerate(metrics_original):
        user_key = user_keys_original[i]
        health = metrics["health"]
        if health <= 0:
            already_liquidated.add(user_key)
        else:
            healthy_users[user_key] = health

    new_maint_weight_raw = int(new_maintenance_asset_weight * 10000)
    spot_market.data.maintenance_asset_weight = new_maint_weight_raw

    try:
        res_new = get_user_metrics_maintenance(vat.users)
        metrics_new = res_new["metrics_maintenance"]
        user_keys_new = res_new["user_keys"]

        newly_liquidated = []
        for i, metrics in enumerate(metrics_new):
            user_key = user_keys_new[i]
            health = metrics["health"]
            if user_key in healthy_users and health <= 0:
                newly_liquidated.append(user_key)

        return {
            "original_maintenance_asset_weight": float(original_maint_weight) / 10000,
            "new_maintenance_asset_weight": new_maintenance_asset_weight,
            "total_users": len(user_keys_new),
            "already_liquidated_count": len(already_liquidated),
            "already_liquidated_user_keys": list(already_liquidated),
            "newly_liquidated_count": len(newly_liquidated),
            "newly_liquidated_user_keys": newly_liquidated,
            "total_liquidated_after_change": len(already_liquidated)
            + len(newly_liquidated),
            "slot": slot,
        }
    finally:
        spot_market.data.maintenance_asset_weight = original_maint_weight


@router.get("/liquidation-simulation")
async def get_liquidation_simulation(
    request: BackendRequest,
    spot_market_index: int,
    new_maintenance_asset_weight: float,
):
    return await _get_liquidation_simulation(
        request.state.backend_state.last_oracle_slot,
        request.state.backend_state.vat,
        spot_market_index,
        new_maintenance_asset_weight,
    )


@router.get("/maintenance-asset-weights")
async def get_maintenance_asset_weights(
    request: BackendRequest,
):
    return {
        "maintenance_asset_weights": {
            market.data.market_index: float(market.data.maintenance_asset_weight)
            / 10000
            for market in request.state.backend_state.vat.spot_markets.values()
        },
    }
