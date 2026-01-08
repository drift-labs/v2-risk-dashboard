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

    # Core columns in display order (always included even if zero)
    core_columns_ordered = [
        'user_key', 'is_high_leverage', 'leverage', 'upnl', 'net_usd_value',
        'perp_liability', 'spot_asset', 'spot_liability', 'health'
    ]
    core_columns = set(core_columns_ordered)

    # Columns to exclude from output (used for computation only)
    internal_columns = {'net_v', 'net_p'}

    # Convert to sparse records format - omit zero values for spot columns
    # This reduces size by ~90%+ since data is 98.6% sparse
    records = []
    for _, row in df.iterrows():
        record = {}

        # Add user_key first
        record['user_key'] = row['user_key']

        # Add sparse perp_positions right after user_key
        # This preserves full perp exposure data that would otherwise be lost
        if 'net_p' in row and row['net_p']:
            perp_positions = {
                str(market_idx): val
                for market_idx, val in row['net_p'].items()
                if val != 0 and val != 0.0
            }
            if perp_positions:
                record['perp_positions'] = perp_positions

        # Add remaining core columns in order
        for col in core_columns_ordered[1:]:  # Skip user_key, already added
            if col in df.columns:
                record[col] = row[col]

        # Add non-zero spot columns
        for col in df.columns:
            if col in internal_columns or col in core_columns:
                continue
            val = row[col]
            if val != 0 and val != 0.0:
                record[col] = val

        records.append(record)

    print("==> Asset liability matrix fetched")

    return {
        "slot": slot,
        "df": records,
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
    already_liquidated_with_liability = []  # (user_key, total_liability)

    for i, metrics in enumerate(metrics_original):
        user_key = user_keys_original[i]
        health = metrics["health"]
        total_liability = metrics["spot_liability"] + metrics["perp_liability"]
        if health <= 0:
            already_liquidated.add(user_key)
            already_liquidated_with_liability.append((user_key, total_liability))
        else:
            healthy_users[user_key] = health

    # Sort already liquidated by total liability (largest first)
    already_liquidated_sorted = [
        user_key
        for user_key, _ in sorted(
            already_liquidated_with_liability, key=lambda x: x[1], reverse=True
        )
    ]
    already_liquidated_liabilities = {
        user_key: liability
        for user_key, liability in sorted(
            already_liquidated_with_liability, key=lambda x: x[1], reverse=True
        )
    }

    new_maint_weight_raw = int(new_maintenance_asset_weight * 10000)
    spot_market.data.maintenance_asset_weight = new_maint_weight_raw

    try:
        res_new = get_user_metrics_maintenance(vat.users)
        metrics_new = res_new["metrics_maintenance"]
        user_keys_new = res_new["user_keys"]

        newly_liquidated_with_liability = []  # (user_key, total_liability)
        for i, metrics in enumerate(metrics_new):
            user_key = user_keys_new[i]
            health = metrics["health"]
            if user_key in healthy_users and health <= 0:
                total_liability = metrics["spot_liability"] + metrics["perp_liability"]
                newly_liquidated_with_liability.append((user_key, total_liability))

        # Sort newly liquidated by total liability (largest first)
        newly_liquidated_sorted = [
            user_key
            for user_key, _ in sorted(
                newly_liquidated_with_liability, key=lambda x: x[1], reverse=True
            )
        ]
        newly_liquidated_liabilities = {
            user_key: liability
            for user_key, liability in sorted(
                newly_liquidated_with_liability, key=lambda x: x[1], reverse=True
            )
        }

        return {
            "original_maintenance_asset_weight": float(original_maint_weight) / 10000,
            "new_maintenance_asset_weight": new_maintenance_asset_weight,
            "total_users": len(user_keys_new),
            "already_liquidated_count": len(already_liquidated),
            "already_liquidated_user_keys": already_liquidated_sorted,
            "already_liquidated_liabilities": already_liquidated_liabilities,
            "newly_liquidated_count": len(newly_liquidated_sorted),
            "newly_liquidated_user_keys": newly_liquidated_sorted,
            "newly_liquidated_liabilities": newly_liquidated_liabilities,
            "total_liquidated_after_change": len(already_liquidated)
            + len(newly_liquidated_sorted),
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
