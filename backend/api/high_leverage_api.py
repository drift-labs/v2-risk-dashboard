import logging
from typing import Optional

from driftpy.addresses import (
    get_high_leverage_mode_config_public_key,
)
from driftpy.constants.numeric_constants import (
    BASE_PRECISION,
    MARGIN_PRECISION,
    PRICE_PRECISION,
    QUOTE_PRECISION,
)
from driftpy.constants.perp_markets import (
    mainnet_perp_market_configs,
)
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.margin import MarginCategory
from driftpy.pickle.vat import Vat
from driftpy.types import (
    OraclePriceData,
    UserAccount,
)
from driftpy.user_map.user_map import UserMap
from fastapi import APIRouter

from backend.state import BackendRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

PERP_DECIMALS = 9
SLOT_INACTIVITY_THRESHOLD = 9000


@router.get("/config")
async def get_high_leverage_config(request: BackendRequest):
    current_slot = getattr(request.state.backend_state, "last_oracle_slot", 0)
    drift_client: DriftClient = getattr(request.state.backend_state, "dc")
    high_leverage_mode_config_pda = get_high_leverage_mode_config_public_key(
        drift_client.program_id
    )
    config_account = await drift_client.program.account["HighLeverageModeConfig"].fetch(
        high_leverage_mode_config_pda
    )
    total_spots = config_account.max_users
    opted_in_users_count = config_account.current_users

    return {
        "slot": current_slot,
        "total_spots": total_spots,
        "opted_in_spots": opted_in_users_count,
    }


@router.get("/positions/detailed")
async def get_high_leverage_positions_detailed(request: BackendRequest):
    user_map: Optional[UserMap] = getattr(request.state.backend_state, "user_map", None)
    vat: Optional[Vat] = getattr(request.state.backend_state, "vat", None)

    if (
        not user_map
        or not hasattr(user_map, "values")
        or not vat
        or not hasattr(vat, "perp_oracles")
    ):
        logger.warning(
            "UserMap or VAT (with perp_oracles) not found/invalid. Returning empty list for /positions/detailed."
        )
        return []

    try:
        user_values = list(user_map.values())
        logger.info(
            f"Processing {len(user_values)} users from UserMap for /positions/detailed."
        )
    except Exception as e:
        logger.error(
            f"Error getting users from UserMap for /positions/detailed: {e}",
            exc_info=True,
        )
        return []

    detailed_hl_positions = []
    for user in user_values:
        if not isinstance(user, DriftUser):
            logger.warning(
                f"Skipping item in user_map values for /positions/detailed, expected DriftUser, got {type(user)}"
            )
            continue

        if user.is_high_leverage_mode() or user.is_high_leverage_maintenance_mode():
            user_account: UserAccount = user.get_user_account()
            user_public_key_str = str(user.user_public_key)
            authority_str = str(user_account.authority)

            account_leverage_raw = user.get_leverage()
            account_leverage_ui = account_leverage_raw / MARGIN_PRECISION

            for position in user_account.perp_positions:
                if position.base_asset_amount == 0:
                    continue

                market_index = position.market_index
                oracle_price_data: Optional[OraclePriceData] = vat.perp_oracles.get(
                    market_index
                )

                if oracle_price_data is None:
                    logger.warning(
                        f"Missing oracle price data for market_index {market_index} (user {user_public_key_str}). Skipping position."
                    )
                    continue

                base_asset_amount_val = position.base_asset_amount
                if base_asset_amount_val is None:
                    logger.warning(
                        f"Position base_asset_amount is None for user {user_public_key_str}, market {market_index}. Skipping position."
                    )
                    continue

                oracle_price = float(oracle_price_data.price) / PRICE_PRECISION
                base_asset_amount_ui = base_asset_amount_val / (10**PERP_DECIMALS)
                position_value_usd = abs(base_asset_amount_ui) * oracle_price

                margin_requirement_raw = (
                    user.calculate_weighted_perp_position_liability(
                        position, MarginCategory.INITIAL, 0, True
                    )
                )

                base_asset_value_raw = (
                    abs(position.base_asset_amount)
                    * oracle_price_data.price
                    // BASE_PRECISION
                )

                position_leverage_ui = 0.0
                if margin_requirement_raw != 0:
                    position_leverage_raw = (
                        base_asset_value_raw * MARGIN_PRECISION
                    ) // margin_requirement_raw
                    position_leverage_ui = position_leverage_raw / MARGIN_PRECISION
                else:
                    logger.debug(
                        f"Margin requirement raw is 0 for user {user_public_key_str}, market {market_index}. Setting position leverage to 0."
                    )

                market_symbol = "N/A"
                if market_index < len(mainnet_perp_market_configs):
                    market_symbol = mainnet_perp_market_configs[market_index].symbol
                else:
                    logger.warning(
                        f"market_index {market_index} out of range for mainnet_perp_market_configs."
                    )

                detailed_hl_positions.append(
                    {
                        "user_public_key": user_public_key_str,
                        "authority": authority_str,
                        "market_index": market_index,
                        "market_symbol": market_symbol,
                        "base_asset_amount_ui": base_asset_amount_ui,
                        "position_value_usd": position_value_usd,
                        "account_leverage": account_leverage_ui,
                        "position_leverage": position_leverage_ui,
                        "leverage_category": "high_leverage"
                        if user.is_high_leverage_mode()
                        else "high_leverage_maintenance",
                    }
                )

    return detailed_hl_positions


@router.get("/bootable-users")
async def get_bootable_user_details(request: BackendRequest):
    current_slot = getattr(request.state.backend_state, "last_oracle_slot", 0)
    user_map: Optional[UserMap] = getattr(request.state.backend_state, "user_map", None)

    if current_slot == 0:
        logger.warning(
            "Current slot is 0, cannot accurately determine bootable users by inactivity. Returning empty list."
        )
        return []

    if not user_map or not hasattr(user_map, "values"):
        logger.warning(
            "UserMap not found or invalid in backend state. Returning empty list for /bootable-users."
        )
        return []

    try:
        user_values = list(user_map.values())
        logger.info(
            f"Processing {len(user_values)} users from UserMap for /bootable-users."
        )
    except Exception as e:
        logger.error(
            f"Error getting users from UserMap for /bootable-users: {e}", exc_info=True
        )
        return []

    bootable_users_list = []
    for user in user_values:
        if not isinstance(user, DriftUser):
            logger.warning(f"Skipping item, expected DriftUser, got {type(user)}")
            continue

        user_account: UserAccount = user.get_user_account()
        if user.is_high_leverage_mode():
            last_active_slot_int = int(str(user_account.last_active_slot))
            activity_staleness_slots = current_slot - last_active_slot_int
            if activity_staleness_slots > SLOT_INACTIVITY_THRESHOLD:
                logger.debug(
                    f"User {user.user_public_key} is bootable. Staleness: {activity_staleness_slots} slots."
                )

                account_leverage_ui = user.get_leverage() / MARGIN_PRECISION
                initial_margin_req_usd = (
                    user.get_margin_requirement(MarginCategory.INITIAL)
                    / QUOTE_PRECISION
                )
                total_collateral_usd = (
                    user.get_total_collateral(MarginCategory.INITIAL) / QUOTE_PRECISION
                )
                health_percent = user.get_health()
                user_public_key_str = str(user.user_public_key)
                authority_str = str(user_account.authority)

                bootable_users_list.append(
                    {
                        "user_public_key": user_public_key_str,
                        "authority": authority_str,
                        "account_leverage": account_leverage_ui,
                        "activity_staleness_slots": activity_staleness_slots,
                        "last_active_slot": last_active_slot_int,
                        "initial_margin_requirement_usd": initial_margin_req_usd,
                        "total_collateral_usd": total_collateral_usd,
                        "health_percent": health_percent,
                    }
                )

    return bootable_users_list
