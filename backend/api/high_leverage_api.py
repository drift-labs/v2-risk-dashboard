import asyncio
from typing import List, Optional # Added List
from fastapi import APIRouter, Depends
from pydantic import BaseModel

# backend.state for BackendRequest
from backend.state import BackendRequest

# Drift imports - adjust paths/names if necessary based on project structure
from driftpy.drift_user import DriftUser 
from driftpy.user_map.user_map import UserMap # Assuming UserMap is accessible
from driftpy.types import PerpPosition, UserAccount, is_variant, OraclePriceData # Added OraclePriceData
from driftpy.constants.numeric_constants import PRICE_PRECISION, MARGIN_PRECISION, BASE_PRECISION # Added BASE_PRECISION
from driftpy.constants.perp_markets import mainnet_perp_market_configs # Added mainnet_perp_market_configs
from driftpy.pickle.vat import Vat # Added Vat for type hinting
from driftpy.math.margin import MarginCategory # Added MarginCategory

# Add logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Decimals for perp market base asset amount
PERP_DECIMALS = 9

class HighLeverageStats(BaseModel):
    slot: int # Added slot field
    total_spots: int
    available_spots: int
    bootable_spots: int
    opted_in_spots: int

class HighLeveragePositionDetail(BaseModel):
    user_public_key: str
    authority: str
    market_index: int
    market_symbol: str
    base_asset_amount_ui: float
    position_value_usd: float
    account_leverage: float # Renamed from leverage
    position_leverage: float # Added position-specific leverage

@router.get("/stats", response_model=HighLeverageStats)
async def get_high_leverage_stats(request: BackendRequest):
    """
    Provides statistics about high leverage usage on the Drift protocol.
    - Total spots: Maximum users allowed in high leverage mode (hardcoded).
    - Opted-in spots: Users currently opted into high leverage mode.
    - Available spots: Spots remaining for users to opt-in.
    - Bootable spots: Users opted-in but not actively using high leverage (no significant perp positions).
    """
    
    total_spots = 400  # Hardcoded maximum number of spots

    opted_in_users_count = 0
    bootable_count = 0
    
    # --- Get Slot --- 
    # Use getattr for safer access, default to 0 or handle appropriately if slot is critical
    slot = getattr(request.state.backend_state, 'last_oracle_slot', 0) 
    if slot == 0:
        logger.warning("Could not retrieve last_oracle_slot from backend state.")
        # Decide how to handle missing slot: raise error, return default, etc.
        # For now, it will return 0 as the slot.

    # --- Access UserMap --- 
    user_map: Optional[UserMap] = getattr(request.state.backend_state, 'user_map', None)
    logger.info(f"UserMap object type from state: {type(user_map)}")

    if not user_map or not hasattr(user_map, 'values'):
        logger.warning("UserMap not found or invalid in backend state. Returning default stats.")
        # Return default values, including the fetched slot (which might be 0)
        return HighLeverageStats(
            slot=slot, 
            total_spots=total_spots,
            available_spots=total_spots, 
            bootable_spots=0,
            opted_in_spots=0
        )
        
    try:
        user_values = list(user_map.values()) 
        logger.info(f"Processing {len(user_values)} users from UserMap for /stats.")
        
        if not user_values:
             logger.info("UserMap is empty for /stats.")
        else:
             logger.info(f"First user object type for /stats: {type(user_values[0])}")

    except Exception as e:
        logger.error(f"Error getting users from UserMap for /stats: {e}", exc_info=True)
        return HighLeverageStats(
            slot=slot,
            total_spots=total_spots,
            available_spots=total_spots, 
            bootable_spots=0,
            opted_in_spots=0
        )

    for user in user_values:
        if not isinstance(user, DriftUser):
             logger.warning(f"Skipping item in user_map values for /stats, expected DriftUser, got {type(user)}")
             continue

        is_high_leverage = False
        try:
            is_high_leverage = user.is_high_leverage_mode()
        except Exception as e:
            logger.error(f"Error checking high leverage status for user {user.user_public_key} in /stats: {e}", exc_info=True)
            continue 

        if is_high_leverage:
            opted_in_users_count += 1
            
            has_open_perp_positions = False
            try:
                user_account: UserAccount = user.get_user_account() 
                perp_positions: list[PerpPosition] = user_account.perp_positions
                
                for position in perp_positions:
                    if position.base_asset_amount != 0:
                        has_open_perp_positions = True
                        break 
            except Exception as e:
                 logger.error(f"Error checking positions for user {user.user_public_key} in /stats: {e}", exc_info=True)
                 has_open_perp_positions = True 
            
            if not has_open_perp_positions:
                bootable_count += 1

    available_spots = total_spots - opted_in_users_count
    logger.info(f"Calculated Stats for /stats: Slot={slot}, Total={total_spots}, OptedIn={opted_in_users_count}, Available={available_spots}, Bootable={bootable_count}")

    # Include slot in the returned object
    return HighLeverageStats(
        slot=slot, 
        total_spots=total_spots,
        available_spots=available_spots,
        bootable_spots=bootable_count,
        opted_in_spots=opted_in_users_count
    )

@router.get("/positions/detailed", response_model=List[HighLeveragePositionDetail])
async def get_high_leverage_positions_detailed(request: BackendRequest):
    """
    Returns detailed information for all open perp positions held by users in high leverage mode,
    including the user's current account leverage and position-specific leverage.
    """
    detailed_hl_positions: List[HighLeveragePositionDetail] = []

    user_map: Optional[UserMap] = getattr(request.state.backend_state, 'user_map', None)
    vat: Optional[Vat] = getattr(request.state.backend_state, 'vat', None)

    logger.info(f"UserMap type for /positions/detailed: {type(user_map)}")
    logger.info(f"VAT type for /positions/detailed: {type(vat)}")

    if not user_map or not hasattr(user_map, 'values') or not vat or not hasattr(vat, 'perp_oracles'):
        logger.warning("UserMap or VAT (with perp_oracles) not found/invalid. Returning empty list for /positions/detailed.")
        return []

    try:
        user_values = list(user_map.values())
        logger.info(f"Processing {len(user_values)} users from UserMap for /positions/detailed.")
    except Exception as e:
        logger.error(f"Error getting users from UserMap for /positions/detailed: {e}", exc_info=True)
        return []

    for user in user_values:
        if not isinstance(user, DriftUser):
            logger.warning(f"Skipping item in user_map values for /positions/detailed, expected DriftUser, got {type(user)}")
            continue
        
        try:
            if user.is_high_leverage_mode():
                user_account: UserAccount = user.get_user_account()
                user_public_key_str = str(user.user_public_key)
                authority_str = str(user_account.authority)
                
                # Calculate user's account leverage
                account_leverage_raw = user.get_leverage()
                account_leverage_ui = account_leverage_raw / MARGIN_PRECISION
                logger.debug(f"User {user_public_key_str} is high leverage. Account Leverage: {account_leverage_ui:.2f}x (raw: {account_leverage_raw})")

                for position in user_account.perp_positions:
                    if position.base_asset_amount == 0:
                        continue

                    market_index = position.market_index
                    oracle_price_data: Optional[OraclePriceData] = vat.perp_oracles.get(market_index)

                    if oracle_price_data is None:
                        logger.warning(f"Missing oracle price data for market_index {market_index} (user {user_public_key_str}). Skipping position.")
                        continue
                    
                    base_asset_amount_val = position.base_asset_amount
                    if base_asset_amount_val is None: 
                        logger.warning(f"Position base_asset_amount is None for user {user_public_key_str}, market {market_index}. Skipping position.")
                        continue
                    
                    try:
                        # Calculate Base Value and Notional (UI)
                        oracle_price = float(oracle_price_data.price) / PRICE_PRECISION
                        base_asset_amount_ui = base_asset_amount_val / (10**PERP_DECIMALS)
                        position_value_usd = abs(base_asset_amount_ui) * oracle_price

                        # Calculate Position Leverage
                        margin_requirement_raw = user.calculate_weighted_perp_position_liability(
                            position, 
                            MarginCategory.INITIAL, 
                            0,  # liquidation_buffer
                            True # include_open_orders
                        )
                        
                        base_asset_value_raw = abs(position.base_asset_amount) * oracle_price_data.price // BASE_PRECISION
                        
                        position_leverage_ui = 0.0
                        if margin_requirement_raw != 0:
                            position_leverage_raw = (base_asset_value_raw * MARGIN_PRECISION) // margin_requirement_raw
                            position_leverage_ui = position_leverage_raw / MARGIN_PRECISION
                        else:
                            # Handle case with zero margin requirement (e.g., negligible position size or specific market state)
                            logger.debug(f"Margin requirement raw is 0 for user {user_public_key_str}, market {market_index}. Setting position leverage to 0.")

                        # Get Market Symbol
                        market_symbol = 'N/A'
                        if market_index < len(mainnet_perp_market_configs):
                            market_symbol = mainnet_perp_market_configs[market_index].symbol
                        else:
                            logger.warning(f"market_index {market_index} out of range for mainnet_perp_market_configs.")

                        detailed_hl_positions.append(
                            HighLeveragePositionDetail(
                                user_public_key=user_public_key_str,
                                authority=authority_str,
                                market_index=market_index,
                                market_symbol=market_symbol,
                                base_asset_amount_ui=base_asset_amount_ui,
                                position_value_usd=position_value_usd,
                                account_leverage=account_leverage_ui, # Account leverage
                                position_leverage=position_leverage_ui # Position leverage
                            )
                        )
                    except (TypeError, ValueError, AttributeError) as calc_e: # Added AttributeError for safety
                        logger.error(f"Error calculating position data or leverage for user {user_public_key_str}, market {market_index}: {calc_e}. Skipping.", exc_info=True)
                        continue
        except Exception as user_proc_e:
            logger.error(f"Error processing user {getattr(user, 'user_public_key', 'UNKNOWN')} for /positions/detailed: {user_proc_e}", exc_info=True)
            continue # Skip to next user on error
            
    logger.info(f"Returning {len(detailed_hl_positions)} high leverage positions.")
    return detailed_hl_positions