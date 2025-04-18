from driftpy.constants import BASE_PRECISION, PRICE_PRECISION, QUOTE_SPOT_MARKET_INDEX, SPOT_BALANCE_PRECISION
from driftpy.pickle.vat import Vat
from driftpy.types import is_variant
from fastapi import APIRouter
import logging
from collections import defaultdict

from backend.state import BackendRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def format_number(number: float, decimals: int = 4, use_commas: bool = True) -> str:
    """Format a number with proper decimal places and optional comma separators"""
    if abs(number) >= 1e6:
        # For large numbers, use millions format
        return f"{number/1e6:,.{decimals}f}M"
    elif abs(number) >= 1e3 and use_commas:
        return f"{number:,.{decimals}f}"
    else:
        return f"{number:.{decimals}f}"


async def _get_aggregated_positions(vat: Vat) -> dict:
    """
    Get aggregated positions for all users in the system.
    
    Args:
        vat: The Vat instance containing user and market data
        
    Returns:
        dict: Dictionary containing aggregated position data including:
            - total_unique_authorities: Number of unique user authorities
            - total_sub_accounts: Total number of sub-accounts
            - total_net_value: Total net value across all accounts
            - perp_markets: Dictionary of perpetual market aggregates
            - spot_markets: Dictionary of spot market aggregates
            - errors: List of any errors encountered during processing
    """
    # Initialize aggregation containers
    perp_aggregates = defaultdict(lambda: {
        "market_name": "",
        "total_long_usd": 0.0,
        "total_short_usd": 0.0,
        "total_lp_shares": 0,
        "current_price": None,  # Changed to None as default
        "unique_users": set(),
        "errors": []  # Track errors per market
    })
    
    spot_aggregates = defaultdict(lambda: {
        "market_name": "",
        "total_deposits_native": 0.0,
        "total_borrows_native": 0.0,
        "total_deposits_usd": 0.0,
        "total_borrows_usd": 0.0,
        "token_price": None,  # Changed to None as default
        "decimals": 0,
        "unique_users": set(),
        "errors": []  # Track errors per market
    })
    
    total_net_value = 0.0
    total_sub_accounts = 0
    unique_authorities = set()
    global_errors = []  # Track global errors

    # Log the available oracles for debugging
    logger.info(f"Available perp oracles: {list(vat.perp_oracles.keys())}")
    logger.info(f"Available spot oracles: {list(vat.spot_oracles.keys())}")

    user_count = sum(1 for _ in vat.users.values())
    logger.info(f"Processing {user_count} users")

    # Process all users
    for user in vat.users.values():
        try:
            user_account = user.get_user_account()
            authority = str(user_account.authority)
            unique_authorities.add(authority)
            total_sub_accounts += 1
            
            try:
                total_net_value += user.get_net_usd_value() / 1e6
            except Exception as e:
                logger.warning(f"Error calculating net value for user {authority}: {str(e)}")
            
            # Process perpetual positions
            perp_positions = user.get_active_perp_positions()
            for position in perp_positions:
                try:
                    market = user.drift_client.get_perp_market_account(position.market_index)
                    market_name = bytes(market.name).decode('utf-8').strip('\x00')
                    
                    agg = perp_aggregates[position.market_index]
                    agg["market_name"] = market_name
                    
                    oracle_price_data = vat.perp_oracles.get(position.market_index)
                    if oracle_price_data is None:
                        error_msg = f"Oracle not found for perp market {position.market_index}"
                        if error_msg not in agg["errors"]:
                            agg["errors"].append(error_msg)
                            logger.warning(error_msg)
                        continue
                    
                    agg["current_price"] = oracle_price_data.price / PRICE_PRECISION
                    
                    position_value = abs(user.get_perp_position_value(
                        position.market_index,
                        oracle_price_data,
                        include_open_orders=True
                    ) / PRICE_PRECISION)
                    
                    base_asset_amount = position.base_asset_amount / BASE_PRECISION
                    if base_asset_amount > 0:
                        agg["total_long_usd"] += position_value
                    else:
                        agg["total_short_usd"] += position_value
                    
                    agg["total_lp_shares"] += position.lp_shares / BASE_PRECISION
                    agg["unique_users"].add(authority)
                except Exception as e:
                    error_msg = f"Error processing perp position for market {position.market_index}: {str(e)}"
                    logger.warning(error_msg)
                    if error_msg not in perp_aggregates[position.market_index]["errors"]:
                        perp_aggregates[position.market_index]["errors"].append(error_msg)
            
            # Process spot positions
            spot_positions = user.get_active_spot_positions()
            for position in spot_positions:
                try:
                    market = user.drift_client.get_spot_market_account(position.market_index)
                    market_name = bytes(market.name).decode('utf-8').strip('\x00')
                    
                    agg = spot_aggregates[position.market_index]
                    agg["market_name"] = market_name
                    agg["decimals"] = market.decimals
                    
                    token_amount = user.get_token_amount(position.market_index)
                    formatted_amount = token_amount / (10 ** market.decimals)
                    
                    if position.market_index == QUOTE_SPOT_MARKET_INDEX:
                        token_price = 1.0
                        token_value = abs(formatted_amount)
                    else:
                        oracle_price_data = vat.spot_oracles.get(position.market_index)
                        if oracle_price_data is None:
                            error_msg = f"Oracle not found for spot market {position.market_index}"
                            if error_msg not in agg["errors"]:
                                agg["errors"].append(error_msg)
                                logger.warning(error_msg)
                            continue
                            
                        token_price = oracle_price_data.price / PRICE_PRECISION
                        if token_amount < 0:
                            token_value = abs(user.get_spot_market_liability_value(
                                market_index=position.market_index,
                                include_open_orders=True
                            ) / PRICE_PRECISION)
                        else:
                            token_value = abs(user.get_spot_market_asset_value(
                                market_index=position.market_index,
                                include_open_orders=True
                            ) / PRICE_PRECISION)
                    
                    agg["token_price"] = token_price
                    
                    if token_amount > 0:
                        agg["total_deposits_native"] += formatted_amount
                        agg["total_deposits_usd"] += token_value
                    else:
                        agg["total_borrows_native"] += abs(formatted_amount)
                        agg["total_borrows_usd"] += token_value
                    
                    agg["unique_users"].add(authority)
                except Exception as e:
                    error_msg = f"Error processing spot position for market {position.market_index}: {str(e)}"
                    logger.warning(error_msg)
                    if error_msg not in spot_aggregates[position.market_index]["errors"]:
                        spot_aggregates[position.market_index]["errors"].append(error_msg)
                
        except Exception as e:
            error_msg = f"Error processing user {authority}: {str(e)}"
            logger.warning(error_msg)
            global_errors.append(error_msg)
            continue

    # Convert sets to counts for JSON serialization
    for agg in perp_aggregates.values():
        agg["unique_users"] = len(agg["unique_users"])
    for agg in spot_aggregates.values():
        agg["unique_users"] = len(agg["unique_users"])

    # Sort markets by index
    sorted_perp_markets = dict(sorted(perp_aggregates.items(), key=lambda x: int(x[0])))
    sorted_spot_markets = dict(sorted(spot_aggregates.items(), key=lambda x: int(x[0])))

    return {
        "total_unique_authorities": len(unique_authorities),
        "total_sub_accounts": total_sub_accounts,
        "total_net_value": total_net_value,
        "perp_markets": sorted_perp_markets,
        "spot_markets": sorted_spot_markets,
        "global_errors": global_errors
    }


@router.get("/aggregated")
async def get_aggregated_positions(request: BackendRequest):
    """
    Get aggregated positions across all users in the Drift Protocol.
    
    This endpoint calculates and returns aggregated position data including:
    - Total unique authorities and sub-accounts
    - Total net value across all accounts
    - Per-market aggregates for both perpetual and spot markets
    - Position values, user counts, and market details
    - Error tracking for missing oracles or calculation issues
    
    Returns:
        dict: Aggregated position data across all markets and users
    """
    return await _get_aggregated_positions(request.state.backend_state.vat) 