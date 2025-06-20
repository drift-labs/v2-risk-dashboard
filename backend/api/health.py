import heapq
import logging

import pandas as pd
from driftpy.constants import BASE_PRECISION, PRICE_PRECISION, SPOT_BALANCE_PRECISION
from driftpy.pickle.vat import Vat
from driftpy.types import is_variant
from fastapi import APIRouter

from backend.state import BackendRequest

router = APIRouter()

logger = logging.getLogger(__name__)


def to_financial(num: float):
    """
    Helper function to format a number to a financial format.
    """
    num_str = str(num)
    if "e" in num_str.lower():
        return round(num, 2)

    decimal_pos = num_str.find(".")
    if decimal_pos != -1:
        return float(num_str[: decimal_pos + 3])
    return num


@router.get("/health_distribution")
def get_account_health_distribution(request: BackendRequest):
    """
    Get the distribution of account health across different ranges.

    This endpoint calculates the distribution of account health for all users,
    categorizing them into health ranges and summing up the total collateral
    in each range.

    Returns:
        list[dict]: A list of dictionaries containing the health distribution data.
        Each dictionary has the following keys:
        - Health Range (str): The health percentage range (e.g., '0-10%')
        - Counts (int): The number of accounts in this range
        - Notional Values (float): The total collateral value in this range
    """
    vat: Vat = request.state.backend_state.vat
    health_notional_distributions = {
        "0-10%": 0.0,
        "10-20%": 0.0,
        "20-30%": 0.0,
        "30-40%": 0.0,
        "40-50%": 0.0,
        "50-60%": 0.0,
        "60-70%": 0.0,
        "70-80%": 0.0,
        "80-90%": 0.0,
        "90-100%": 0.0,
    }
    health_counts = {
        "0-10%": 0.0,
        "10-20%": 0.0,
        "20-30%": 0.0,
        "30-40%": 0.0,
        "40-50%": 0.0,
        "50-60%": 0.0,
        "60-70%": 0.0,
        "70-80%": 0.0,
        "80-90%": 0.0,
        "90-100%": 0.0,
    }

    for user in vat.users.values():
        try:
            total_collateral = user.get_total_collateral() / PRICE_PRECISION
            current_health = user.get_health()
        except Exception as e:
            print(f"==> Error from health [{user.user_public_key}] ", e)
            continue
        match current_health:
            case _ if current_health < 10:
                health_notional_distributions["0-10%"] += total_collateral
                health_counts["0-10%"] += 1
            case _ if current_health < 20:
                health_notional_distributions["10-20%"] += total_collateral
                health_counts["10-20%"] += 1
            case _ if current_health < 30:
                health_notional_distributions["20-30%"] += total_collateral
                health_counts["20-30%"] += 1
            case _ if current_health < 40:
                health_notional_distributions["30-40%"] += total_collateral
                health_counts["30-40%"] += 1
            case _ if current_health < 50:
                health_notional_distributions["40-50%"] += total_collateral
                health_counts["40-50%"] += 1
            case _ if current_health < 60:
                health_notional_distributions["50-60%"] += total_collateral
                health_counts["50-60%"] += 1
            case _ if current_health < 70:
                health_notional_distributions["60-70%"] += total_collateral
                health_counts["60-70%"] += 1
            case _ if current_health < 80:
                health_notional_distributions["70-80%"] += total_collateral
                health_counts["70-80%"] += 1
            case _ if current_health < 90:
                health_notional_distributions["80-90%"] += total_collateral
                health_counts["80-90%"] += 1
            case _:
                health_notional_distributions["90-100%"] += total_collateral
                health_counts["90-100%"] += 1
    df = pd.DataFrame(
        {
            "Health Range": list(health_counts.keys()),
            "Counts": list(health_counts.values()),
            "Notional Values": list(health_notional_distributions.values()),
        }
    )

    return df.to_dict(orient="records")


@router.get("/largest_perp_positions")
def get_largest_perp_positions(request: BackendRequest, market_index: int = None):
    """
    Get the largest perp positions by notional value across all users or for a specific market if market_index is provided.
    """
    vat: Vat = request.state.backend_state.vat
    logger.info(
        f"==> [largest_perp_positions] Called with "
        f"market_index={market_index}, current_pickle={request.state.backend_state.current_pickle_path}"
    )
    try:
        user_count = sum(1 for _ in vat.users.values())
        logger.info(f"==> [largest_perp_positions] # of loaded users: {user_count}")
    except Exception as e:
        logger.info(f"==> [largest_perp_positions] Unable to count users: {str(e)}")

    # Collect all positions first
    all_positions = []
    total_positions_checked = 0
    positions_meeting_criteria = 0

    # Log the markets we have
    logger.info(f"==> [largest_perp_positions] Available perp oracles: {list(vat.perp_oracles.keys())}")

    for user in vat.users.values():
        for position in user.get_user_account().perp_positions:
            total_positions_checked += 1
            # Skip if filtering by market_index
            if market_index is not None and position.market_index != market_index:
                continue

            # Process all non-zero positions (both long and short)
            if position.base_asset_amount != 0:
                market_price = vat.perp_oracles.get(position.market_index)
                if market_price is not None:
                    positions_meeting_criteria += 1
                    market_price_ui = market_price.price / PRICE_PRECISION
                    base_asset_value = (
                        abs(position.base_asset_amount) / BASE_PRECISION
                    ) * market_price_ui
                    
                    # Store position info with actual value
                    all_positions.append((
                        base_asset_value,  # Actual value for sorting
                        user.user_public_key,
                        position.market_index,
                        position.base_asset_amount / BASE_PRECISION,  # Keep original sign for display
                    ))

    # Sort all positions by value (descending)
    positions = sorted(all_positions, key=lambda x: x[0], reverse=True)

    # Add summary logging
    total_value_returned = sum(pos[0] for pos in positions)
    logger.info(
        f"==> [largest_perp_positions] Stats => total_checked={total_positions_checked}, "
        f"positions_meeting_criteria={positions_meeting_criteria}, positions_returned={len(positions)}, "
        f"total_notional_value=${total_value_returned:,.2f}"
    )

    data = {
        "Market Index": [pos[2] for pos in positions],
        "Value": [f"${pos[0]:,.2f}" for pos in positions],
        "Base Asset Amount": [f"{pos[3]:,.2f}" for pos in positions],
        "Public Key": [pos[1] for pos in positions],
    }

    return data


@router.get("/most_levered_perp_positions_above_1m")
def get_most_levered_perp_positions_above_1m(request: BackendRequest, market_index: int = None):
    """
    Get the most leveraged perpetual positions with value above $1 million.

    This endpoint calculates the leverage of each perpetual position with a value
    over $1 million and returns the most leveraged positions.
    Results can be filtered by market_index if provided.

    Args:
        request: The backend request object
        market_index: Optional market index to filter by

    Returns:
        dict: A dictionary containing lists of data for the top leveraged positions:
        - Market Index (list[int]): The market indices of the top positions
        - Value (list[str]): The formatted dollar values of the positions
        - Base Asset Amount (list[str]): The formatted base asset amounts
        - Leverage (list[str]): The formatted leverage ratios
        - Public Key (list[str]): The public keys of the position holders
    """
    vat: Vat = request.state.backend_state.vat
    all_positions: list[tuple[float, str, int, float, float]] = []

    for user in vat.users.values():
        try:
            total_collateral = user.get_total_collateral() / PRICE_PRECISION
        except Exception as e:
            print(
                f"==> Error from get_most_levered_perp_positions_above_1m [{user.user_public_key}] ",
                e,
            )
            continue
        if total_collateral > 0:
            for position in user.get_user_account().perp_positions:
                # Skip if filtering by market_index
                if market_index is not None and position.market_index != market_index:
                    continue
                    
                if position.base_asset_amount > 0:
                    market_price = vat.perp_oracles.get(position.market_index)
                    if market_price is not None:
                        market_price_ui = market_price.price / PRICE_PRECISION
                        base_asset_value = (
                            abs(position.base_asset_amount) / BASE_PRECISION
                        ) * market_price_ui
                        leverage = base_asset_value / total_collateral
                        if base_asset_value > 1_000_000:
                            item = (
                                to_financial(base_asset_value),
                                user.user_public_key,
                                position.market_index,
                                position.base_asset_amount / BASE_PRECISION,
                                leverage,
                            )
                            all_positions.append(item)

    positions = sorted(
        all_positions,
        key=lambda x: x[4],
        reverse=True
    )

    data = {
        "Market Index": [pos[2] for pos in positions],
        "Value": [f"${pos[0]:,.2f}" for pos in positions],
        "Base Asset Amount": [f"{pos[3]:,.2f}" for pos in positions],
        "Leverage": [f"{pos[4]:,.2f}" for pos in positions],
        "Public Key": [pos[1] for pos in positions],
    }

    return data


@router.get("/largest_spot_borrows")
def get_largest_spot_borrows(request: BackendRequest, market_index: int = None):
    """
    Get the largest spot borrowing positions by value.

    This endpoint retrieves the largest spot borrowing positions across all users,
    calculated based on the current market prices. Results can be filtered by 
    market_index if provided.

    Args:
        request: The backend request object
        market_index: Optional market index to filter by

    Returns:
        dict: A dictionary containing lists of data for the top borrowing positions:
        - Market Index (list[int]): The market indices of the top borrows
        - Value (list[str]): The formatted dollar values of the borrows
        - Scaled Balance (list[str]): The formatted scaled balances of the borrows
        - Public Key (list[str]): The public keys of the borrowers
    """
    vat: Vat = request.state.backend_state.vat
    all_borrows: list[tuple[float, str, int, float]] = []

    for user in vat.users.values():
        for position in user.get_user_account().spot_positions:
            # Skip if filtering by market_index
            if market_index is not None and position.market_index != market_index:
                continue
                
            if position.scaled_balance > 0 and is_variant(
                position.balance_type, "Borrow"
            ):
                market_price = vat.spot_oracles.get(position.market_index)
                if market_price is not None:
                    market_price_ui = market_price.price / PRICE_PRECISION
                    borrow_value = (
                        position.scaled_balance / SPOT_BALANCE_PRECISION
                    ) * market_price_ui
                    item = (
                        to_financial(borrow_value),
                        user.user_public_key,
                        position.market_index,
                        position.scaled_balance / SPOT_BALANCE_PRECISION,
                    )
                    all_borrows.append(item)

    borrows = sorted(all_borrows, key=lambda x: x[0], reverse=True)

    data = {
        "Market Index": [pos[2] for pos in borrows],
        "Value": [f"${pos[0]:,.2f}" for pos in borrows],
        "Scaled Balance": [f"{pos[3]:,.2f}" for pos in borrows],
        "Public Key": [pos[1] for pos in borrows],
    }

    return data


@router.get("/most_levered_spot_borrows_above_1m")
def get_most_levered_spot_borrows_above_1m(request: BackendRequest, market_index: int = None):
    """
    Get the most leveraged spot borrowing positions with value above $750,000.

    This endpoint calculates the leverage of each spot borrowing position with a value
    over $750,000 and returns the most leveraged positions.
    Results can be filtered by market_index if provided.

    Args:
        request: The backend request object
        market_index: Optional market index to filter by

    Returns:
        dict: A dictionary containing lists of data for the leveraged borrowing positions:
        - Market Index (list[int]): The market indices of the top borrows
        - Value (list[str]): The formatted dollar values of the borrows
        - Scaled Balance (list[str]): The formatted scaled balances of the borrows
        - Leverage (list[str]): The formatted leverage ratios
        - Public Key (list[str]): The public keys of the borrowers
        - Error (list[str]): Error details if any (empty string if no error)
    """
    vat: Vat = request.state.backend_state.vat
    all_borrows: list[tuple[float, str, int, float, float, str]] = []  # Added error field
    error_positions = []  # Track positions with errors for logging

    for user in vat.users.values():
        user_collateral = 0
        collateral_error = ""
        
        try:
            user_collateral = user.get_total_collateral() / PRICE_PRECISION
        except Exception as e:
            collateral_error = f"Collateral error: {str(e)}"
            logger.warning(
                f"Error calculating collateral for user [{user.user_public_key}]: {str(e)}"
            )
            
        for position in user.get_user_account().spot_positions:
            # Skip if filtering by market_index
            if market_index is not None and position.market_index != market_index:
                continue
                
            if is_variant(position.balance_type, "Borrow") and position.scaled_balance > 0:
                position_error = collateral_error  # Start with any collateral error
                market_price = vat.spot_oracles.get(position.market_index)
                
                if market_price is None:
                    oracle_error = f"Oracle for market {position.market_index} not found"
                    position_error = oracle_error if not position_error else f"{position_error}; {oracle_error}"
                    logger.warning(f"{oracle_error} for user [{user.user_public_key}]")
                    
                    # Add position with error
                    borrow_value = 0  # Default value when price is unknown
                    scaled_balance = position.scaled_balance / SPOT_BALANCE_PRECISION
                    leverage = 0  # Default leverage when calculation is impossible
                    
                    error_positions.append({
                        "market_index": position.market_index,
                        "public_key": user.user_public_key,
                        "scaled_balance": scaled_balance,
                        "error": position_error
                    })
                    
                    # Add this one with error
                    item = (
                        borrow_value,  # Will be sorted last due to 0 value
                        user.user_public_key,
                        position.market_index,
                        scaled_balance,
                        leverage,
                        position_error,
                    )
                    all_borrows.append(item)
                else:
                    try:
                        market_price_ui = market_price.price / PRICE_PRECISION
                        borrow_value = (
                            position.scaled_balance / SPOT_BALANCE_PRECISION
                        ) * market_price_ui
                        
                        if user_collateral > 0:
                            leverage = borrow_value / user_collateral
                        else:
                            leverage = float('inf')  # Infinite leverage when collateral is 0
                            if not position_error:
                                position_error = "Zero collateral"
                        
                        if borrow_value > 750_000:
                            item = (
                                to_financial(borrow_value),
                                user.user_public_key,
                                position.market_index,
                                position.scaled_balance / SPOT_BALANCE_PRECISION,
                                leverage,
                                position_error,  # Empty string if no error
                            )
                            all_borrows.append(item)
                    except Exception as e:
                        calc_error = f"Calculation error: {str(e)}"
                        position_error = calc_error if not position_error else f"{position_error}; {calc_error}"
                        logger.warning(
                            f"Error processing borrow position for user [{user.user_public_key}] market [{position.market_index}]: {str(e)}"
                        )
                        
                        # Add error position
                        error_positions.append({
                            "market_index": position.market_index,
                            "public_key": user.user_public_key,
                            "scaled_balance": position.scaled_balance / SPOT_BALANCE_PRECISION,
                            "error": position_error
                        })

    # Log all error positions for debugging
    if error_positions:
        logger.warning(f"Found {len(error_positions)} positions with errors: {error_positions}")

    positions = sorted(
        all_borrows,
        key=lambda x: x[4] if not x[5] else float('-inf'),  # Sort error positions first
        reverse=True,
    )

    data = {
        "Market Index": [pos[2] for pos in positions],
        "Value": [f"${pos[0]:,.2f}" if not pos[5] else "N/A" for pos in positions],
        "Scaled Balance": [f"{pos[3]:,.2f}" for pos in positions],
        "Leverage": [f"{pos[4]:,.2f}" if not pos[5] and pos[4] != float('inf') else "∞" if pos[4] == float('inf') else "N/A" for pos in positions],
        "Public Key": [pos[1] for pos in positions],
        "Error": [pos[5] for pos in positions],
    }

    return data


@router.get("/largest_spot_borrow_per_market")
def get_largest_spot_borrow_per_market(request: BackendRequest):
    """
    Get the largest spot borrowing position for each market index.

    This endpoint retrieves the single largest spot borrowing position for each market,
    calculated based on the current market prices.

    Returns:
        dict: A dictionary containing lists of data for the largest borrow per market:
        - Market Index (list[int]): The market indices
        - Value (list[str]): The formatted dollar values of the borrows
        - Scaled Balance (list[str]): The formatted scaled balances of the borrows
        - Public Key (list[str]): The public keys of the borrowers
    """
    vat: Vat = request.state.backend_state.vat
    market_largest_borrows: dict[int, tuple[float, str, int, float]] = {}

    for user in vat.users.values():
        for position in user.get_user_account().spot_positions:
            if position.scaled_balance > 0 and is_variant(
                position.balance_type, "Borrow"
            ):
                market_price = vat.spot_oracles.get(position.market_index)
                if market_price is not None:
                    market_price_ui = market_price.price / PRICE_PRECISION
                    borrow_value = (
                        position.scaled_balance / SPOT_BALANCE_PRECISION
                    ) * market_price_ui
                    borrow_item = (
                        to_financial(borrow_value),
                        user.user_public_key,
                        position.market_index,
                        position.scaled_balance / SPOT_BALANCE_PRECISION,
                    )

                    # Update if this is the largest borrow for this market
                    if (position.market_index not in market_largest_borrows or
                        borrow_value > market_largest_borrows[position.market_index][0]):
                        market_largest_borrows[position.market_index] = borrow_item

    borrows = sorted(
        market_largest_borrows.values(),
        key=lambda x: x[2]
    )

    data = {
        "Market Index": [pos[2] for pos in borrows],
        "Value": [f"${pos[0]:,.2f}" for pos in borrows],
        "Scaled Balance": [f"{pos[3]:,.2f}" for pos in borrows],
        "Public Key": [pos[1] for pos in borrows],
    }

    return data