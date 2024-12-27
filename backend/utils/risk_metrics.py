from typing import Dict, Tuple
from driftpy.drift_user import DriftUser
from driftpy.math.margin import MarginCategory
from driftpy.constants.numeric_constants import MARGIN_PRECISION, QUOTE_PRECISION
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.constants.perp_markets import mainnet_perp_market_configs

def check_liquidity(user: DriftUser, price_impact: float, market_index: int) -> bool:
    """
    Check if the liquidity criteria passes for a given market.
    
    Args:
        user: DriftUser instance
        price_impact: Price impact percentage from Jupiter
        market_index: Spot market index
    
    Returns:
        bool: True if liquidity check passes
    """
    market_config = mainnet_spot_market_configs[market_index]
    maint_asset_weight = market_config.maintenance_asset_weight / MARGIN_PRECISION
    return price_impact < (1 - maint_asset_weight)

def check_spot_leverage(user: DriftUser, market_index: int) -> bool:
    """
    Check if spot position's effective leverage is within limits.
    
    Args:
        user: DriftUser instance
        market_index: Spot market index
    
    Returns:
        bool: True if spot leverage check passes
    """
    market_config = mainnet_spot_market_configs[market_index]
    maint_asset_weight = market_config.maintenance_asset_weight / MARGIN_PRECISION
    
    # Calculate effective leverage for spot positions
    asset_value = user.get_spot_market_asset_value(market_index, MarginCategory.MAINTENANCE)
    liability_value = user.get_spot_market_liability_value(market_index, MarginCategory.MAINTENANCE)
    
    if asset_value == 0:
        return True  # No position means no leverage
        
    effective_leverage = liability_value / asset_value
    return effective_leverage < (0.5 * maint_asset_weight)

def check_perp_leverage(user: DriftUser, market_index: int) -> bool:
    """
    Check if perp position's effective leverage is within limits.
    
    Args:
        user: DriftUser instance
        market_index: Perp market index
    
    Returns:
        bool: True if perp leverage check passes
    """
    # Calculate effective leverage for perp positions
    perp_value = abs(user.get_perp_market_value(market_index))
    total_collateral = user.get_total_collateral(MarginCategory.MAINTENANCE)
    
    if total_collateral == 0:
        return False  # No collateral means infinite leverage
        
    effective_leverage = perp_value / total_collateral
    return 1 <= effective_leverage <= 2

def check_insurance_coverage(user: DriftUser, market_index: int, insurance_fund_balance: int) -> bool:
    """
    Check if high leverage positions are covered by insurance fund.
    
    Args:
        user: DriftUser instance
        market_index: Perp market index
        insurance_fund_balance: Current insurance fund balance for the market
    
    Returns:
        bool: True if insurance coverage is sufficient
    """
    leverage = user.get_leverage() / MARGIN_PRECISION
    if leverage <= 2:
        return True
        
    position_notional = abs(user.get_perp_market_value(market_index))
    return position_notional <= insurance_fund_balance

def calculate_target_scale_iaw(
    user: DriftUser,
    market_index: int,
    price_impact: float,
    insurance_fund_balance: int
) -> Tuple[float, Dict[str, bool]]:
    """
    Calculate Target Scale IAW based on all safety criteria.
    
    Args:
        user: DriftUser instance
        market_index: Market index
        price_impact: Price impact percentage from Jupiter
        insurance_fund_balance: Current insurance fund balance
    
    Returns:
        Tuple[float, Dict[str, bool]]: Target Scale IAW value and dict of criteria results
    """
    criteria_results = {
        "liquidity_check": check_liquidity(user, price_impact, market_index),
        "spot_leverage": check_spot_leverage(user, market_index),
        "perp_leverage": check_perp_leverage(user, market_index),
        "insurance_coverage": check_insurance_coverage(user, market_index, insurance_fund_balance)
    }
    
    # If all criteria pass, set to 1.2x total deposits notional
    if all(criteria_results.values()):
        total_deposits = user.get_spot_market_asset_value(None, MarginCategory.INITIAL) / QUOTE_PRECISION
        return (1.2 * total_deposits, criteria_results)
    
    return (0.0, criteria_results) 