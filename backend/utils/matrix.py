import pandas as pd
import logging
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.constants import BASE_PRECISION
from driftpy.pickle.vat import Vat
from driftpy.math.margin import MarginCategory

from backend.utils.user_metrics import get_user_leverages_for_asset_liability

logger = logging.getLogger(__name__)


def calculate_effective_leverage(assets: float, liabilities: float) -> float:
    return liabilities / assets if assets != 0 else 0


def format_metric(
    value: float, should_highlight: bool, mode: int, financial: bool = False
) -> str:
    formatted = f"{value:,.2f}" if financial else f"{value:.2f}"
    return f"{formatted} ✅" if should_highlight and mode > 0 else formatted


async def get_leverage_data(vat, mode):
    """Get leverage data for the given mode."""
    try:
        result = get_user_leverages_for_asset_liability(vat.users)
        if not result:
            return None
            
        leverage_data = {
            0: result["leverages_none"],
            1: result["leverages_none"],
            2: [x for x in result["leverages_initial"] if int(x["health"]) <= 10],
            3: [x for x in result["leverages_maintenance"] if int(x["health"]) <= 10],
        }
        return leverage_data
    except Exception as e:
        logger.error(f"Error getting leverage data: {str(e)}")
        return None


async def get_matrix(vat: Vat, mode: int = 0, perp_market_index: int = 0):
    """
    Get the matrix of user positions and metrics.
    """
    try:
        # Get spot markets
        spot_markets = set()
        if hasattr(vat, "spot_markets") and hasattr(vat.spot_markets, "markets"):
            for market_index in vat.spot_markets.markets:
                spot_markets.add(market_index)
        
        # Get perp markets
        perp_markets = set()
        if hasattr(vat, "perp_markets") and hasattr(vat.perp_markets, "markets"):
            for market_index in vat.perp_markets.markets:
                perp_markets.add(market_index)
        
        # Get user keys
        user_keys = []
        if hasattr(vat, "users") and hasattr(vat.users, "users"):
            for user_key in vat.users.users:
                user_keys.append(user_key)
        
        # Get leverage data
        leverage_data = await get_leverage_data(vat, mode)
        if not leverage_data:
            return None
        
        # Get user keys based on mode
        user_keys = (
            [x["user_key"] for x in leverage_data[mode]]
            if mode in [2, 3]
            else user_keys
        )
        logger.info(f"Processing data for {len(user_keys)} users...")
        
        df = pd.DataFrame(leverage_data[mode], index=user_keys)
        
        logger.info("Initializing market columns...")
        new_columns = {}
        for market_id in spot_markets:
            prefix = f"spot_{market_id}"
            column_names = [
                f"{prefix}_all_assets",
                f"{prefix}_all",
                f"{prefix}_all_perp",
                f"{prefix}_all_spot",
                f"{prefix}_perp_{perp_market_index}_long",
                f"{prefix}_perp_{perp_market_index}_short",
            ]
            for col in column_names:
                new_columns[col] = pd.Series(0.0, index=df.index)
        
        logger.info("Calculating market metrics for each user...")
        for user_key in user_keys:
            try:
                user = vat.users.users[user_key]
                for market_id in spot_markets:
                    prefix = f"spot_{market_id}"
                    
                    # Calculate metrics for each market
                    all_assets = user.get_spot_market_asset_value(market_id, MarginCategory.MAINTENANCE)
                    all_liabilities = user.get_spot_market_liability_value(market_id, MarginCategory.MAINTENANCE)
                    all_perp = user.get_perp_market_value(market_id, MarginCategory.MAINTENANCE)
                    all_spot = user.get_spot_market_value(market_id, MarginCategory.MAINTENANCE)
                    
                    # Get perp position details
                    perp_long = 0.0
                    perp_short = 0.0
                    perp_position = user.get_perp_position(perp_market_index)
                    if perp_position:
                        base_asset_amount = perp_position.base_asset_amount / BASE_PRECISION
                        if base_asset_amount > 0:
                            perp_long = base_asset_amount
                        else:
                            perp_short = abs(base_asset_amount)
                    
                    # Update DataFrame
                    new_columns[f"{prefix}_all_assets"][user_key] = all_assets
                    new_columns[f"{prefix}_all"][user_key] = all_liabilities
                    new_columns[f"{prefix}_all_perp"][user_key] = all_perp
                    new_columns[f"{prefix}_all_spot"][user_key] = all_spot
                    new_columns[f"{prefix}_perp_{perp_market_index}_long"][user_key] = perp_long
                    new_columns[f"{prefix}_perp_{perp_market_index}_short"][user_key] = perp_short
                    
            except Exception as e:
                logger.error(f"Error processing user {user_key}: {str(e)}")
                continue
        
        # Add new columns to DataFrame
        for col_name, col_data in new_columns.items():
            df[col_name] = col_data
        
        return df
        
    except Exception as e:
        logger.error(f"Error in get_matrix: {str(e)}")
        return None
