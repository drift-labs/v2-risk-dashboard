# This is the new version of the gecko market recommender which makes use of the /backend/utils/coingecko_api.py utility script.

"""
Follow this high level structure when writing the new script. 
Only focus on one section at a time. 
I will instruct you which section we are working on.
If I do not instruct you, ask me which section you are working on.
This script will use synchronous code instead of asynchronous code in the previous version.


def fetch_market_data():
    # Minimal working data retrieval with ccxt
    pass

def score_assets(market_data):
    # Single simplified scoring method for all assets
    pass

def categorize_assets(scored_assets):
    # Clearly separate lists: to_list, to_delist, leverage_up, leverage_down
    pass

def main():
    data = fetch_market_data()
    scored = score_assets(data)
    categorized = categorize_assets(scored)
    # output simple DataFrames for Streamlit
    return categorized

"""

import logging
from typing import Dict, List
from fastapi import APIRouter
from backend.utils.coingecko_api import fetch_all_coingecko_market_data, fetch_all_coingecko_volumes
from driftpy.pickle.vat import Vat
from backend.state import BackendRequest

# Constants
NUMBER_OF_TOKENS = 1
PRICE_PRECISION = 1e6  # Add price precision constant
MARKET_BASE_DECIMALS = {
    0: 9,  # Default to 9 decimals if not specified
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter()

@router.get("/market-data")
async def get_market_data(request: BackendRequest):
    """
    Get comprehensive market data including CoinGecko data, Drift metrics, and listing recommendations.
    
    Returns:
        List[Dict]: List of dictionaries containing market data, scoring, and recommendations for each coin
    """
    return main(request.state.backend_state.vat)

def fetch_coingecko_market_data() -> List[Dict]:
    """
    Fetches market data for top 250 coins from CoinGecko API.
    Returns data in a standardized format for the dashboard.
    
    Returns:
        List of dictionaries containing market data for each coin
    """
    logger.info("Fetching market data for top 250 coins from CoinGecko...")
    
    try:
        # Fetch market data
        market_data = fetch_all_coingecko_market_data(NUMBER_OF_TOKENS)
        
        if not market_data:
            logger.error("Failed to fetch market data from CoinGecko")
            return []
        
        # Get list of coin IDs for volume fetching
        coin_ids = list(market_data.keys())
        
        # Fetch 30-day volume data
        volume_data = fetch_all_coingecko_volumes(coin_ids)
        
        # Format response data
        formatted_data = []
        for coin_id, data in market_data.items():
            try:
                formatted_entry = {
                    "symbol": data.get('symbol', '').upper(),
                    "coingecko_data": {
                        "coingecko_name": data.get('name'),
                        "coingecko_id": coin_id,
                        "coingecko_image_url": data.get('image_url'),
                        "coingecko_current_price": data.get('current_price'),
                        "coingecko_market_cap_rank": data.get('market_cap_rank'),
                        "coingecko_market_cap": data.get('market_cap'),
                        "coingecko_fully_diluted_valuation": data.get('fully_diluted_valuation'),
                        "coingecko_total_volume_24h": data.get('total_volume_24h'),
                        "coingecko_mc_derived": data.get('market_cap'),  # Same as market_cap for now
                        "coingecko_circulating": data.get('circulating_supply'),
                        "coingecko_total_supply": data.get('total_supply'),
                        "coingecko_max_supply": data.get('max_supply'),
                        "coingecko_ath_price": data.get('ath_price'),
                        "coingecko_30d_volume": volume_data.get(coin_id, 0)
                    }
                }
                formatted_data.append(formatted_entry)
                logger.info(f"Processed market data for {formatted_entry['symbol']}")
            except Exception as e:
                logger.error(f"Error formatting data for coin {coin_id}: {e}")
                continue
        
        logger.info(f"Successfully processed market data for {len(formatted_data)} coins")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error in fetch_coingecko_market_data: {e}", exc_info=True)
        return []
    
def fetch_driftpy_data(vat: Vat) -> Dict:
    """
    Fetches market data from DriftPy to check for perp market listings.
    
    Args:
        vat: Vat instance containing market data
    
    Returns:
        Dict: Dictionary containing Drift perp market listing status for each coin
    """
    logger.info("Fetching DriftPy data to check perp market listings...")
    
    try:
        # Track long and short positions separately for each market
        market_long_positions = {}   # Track sum of long positions
        market_short_positions = {}  # Track sum of short positions
        market_position_counts = {}  # Track number of positions
        
        # Process each user's positions
        processed_users = 0
        logger.info("Processing positions from users via Vat...")
        
        # Iterate through all users to aggregate positions
        for user in vat.users.values():
            try:
                processed_users += 1
                if processed_users % 5000 == 0:
                    logger.info(f"Processed {processed_users} users...")

                # Get perp positions directly from user account
                for position in user.get_user_account().perp_positions:
                    if position.base_asset_amount != 0:
                        market_idx = position.market_index
                        base_amount = position.base_asset_amount

                        # Initialize market tracking if first time seeing this market
                        if market_idx not in market_long_positions:
                            market_long_positions[market_idx] = 0
                            market_short_positions[market_idx] = 0
                            market_position_counts[market_idx] = 0

                        # Add to appropriate direction (long or short)
                        if base_amount > 0:  # Long position
                            market_long_positions[market_idx] += base_amount
                        else:  # Short position
                            market_short_positions[market_idx] += abs(base_amount)

                        market_position_counts[market_idx] += 1

            except Exception as e:
                logger.debug(f"Error processing user positions: {str(e)}")
                continue

        logger.info(f"Found positions in {len(market_long_positions)} markets")
        
        # Get all perp markets from vat
        perp_markets = {}
        # Use values() instead of items() since MarketMap is an object with specific methods
        for market in vat.perp_markets.values():
            try:
                # Get market name and clean it
                market_name = bytes(market.data.name).decode('utf-8').strip('\x00')
                # Extract symbol from market name (e.g., "BTC-PERP" -> "BTC")
                symbol = market_name.split('-')[0].upper()
                
                # Get oracle price from perp_oracles
                market_price = vat.perp_oracles.get(market.data.market_index)
                if market_price is None:
                    logger.warning(f"Oracle price not found for market {market.data.market_index}")
                    continue
                
                # Get oracle price from the OraclePriceData object and apply precision
                oracle_price = float(market_price.price) / PRICE_PRECISION
                
                # Calculate OI
                market_index = market.data.market_index
                long_amount = market_long_positions.get(market_index, 0)
                short_amount = market_short_positions.get(market_index, 0)
                base_oi = max(long_amount, short_amount)
                
                # Get base decimals
                base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)
                
                # Calculate OI in USD
                base_oi_readable = base_oi / (10 ** base_decimals)
                oi_usd = base_oi_readable * oracle_price
                
                # Include all required drift data fields with default values
                perp_markets[symbol] = {
                    "drift_is_listed_perp": "true",
                    "drift_perp_market": market_name,
                    "drift_is_listed_spot": "false",  # Default value
                    "drift_spot_market": None,
                    "drift_oracle_price": oracle_price,
                    "drift_volume_30d": 0.0,  # Default value, should be calculated if available
                    "drift_max_leverage": float(market.data.margin_ratio_initial),  # This might need conversion
                    "drift_open_interest": oi_usd,
                    "drift_funding_rate_1h": float(market.data.amm.last_funding_rate) / 1e6 * 100  # Convert to hourly percentage
                }
            except Exception as e:
                logger.error(f"Error processing perp market: {e}")
                continue
        
        logger.info(f"Found {len(perp_markets)} perp markets on Drift")
        return perp_markets
        
    except Exception as e:
        logger.error(f"Error in fetch_driftpy_data: {e}")
        return {}

def fetch_drift_data_api_data() -> Dict:
    """
    Fetches market data from Drift API (placeholder implementation).
    
    Returns:
        Dict: Dictionary containing Drift API data for each coin
    """
    logger.info("Fetching Drift API data (placeholder)...")
    
    # Placeholder data structure matching expected format
    return {
        "BTC": {
            "drift_is_listed_spot": "true",
            "drift_is_listed_perp": "true",
            "drift_spot_market": "wBTC",
            "drift_perp_market": "BTC-PERP",
            "drift_oracle_price": 93500.0,
            "drift_volume_30d": 200000000.0,
            "drift_max_leverage": 10.0,
            "drift_open_interest": 150000000.0,
            "drift_funding_rate_1h": 0.001
        },
        "ETH": {
            "drift_is_listed_spot": "true",
            "drift_is_listed_perp": "true",
            "drift_spot_market": "wETH",
            "drift_perp_market": "ETH-PERP",
            "drift_oracle_price": 5200.0,
            "drift_volume_30d": 150000000.0,
            "drift_max_leverage": 10.0,
            "drift_open_interest": 100000000.0,
            "drift_funding_rate_1h": 0.0008
        }
    }


def score_assets(market_data: List[Dict]) -> List[Dict]:
    """
    Scores assets based on aggregated market data and provides Drift-specific recommendations.
    
    Args:
        market_data: List of dictionaries containing aggregated market data
        
    Returns:
        List[Dict]: Original market data with added scoring and Drift-specific recommendations
    """
    logger.info("Scoring assets and generating Drift recommendations...")
    
    scored_data = []
    for asset in market_data:
        asset_copy = asset.copy()
        
        # Calculate component scores (placeholder logic)
        market_cap_score = min(20.0, asset_copy["coingecko_data"]["coingecko_market_cap"] / 1e11)
        global_vol_score = min(35.0, asset_copy["coingecko_data"]["coingecko_total_volume_24h"] / 1e9)
        
        # Calculate Drift activity scores if the asset is listed
        drift_activity_score = 0.0
        if "drift_data" in asset_copy and asset_copy["drift_data"].get("drift_is_listed_perp") == "true":
            volume_on_drift = min(15.0, asset_copy["drift_data"]["drift_volume_30d"] / 1e8)
            oi_on_drift = min(15.5, asset_copy["drift_data"]["drift_open_interest"] / 1e8)
            drift_activity_score = volume_on_drift + oi_on_drift
        
        # Calculate partial scores
        partial_mc = min(20.0, market_cap_score)
        partial_global_volume = min(18.0, global_vol_score / 2)
        partial_volume_on_drift = min(15.0, drift_activity_score / 2)
        partial_oi_on_drift = min(15.5, drift_activity_score / 2)
        
        # Calculate overall score
        overall_score = (
            partial_mc +
            partial_global_volume +
            partial_volume_on_drift +
            partial_oi_on_drift
        )
        
        # Add scoring information
        asset_copy["scoring"] = {
            "overall_score": overall_score,
            "market_cap_score": market_cap_score,
            "global_vol_score": global_vol_score,
            "drift_activity_score": drift_activity_score,
            "partial_mc": partial_mc,
            "partial_global_volume": partial_global_volume,
            "partial_volume_on_drift": partial_volume_on_drift,
            "partial_oi_on_drift": partial_oi_on_drift
        }
        
        # Determine recommendation based on scores
        if overall_score >= 80:
            if not asset_copy["drift_data"].get("drift_is_listed_perp") == "true":
                recommendation = "List"
            elif asset_copy["drift_data"]["drift_max_leverage"] < 20:
                recommendation = "Increase_Leverage"
            else:
                recommendation = "Keep"
        elif overall_score >= 60:
            if asset_copy["drift_data"].get("drift_is_listed_perp") == "true":
                recommendation = "Keep"
            else:
                recommendation = "Monitor"
        elif overall_score >= 40:
            if asset_copy["drift_data"].get("drift_is_listed_perp") == "true":
                if asset_copy["drift_data"]["drift_max_leverage"] > 5:
                    recommendation = "Decrease_Leverage"
                else:
                    recommendation = "Monitor"
            else:
                recommendation = "Monitor"
        else:
            if asset_copy["drift_data"].get("drift_is_listed_perp") == "true":
                recommendation = "Delist"
            else:
                recommendation = "Ignore"
        
        asset_copy["recommendation"] = recommendation
        scored_data.append(asset_copy)
    
    return scored_data

def main(vat: Vat) -> List[Dict]:
    """
    Main function to fetch, aggregate, and score market data from multiple sources.
    Provides recommendations for Drift protocol listing decisions.
    
    Args:
        vat: Vat instance containing market data
    
    Returns:
        List[Dict]: List of dictionaries containing comprehensive market data and Drift-specific recommendations
    """
    logger.info("Starting market data aggregation process...")
    
    try:
        # Fetch data from all sources
        coingecko_data = fetch_coingecko_market_data()
        drift_data = fetch_driftpy_data(vat)
        
        # Integrate drift data into coingecko data
        for asset in coingecko_data:
            symbol = asset["symbol"]
            
            # Add Drift data if available
            if symbol in drift_data:
                asset["drift_data"] = drift_data[symbol]
            else:
                # Add default drift data for non-listed assets
                asset["drift_data"] = {
                    "drift_is_listed_spot": "false",
                    "drift_is_listed_perp": "false",
                    "drift_spot_market": None,
                    "drift_perp_market": None,
                    "drift_oracle_price": None,
                    "drift_volume_30d": 0.0,
                    "drift_max_leverage": 0.0,
                    "drift_open_interest": 0.0,
                    "drift_funding_rate_1h": 0.0
                }
        
        # Score the aggregated data
        scored_data = score_assets(coingecko_data)
        
        logger.info(f"Successfully aggregated and scored data for {len(scored_data)} assets")
        return scored_data
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    main()