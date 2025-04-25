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
    Fetches market data from DriftPy to check for perp and spot market listings.
    
    Args:
        vat: Vat instance containing market data
    
    Returns:
        Dict: Dictionary containing Drift market data with nested perp and spot markets
    """
    logger.info("Fetching DriftPy data to check perp and spot market listings...")
    
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
        
        # Initialize drift markets dictionary
        drift_markets = {}
        
        # Process perpetual markets
        # Use values() instead of items() since MarketMap is an object with specific methods
        for market in vat.perp_markets.values():
            try:
                # Get market name and clean it
                market_name = bytes(market.data.name).decode('utf-8').strip('\x00').strip()
                # Extract symbol from market name (e.g., "BTC-PERP" -> "BTC")
                symbol = market_name.split('-')[0].upper().strip()
                
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
                
                # Initialize symbol entry if not exists
                if symbol not in drift_markets:
                    drift_markets[symbol] = {
                        "drift_is_listed_perp": "true",
                        "drift_is_listed_spot": "false",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_volume_30d": 0.0,
                        "drift_max_leverage": float(market.data.margin_ratio_initial),
                        "drift_open_interest": oi_usd,
                        "drift_funding_rate_1h": float(market.data.amm.last_funding_rate) / 1e6 * 100
                    }
                
                # Add perp market data
                drift_markets[symbol]["drift_perp_markets"][market_name] = {
                    "drift_perp_oracle_price": oracle_price,
                    "drift_perp_volume_30d": 0.0,  # Will be updated when volume data is available
                }
                
            except Exception as e:
                logger.error(f"Error processing perp market: {e}")
                continue
        
        logger.info(f"Found {len(drift_markets)} perpetual markets on Drift")
        
        # Process spot markets
        logger.info("Processing spot markets from Vat...")
        spot_market_count = sum(1 for _ in vat.spot_markets.values())
        logger.info(f"Found {spot_market_count} spot markets in vat.spot_markets")
        
        # Process each spot market
        for market in vat.spot_markets.values():
            try:
                # Get market name and clean it
                market_name = bytes(market.data.name).decode('utf-8').strip('\x00').strip()
                logger.info(f"Processing spot market: {market_name} (market_index: {market.data.market_index})")
                
                # Extract symbol with special handling for BTC variants
                raw_symbol = market_name.upper().strip()
                logger.info(f"Raw symbol before BTC check: {raw_symbol}")
                
                if raw_symbol in ['WBTC', 'CBBTC']:
                    symbol = 'BTC'
                    logger.info(f"Found BTC variant: {raw_symbol}, mapping to {symbol}")
                else:
                    symbol = raw_symbol.replace('W', '').strip()
                
                # Get spot oracle price
                market_price = vat.spot_oracles.get(market.data.market_index)
                if market_price is None:
                    logger.warning(f"No oracle price found for market {raw_symbol} (index: {market.data.market_index})")
                    continue
                
                spot_oracle_price = float(market_price.price) / PRICE_PRECISION
                logger.info(f"Got oracle price for {raw_symbol}: {spot_oracle_price}")
                
                # Initialize or update symbol entry
                if symbol not in drift_markets:
                    logger.info(f"Initializing new market entry for symbol {symbol}")
                    drift_markets[symbol] = {
                        "drift_is_listed_perp": "false",
                        "drift_is_listed_spot": "true",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_volume_30d": 0.0,
                        "drift_max_leverage": 0.0,
                        "drift_open_interest": 0.0,
                        "drift_funding_rate_1h": 0.0
                    }
                else:
                    logger.info(f"Updating existing market entry for symbol {symbol}")
                    drift_markets[symbol]["drift_is_listed_spot"] = "true"
                
                # Add spot market data - use the original market name to preserve WBTC/CBBTC distinction
                logger.info(f"Adding spot market {raw_symbol} to {symbol}'s drift_spot_markets")
                drift_markets[symbol]["drift_spot_markets"][raw_symbol] = {
                    "drift_spot_oracle_price": spot_oracle_price,
                    "drift_spot_volume_30d": 0.0,  # Will be updated when volume data is available
                }
                logger.info(f"Current spot markets for {symbol}: {list(drift_markets[symbol]['drift_spot_markets'].keys())}")
                
            except Exception as e:
                logger.error(f"Error processing spot market: {e}")
                continue
        
        logger.info(f"Completed market processing. Final market count: {len(drift_markets)}")
        return drift_markets
        
    except Exception as e:
        logger.error(f"Error in fetch_driftpy_data: {e}")
        return {}

def fetch_drift_data_api_data() -> Dict:
    """
    Fetches market data from Drift API (placeholder implementation).
    
    Returns:
        Dict: Dictionary containing Drift API data for each coin with nested market data
    """
    logger.info("Fetching Drift API data (placeholder)...")
    
    # Placeholder data structure matching new nested format
    return {
        "BTC": {
            "drift_is_listed_spot": "true",
            "drift_is_listed_perp": "true",
            "drift_perp_markets": {
                "BTC-PERP": {
                    "drift_perp_volume_30d": 150000000.0,
                    "drift_is_listed_perp": True,
                    "drift_perp_oi": 100000000.0,
                }
            },
            "drift_spot_markets": {
                "WBTC": {
                    "drift_spot_volume_30d": 25000000.0,
                    "drift_is_listed_spot": True,
                },
                "CBBTC": {
                    "drift_spot_volume_30d": 25000000.0,
                    "drift_is_listed_spot": True,
                }
            },
            "drift_total_volume_30d": 200000000.0,  # Sum of all market volumes
            "drift_max_leverage": 10.0,
            "drift_open_interest": 150000000.0,
            "drift_funding_rate_1h": 0.001
        },
        "ETH": {
            "drift_is_listed_spot": "true",
            "drift_is_listed_perp": "true",
            "drift_perp_markets": {
                "ETH-PERP": {
                    "drift_perp_volume_30d": 100000000.0,
                    "drift_is_listed_perp": True,
                    "drift_perp_oi": 50000000.0,
                }
            },
            "drift_spot_markets": {
                "WETH": {
                    "drift_spot_volume_30d": 50000000.0,
                    "drift_is_listed_spot": True,
                }
            },
            "drift_total_volume_30d": 150000000.0,  # Sum of all market volumes
            "drift_max_leverage": 10.0,
            "drift_open_interest": 100000000.0,
            "drift_funding_rate_1h": 0.0008
        },
        "USDC": {
            "drift_spot_volume_30d": 0.0,
            "drift_is_listed_spot": True,
        }
    }

def calculate_volume_score(volume_30d: float) -> float:
    """
    Calculate a score based on 30-day trading volume.
    
    Args:
        volume_30d (float): 30-day trading volume in USD
        
    Returns:
        float: Volume score between 0 and 50
    """
    # Score based on volume tiers (adjust thresholds as needed)
    if volume_30d >= 1_000_000_000:  # $1B+
        return 50.0
    elif volume_30d >= 500_000_000:  # $500M+
        return 40.0
    elif volume_30d >= 100_000_000:  # $100M+
        return 30.0
    elif volume_30d >= 50_000_000:   # $50M+
        return 20.0
    elif volume_30d >= 10_000_000:   # $10M+
        return 10.0
    else:
        return max(0.0, min(10.0, volume_30d / 1_000_000))  # Linear score up to $10M

def calculate_leverage_score(max_leverage: float) -> float:
    """
    Calculate a score based on maximum leverage offered.
    
    Args:
        max_leverage (float): Maximum leverage offered
        
    Returns:
        float: Leverage score between 0 and 50
    """
    # Score based on leverage tiers
    if max_leverage >= 20:
        return 50.0
    elif max_leverage >= 15:
        return 40.0
    elif max_leverage >= 10:
        return 30.0
    elif max_leverage >= 5:
        return 20.0
    elif max_leverage > 0:
        return 10.0
    else:
        return 0.0

def score_assets(assets: List[Dict], drift_data: Dict) -> List[Dict]:
    """
    Score assets based on Drift market data and other metrics.
    
    Args:
        assets (List[Dict]): List of asset dictionaries with market data
        drift_data (Dict): Nested dictionary containing Drift market data for perp and spot markets
        
    Returns:
        List[Dict]: List of scored assets with additional metrics
    """
    scored_assets = []
    
    for asset in assets:
        symbol = asset.get('symbol', '').upper()
        
        # Get Drift market data if available
        market_info = drift_data.get(symbol, {})
        
        # Calculate total volume from all markets
        total_volume = 0.0
        
        # Process perp markets
        perp_markets = market_info.get('drift_perp_markets', {})
        for perp_data in perp_markets.values():
            total_volume += perp_data.get('drift_perp_volume_30d', 0.0)
            
        # Process spot markets
        spot_markets = market_info.get('drift_spot_markets', {})
        for spot_data in spot_markets.values():
            total_volume += spot_data.get('drift_spot_volume_30d', 0.0)
        
        # Calculate scores using helper functions
        volume_score = calculate_volume_score(total_volume)
        leverage_score = calculate_leverage_score(market_info.get('drift_max_leverage', 0.0))
        
        # Combine scores (equal weighting for now)
        total_score = volume_score + leverage_score
        
        # Create scored asset dictionary with nested drift_data
        scored_asset = {
            **asset,  # Include all original asset data
            'volume_score': volume_score,
            'leverage_score': leverage_score,
            'total_score': total_score
        }
        
        # If there's no drift data, initialize with default structure
        if symbol not in drift_data:
            scored_asset['drift_data'] = {
                "drift_is_listed_spot": "false",
                "drift_is_listed_perp": "false",
                "drift_perp_markets": {},
                "drift_spot_markets": {},
                "drift_total_volume_30d": 0.0,
                "drift_max_leverage": 0.0,
                "drift_open_interest": 0.0,
                "drift_funding_rate_1h": 0.0
            }
        else:
            scored_asset['drift_data'] = drift_data[symbol]
            # Update total volume in drift_data
            scored_asset['drift_data']['drift_total_volume_30d'] = total_volume
        
        scored_assets.append(scored_asset)
    
    # Sort assets by total score in descending order
    scored_assets.sort(key=lambda x: x['total_score'], reverse=True)
    
    return scored_assets

def process_drift_markets(scored_data: List[Dict], drift_data: Dict) -> Dict:
    """
    Process Drift markets data and update drift_data dictionary with the new nested structure.
    
    Args:
        scored_data (List[Dict]): List of dictionaries containing scored market data
        drift_data (Dict): Dictionary containing Drift market data with nested perp and spot markets
        
    Returns:
        Dict: Updated drift_data dictionary
    """
    processed_drift_data = {}
    
    for asset in scored_data:
        symbol = asset.get('symbol', '')
        if not symbol:
            continue
            
        # Get drift data for this symbol if it exists
        symbol_drift_data = drift_data.get(symbol, {})
        
        # Initialize default structure
        processed_drift_data[symbol] = {
            "drift_is_listed_perp": symbol_drift_data.get('drift_is_listed_perp', 'false'),
            "drift_is_listed_spot": symbol_drift_data.get('drift_is_listed_spot', 'false'),
            "drift_perp_markets": symbol_drift_data.get('drift_perp_markets', {}),
            "drift_spot_markets": symbol_drift_data.get('drift_spot_markets', {}),
            "drift_total_volume_30d": 0.0,  # Will be calculated from market volumes
            "drift_max_leverage": symbol_drift_data.get('drift_max_leverage', 0.0),
            "drift_open_interest": symbol_drift_data.get('drift_open_interest', 0.0),
            "drift_funding_rate_1h": symbol_drift_data.get('drift_funding_rate_1h', 0.0)
        }
        
        # Calculate total volume across all markets
        total_volume = 0.0
        
        # Sum volumes from perp markets
        for market_data in processed_drift_data[symbol]["drift_perp_markets"].values():
            total_volume += market_data.get("drift_perp_volume_30d", 0.0)
            
        # Sum volumes from spot markets
        for market_data in processed_drift_data[symbol]["drift_spot_markets"].values():
            total_volume += market_data.get("drift_spot_volume_30d", 0.0)
            
        processed_drift_data[symbol]["drift_total_volume_30d"] = total_volume
    
    return processed_drift_data

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
                    "drift_perp_markets": {},
                    "drift_spot_markets": {},
                    "drift_total_volume_30d": 0.0,
                    "drift_max_leverage": 0.0,
                    "drift_open_interest": 0.0,
                    "drift_funding_rate_1h": 0.0
                }
        
        # Score the aggregated data
        scored_data = score_assets(coingecko_data, drift_data)
        
        # Process Drift markets data and update drift_data dictionary
        drift_data = process_drift_markets(scored_data, drift_data)
        
        logger.info(f"Successfully aggregated and scored data for {len(scored_data)} assets")
        return scored_data
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    main()