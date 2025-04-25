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
import os
from datetime import datetime, timedelta

# Constants
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
async def get_market_data(request: BackendRequest, number_of_tokens: int = 2):
    """
    Get comprehensive market data including CoinGecko data, Drift metrics, and listing recommendations.
    
    Args:
        request (BackendRequest): The backend request object containing state
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List[Dict]: List of dictionaries containing market data, scoring, and recommendations for each coin
    """
    return main(request.state.backend_state.vat, number_of_tokens)

def fetch_coingecko_market_data(number_of_tokens: int = 2) -> List[Dict]:
    """
    Fetches market data for top N coins from CoinGecko API.
    
    Args:
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List of dictionaries containing market data for each coin
    """
    logger.info(f"Fetching market data for top {number_of_tokens} coins from CoinGecko...")
    
    try:
        # Fetch market data
        market_data = fetch_all_coingecko_market_data(number_of_tokens)
        
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
                # Preserve original case for API calls
                original_market_name = market_name
                
                # Extract symbol from market name (e.g., "BTC-PERP" -> "BTC")
                # Keep original case for API calls, but use uppercase for dictionary keys
                symbol_parts = market_name.split('-')
                if len(symbol_parts) > 0:
                    # Use uppercase for normalized symbol (dictionary keys)
                    normalized_symbol = symbol_parts[0].upper().strip()
                else:
                    normalized_symbol = market_name.upper().strip()
                
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
                if normalized_symbol not in drift_markets:
                    drift_markets[normalized_symbol] = {
                        "drift_is_listed_perp": "true",
                        "drift_is_listed_spot": "false",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_quote_volume_30d": 0.0,
                        "drift_total_base_volume_30d": 0.0,
                        "drift_max_leverage": float(market.data.margin_ratio_initial),
                        "drift_open_interest": oi_usd,
                        "drift_funding_rate_1h": float(market.data.amm.last_funding_rate) / 1e6 * 100
                    }
                
                # Add perp market data - use original market name with preserved case for API calls
                drift_markets[normalized_symbol]["drift_perp_markets"][original_market_name] = {
                    "drift_perp_oracle_price": oracle_price,
                    "drift_perp_quote_volume_30d": 0.0,
                    "drift_perp_base_volume_30d": 0.0
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
                # IMPORTANT: Preserve original case for API calls
                raw_symbol = market_name.strip()  # Preserve case for API calls
                symbol_key = raw_symbol.upper()  # Use uppercase for dictionary keys for consistent lookups
                logger.info(f"Raw symbol before BTC check: {raw_symbol}")
                
                if symbol_key in ['WBTC', 'CBBTC']:
                    normalized_symbol = 'BTC'
                    logger.info(f"Found BTC variant: {raw_symbol}, mapping to {normalized_symbol}")
                else:
                    normalized_symbol = symbol_key.replace('W', '').strip()
                
                # Get spot oracle price
                market_price = vat.spot_oracles.get(market.data.market_index)
                if market_price is None:
                    logger.warning(f"No oracle price found for market {raw_symbol} (index: {market.data.market_index})")
                    continue
                
                spot_oracle_price = float(market_price.price) / PRICE_PRECISION
                logger.info(f"Got oracle price for {raw_symbol}: {spot_oracle_price}")
                
                # Initialize or update symbol entry
                if normalized_symbol not in drift_markets:
                    logger.info(f"Initializing new market entry for symbol {normalized_symbol}")
                    drift_markets[normalized_symbol] = {
                        "drift_is_listed_perp": "false",
                        "drift_is_listed_spot": "true",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_quote_volume_30d": 0.0,
                        "drift_total_base_volume_30d": 0.0,
                        "drift_max_leverage": 0.0,
                        "drift_open_interest": 0.0,
                        "drift_funding_rate_1h": 0.0
                    }
                else:
                    logger.info(f"Updating existing market entry for symbol {normalized_symbol}")
                    drift_markets[normalized_symbol]["drift_is_listed_spot"] = "true"
                
                # Add spot market data - use the original market name with preserved case for API calls
                logger.info(f"Adding spot market {raw_symbol} to {normalized_symbol}'s drift_spot_markets")
                drift_markets[normalized_symbol]["drift_spot_markets"][raw_symbol] = {
                    "drift_spot_oracle_price": spot_oracle_price,
                    "drift_spot_quote_volume_30d": 0.0,
                    "drift_spot_base_volume_30d": 0.0
                }
                logger.info(f"Current spot markets for {normalized_symbol}: {list(drift_markets[normalized_symbol]['drift_spot_markets'].keys())}")
                
            except Exception as e:
                logger.error(f"Error processing spot market: {e}")
                continue
        
        logger.info(f"Completed market processing. Final market count: {len(drift_markets)}")
        return drift_markets
        
    except Exception as e:
        logger.error(f"Error in fetch_driftpy_data: {e}")
        return {}

def fetch_drift_data_api_data(discovered_markets: Dict = None) -> Dict:
    """
    Fetches market data from Drift API including historical trade data.
    Synchronous version of the trade fetching functionality.
    
    Args:
        discovered_markets (Dict): Dictionary of markets discovered from fetch_driftpy_data
                                Contains market structure with perp and spot listings
    
    Returns:
        Dict: Dictionary containing Drift API data for each coin with nested market data
    """
    logger.info("Fetching Drift API data...")
    
    # Initialize result dictionary
    drift_markets = {}
    
    # Constants for trade fetching
    DAYS_TO_CONSIDER = 30
    DRIFT_DATA_API_BASE = os.getenv("DRIFT_DATA_API_BASE_URL", "https://data.api.drift.trade")
    DRIFT_DATA_API_AUTH = os.getenv("DRIFT_DATA_API_AUTH", "")  # Get auth token from env
    DRIFT_DATA_API_HEADERS = {
        "accept": "application/json",
        "x-origin-verify": DRIFT_DATA_API_AUTH  # Use x-origin-verify header
    }
    DRIFT_API_RATE_LIMIT_INTERVAL = 0.2  # seconds between requests
    
    def fetch_api_page(url: str, retries: int = 5) -> Dict:
        """
        Synchronous version of API page fetching with retries and rate limiting.
        """
        import requests
        import time
        
        if not DRIFT_DATA_API_AUTH:
            logger.error("DRIFT_DATA_API_AUTH environment variable not set")
            return {"success": False, "records": [], "meta": {"totalRecords": 0}}
        
        last_request_time = getattr(fetch_api_page, 'last_request_time', 0)
        
        for attempt in range(retries):
            # Apply rate limiting
            current_time = time.time()
            wait_time = DRIFT_API_RATE_LIMIT_INTERVAL - (current_time - last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            
            try:
                logger.info(f"Fetching data from: {url}")
                response = requests.get(url, headers=DRIFT_DATA_API_HEADERS, timeout=10)
                fetch_api_page.last_request_time = time.time()
                
                if response.status_code != 200:
                    logger.warning(f"API request failed: {url}, status: {response.status_code}, response: {response.text[:200]}")
                    if attempt < retries - 1:
                        logger.info(f"Retrying request (attempt {attempt + 2}/{retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {"success": False, "records": [], "meta": {"totalRecords": 0}}
                
                data = response.json()
                logger.info(f"Successfully fetched data from {url} - got {len(data.get('records', []))} records")
                return data
                
            except Exception as e:
                logger.warning(f"Error fetching {url}: {str(e)}")
                if attempt < retries - 1:
                    logger.info(f"Retrying request (attempt {attempt + 2}/{retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {"success": False, "records": [], "meta": {"totalRecords": 0}}
        
        return {"success": False, "records": [], "meta": {"totalRecords": 0}}

    def fetch_market_trades(symbol: str, start_date: datetime, end_date: datetime = None) -> List[Dict]:
        """
        Fetch market candle data for a symbol using the candles endpoint.
        This replaces the inefficient day-by-day trade fetching with a single API call.
        
        Args:
            symbol: Market symbol (e.g., "SOL-PERP")
            start_date: Start date for candle data
            end_date: End date for candle data (defaults to current time)
            
        Returns:
            List[Dict]: List of candle data records
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Log the start of fetching for this symbol
        logger.info(f"Starting to fetch candle data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to Unix timestamps (seconds)
        # NOTE: For the Drift API, startTs should be the more recent timestamp (end_date)
        # and endTs should be the older timestamp (start_date), which is counter-intuitive
        end_ts = int(start_date.timestamp())    # Earlier date - goes in endTs
        start_ts = int(end_date.timestamp())    # Later date - goes in startTs
        
        # Calculate number of days to request (plus 1 to include both start and end)
        days = (end_date - start_date).days + 1
        
        if not DRIFT_DATA_API_BASE:
            logger.error("DRIFT_DATA_API_BASE environment variable not set")
            return []
        
        # Use the candles endpoint with daily resolution (D)
        # The API expects: startTs = most recent, endTs = oldest (reverse of what might be expected)
        url = f"{DRIFT_DATA_API_BASE}/market/{symbol}/candles/D?startTs={start_ts}&endTs={end_ts}&limit={min(days, 31)}"
        
        logger.info(f"Requesting candles with startTs={start_ts} (now), endTs={end_ts} (past)")
        
        # Fetch candle data
        data = fetch_api_page(url)
        
        if data.get("success") and "records" in data:
            records_count = len(data["records"])
            logger.info(f"Successfully fetched {records_count} candles for {symbol}")
            return data["records"]
        else:
            logger.warning(f"Failed to fetch candle data for {symbol}")
            return []
    
    def calculate_market_volume(symbol: str) -> dict:
        """
        Calculate total trading volume for a market over the past 30 days using candle data.
        Returns both quote volume (in USD) and base volume (in token units).
        
        Args:
            symbol: Market symbol (e.g., "SOL-PERP")
            
        Returns:
            dict: Dictionary with quote_volume and base_volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_TO_CONSIDER)
        
        try:
            candles = fetch_market_trades(symbol, start_date, end_date)
            
            total_quote_volume = 0.0
            total_base_volume = 0.0
            
            for candle in candles:
                try:
                    # Extract both quote and base volumes
                    quote_volume = float(candle.get("quoteVolume", 0))
                    base_volume = float(candle.get("baseVolume", 0))
                    
                    total_quote_volume += quote_volume
                    total_base_volume += base_volume
                except (ValueError, TypeError):
                    continue
            
            logger.info(f"Calculated {DAYS_TO_CONSIDER}-day volumes for {symbol}: " 
                        f"${total_quote_volume:,.2f} (quote), {total_base_volume:,.2f} (base)")
            
            return {
                "quote_volume": total_quote_volume,
                "base_volume": total_base_volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume for {symbol}: {str(e)}")
            return {
                "quote_volume": 0.0,
                "base_volume": 0.0
            }
    
    # Process each market
    try:
        if not discovered_markets:
            logger.warning("No discovered markets provided, skipping market processing")
            return {}

        logger.info(f"Processing {len(discovered_markets)} discovered markets")
        
        for symbol, market_info in discovered_markets.items():
            drift_markets[symbol] = {
                "drift_is_listed_spot": market_info.get("drift_is_listed_spot", "false"),
                "drift_is_listed_perp": market_info.get("drift_is_listed_perp", "false"),
                "drift_perp_markets": {},
                "drift_spot_markets": {},
                "drift_total_quote_volume_30d": 0.0,
                "drift_total_base_volume_30d": 0.0,
                "drift_max_leverage": market_info.get("drift_max_leverage", 0.0),
                "drift_open_interest": market_info.get("drift_open_interest", 0.0),
                "drift_funding_rate_1h": market_info.get("drift_funding_rate_1h", 0.0)
            }
            
            # Process perp markets
            total_quote_volume = 0.0
            total_base_volume = 0.0
            for perp_market_name in market_info.get("drift_perp_markets", {}):
                # Use original market name (case-preserved) for API call
                logger.info(f"Calculating volume for perp market: {perp_market_name} (preserving case)")
                volume_data = calculate_market_volume(perp_market_name)  # Returns dict with quote and base volumes
                
                drift_markets[symbol]["drift_perp_markets"][perp_market_name] = {
                    "drift_perp_oracle_price": market_info["drift_perp_markets"][perp_market_name].get("drift_perp_oracle_price", 0.0),
                    "drift_perp_quote_volume_30d": volume_data["quote_volume"],
                    "drift_perp_base_volume_30d": volume_data["base_volume"],
                    "drift_is_listed_perp": True,
                    "drift_perp_oi": volume_data["quote_volume"] * 0.5  # Example calculation
                }
                total_quote_volume += volume_data["quote_volume"]
                total_base_volume += volume_data["base_volume"]
            
            # Process spot markets
            for spot_market_name in market_info.get("drift_spot_markets", {}):
                # Use original market name (case-preserved) for API call
                logger.info(f"Calculating volume for spot market: {spot_market_name} (preserving case)")
                volume_data = calculate_market_volume(spot_market_name)  # Returns dict with quote and base volumes
                
                drift_markets[symbol]["drift_spot_markets"][spot_market_name] = {
                    "drift_spot_oracle_price": market_info["drift_spot_markets"][spot_market_name].get("drift_spot_oracle_price", 0.0),
                    "drift_spot_quote_volume_30d": volume_data["quote_volume"],
                    "drift_spot_base_volume_30d": volume_data["base_volume"],
                    "drift_is_listed_spot": True
                }
                total_quote_volume += volume_data["quote_volume"]
                total_base_volume += volume_data["base_volume"]
            
            # Update total volumes
            drift_markets[symbol]["drift_total_quote_volume_30d"] = total_quote_volume
            drift_markets[symbol]["drift_total_base_volume_30d"] = total_base_volume
            
            # Update OI and funding rate if perp markets exist
            if market_info.get("drift_is_listed_perp") == "true":
                drift_markets[symbol]["drift_open_interest"] = total_quote_volume * 0.75  # Example calculation
                drift_markets[symbol]["drift_funding_rate_1h"] = market_info.get("drift_funding_rate_1h", 0.0)
    
    except Exception as e:
        logger.error(f"Error in fetch_drift_data_api_data: {e}")
        return {}
    
    return drift_markets

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
        
        # Calculate total volumes from all markets
        total_quote_volume = market_info.get('drift_total_quote_volume_30d', 0.0)
        total_base_volume = market_info.get('drift_total_base_volume_30d', 0.0)
            
        # Calculate scores using helper functions
        volume_score = calculate_volume_score(total_quote_volume)
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
                "drift_total_quote_volume_30d": 0.0,
                "drift_total_base_volume_30d": 0.0,
                "drift_max_leverage": 0.0,
                "drift_open_interest": 0.0,
                "drift_funding_rate_1h": 0.0
            }
        else:
            scored_asset['drift_data'] = drift_data[symbol]
        
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
            "drift_total_quote_volume_30d": 0.0,  # Will be calculated from market volumes
            "drift_total_base_volume_30d": 0.0,   # Will be calculated from market volumes
            "drift_max_leverage": symbol_drift_data.get('drift_max_leverage', 0.0),
            "drift_open_interest": symbol_drift_data.get('drift_open_interest', 0.0),
            "drift_funding_rate_1h": symbol_drift_data.get('drift_funding_rate_1h', 0.0)
        }
        
        # Calculate total volumes across all markets
        total_quote_volume = 0.0
        total_base_volume = 0.0
        
        # Clean up perp markets and sum volumes
        for market_name, market_data in list(processed_drift_data[symbol]["drift_perp_markets"].items()):
            # Remove redundant volume field
            if "drift_perp_volume_30d" in market_data:
                del market_data["drift_perp_volume_30d"]
                
            # Sum quote and base volumes    
            total_quote_volume += market_data.get("drift_perp_quote_volume_30d", 0.0)
            total_base_volume += market_data.get("drift_perp_base_volume_30d", 0.0)
            
        # Clean up spot markets and sum volumes
        for market_name, market_data in list(processed_drift_data[symbol]["drift_spot_markets"].items()):
            # Remove redundant volume field
            if "drift_spot_volume_30d" in market_data:
                del market_data["drift_spot_volume_30d"]
                
            # Sum quote and base volumes
            total_quote_volume += market_data.get("drift_spot_quote_volume_30d", 0.0)
            total_base_volume += market_data.get("drift_spot_base_volume_30d", 0.0)
            
        # Update total volumes
        processed_drift_data[symbol]["drift_total_quote_volume_30d"] = total_quote_volume
        processed_drift_data[symbol]["drift_total_base_volume_30d"] = total_base_volume
    
    return processed_drift_data

def main(vat: Vat, number_of_tokens: int = 2) -> List[Dict]:
    """
    Main function to fetch, aggregate, and score market data from multiple sources.
    Provides recommendations for Drift protocol listing decisions.
    
    Args:
        vat: Vat instance containing market data
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List[Dict]: List of dictionaries containing comprehensive market data and Drift-specific recommendations
    """
    logger.info("Starting market data aggregation process...")
    
    try:
        # Fetch CoinGecko data first to get the list of tokens we're interested in
        coingecko_data = fetch_coingecko_market_data(number_of_tokens)
        if not coingecko_data:
            logger.error("Failed to fetch CoinGecko data")
            return []
            
        # Get the list of symbols we're processing
        processed_symbols = {asset["symbol"].upper() for asset in coingecko_data}
        logger.info(f"Processing {len(processed_symbols)} symbols from CoinGecko: {', '.join(processed_symbols)}")
        
        # Fetch DriftPy data for market discovery
        drift_data = fetch_driftpy_data(vat)
        
        # Filter drift_data to only include symbols we got from CoinGecko
        filtered_drift_data = {
            symbol: data for symbol, data in drift_data.items()
            if symbol in processed_symbols
        }
        logger.info(f"Filtered Drift markets to {len(filtered_drift_data)} matching symbols")
        
        # Fetch Drift API data using the discovered and filtered markets
        drift_api_data = fetch_drift_data_api_data(filtered_drift_data)
        
        # Merge drift_api_data volumes into drift_data
        for symbol, api_market_data in drift_api_data.items():
            if symbol in drift_data:
                # Update perp market volumes
                for market_name, perp_data in api_market_data.get('drift_perp_markets', {}).items():
                    if market_name in drift_data[symbol].get('drift_perp_markets', {}):
                        # Update with both quote and base volumes
                        if "drift_perp_quote_volume_30d" in perp_data:
                            drift_data[symbol]['drift_perp_markets'][market_name]['drift_perp_quote_volume_30d'] = perp_data.get('drift_perp_quote_volume_30d', 0.0)
                        if "drift_perp_base_volume_30d" in perp_data:
                            drift_data[symbol]['drift_perp_markets'][market_name]['drift_perp_base_volume_30d'] = perp_data.get('drift_perp_base_volume_30d', 0.0)
                
                # Update spot market volumes
                for market_name, spot_data in api_market_data.get('drift_spot_markets', {}).items():
                    if market_name in drift_data[symbol].get('drift_spot_markets', {}):
                        # Update with both quote and base volumes
                        if "drift_spot_quote_volume_30d" in spot_data:
                            drift_data[symbol]['drift_spot_markets'][market_name]['drift_spot_quote_volume_30d'] = spot_data.get('drift_spot_quote_volume_30d', 0.0)
                        if "drift_spot_base_volume_30d" in spot_data:
                            drift_data[symbol]['drift_spot_markets'][market_name]['drift_spot_base_volume_30d'] = spot_data.get('drift_spot_base_volume_30d', 0.0)
                
                # Update total volumes
                if "drift_total_quote_volume_30d" in api_market_data:
                    drift_data[symbol]['drift_total_quote_volume_30d'] = api_market_data.get('drift_total_quote_volume_30d', 0.0)
                if "drift_total_base_volume_30d" in api_market_data:
                    drift_data[symbol]['drift_total_base_volume_30d'] = api_market_data.get('drift_total_base_volume_30d', 0.0)
        
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
                    "drift_total_quote_volume_30d": 0.0,
                    "drift_total_base_volume_30d": 0.0,
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