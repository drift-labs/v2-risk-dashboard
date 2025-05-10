"""
This module handles all Drift Data API interactions for fetching market data.
It provides functions to fetch current market data, historical volumes, and other metrics.
"""

import os
import time
import logging
import requests
from typing import Dict, List
from datetime import datetime, timedelta
from backend.utils.cache_utils import ttl_cache

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Configuration ---
DRIFT_DATA_API_BASE = os.getenv("DRIFT_DATA_API_BASE_URL", "https://data.api.drift.trade")
DRIFT_DATA_API_AUTH = os.getenv("DRIFT_DATA_API_AUTH_KEY", "")  # Get auth token from env
DRIFT_DATA_API_HEADERS = {
    "accept": "application/json",
    "x-origin-verify": DRIFT_DATA_API_AUTH  # Use x-origin-verify header
}
DRIFT_API_RATE_LIMIT_INTERVAL = 0.2  # seconds between requests

def fetch_api_page(url: str, retries: int = 5) -> Dict:
    """
    Synchronous version of API page fetching with retries and rate limiting.
    """
    if not DRIFT_DATA_API_AUTH:
        logger.error("DRIFT_DATA_API_AUTH environment variable not set")
    
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

@ttl_cache(ttl_seconds=43200)  # Cache for 12 hours
def fetch_market_trades(market_name: str, start_date: datetime, end_date: datetime = None) -> List[Dict]:
    """
    Fetch market candle data for a market using the candles endpoint.
    Uses the original market name (e.g., "1MPEPE-PERP") for API calls.
    Responses are cached for 12 hours.
    
    Args:
        market_name: Original market name (e.g., "1MPEPE-PERP", "SOL-PERP")
        start_date: Start date for candle data
        end_date: End date for candle data (defaults to current time)
        
    Returns:
        List[Dict]: List of candle data records
    """
    # Normalize dates to day boundaries for consistent cache keys
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if end_date is None:
        end_date = datetime.now()
    end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Log the start of fetching for this market
    logger.info(f"Starting to fetch candle data for {market_name} from {start_date} to {end_date}")
    
    # Convert dates to Unix timestamps (seconds)
    # NOTE: For the Drift API, startTs should be the more recent timestamp (end_date)
    # and endTs should be the older timestamp (start_date), which is counter-intuitive
    end_ts = int(start_date.timestamp())     # Earlier date - goes in endTs
    start_ts = int(end_date.timestamp())     # Later date - goes in startTs
    
    # Calculate number of days to request (plus 1 to include both start and end)
    days = (end_date - start_date).days + 1
    
    if not DRIFT_DATA_API_BASE:
        logger.error("DRIFT_DATA_API_BASE environment variable not set")
        return []
    
    # Ensure market name is uppercase for API call
    api_market_name = market_name.upper()
    logger.info(f"Using uppercase market name for API call: {api_market_name}")
    
    # Use the candles endpoint with daily resolution (D)
    # The API expects: startTs = most recent, endTs = oldest (reverse of what might be expected)
    url = f"{DRIFT_DATA_API_BASE}/market/{api_market_name}/candles/D?startTs={start_ts}&endTs={end_ts}&limit={min(days, 31)}"
    
    logger.info(f"Requesting candles with startTs={start_ts} (now), endTs={end_ts} (past)")
    
    # Fetch candle data
    data = fetch_api_page(url)
    
    if data.get("success") and "records" in data:
        records_count = len(data["records"])
        logger.info(f"Successfully fetched {records_count} candles for {api_market_name}")
        return data["records"]
    else:
        logger.warning(f"Failed to fetch candle data for {api_market_name}")
        return []

@ttl_cache(ttl_seconds=43200)  # Cache for 12 hours
def calculate_market_volume(market_name: str, days_to_consider: int = 30) -> dict:
    """
    Calculate total trading volume for a market over the past N days using candle data.
    Uses the original market name for API calls.
    Responses are cached for 12 hours.
    
    Args:
        market_name: Original market name (e.g., "1MPEPE-PERP", "SOL-PERP")
        days_to_consider: Number of days to calculate volume for (default: 30)
            
    Returns:
        dict: Dictionary with quote_volume and base_volume
    """
    # Normalize end date to start of current day for consistent cache keys
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = (end_date - timedelta(days=days_to_consider)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    try:
        # Ensure market name is uppercase for API call
        api_market_name = market_name.upper()
        logger.info(f"Calculating volume for market {api_market_name} (original: {market_name})")
        
        candles = fetch_market_trades(api_market_name, start_date, end_date)
        
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
        
        logger.info(f"Calculated {days_to_consider}-day volumes for {api_market_name}: " 
                    f"${total_quote_volume:,.2f} (quote), {total_base_volume:,.2f} (base)")
        
        return {
            "quote_volume": total_quote_volume,
            "base_volume": total_base_volume
        }
        
    except Exception as e:
        logger.error(f"Error calculating volume for {market_name}: {str(e)}")
        return {
            "quote_volume": 0.0,
            "base_volume": 0.0
        }

@ttl_cache(ttl_seconds=43200)  # Cache for 12 hours
def fetch_drift_data_api_data(discovered_markets: Dict = None) -> Dict:
    """
    Fetches market data from Drift API including historical trade data.
    Synchronous version of the trade fetching functionality.
    Responses are cached for 12 hours.
    
    Args:
        discovered_markets (Dict): Dictionary of markets discovered from fetch_driftpy_data
                                   Contains market structure with perp and spot listings
    
    Returns:
        Dict: Dictionary containing Drift API data for each coin with nested market data
    """
    logger.info("Fetching Drift API data...")
    
    # Initialize result dictionary
    drift_markets = {}
    
    # Process each market
    try:
        if not discovered_markets:
            logger.warning("No discovered markets provided, skipping market processing")
            return {}

        logger.info(f"Processing {len(discovered_markets)} discovered markets")
        
        for symbol, market_info in discovered_markets.items():
            # Clean the symbol to handle basket markets
            normalized_symbol = clean_market_name(symbol).upper()
            
            drift_markets[normalized_symbol] = {
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
                # Check if this market should be ignored
                market_index = next((market["index"] for market in IGNORED_DRIFT_PERP_MARKETS if market["name"] == perp_market_name), None)
                if market_index is not None:
                    logger.info(f"Skipping ignored perp market: {perp_market_name}")
                    continue

                # Use original market name (case-preserved) for API call
                logger.info(f"Calculating volume for perp market: {perp_market_name} (preserving case)")
                volume_data = calculate_market_volume(perp_market_name)  # Returns dict with quote and base volumes
                
                drift_markets[normalized_symbol]["drift_perp_markets"][perp_market_name] = {
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
                
                drift_markets[normalized_symbol]["drift_spot_markets"][spot_market_name] = {
                    "drift_spot_oracle_price": market_info["drift_spot_markets"][spot_market_name].get("drift_spot_oracle_price", 0.0),
                    "drift_spot_quote_volume_30d": volume_data["quote_volume"],
                    "drift_spot_base_volume_30d": volume_data["base_volume"],
                    "drift_is_listed_spot": True
                }
                total_quote_volume += volume_data["quote_volume"]
                total_base_volume += volume_data["base_volume"]
            
            # Update total volumes
            drift_markets[normalized_symbol]["drift_total_quote_volume_30d"] = total_quote_volume
            drift_markets[normalized_symbol]["drift_total_base_volume_30d"] = total_base_volume
            
            # Update OI and funding rate if perp markets exist
            if market_info.get("drift_is_listed_perp") == "true":
                drift_markets[normalized_symbol]["drift_open_interest"] = total_quote_volume * 0.75  # Example calculation
                drift_markets[normalized_symbol]["drift_funding_rate_1h"] = market_info.get("drift_funding_rate_1h", 0.0)
    
    except Exception as e:
        logger.error(f"Error in fetch_drift_data_api_data: {e}")
        return {}
    
    return drift_markets

def clean_market_name(market_name: str) -> str:
    """Clean market name by removing basket prefixes and -PERP suffix."""
    from backend.api.market_recommender_api import BASKET_MARKET_PREFIXES
    
    # First remove any -PERP suffix
    name = market_name.split('-')[0].strip()
    
    # Then check for and remove any basket prefixes
    for prefix in BASKET_MARKET_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    return name