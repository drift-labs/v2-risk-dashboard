"""
This module handles all CoinGecko API interactions for fetching cryptocurrency market data.
It provides functions to fetch current market data, historical volumes, and other metrics.
"""

import os
import json
import time
import requests
import logging
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from backend.utils.cache_utils import ttl_cache

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Configuration ---
if os.getenv("DEV") == "false":
    COINGECKO_API_BASE = os.getenv("COINGECKO_DEMO_API_BASE_URL", "")
    COINGECKO_API_KEY = os.getenv("COINGECKO_DEMO_API_KEY", "")
    COINGECKO_API_HEADER = os.getenv("COINGECKO_DEMO_API_HEADER", "")
    COINGECKO_RATE_LIMIT_DELAY = 2 # Demo API rate limit is 30
    logger.info("CoinGecko Demo API configuration loaded")
else:
    COINGECKO_API_BASE = os.getenv("COINGECKO_PRO_API_BASE_URL", "")
    COINGECKO_API_KEY = os.getenv("COINGECKO_PRO_API_KEY", "")
    COINGECKO_API_HEADER = os.getenv("COINGECKO_PRO_API_HEADER", "")
    COINGECKO_RATE_LIMIT_DELAY = .25 # Pro API rate limit is 500 RPM but is shared with app.drift.trade
    logger.info("CoinGecko Pro API configuration loaded")

MAX_COINS_PER_PAGE = 250  # Maximum allowed by CoinGecko API

# --- Global Rate Limiter Variables ---
cg_last_request_time = 0.0
cg_api_calls = 0  # Counter for API calls

# --- Helper Functions ---
def increment_api_calls():
    """Increment the API call counter and log the current count."""
    global cg_api_calls
    cg_api_calls += 1
    logger.info(f"CoinGecko API calls made: {cg_api_calls}")

def enforce_rate_limit(last_request_time: float, rate_limit: float) -> None:
    """
    Enforce rate limiting by waiting if necessary.
    
    Args:
        last_request_time: Time of the last request in seconds since epoch
        rate_limit: Minimum time between requests in seconds
    """
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    
    if time_since_last_request < rate_limit:
        wait_time = rate_limit - time_since_last_request
        logger.info(f"Rate limiting: Waiting {wait_time:.2f}s")
        time.sleep(wait_time)

def make_api_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    rate_limit: float = 0.1,
    last_request_time: float = 0,
) -> Dict:
    """
    Make an API request with rate limiting and error handling.
    
    Args:
        url: The URL to make the request to
        method: HTTP method to use (default: GET)
        headers: Optional headers to include
        params: Optional query parameters
        rate_limit: Minimum time between requests in seconds
        last_request_time: Time of the last request in seconds since epoch
        
    Returns:
        Dict containing the response data or error information
    """
    enforce_rate_limit(last_request_time, rate_limit)
    increment_api_calls()  # Track API call
    
    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            timeout=30
        )
        
        if response.status_code == 429:  # Rate limit exceeded
            retry_after = int(response.headers.get("Retry-After", "10"))
            logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds.")
            time.sleep(retry_after)
            return make_api_request(url, method, headers, params, rate_limit)
        
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error during API request: {str(e)}")
        return {"error": str(e)}

def fetch_coingecko(endpoint: str, params: Dict = None) -> Optional[Union[List, Dict]]:
    """
    Generic CoinGecko fetcher that exactly matches the successful example.
    
    Args:
        endpoint: API endpoint to fetch
        params: Optional query parameters
        
    Returns:
        API response data or None if request fails
    """
    global cg_last_request_time
    
    if params is None:
        params = {}
    
    url = f"{COINGECKO_API_BASE}{endpoint}"
    
    # Exact headers from the successful example
    headers = {
        'accept': 'application/json',
        COINGECKO_API_HEADER: COINGECKO_API_KEY,
    }
    
    max_retries = 3
    retry_delay = 2.0
    
    for retry in range(max_retries):
        # Apply rate limiting
        wait_time = COINGECKO_RATE_LIMIT_DELAY - (time.time() - cg_last_request_time)
        if wait_time > 0:
            logger.debug(f"CG Rate limiting. Wait: {wait_time:.2f}s")
            time.sleep(wait_time)
        cg_last_request_time = time.time()
        
        try:
            logger.info(f"Fetching CG URL: {url} with params: {params}")
            logger.info(f"Headers: {headers}")
            
            # Make the request exactly as in the example
            response = requests.get(url, headers=headers, params=params, timeout=20)
            response_text = response.text
            
            if response.status_code == 200:
                try:
                    data = json.loads(response_text)
                    logger.info(f"Successfully received data from CoinGecko API")
                    return data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {response_text[:100]}...")
                    return None
            else:
                logger.error(f"CG request failed {url}: Status {response.status_code}, Response: {response_text[:100]}...")
                
                if response.status_code == 429:
                    # Rate limit hit
                    logger.warning(f"CG rate limit hit for {url}. Retrying after delay.")
                    retry_delay *= 2  # Exponential backoff
                    time.sleep(retry_delay)
                    continue
                elif retry < max_retries - 1:
                    retry_delay *= 1.5
                    time.sleep(retry_delay)
                    continue
                
                return None
        
        except requests.Timeout:
            logger.warning(f"CG request timeout for {url}")
            if retry < max_retries - 1:
                retry_delay *= 1.5
                time.sleep(retry_delay)
                continue
            return None
        except Exception as e:
            logger.error(f"Error fetching CG {url}: {e}")
            if retry < max_retries - 1:
                retry_delay *= 1.5
                time.sleep(retry_delay)
                continue
            return None
    
    return None

def calculate_pagination(total_tokens: int) -> List[Tuple[int, int]]:
    """
    Calculate the optimal pagination strategy for fetching the desired number of tokens.
    
    Args:
        total_tokens: Total number of tokens to fetch
        
    Returns:
        List of tuples (page_number, tokens_for_this_page)
    """
    if total_tokens <= 0:
        return []
    
    # Calculate number of full pages needed (250 tokens each)
    full_pages = total_tokens // MAX_COINS_PER_PAGE
    remaining_tokens = total_tokens % MAX_COINS_PER_PAGE
    
    # Create pagination strategy
    pagination = []
    
    # Add full pages
    for page in range(1, full_pages + 1):
        pagination.append((page, MAX_COINS_PER_PAGE))
    
    # Add final page with remaining tokens if any
    if remaining_tokens > 0:
        pagination.append((full_pages + 1, remaining_tokens))
    
    return pagination

@ttl_cache(ttl_seconds=43200)  # Cache for 12 hours
def fetch_all_coingecko_market_data(number_of_tokens: int = 250) -> dict:
    """
    Fetches data from CG /coins/markets using the exact parameters that worked.
    Filters out any tokens whose symbols are in IGNORED_SYMBOLS.
    Responses are cached for 12 hours.
    
    Args:
        number_of_tokens: Number of top tokens by market cap to fetch (default: 250)
        
    Returns:
        Dict mapping coin IDs to their market data
    """
    from backend.api.market_recommender import IGNORED_SYMBOLS
    
    all_markets_data = {}
    total_tokens_received = 0
    total_tokens_after_filtering = 0
    
    # Calculate pagination strategy - request more tokens to account for filtering
    # Add the number of ignored symbols to ensure we still get the requested number after filtering
    adjusted_token_count = number_of_tokens + len(IGNORED_SYMBOLS)
    pagination = calculate_pagination(adjusted_token_count)
    total_pages = len(pagination)
    
    if not pagination:
        logger.warning("Invalid number of tokens requested. Returning empty result.")
        return all_markets_data
    
    logger.info(f"Fetching market data from CoinGecko for {adjusted_token_count} tokens across {total_pages} pages...")
    
    try:
        for page_num, tokens_this_page in pagination:
            logger.info(f"Fetching CoinGecko markets page {page_num}/{total_pages} ({tokens_this_page} tokens)")
            
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': tokens_this_page,
                'precision': '2',
                'page': page_num
            }
            
            # Make API request
            endpoint = '/coins/markets'
            data = fetch_coingecko(endpoint, params)
            
            if not data or not isinstance(data, list):
                logger.warning(f"No valid data received for page {page_num}. Skipping.")
                # If we fail to get data for a page, we should stop trying further pages
                break
            
            # Update total tokens received
            tokens_received_this_page = len(data)
            total_tokens_received += tokens_received_this_page
            
            # If we received fewer tokens than requested for this page, we've hit the end
            if tokens_received_this_page < tokens_this_page:
                logger.info(f"Received {tokens_received_this_page} tokens instead of {tokens_this_page} requested. This appears to be all available tokens.")
            
            # Process each market in the response
            for market in data:
                try:
                    coin_id = market.get('id')
                    if not coin_id:
                        continue
                    
                    symbol = market.get('symbol', '').upper()
                    
                    # Skip ignored symbols
                    if symbol in IGNORED_SYMBOLS:
                        logger.info(f"Skipping ignored symbol: {symbol}")
                        continue
                    
                    market_data = {
                        'symbol': symbol,
                        'name': market.get('name'),
                        'image_url': market.get('image'),
                        'current_price': market.get('current_price'),
                        'market_cap_rank': market.get('market_cap_rank'),
                        'market_cap': market.get('market_cap'),
                        'fully_diluted_valuation': market.get('fully_diluted_valuation'),
                        'total_volume_24h': market.get('total_volume'),
                        'circulating_supply': market.get('circulating_supply'),
                        'total_supply': market.get('total_supply'),
                        'max_supply': market.get('max_supply'),
                        'ath_price': market.get('ath'),
                        'ath_change_percentage': market.get('ath_change_percentage')
                    }
                    
                    all_markets_data[coin_id] = market_data
                    total_tokens_after_filtering += 1
                    logger.info(f"Processed market data for {symbol} (ID: {coin_id})")
                    
                    # If we have enough tokens after filtering, stop processing
                    if total_tokens_after_filtering >= number_of_tokens:
                        logger.info(f"Reached requested number of tokens ({number_of_tokens}) after filtering. Stopping.")
                        return all_markets_data
                        
                except Exception as e:
                    logger.warning(f"Error processing market: {e}")
                    continue
            
            # If we received fewer tokens than requested, no need to continue to next page
            if tokens_received_this_page < tokens_this_page:
                break
            
            # Respect rate limiting between page requests
            if page_num < total_pages:
                time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        if total_tokens_after_filtering < number_of_tokens:
            logger.warning(f"Requested {number_of_tokens} tokens but only received {total_tokens_after_filtering} after filtering ignored symbols. This appears to be all available tokens.")
        
        logger.info(f"Successfully fetched data for {len(all_markets_data)} markets from CoinGecko after filtering ignored symbols")
        return all_markets_data
    except Exception as e:
        logger.error(f"Error fetching CoinGecko market data: {e}", exc_info=True)
        return all_markets_data

def fetch_coin_volume(coin_id: str) -> Optional[float]:
    """
    Helper to fetch 30d volume data for a single coin.
    
    Args:
        coin_id: CoinGecko coin ID
        
    Returns:
        float: Total volume over 30 days or None if data fetch fails
    """
    try:
        # Request 30 days of market data with exact parameters from the successful example
        params = {
            'vs_currency': 'usd',
            'days': '30',
            'interval': 'daily',
            'precision': '2'  # Adding precision parameter from the successful example
        }
        
        endpoint = f"/coins/{coin_id}/market_chart"
        data = fetch_coingecko(endpoint, params)
        
        if not data or 'total_volumes' not in data or not data['total_volumes']:
            logger.warning(f"No volume data received for {coin_id}")
            return None
        
        # Extract daily volumes (data format is [[timestamp, volume], ...])
        volumes = [entry[1] for entry in data['total_volumes'] if len(entry) >= 2]
        
        if not volumes:
            logger.warning(f"Empty volumes list for {coin_id}")
            return None
        
        # Calculate total volume
        total_volume = sum(volumes)
        logger.info(f"Calculated 30d total volume for {coin_id}: ${total_volume:,.2f}")
        
        return total_volume
    
    except Exception as e:
        logger.error(f"Error fetching volume data for {coin_id}: {e}")
        return None

@ttl_cache(ttl_seconds=43200)  # Cache for 12 hours
def fetch_all_coingecko_volumes(coin_ids: list) -> dict:
    """
    Fetches 30d volume data from CG /coins/{id}/market_chart.
    Responses are cached for 12 hours.
    
    Args:
        coin_ids: List of CoinGecko coin IDs
        
    Returns:
        dict: Dictionary mapping coingecko_id to total_30d_volume
    """
    volume_data = {}  # Dict: coingecko_id -> total_30d_volume
    logger.info(f"Fetching 30d volume data from CoinGecko for {len(coin_ids)} coins...")
    
    # Process coins one by one to avoid rate limiting issues
    for coin_id in coin_ids:
        try:
            total_volume = fetch_coin_volume(coin_id)
            if total_volume is not None:
                volume_data[coin_id] = total_volume
                logger.info(f"Processed 30d volume for {coin_id}: ${total_volume:,.2f}")
            
            # Respect rate limiting between requests
            time.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        except Exception as e:
            logger.warning(f"Error processing volume for {coin_id}: {str(e)}")
            continue
    
    logger.info(f"Successfully fetched 30d volume data for {len(volume_data)}/{len(coin_ids)} coins")
    return volume_data

def fetch_coingecko_data(symbol: str) -> Dict:
    """
    Fetch market data from CoinGecko API for a given symbol.
    
    Args:
        symbol: The cryptocurrency symbol to fetch data for
        
    Returns:
        Dict containing the market data or error information
    """
    global cg_last_request_time
    
    # First, get the coin ID
    search_url = f"{COINGECKO_API_BASE}/search"
    search_response = make_api_request(
        url=search_url,
        params={"query": symbol},
        rate_limit=COINGECKO_RATE_LIMIT_DELAY,
        last_request_time=cg_last_request_time
    )
    cg_last_request_time = time.time()
    
    if "error" in search_response:
        return {"error": f"CoinGecko search failed: {search_response['error']}"}
    
    coins = search_response.get("coins", [])
    if not coins:
        return {"error": f"No coins found for symbol {symbol}"}
    
    # Find the exact match or best match
    coin_id = None
    for coin in coins:
        if coin["symbol"].lower() == symbol.lower():
            coin_id = coin["id"]
            break
    
    if not coin_id:
        return {"error": f"Could not find exact match for symbol {symbol}"}
    
    # Fetch detailed market data
    market_url = f"{COINGECKO_API_BASE}/coins/{coin_id}"
    market_response = make_api_request(
        url=market_url,
        params={
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false",
            "sparkline": "false"
        },
        rate_limit=COINGECKO_RATE_LIMIT_DELAY,
        last_request_time=cg_last_request_time
    )
    cg_last_request_time = time.time()
    
    if "error" in market_response:
        return {"error": f"CoinGecko market data failed: {market_response['error']}"}
    
    # Extract relevant data
    market_data = market_response.get("market_data", {})
    return {
        "coingecko_name": market_response.get("name"),
        "coingecko_id": coin_id,
        "coingecko_image_url": market_response.get("image", {}).get("large"),
        "coingecko_current_price": market_data.get("current_price", {}).get("usd"),
        "coingecko_market_cap_rank": market_data.get("market_cap_rank"),
        "coingecko_market_cap": market_data.get("market_cap", {}).get("usd"),
        "coingecko_fully_diluted_valuation": market_data.get("fully_diluted_valuation", {}).get("usd"),
        "coingecko_total_volume_24h": market_data.get("total_volume", {}).get("usd"),
        "coingecko_30d_volume_total": None,  # Will be calculated from historical data
        "coingecko_circulating_supply": market_data.get("circulating_supply"),
        "coingecko_total_supply": market_data.get("total_supply"),
        "coingecko_max_supply": market_data.get("max_supply"),
        "coingecko_ath_price": market_data.get("ath", {}).get("usd"),
        "coingecko_ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd")
    }

def fetch_coingecko_historical_volume(coin_id: str, number_of_days: int = 30) -> float:
    """
    Fetch historical volume data from CoinGecko API.
    
    Args:
        coin_id: The CoinGecko coin ID
        number_of_days: Number of days of historical data to fetch (default: 30)
        
    Returns:
        Total volume over the specified period or 0 if error
    """
    global cg_last_request_time
    
    url = f"{COINGECKO_API_BASE}/coins/{coin_id}/market_chart"
    response = make_api_request(
        url=url,
        params={"vs_currency": "usd", "days": str(number_of_days), "interval": "daily"},
        rate_limit=COINGECKO_RATE_LIMIT_DELAY,
        last_request_time=cg_last_request_time
    )
    cg_last_request_time = time.time()
    
    if "error" in response:
        logger.error(f"Failed to fetch historical volume: {response['error']}")
        return 0.0
    
    try:
        # Sum up the daily volumes
        volumes = response.get("total_volumes", [])
        total_volume = sum(volume[1] for volume in volumes)
        return total_volume
    except Exception as e:
        logger.error(f"Error processing historical volume data: {str(e)}")
        return 0.0