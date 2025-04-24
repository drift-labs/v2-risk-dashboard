"""
This module handles all CoinGecko API interactions for fetching cryptocurrency market data.
It provides functions to fetch current market data, historical volumes, and other metrics.
"""

import os
import json
import time
import asyncio
import aiohttp
import logging
import math
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API Configuration ---
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_DEMO_API_KEY = "CG-oWyNSQuvyZMKCDzL3yqGzyrh"  # Demo key
COINGECKO_REQ_PER_MINUTE = 30  # Conservative rate limit
COINGECKO_RATE_LIMIT_DELAY = 60.0 / COINGECKO_REQ_PER_MINUTE
MAX_COINS_PER_PAGE = 250  # Maximum allowed by CoinGecko API

# --- Global Rate Limiter Variables ---
cg_rate_limit_lock = asyncio.Lock()
cg_last_request_time = 0.0

# --- Helper Functions ---
async def enforce_rate_limit(last_request_time: float, rate_limit: float) -> None:
    """
    Enforce rate limiting by waiting if necessary.
    
    Args:
        last_request_time: Time of the last request in seconds since epoch
        rate_limit: Minimum time between requests in seconds
    """
    current_time = time.time()
    time_since_last_request = current_time - last_request_time
    
    if time_since_last_request < rate_limit:
        await asyncio.sleep(rate_limit - time_since_last_request)

async def make_api_request(
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
    await enforce_rate_limit(last_request_time, rate_limit)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 429:  # Rate limit exceeded
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    return await make_api_request(url, method, headers, params, rate_limit)
                
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": str(e)}
    except asyncio.TimeoutError:
        logger.error(f"API request timed out: {url}")
        return {"error": "Request timed out"}
    except Exception as e:
        logger.error(f"Unexpected error during API request: {str(e)}")
        return {"error": str(e)}

async def fetch_coingecko(session: aiohttp.ClientSession, endpoint: str, params: Dict = None) -> Optional[Union[List, Dict]]:
    """
    Generic CoinGecko fetcher that exactly matches the successful example.
    
    Args:
        session: aiohttp client session
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
        'x-cg-demo-api-key': COINGECKO_DEMO_API_KEY,
    }
    
    max_retries = 3
    retry_delay = 2.0
    
    for retry in range(max_retries):
        # Apply rate limiting
        async with cg_rate_limit_lock:
            wait_time = COINGECKO_RATE_LIMIT_DELAY - (time.time() - cg_last_request_time)
            if wait_time > 0:
                logger.debug(f"CG Rate limiting. Wait: {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            cg_last_request_time = time.time()
        
        try:
            logger.info(f"Fetching CG URL: {url} with params: {params}")
            logger.info(f"Headers: {headers}")
            
            # Make the request exactly as in the example
            async with session.get(url, headers=headers, params=params, timeout=20) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        data = json.loads(response_text)
                        logger.info(f"Successfully received data from CoinGecko API")
                        return data
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON response: {response_text[:100]}...")
                        return None
                else:
                    logger.error(f"CG request failed {url}: Status {response.status}, Response: {response_text[:100]}...")
                    
                    if response.status == 429:
                        # Rate limit hit
                        logger.warning(f"CG rate limit hit for {url}. Retrying after delay.")
                        retry_delay *= 2  # Exponential backoff
                        await asyncio.sleep(retry_delay)
                        continue
                    elif retry < max_retries - 1:
                        retry_delay *= 1.5
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    return None
        
        except asyncio.TimeoutError:
            logger.warning(f"CG request timeout for {url}")
            if retry < max_retries - 1:
                retry_delay *= 1.5
                await asyncio.sleep(retry_delay)
                continue
            return None
        except Exception as e:
            logger.error(f"Error fetching CG {url}: {e}")
            if retry < max_retries - 1:
                retry_delay *= 1.5
                await asyncio.sleep(retry_delay)
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

async def fetch_all_coingecko_market_data(session: aiohttp.ClientSession, number_of_tokens: int = 250) -> dict:
    """
    Fetches data from CG /coins/markets using the exact parameters that worked.
    
    Args:
        session: aiohttp client session
        number_of_tokens: Number of top tokens by market cap to fetch (default: 250)
        
    Returns:
        Dict mapping coin IDs to their market data
    """
    all_markets_data = {}
    total_tokens_received = 0
    
    # Calculate pagination strategy
    pagination = calculate_pagination(number_of_tokens)
    total_pages = len(pagination)
    
    if not pagination:
        logger.warning("Invalid number of tokens requested. Returning empty result.")
        return all_markets_data
    
    logger.info(f"Fetching market data from CoinGecko for {number_of_tokens} tokens across {total_pages} pages...")
    
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
            data = await fetch_coingecko(session, endpoint, params)
            
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
                    logger.info(f"Processed market data for {symbol} (ID: {coin_id})")
                except Exception as e:
                    logger.warning(f"Error processing market: {e}")
                    continue
            
            # If we received fewer tokens than requested, no need to continue to next page
            if tokens_received_this_page < tokens_this_page:
                break
            
            # Respect rate limiting between page requests
            if page_num < total_pages:
                await asyncio.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        if total_tokens_received < number_of_tokens:
            logger.warning(f"Requested {number_of_tokens} tokens but only received {total_tokens_received}. This appears to be all available tokens.")
        
        logger.info(f"Successfully fetched data for {len(all_markets_data)} markets from CoinGecko")
        return all_markets_data
    except Exception as e:
        logger.error(f"Error fetching CoinGecko market data: {e}", exc_info=True)
        return all_markets_data

async def fetch_coin_volume(session: aiohttp.ClientSession, coin_id: str) -> Optional[float]:
    """
    Helper to fetch 30d volume data for a single coin.
    
    Args:
        session: aiohttp client session
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
        data = await fetch_coingecko(session, endpoint, params)
        
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
        raise  # Re-raise to be caught by gather

async def fetch_all_coingecko_volumes(session: aiohttp.ClientSession, coin_ids: list) -> dict:
    """
    Fetches 30d volume data from CG /coins/{id}/market_chart.
    
    Args:
        session: aiohttp client session
        coin_ids: List of CoinGecko coin IDs
        
    Returns:
        dict: Dictionary mapping coingecko_id to total_30d_volume
    """
    volume_data = {}  # Dict: coingecko_id -> total_30d_volume
    logger.info(f"Fetching 30d volume data from CoinGecko for {len(coin_ids)} coins...")
    
    # Process coins one by one to avoid rate limiting issues
    for coin_id in coin_ids:
        try:
            total_volume = await fetch_coin_volume(session, coin_id)
            if total_volume is not None:
                volume_data[coin_id] = total_volume
                logger.info(f"Processed 30d volume for {coin_id}: ${total_volume:,.2f}")
            
            # Respect rate limiting between requests
            await asyncio.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        except Exception as e:
            logger.warning(f"Error processing volume for {coin_id}: {str(e)}")
            continue
    
    logger.info(f"Successfully fetched 30d volume data for {len(volume_data)}/{len(coin_ids)} coins")
    return volume_data

async def fetch_coingecko_data(symbol: str) -> Dict:
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
    search_response = await make_api_request(
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
    market_response = await make_api_request(
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

async def fetch_coingecko_historical_volume(coin_id: str, days: int = 30) -> float:
    """
    Fetch historical volume data from CoinGecko API.
    
    Args:
        coin_id: The CoinGecko coin ID
        days: Number of days of historical data to fetch
        
    Returns:
        Total volume over the specified period or 0 if error
    """
    global cg_last_request_time
    
    url = f"{COINGECKO_API_BASE}/coins/{coin_id}/market_chart"
    response = await make_api_request(
        url=url,
        params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
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