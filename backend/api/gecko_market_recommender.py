import os
import json
import asyncio
import aiohttp
import aiofiles
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
# Assuming DriftPy and related types are available
# from driftpy.drift_client import DriftClient
# from driftpy.constants.numeric_constants import * # Example
# from driftpy.types import MarketType, OraclePriceData # etc.

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Add FileHandler if needed
)
logger = logging.getLogger("market_data_engine")

MARKET_DATA_FILE = Path("../../cache/market_data.json") # Store in project root's cache directory

# Reuse configurations from the original script (API keys, URLs, rate limits, etc.)
COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_DEMO_API_KEY = "CG-oWyNSQuvyZMKCDzL3yqGzyrh"  # Demo key that worked in the example
COINGECKO_REQ_PER_MINUTE = 30  # Conservative rate limit
COINGECKO_RATE_LIMIT_DELAY = 60.0 / COINGECKO_REQ_PER_MINUTE
COINGECKO_MARKETS_PER_PAGE = 1  # Starting with 1 to ensure it works
COINGECKO_NUM_PAGES = 1  # Start with 1 page to verify

DRIFT_DATA_API_BASE_URL = os.environ.get("DRIFT_DATA_API_BASE_URL")
DRIFT_DATA_API_HEADERS = json.loads(os.environ.get("DRIFT_DATA_API_HEADERS", '{}'))
DRIFT_API_RATE_LIMIT_INTERVAL = 0.1 # seconds

# Reuse SCORE_CUTOFFS and other scoring params from original script
SCORE_CUTOFFS = {
    'Market Cap Score': {
        'coingecko_mc_derived': {'kind': 'exp', 'start': 1_000_000, 'end': 5_000_000_000, 'steps': 20},
    },
    'Global Vol Score': {
        'coingecko_global_volume_30d_avg': {'kind': 'exp', 'start': 10_000, 'end': 1_000_000_000, 'steps': 20},
    },
    'Drift Activity Score': {
        'drift_volume_30d': {'kind': 'exp', 'start': 1_000, 'end': 500_000_000, 'steps': 10},
        'drift_open_interest': {'kind': 'exp', 'start': 1_000, 'end': 500_000_000, 'steps': 10},
    },
}

# Score boundaries for recommendations
SCORE_UB = {0: 62, 3: 75, 5: 85, 10: 101}  # Upper Bound: If score >= this, consider increasing leverage
SCORE_LB = {0: 0, 5: 31, 10: 48, 20: 60}   # Lower Bound: If score < this, consider decreasing leverage/delisting

# --- Global Rate Limiter Variables ---
cg_rate_limit_lock = asyncio.Lock()
cg_last_request_time = 0.0
drift_rate_limit_lock = asyncio.Lock()
drift_last_request_time = 0.0

# --- Utility Functions ---

async def load_market_data(filepath: Path) -> list:
    """Loads the market data list from the JSON file."""
    if not filepath.exists():
        logger.warning(f"{filepath} not found. Returning empty list.")
        return []
    try:
        async with aiofiles.open(filepath, mode='r') as f:
            content = await f.read()
            data = json.loads(content)
            if not isinstance(data, list):
                logger.error(f"Data in {filepath} is not a list. Returning empty list.")
                return []
            logger.info(f"Successfully loaded {len(data)} records from {filepath}")
            return data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {filepath}. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}", exc_info=True)
        return []

async def save_market_data(data: list, filepath: Path):
    """Saves the updated market data list to the JSON file."""
    try:
        # Ensure parent directory exists if filepath includes directories
        filepath.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(filepath, mode='w') as f:
            await f.write(json.dumps(data, indent=4))
        logger.info(f"Successfully saved {len(data)} records to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}", exc_info=True)

def get_default_market_structure(symbol: str) -> dict:
    """Returns a default nested dictionary structure for a new symbol."""
    # Define the structure based on the final JSON format
    return {
        "symbol": symbol.upper(),
        "coingecko_data": {
            "coingecko_name": None, "coingecko_id": None, "coingecko_image_url": None,
            "coingecko_current_price": None, "coingecko_market_cap_rank": None,
            "coingecko_market_cap": None, "coingecko_fully_diluted_valuation": None,
            "coingecko_total_volume_24h": None, "coingecko_global_volume_30d_avg": None,
            "coingecko_mc_derived": None, "coingecko_circulating_supply": None,
            "coingecko_total_supply": None, "coingecko_max_supply": None,
            "coingecko_ath_price": None, "coingecko_ath_change_percentage": None
        },
        "drift_data": {
            "drift_is_listed_spot": False, "drift_is_listed_perp": False,
            "drift_spot_market": None, "drift_perp_market": None,
            "drift_oracle_price": None, "drift_volume_30d": None,
            "drift_max_leverage": None, "drift_open_interest": None,
            "drift_funding_rate_1h": None
        },
        "scoring": {
            "scoring_overall_score": 0.0, "scoring_market_cap_score": 0.0,
            "scoring_global_vol_score": 0.0, "scoring_drift_activity_score": 0.0,
            "scoring_partial_mc": 0.0, "scoring_partial_global_volume": 0.0,
            "scoring_partial_volume_on_drift": 0.0, "scoring_partial_oi_on_drift": 0.0
        },
        "recommendation": "Monitor" # Default recommendation
    }

# --- Data Fetching Functions (Async) ---

async def fetch_coingecko(session: aiohttp.ClientSession, endpoint: str, params: Dict = None) -> Optional[Union[List, Dict]]:
    """Generic CoinGecko fetcher that exactly matches the successful example."""
    global cg_last_request_time
    
    if params is None:
        params = {}
    
    url = f"{COINGECKO_API_BASE_URL}{endpoint}"
    
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


async def fetch_all_coingecko_market_data(session: aiohttp.ClientSession) -> dict:
    """Fetches data from CG /coins/markets using the exact parameters that worked."""
    all_markets_data = {}
    logger.info(f"Fetching market data from CoinGecko ({COINGECKO_NUM_PAGES} pages with {COINGECKO_MARKETS_PER_PAGE} markets per page)...")
    
    try:
        for page in range(1, COINGECKO_NUM_PAGES + 1):
            logger.info(f"Fetching CoinGecko markets page {page}/{COINGECKO_NUM_PAGES}")
            
            # Exact parameters from the successful example
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': COINGECKO_MARKETS_PER_PAGE,
                'precision': '2',
                'page': page
            }
            
            # Make API request
            endpoint = '/coins/markets'
            data = await fetch_coingecko(session, endpoint, params)
            
            if not data or not isinstance(data, list):
                logger.warning(f"No valid data received for page {page}. Skipping.")
                continue
            
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
            
            # Respect rate limiting between page requests
            if page < COINGECKO_NUM_PAGES:
                await asyncio.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        logger.info(f"Successfully fetched data for {len(all_markets_data)} markets from CoinGecko")
        return all_markets_data
    except Exception as e:
        logger.error(f"Error fetching CoinGecko market data: {e}", exc_info=True)
        return all_markets_data


async def fetch_coin_volume(session: aiohttp.ClientSession, coin_id: str) -> Optional[float]:
    """Helper to fetch 30d volume data for a single coin."""
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
        
        # Calculate average
        avg_volume = sum(volumes) / len(volumes)
        logger.info(f"Calculated 30d avg volume for {coin_id}: ${avg_volume:,.2f}")
        return avg_volume
    
    except Exception as e:
        logger.error(f"Error fetching volume data for {coin_id}: {e}")
        raise  # Re-raise to be caught by gather

async def fetch_all_coingecko_volumes(session: aiohttp.ClientSession, coin_ids: list) -> dict:
    """Fetches 30d volume data from CG /coins/{id}/market_chart and calculates average."""
    volume_averages = {}  # Dict: coingecko_id -> avg_30d_volume
    logger.info(f"Fetching 30d volume data from CoinGecko for {len(coin_ids)} coins...")
    
    # Process coins one by one to avoid rate limiting issues
    for coin_id in coin_ids:
        try:
            avg_volume = await fetch_coin_volume(session, coin_id)
            if avg_volume is not None:
                volume_averages[coin_id] = avg_volume
                logger.info(f"Processed 30d volume for {coin_id}: ${avg_volume:,.2f}")
            
            # Respect rate limiting between requests
            await asyncio.sleep(COINGECKO_RATE_LIMIT_DELAY)
        
        except Exception as e:
            logger.warning(f"Error processing volume for {coin_id}: {str(e)}")
            continue
    
    logger.info(f"Successfully fetched 30d volume data for {len(volume_averages)}/{len(coin_ids)} coins")
    return volume_averages


async def fetch_drift_sdk_data(drift_client) -> dict:
    """Fetches current on-chain data using DriftPy SDK."""
    drift_sdk_metrics = {} # Dict: symbol -> sdk_data_dict
    logger.info("Fetching data from Drift SDK...")
    if not drift_client: # or not drift_client.is_initialized(): # Add checks
        logger.error("Drift client not available or initialized.")
        return {}

    try:
        # Placeholder implementation for Drift SDK data
        logger.info("Using placeholder data for Drift SDK metrics")
        
        # Example data for BTC
        drift_sdk_metrics['BTC'] = {
            'is_listed_perp': True,
            'perp_market': 'BTC-PERP',
            'oracle_price': 93500.0,
            'max_leverage': 10.0,
            'funding_rate_1h': 0.001,
            'is_listed_spot': True,
            'spot_market': 'SOL'
        }
        
        # Example data for ETH
        drift_sdk_metrics['ETH'] = { 
            'is_listed_perp': True, 
            'perp_market': 'ETH-PERP', 
            'oracle_price': 5000.0,
            'max_leverage': 20.0,
            'funding_rate_1h': 0.0015,
            'is_listed_spot': False,
            'spot_market': None
        }
        
    except Exception as e:
        logger.error(f"Error fetching Drift SDK data: {e}", exc_info=True)
    
    return drift_sdk_metrics


async def fetch_drift_api_data(session: aiohttp.ClientSession, symbols_to_fetch: list) -> dict:
    """Fetches historical/aggregated data like 30d vol and OI from Drift Data API."""
    drift_api_metrics = {} # Dict: symbol -> api_data_dict
    logger.info(f"Fetching 30d volume/OI from Drift Data API for {len(symbols_to_fetch)} symbols...")

    # Placeholder implementation for Drift API data
    logger.info("Using placeholder data for Drift API metrics")
    
    # Example data for BTC
    drift_api_metrics['BTC'] = {
        'volume_30d': 200000000.0,
        'open_interest': 150000000.0
    }
    
    # Example data for ETH
    drift_api_metrics['ETH'] = {
        'volume_30d': 100000000.0,
        'open_interest': 70000000.0
    }
    
    return drift_api_metrics


# --- Data Update/Merge Function ---

def update_market_data(existing_data: list, fetched_cg_market: dict, fetched_cg_volume: dict, fetched_drift_sdk: dict, fetched_drift_api: dict) -> list:
    """Updates the existing market data list with fetched data."""
    logger.info("Updating market data with fetched results...")
    updated_data_map = {item['symbol']: item for item in existing_data}
    all_symbols = set(updated_data_map.keys())

    # --- Integrate CoinGecko Market Data ---
    cg_ids_processed = set()
    for cg_id, cg_data in fetched_cg_market.items():
        # --- Use clean_symbol() logic if needed, map cg_id to symbol ---
        symbol = cg_data.get('symbol','').upper() # Simplistic mapping for outline
        if not symbol: continue

        all_symbols.add(symbol)
        cg_ids_processed.add(cg_id)
        if symbol not in updated_data_map:
            updated_data_map[symbol] = get_default_market_structure(symbol)

        # Update coingecko_data section
        target = updated_data_map[symbol]['coingecko_data']
        target['coingecko_name'] = cg_data.get('name')
        target['coingecko_id'] = cg_id
        target['coingecko_image_url'] = cg_data.get('image_url')
        target['coingecko_current_price'] = cg_data.get('current_price')
        target['coingecko_market_cap_rank'] = cg_data.get('market_cap_rank')
        target['coingecko_market_cap'] = cg_data.get('market_cap')
        target['coingecko_fully_diluted_valuation'] = cg_data.get('fully_diluted_valuation')
        target['coingecko_total_volume_24h'] = cg_data.get('total_volume_24h')
        target['coingecko_circulating_supply'] = cg_data.get('circulating_supply')
        target['coingecko_total_supply'] = cg_data.get('total_supply')
        target['coingecko_max_supply'] = cg_data.get('max_supply')
        target['coingecko_ath_price'] = cg_data.get('ath_price')
        target['coingecko_ath_change_percentage'] = cg_data.get('ath_change_percentage')

        # Calculate derived MC (example)
        mc = target['coingecko_market_cap'] or 0
        fdv = target['coingecko_fully_diluted_valuation'] or 0
        target['coingecko_mc_derived'] = max(mc, fdv) if (mc or fdv) else None


    # --- Integrate CoinGecko Volume Data ---
    for cg_id, avg_vol in fetched_cg_volume.items():
        # Find corresponding symbol(s) - might need a better mapping
        found = False
        for symbol, data in updated_data_map.items():
            if data['coingecko_data'].get('coingecko_id') == cg_id:
                 data['coingecko_data']['coingecko_global_volume_30d_avg'] = avg_vol
                 found = True
                 break # Assume 1-1 mapping for now
        # if not found: logger.warning(f"No symbol found for CG volume ID: {cg_id}") # Causes too much noise initially


    # --- Integrate Drift SDK Data ---
    for symbol, sdk_data in fetched_drift_sdk.items():
        symbol_upper = symbol.upper()
        all_symbols.add(symbol_upper)
        if symbol_upper not in updated_data_map:
             updated_data_map[symbol_upper] = get_default_market_structure(symbol_upper)

        target = updated_data_map[symbol_upper]['drift_data']
        target['drift_is_listed_spot'] = sdk_data.get('is_listed_spot', False)
        target['drift_is_listed_perp'] = sdk_data.get('is_listed_perp', False)
        target['drift_spot_market'] = sdk_data.get('spot_market')
        target['drift_perp_market'] = sdk_data.get('perp_market')
        target['drift_oracle_price'] = sdk_data.get('oracle_price')
        target['drift_max_leverage'] = sdk_data.get('max_leverage')
        target['drift_funding_rate_1h'] = sdk_data.get('funding_rate_1h')
        # Note: OI not updated here, assuming it comes from API


    # --- Integrate Drift API Data (Volume, OI) ---
    for symbol, api_data in fetched_drift_api.items():
         symbol_upper = symbol.upper()
         if symbol_upper in updated_data_map: # Only update if symbol exists
              target = updated_data_map[symbol_upper]['drift_data']
              target['drift_volume_30d'] = api_data.get('volume_30d')
              target['drift_open_interest'] = api_data.get('open_interest')
         # else: logger.warning(f"Symbol {symbol_upper} from Drift API not found in main map.") # Could happen if CG data missing


    # --- Ensure all symbols processed have a structure ---
    final_data_list = []
    for symbol in sorted(list(all_symbols)):
         if symbol not in updated_data_map:
              logger.warning(f"Creating default structure for symbol {symbol} that wasn't fully processed.")
              updated_data_map[symbol] = get_default_market_structure(symbol)
         final_data_list.append(updated_data_map[symbol])


    logger.info(f"Finished updating data. Total symbols: {len(final_data_list)}")
    return final_data_list


# --- Scoring Function ---

def calculate_scores(market_data: list) -> list:
    """Calculates scores based on the data in the nested structure."""
    logger.info(f"Calculating scores for {len(market_data)} markets...")

    for market in market_data:
        symbol = market['symbol']
        logger.debug(f"Scoring symbol: {symbol}")
        scores = market['scoring']  # Get the scoring sub-dict
        cg_data = market['coingecko_data']
        drift_data = market['drift_data']

        # Reset scores before calculation
        scores['scoring_overall_score'] = 0.0
        scores['scoring_market_cap_score'] = 0.0
        scores['scoring_global_vol_score'] = 0.0
        scores['scoring_drift_activity_score'] = 0.0
        scores['scoring_partial_mc'] = 0.0
        scores['scoring_partial_global_volume'] = 0.0
        scores['scoring_partial_volume_on_drift'] = 0.0
        scores['scoring_partial_oi_on_drift'] = 0.0

        try:
            # Market Cap Score Calculation
            mc_metric = cg_data.get('coingecko_mc_derived') or 0
            mc_config = SCORE_CUTOFFS.get('Market Cap Score', {}).get('coingecko_mc_derived', {})
            if mc_config and mc_metric > 0:
                partial_mc_score = calculate_partial_score(mc_metric, mc_config)
                scores['scoring_partial_mc'] = partial_mc_score
                scores['scoring_market_cap_score'] += partial_mc_score

            # Global Volume Score Calculation
            vol_30d_avg_metric = cg_data.get('coingecko_global_volume_30d_avg') or 0
            vol_config = SCORE_CUTOFFS.get('Global Vol Score', {}).get('coingecko_global_volume_30d_avg', {})
            if vol_config and vol_30d_avg_metric > 0:
                partial_global_vol_score = calculate_partial_score(vol_30d_avg_metric, vol_config)
                scores['scoring_partial_global_volume'] = partial_global_vol_score
                scores['scoring_global_vol_score'] += partial_global_vol_score

            # Drift Volume Score Calculation
            drift_vol_metric = drift_data.get('drift_volume_30d') or 0
            drift_vol_config = SCORE_CUTOFFS.get('Drift Activity Score', {}).get('drift_volume_30d', {})
            if drift_vol_config and drift_vol_metric > 0:
                partial_drift_vol_score = calculate_partial_score(drift_vol_metric, drift_vol_config)
                scores['scoring_partial_volume_on_drift'] = partial_drift_vol_score
                scores['scoring_drift_activity_score'] += partial_drift_vol_score

            # Drift OI Score Calculation
            drift_oi_metric = drift_data.get('drift_open_interest') or 0
            drift_oi_config = SCORE_CUTOFFS.get('Drift Activity Score', {}).get('drift_open_interest', {})
            if drift_oi_config and drift_oi_metric > 0:
                partial_drift_oi_score = calculate_partial_score(drift_oi_metric, drift_oi_config)
                scores['scoring_partial_oi_on_drift'] = partial_drift_oi_score
                scores['scoring_drift_activity_score'] += partial_drift_oi_score

            # Calculate Overall Score
            scores['scoring_overall_score'] = (
                scores['scoring_market_cap_score'] +
                scores['scoring_global_vol_score'] +
                scores['scoring_drift_activity_score']
            )

            # Generate Recommendation
            market['recommendation'] = calculate_recommendation(
                scores['scoring_overall_score'],
                drift_data.get('drift_max_leverage'),
                drift_data.get('drift_is_listed_perp', False)
            )

        except Exception as e:
            logger.error(f"Error scoring symbol {symbol}: {e}", exc_info=True)
            # Keep scores at 0 if error occurs

    logger.info("Finished calculating scores.")
    return market_data  # Return the list with updated scoring sections

def calculate_partial_score(metric_value: float, config: dict) -> float:
    """Calculates score for a single metric based on its configuration."""
    if not metric_value or metric_value <= 0:
        return 0.0
        
    steps = config.get('steps', 10)
    start = config.get('start', 0)
    end = config.get('end', 1)
    kind = config.get('kind', 'exp')
    
    point_thresholds = []
    
    # Generate threshold points
    if kind == 'exp' and start > 0:
        # Exponential spacing
        ratio = end / start
        for k in range(steps + 1):
            threshold = start * (ratio ** (k / steps))
            point_thresholds.append((threshold, k))
    else:
        # Linear spacing as fallback
        for k in range(steps + 1):
            threshold = start + (end - start) * (k / steps)
            point_thresholds.append((threshold, k))
    
    # Find the highest threshold that the metric exceeds
    points = 0
    for threshold, pts in point_thresholds:
        if metric_value >= threshold:
            points = pts
    
    return float(points)

def calculate_recommendation(overall_score: float, max_leverage: Optional[float], is_listed: bool) -> str:
    """Determines recommendation based on overall score and current market status."""
    if not is_listed:
        # If not listed, decide if should be listed
        if overall_score >= SCORE_LB.get(5, 31):  # Use threshold for 5x leverage
            return "List"
        return "Monitor"
        
    # Token is already listed, determine if it should remain
    if max_leverage is None or max_leverage <= 0:
        return "Monitor"
        
    current_leverage = max_leverage
    
    # Find the largest key in SCORE_LB that is less than or equal to the current leverage
    lower_bound_key = 0
    for k in sorted(SCORE_LB.keys()):
        if k <= current_leverage:
            lower_bound_key = k
        else:
            break
    
    # Check if score is below lower bound
    if overall_score < SCORE_LB[lower_bound_key]:
        if current_leverage > 5:  # If leverage is higher than 5x and score is low
            return "Decrease Leverage"
        else:  # If leverage is 5x or lower and score is low
            return "Delist"
    
    # Check if score exceeds upper bound for potential leverage increase
    upper_bound_key = 0
    for k in sorted(SCORE_UB.keys()):
        if k <= current_leverage:
            upper_bound_key = k
        else:
            break
    
    if overall_score >= SCORE_UB[upper_bound_key]:
        return "Increase Leverage"
        
    # Default: keep as is
    return "Keep"


# --- Main Orchestration Function (Async) ---

async def main():
    """Main function to orchestrate loading, fetching, updating, scoring, and saving."""
    start_time = time.time()
    logger.info("=== Starting Market Data Engine Run ===")

    # Initialize Drift client if needed (commented out for now)
    drift_client = None
    logger.info("Drift Client setup placeholder - not initialized for this run.")

    # Load existing data
    market_data = await load_market_data(MARKET_DATA_FILE)
    symbols_in_file = {item['symbol'] for item in market_data}
    logger.info(f"Found {len(symbols_in_file)} symbols in existing file.")

    # Fetch data concurrently
    async with aiohttp.ClientSession() as session:
        # Step 1: Fetch CoinGecko market data
        logger.info("Step 1: Fetching CoinGecko market data...")
        fetched_cg_market = await fetch_all_coingecko_market_data(session)
        
        if not fetched_cg_market:
            logger.error("Failed to fetch CoinGecko market data. Aborting.")
            return
            
        logger.info(f"Successfully fetched data for {len(fetched_cg_market)} markets from CoinGecko")
        
        # Step 2: Prepare list of CoinGecko IDs for volume fetching
        coingecko_ids_to_fetch_volume = list(fetched_cg_market.keys())
        if len(coingecko_ids_to_fetch_volume) > 1:  # Limit to 1 to avoid rate limits while testing
            logger.info(f"Limiting volume fetching to 1 coin out of {len(coingecko_ids_to_fetch_volume)}")
            coingecko_ids_to_fetch_volume = coingecko_ids_to_fetch_volume[:1]
        
        logger.info(f"Step 2: Fetching volumes for {len(coingecko_ids_to_fetch_volume)} CoinGecko IDs...")
        fetched_cg_volume = await fetch_all_coingecko_volumes(session, coingecko_ids_to_fetch_volume)
        
        # Step 3: Fetch Drift SDK data (placeholder)
        logger.info("Step 3: Fetching Drift SDK data (placeholder)...")
        fetched_drift_sdk = await fetch_drift_sdk_data(drift_client)
        
        # Step 4: Fetch Drift API data (placeholder)
        logger.info("Step 4: Fetching Drift API data (placeholder)...")
        # In a real implementation, we would extract symbols from fetched_cg_market to pass to this function
        drift_perp_symbols_to_fetch = ['BTC-PERP', 'ETH-PERP']  # Placeholder example
        fetched_drift_api = await fetch_drift_api_data(session, drift_perp_symbols_to_fetch)

    # Step 5: Update data structure
    logger.info("Step 5: Updating market data with fetched results...")
    market_data = update_market_data(
        market_data,
        fetched_cg_market or {},
        fetched_cg_volume or {},
        fetched_drift_sdk or {},
        fetched_drift_api or {}
    )

    # Step 6: Calculate scores
    logger.info("Step 6: Calculating scores and recommendations...")
    market_data = calculate_scores(market_data)

    # Step 7: Save updated data
    logger.info(f"Step 7: Saving {len(market_data)} records to {MARKET_DATA_FILE}...")
    await save_market_data(market_data, MARKET_DATA_FILE)

    end_time = time.time()
    logger.info(f"=== Market Data Engine Run Finished in {end_time - start_time:.2f} seconds ===")
    logger.info(f"=== Saved {len(market_data)} records to {MARKET_DATA_FILE} ===")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Run interrupted by user.")
    except Exception as e:
        logger.critical(f"Unhandled error in main execution: {e}", exc_info=True)