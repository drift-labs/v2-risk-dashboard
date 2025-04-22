# Extend from build_delisting_recos_v2.1.py

import os
import math
import sys
import json
import time
import asyncio
import aiofiles
import requests
import logging
import random
import traceback
from typing import Dict, Optional, Any, List, Tuple, Set, Union
from pathlib import Path
from datetime import datetime, timedelta

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
import aiohttp

from backend.state import BackendRequest
# Add driftpy imports
from driftpy.pickle.vat import Vat
from driftpy.drift_client import DriftClient
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.types import MarketType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('list_recommender.log')
    ]
)
logger = logging.getLogger("backend.api.list_recommender")

# Create FastAPI router
router = APIRouter()

# --- Configuration Constants ---
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
DAYS_TO_CONSIDER = 30

OUTPUT_COLS = [
    'Symbol', 'Max Lev.', 'Strict', 'Recommendation',
    'Market Cap Score', 'Spot Vol Score', 'Futures Vol Score',
    'Activity Score', 'Liquidity Score', 'Score',
    'MC $m', 'Spot Volume $m', 'Spot Vol Geomean $m',
    'Fut Volume $m', 'Fut Vol Geomean $m', 'OI $m',
    'Volume $m', 'Slip. $3k', 'Slip. $30k'
]

# Score boundaries
SCORE_UB = {0: 62, 3: 75, 5: 85, 10: 101}
SCORE_LB = {0: 0, 3: 37, 5: 48, 10: 60}

# Reference exchanges
REFERENCE_SPOT_EXCH = {
    'binance', 'bybit', 'okx', 'gate', 'kucoin', 'mexc',
    'coinbase', 'kraken'
}
REFERENCE_FUT_EXCH = {
    'bybit', 'binance', 'gate', 'mexc', 'okx',
    'htx', 'krakenfutures', 'bitmex'
}

# Strict tokens (score boost)
STRICT_TOKENS = {'PURR', 'CATBAL', 'HFUN', 'PIP', 'JEFF', 'VAPOR', 'SOLV',
                 'FARM', 'ATEHUN', 'SCHIZO', 'OMNIX', 'POINTS', 'RAGE'}
STRICT_BOOST = 5

# --- Add Drift-specific Constants ---
# Drift API Configuration
DRIFT_DATA_API_BASE_URL = os.environ.get("DRIFT_DATA_API_BASE_URL") # DO NOT MODIFY THIS
DRIFT_DATA_API_HEADERS = json.loads(os.environ.get("DRIFT_DATA_API_HEADERS", "{}")) # DO NOT MODIFY THIS
API_RATE_LIMIT_INTERVAL = 0.1  # seconds between requests

# Drift Score Boost - These are symbols that get a score boost in the delist recommender
DRIFT_SCORE_BOOST_SYMBOLS = {}

# Drift Score Boost Amount - The amount of score boost to apply to the symbols in DRIFT_SCORE_BOOST_SYMBOLS
DRIFT_SCORE_BOOST_AMOUNT = 10

# Global rate limiter variables for API calls
rate_limit_lock = asyncio.Lock()
last_request_time = 0.0

# Prediction Market Symbols to ignore completely during analysis
IGNORED_SYMBOLS = {
    "TRUMP-WIN-2024-BET",
    "KAMALA-POPULAR-VOTE-2024-BET",
    "FED-CUT-50-SEPT-2024-BET",
    "REPUBLICAN-POPULAR-AND-WIN-BET",
    "BREAKPOINT-IGGYERIC-BET",
    "DEMOCRATS-WIN-MICHIGAN-BET",
    "LANDO-F1-SGP-WIN-BET",
    "WARWICK-FIGHT-WIN-BET",
    "WLF-5B-1W-BET",
    "VRSTPN-WIN-F1-24-DRVRS-CHMP",
    "LNDO-WIN-F1-24-US-GP",
    "SUPERBOWL-LIX-LIONS-BET",
    "SUPERBOWL-LIX-CHIEFS-BET",
    "NBAFINALS25-OKC-BET",
    "NBAFINALS25-BOS-BET",
}

# Drift-specific score cutoffs - simplified for delisting focus
# Note: Inputs to scoring ('MC', 'Spot Volume', etc.) are expected in full dollar amounts, not millions.
DRIFT_SCORE_CUTOFFS = {
    'Market Cap Score': {
        'MC': {'kind': 'exp', 'start': 1_000_000, 'end': 5_000_000_000, 'steps': 20}, # $1M to $5B
    },
    'Spot Vol Score': {
        # Expects 'Spot Volume' (sum of avg daily vol) and 'Spot Vol Geomean' (geomean of top 3 avg daily vol) in full dollars
        'Spot Volume': {'kind': 'exp', 'start': 10_000, 'end': 1_000_000_000, 'steps': 10}, # $10k to $1B
        'Spot Vol Geomean': {'kind': 'exp', 'start': 10_000, 'end': 1_000_000_000, 'steps': 10}, # $10k to $1B
    },
    'Futures Vol Score': {
         # Expects 'Fut Volume' (sum of avg daily vol) and 'Fut Vol Geomean' (geomean of top 3 avg daily vol) in full dollars
        'Fut Volume': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10}, # $10k to $1B
        'Fut Vol Geomean': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10}, # $10k to $1B
    },
    'Drift Activity Score': {
        # Expects 'Volume on Drift' (estimated 30d vol) and 'OI on Drift' in full dollars
        'Volume on Drift': {'kind': 'exp', 'start': 1_000, 'end': 500_000_000, 'steps': 10}, # $1k to $500M
        'OI on Drift': {'kind': 'exp', 'start': 1_000, 'end': 500_000_000, 'steps': 10}, # $1k to $500M
    },
}

# Market-specific base decimals overrides based on known values
MARKET_BASE_DECIMALS = {
    0: 9,  # SOL-PERP
    1: 6,  # BTC-PERP - likely 6 decimals instead of 9 based on expected OI
    2: 9,  # ETH-PERP
}

# Score cutoffs
SCORE_CUTOFFS = {
    'Market Cap Score': {
        'MC $m': {'kind': 'exp', 'start': 1, 'end': 5000, 'steps': 20},
    },
    'Spot Vol Score': {
        'Spot Volume $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10},
        'Spot Vol Geomean $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10},
    },
    'Futures Vol Score': {
        'Fut Volume $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10},
        'Fut Vol Geomean $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10},
    },
    'Activity Score': {
        'Volume $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10},
        'OI $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10},
    },
    'Liquidity Score': {
        'Slip. $3k': {'kind': 'linear', 'start': 5, 'end': 0, 'steps': 5},
        'Slip. $30k': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5},
    }
}

# Cache settings
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds
CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)

# Global variables for rate limiting and progress tracking
exchange_rate_limits = {}
completed_items_tracker = {}
last_recommendation_data = None
last_recommendation_time = 0

# Calculate earliest timestamp to keep data for
earliest_ts_to_keep = time.time() - (DAYS_TO_CONSIDER + 5) * 24 * 60 * 60

# --- Utility Functions ---
def sig_figs(number, sig_figs=3):
    """Round a number to specified significant figures."""
    if np.isnan(number) or number <= 0:
        return 0
    return round(number, int(sig_figs - 1 - math.log10(number)))

def clean_symbol(symbol, exch=''):
    """Clean and standardize cryptocurrency symbol names."""
    TOKEN_ALIASES = {
        'HPOS10I': 'BITCOIN', 'HPOS': 'HPOS', 'HPO': 'HPOS',
        'BITCOIN': 'HPOS', 'NEIROCTO': 'NEIRO', '1MCHEEMS': 'CHEEMS',
        '1MBABYDOGE': 'BABYDOGE', 'JELLYJELLY': 'JELLY'
    }
    EXCH_TOKEN_ALIASES = {
        ('NEIRO', 'bybit'): 'NEIROETH',
        ('NEIRO', 'gate'): 'NEIROETH',
        ('NEIRO', 'kucoin'): 'NEIROETH'
    }
    redone = symbol.split('/')[0]
    for suffix in ['10000000', '1000000', '1000', 'k']:
        redone = redone.replace(suffix, '')
    redone = EXCH_TOKEN_ALIASES.get((redone, exch), redone)
    return TOKEN_ALIASES.get(redone, redone)

def safe_count(obj) -> int:
    """Safely count items in an object that may not directly support len()"""
    if obj is None:
        return 0
    
    # If object has values() method, count items through that
    if hasattr(obj, 'values'):
        return sum(1 for _ in obj.values())
    
    # If object is iterable but not a string, try counting its items
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        try:
            return sum(1 for _ in obj)
        except Exception:
            pass
    
    # Last resort - try len, but catch errors
    try:
        return len(obj)
    except Exception:
        return 0

def ensure_data_dir():
    """Ensure data directory exists for storing exchange data."""
    data_dir = Path('exchange_data')
    data_dir.mkdir(exist_ok=True)
    return data_dir

def get_cache_path(exch: str, spot: bool) -> Path:
    """Get the path for cached exchange data file."""
    return CACHE_DIR / f'exch_candles_{exch}_{"s" if spot else "f"}.json'

def geomean_three(series):
    """Calculate geometric mean of top 3 values in a series."""
    return np.exp(np.log(series+1).sort_values()[-3:].sum()/3) - 1

def adjust_rate_limits(exchange_name, failure=False):
    """Dynamically adjust rate limits based on success/failure."""
    if failure:
        # Increase failure count and adjust limits
        exchange_rate_limits[exchange_name]['failures'] += 1
        if exchange_rate_limits[exchange_name]['failures'] >= 3:
            # Reduce concurrency and increase delay
            current_value = exchange_rate_limits[exchange_name]['semaphore']._value
            if current_value > 1:
                exchange_rate_limits[exchange_name]['semaphore'] = asyncio.Semaphore(current_value - 1)
            exchange_rate_limits[exchange_name]['delay'] *= 1.5
            exchange_rate_limits[exchange_name]['failures'] = 0
            logger.info(f"Reducing request rate for {exchange_name}: {current_value-1} concurrent, {exchange_rate_limits[exchange_name]['delay']:.2f}s delay")
    else:
        # Successful request - slowly recover if we've been too conservative
        exchange_rate_limits[exchange_name]['failures'] = max(0, exchange_rate_limits[exchange_name]['failures'] - 0.2)
        if exchange_rate_limits[exchange_name]['failures'] < 0.5 and random.random() < 0.05:
            # Occasionally try faster rates (only 5% of the time to avoid oscillation)
            exchange_rate_limits[exchange_name]['delay'] = max(0.2, exchange_rate_limits[exchange_name]['delay'] * 0.9)

async def retry_with_backoff_async(func, max_retries=3, initial_delay=1, on_failure=None):
    """Retry an async function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            if on_failure:
                on_failure()
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

async def log_periodic_progress(task_name, total_items, completed_items, interval=30):
    """Log progress periodically to show the script is still running."""
    logger.info(f"{task_name}: Processed {completed_items}/{total_items} ({(completed_items/total_items)*100:.1f}%)")
    if completed_items < total_items:
        await asyncio.sleep(interval)
        # Get the latest count from the tracker
        current_count = completed_items_tracker[task_name]
        # Schedule another progress update
        asyncio.create_task(log_periodic_progress(task_name, total_items, current_count, interval))

async def close_exchange(api):
    """Safely close an asynchronous exchange connection."""
    if api:
        try:
            await api.close()
        except Exception as e:
            logger.error(f"Error closing exchange: {str(e)}")

async def get_hot_ccxt_async_api(exch: str) -> Optional[ccxt_async.Exchange]:
    """Initialize and test an asynchronous CCXT exchange API with proper error handling."""
    try:
        api = getattr(ccxt_async, exch)()
        logger.info(f"Initializing {exch} exchange asynchronously...")
        await api.load_markets()
        return api
    except Exception as e:
        logger.error(f"Failed to initialize {exch} exchange asynchronously: {str(e)}")
        return None

# --- Data Fetching Functions ---
async def load_market_data(exch: str, spot: bool, market: str) -> Optional[List]:
    """Load market data from its own file asynchronously."""
    data_dir = ensure_data_dir() / exch / ("spot" if spot else "futures")
    
    # Sanitize the market name for file paths
    safe_market = market.replace('/', '_').replace(':', '_')
    file_path = data_dir / f"{safe_market}.json"
    
    if not file_path.exists():
        return None
    
    try:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.warning(f"Failed to load market data for {exch} {market}: {str(e)}")
        return None

async def save_market_data(exch: str, spot: bool, market: str, data: List) -> None:
    """Save market data to its own file asynchronously."""
    if not data:
        return
    
    data_dir = ensure_data_dir() / exch / ("spot" if spot else "futures")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize the market name for file paths
    safe_market = market.replace('/', '_').replace(':', '_')
    file_path = data_dir / f"{safe_market}.json"
    
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(data))

async def fetch_market_data(api, market: str, semaphore, exchange_name: str) -> Tuple[str, List]:
    """Fetch OHLCV data for a single market with adaptive rate limiting."""
    global exchange_rate_limits
    
    # Initialize rate limiting parameters if needed
    if exchange_name not in exchange_rate_limits:
        exchange_rate_limits[exchange_name] = {
            'semaphore': asyncio.Semaphore(5),  # Start with default limit
            'delay': 0.2,                       # Initial delay between requests
            'failures': 0,                      # Count of recent failures
        }
    
    try:
        async with exchange_rate_limits[exchange_name]['semaphore']:
            # Dynamic delay based on recent failures
            await asyncio.sleep(exchange_rate_limits[exchange_name]['delay'])
            
            def fetch_ohlcv():
                return api.fetch_ohlcv(market, '1d')
            
            data = await retry_with_backoff_async(
                fetch_ohlcv, 
                max_retries=3, 
                on_failure=lambda: adjust_rate_limits(exchange_name, failure=True)
            )
            
            # Success - potentially reduce delay if we've been conservative
            adjust_rate_limits(exchange_name, failure=False)
            
            # Update progress tracker
            if exchange_name in completed_items_tracker:
                completed_items_tracker[exchange_name] += 1
            
            return market, data
    except Exception as e:
        logger.error(f"Error fetching {market} from {exchange_name}: {str(e)}")
        return market, []

async def download_exch_async(exch: str, spot: bool) -> Optional[Dict]:
    """Download exchange data asynchronously with caching."""
    global completed_items_tracker
    
    cache_path = get_cache_path(exch, spot)
    
    # Check if we have valid cached data (less than 24 hours old)
    if cache_path.exists():
        cache_mod_time = cache_path.stat().st_mtime
        cache_age = time.time() - cache_mod_time
        
        if cache_age < CACHE_DURATION:
            logger.info(f'Using cached data for {exch} {"spot" if spot else "futures"} (age: {cache_age/3600:.1f} hours)')
            try:
                with open(cache_path, 'r') as f:
                    return json.loads(f.read())
            except Exception as e:
                logger.warning(f'Failed to load cached data for {exch}: {str(e)}')
                # Continue to fetch fresh data
        else:
            logger.info(f'Cached data for {exch} is {cache_age/3600:.1f} hours old, fetching fresh data')
    else:
        logger.info(f'No cached data found for {exch}, fetching fresh data')

    logger.info(f'Downloading {exch} {"spot" if spot else "futures"} data asynchronously...')
    
    api = await get_hot_ccxt_async_api(exch)
    if api is None:
        return None

    exchange_data = {}
    try:
        # Filter markets that match our criteria
        eligible_markets = []
        for market in api.markets:
            if spot and ':' in market:
                continue
            if not spot and ':USD' not in market:
                continue
            if '/USD' not in market:
                continue
            if '-' in market:
                continue
            eligible_markets.append(market)
            
        total_markets = len(eligible_markets)
        logger.info(f"Found {total_markets} eligible markets for {exch}")
        
        # Initialize progress tracking for this exchange
        exchange_key = f"{exch}_{'spot' if spot else 'futures'}"
        completed_items_tracker[exchange_key] = 0
        
        # Start periodic progress logging
        if total_markets > 20:  # Only for exchanges with many markets
            asyncio.create_task(log_periodic_progress(
                exchange_key, 
                total_markets,
                completed_items_tracker[exchange_key],
                interval=30  # Log every 30 seconds
            ))
        
        # Create tasks for each market
        tasks = []
        for market in eligible_markets:
            # First try to load from individual file
            data = await load_market_data(exch, spot, market)
            if data:
                exchange_data[market] = data
                completed_items_tracker[exchange_key] += 1
                logger.info(f"Loaded existing data for {exch} {market}")
            else:
                # If not found, add to tasks to fetch
                task = fetch_market_data(api, market, None, exchange_key)
                tasks.append(task)
        
        # Execute all tasks in parallel and collect results
        if tasks:
            logger.info(f"Fetching {len(tasks)} markets for {exch} in parallel")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and save individual market data
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in market fetch for {exch}: {str(result)}")
                    continue
                    
                market, data = result
                if data:
                    exchange_data[market] = data
                    # Save market data individually
                    await save_market_data(exch, spot, market, data)
        
        # Save aggregated data to cache
        if exchange_data:
            with open(cache_path, 'w') as f:
                f.write(json.dumps(exchange_data))
                
        logger.info(f"Completed downloading data for {exch} {'spot' if spot else 'futures'}")
        return exchange_data
    
    except Exception as e:
        logger.error(f"Error downloading {exch} data: {str(e)}")
        return None
    finally:
        await close_exchange(api)

async def dl_reference_exch_data_async():
    """Download reference exchange data asynchronously with improved error handling and parallel processing."""
    raw_reference_exch_df = {}
    
    # Create tasks for all exchanges (both spot and futures)
    tasks = []
    
    for spot, exchs in {True: REFERENCE_SPOT_EXCH, False: REFERENCE_FUT_EXCH}.items():
        for exch in exchs:
            tasks.append((exch, spot, download_exch_async(exch, spot)))
    
    # Execute all tasks concurrently
    logger.info(f"Starting parallel download for {len(tasks)} exchange configurations")
    results = await asyncio.gather(*[task[2] for task in tasks], return_exceptions=True)
    
    # Process results
    for (exch, spot, _), result in zip(tasks, results):
        try:
            if isinstance(result, Exception):
                logger.error(f"Failed to download {exch} {'spot' if spot else 'futures'}: {str(result)}")
                continue
                
            if result:
                raw_reference_exch_df[spot, exch] = result
                logger.info(f"Successfully downloaded data for {exch} {'spot' if spot else 'futures'}")
        except Exception as e:
            logger.error(f"Error processing result for {exch}: {str(e)}")

    return raw_reference_exch_df

def get_hot_ccxt_api(exch: str) -> Optional[ccxt.Exchange]:
    """Initialize and test a CCXT exchange API with proper error handling."""
    try:
        api = getattr(ccxt, exch)()
        logger.info(f"Initializing {exch} exchange...")
        api.load_markets()
        return api
    except Exception as e:
        logger.error(f"Failed to initialize {exch} exchange: {str(e)}")
        return None

def process_reference_exch_data(raw_reference_exch_df):
    """Process reference exchange data with improved error handling."""
    logger.info("Processing reference exchange data...")
    all_candle_data = {}

    for (spot, exch), exch_data in raw_reference_exch_df.items():
        logger.info(f'Processing {exch} {("spot" if spot else "futures")}')
        api = get_hot_ccxt_api(exch)
        if api is None:
            continue
            
        for symbol, market in exch_data.items():
            try:
                coin = clean_symbol(symbol, exch)
                if not len(market):
                    continue
                market_df = (pd.DataFrame(market, columns=[*'tohlcv'])
                             .set_index('t').sort_index()
                             .loc[earliest_ts_to_keep*1000:]
                             .iloc[-DAYS_TO_CONSIDER-1:-1])
                if not len(market_df):
                    continue
                contractsize = min(api.markets.get(
                    symbol, {}).get('contractSize', None) or 1, 1)
                my_val = (np.minimum(market_df.l, market_df.c.iloc[-1]) 
                          * market_df.v).mean() * contractsize
                if my_val >= all_candle_data.get((exch, spot, coin), 0):
                    all_candle_data[exch, spot, coin] = my_val
            except Exception as e:
                logger.error(f"Error processing {symbol} from {exch}: {str(e)}")

    # Create DataFrame from candle data
    df_coins = pd.Series(all_candle_data).sort_values(ascending=False)
    df_coins.index.names = ['exch', 'spot', 'coin']
    
    # Group by spot/coin and calculate metrics
    grouped = df_coins.groupby(['spot', 'coin'])
    
    # Create a DataFrame for each metric
    volume_df = grouped.sum().unstack(0).fillna(0)
    volume_df.columns = [f"{'Spot' if b else 'Fut'} Volume $m" for b in volume_df.columns]
    
    geomean_df = grouped.agg(geomean_three).unstack(0).fillna(0)
    geomean_df.columns = [f"{'Spot' if b else 'Fut'} Vol Geomean $m" for b in geomean_df.columns]
    
    # Combine metrics
    output_df = pd.concat([volume_df, geomean_df], axis=1)
    
    # Convert to millions
    output_df = output_df / 1e6
    
    # Ensure consistent column order
    desired_cols = [
        'Spot Volume $m',
        'Spot Vol Geomean $m',
        'Fut Volume $m',
        'Fut Vol Geomean $m'
    ]
    
    # Add any missing columns with zeros
    for col in desired_cols:
        if col not in output_df.columns:
            output_df[col] = 0
    
    # Select and order columns
    output_df = output_df[desired_cols]
    
    logger.info(f"Processed exchange data for {len(output_df)} tokens")
    logger.debug(f"Exchange data columns: {output_df.columns.tolist()}")
    return output_df

async def dl_cmc_data_async():
    """Download CoinMarketCap data with retry logic asynchronously."""
    logger.info("Downloading CoinMarketCap data...")
    
    # Cache path for CMC data
    cmc_cache_path = CACHE_DIR / 'cmc_data.json'
    
    # Check if we have valid cached data
    if cmc_cache_path.exists():
        cache_mod_time = cmc_cache_path.stat().st_mtime
        cache_age = time.time() - cache_mod_time
        
        if cache_age < CACHE_DURATION:
            logger.info(f'Using cached CMC data (age: {cache_age/3600:.1f} hours)')
            try:
                with open(cmc_cache_path, 'r') as f:
                    return json.loads(f.read())
            except Exception as e:
                logger.warning(f'Failed to load cached CMC data: {str(e)}')
                # Continue to fetch fresh data
        else:
            logger.info(f'Cached CMC data is {cache_age/3600:.1f} hours old, fetching fresh data')
    else:
        logger.info('No cached CMC data found, fetching fresh data')
    
    # If API key is available as environment variable, use it; otherwise use a default value
    cmc_api_key = os.environ.get('CMC_API_KEY', '973f2ad5-6b18-4f01-a8d9-c1d50460ae3a')
    CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    
    # Symbol overrides
    CMC_SYMBOL_OVERRIDES = {
        'Neiro Ethereum': 'NEIROETH',
        'HarryPotterObamaSonic10Inu (ERC-20)': 'HPOS'
    }

    async def fetch_cmc():
        try:
            def request_task():
                response = requests.get(
                    f"{CMC_API_URL}?CMC_PRO_API_KEY={cmc_api_key}&limit=5000",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json().get('data', [])
                
                for item in data:
                    item['symbol'] = CMC_SYMBOL_OVERRIDES.get(item['name'], item['symbol'])
                    
                return data
                
            # Use asyncio.to_thread instead of TaskGroup
            return await asyncio.to_thread(request_task)
        except Exception as e:
            logger.error(f"Failed to fetch CMC data: {str(e)}")
            raise

    try:
        data = await retry_with_backoff_async(fetch_cmc)
        
        # Cache the fetched data
        with open(cmc_cache_path, 'w') as f:
            json.dump(data, f)
            
        return data
    except Exception as e:
        logger.error(f"Failed to fetch CMC data: {str(e)}")
        raise

def process_cmc_data(cmc_data):
    """Process CoinMarketCap data with error handling."""
    logger.info("Processing CoinMarketCap data...")
    try:
        # First process basic CMC data
        df_list = []
        for a in cmc_data:
            try:
                symbol = a['symbol']
                mc = a['quote']['USD'].get('market_cap', 0) or 0
                fd_mc = a['quote']['USD'].get('fully_diluted_market_cap', 0) or 0
                df_list.append({
                    'symbol': symbol,
                    'mc': mc,
                    'fd_mc': fd_mc,
                })
            except Exception as e:
                logger.debug(f"Error processing CMC entry: {str(e)}")
                continue

        # Create DataFrame and handle duplicates
        output_df = pd.DataFrame(df_list)
        if not output_df.empty:
            # Group by symbol and take max values
            output_df = output_df.groupby('symbol').agg({
                'mc': 'max',
                'fd_mc': 'max'
            })
            
            # Use fd_mc if mc is 0
            output_df.loc[output_df['mc']==0, 'mc'] = output_df['fd_mc']
            output_df['MC $m'] = output_df['mc'].fillna(0)/1e6
            
            logger.info(f"Processed {len(output_df)} tokens from CMC data")
            return output_df[['MC $m']]
        else:
            logger.warning("No valid CMC data to process")
            return pd.DataFrame(columns=['MC $m'])
    except Exception as e:
        logger.error(f"Error processing CMC data: {str(e)}")
        raise

async def dl_thunderhead_data_async():
    """Download Thunderhead data with retry logic asynchronously."""
    logger.info("Downloading Thunderhead data...")
    
    # Cache path for Thunderhead data
    thunder_cache_path = CACHE_DIR / 'thunderhead_data.json'
    
    # Check if we have valid cached data
    if thunder_cache_path.exists():
        cache_mod_time = thunder_cache_path.stat().st_mtime
        cache_age = time.time() - cache_mod_time
        
        if cache_age < CACHE_DURATION:
            logger.info(f'Using cached Thunderhead data (age: {cache_age/3600:.1f} hours)')
            try:
                with open(thunder_cache_path, 'r') as f:
                    return json.loads(f.read())
            except Exception as e:
                logger.warning(f'Failed to load cached Thunderhead data: {str(e)}')
                # Continue to fetch fresh data
        else:
            logger.info(f'Cached Thunderhead data is {cache_age/3600:.1f} hours old, fetching fresh data')
    else:
        logger.info('No cached Thunderhead data found, fetching fresh data')
    
    THUNDERHEAD_URL = "https://d2v1fiwobg9w6.cloudfront.net"
    THUNDERHEAD_HEADERS = {"accept": "*/*"}
    THUNDERHEAD_QUERIES = {
        'daily_usd_volume_by_coin',
        'total_volume',
        'asset_ctxs',
        'hlp_positions',
        'liquidity_by_coin'
    }

    raw_thunder_data = {}
    
    async def fetch_query(query):
        try:
            def request_task():
                response = requests.get(
                    f"{THUNDERHEAD_URL}/{query}",
                    headers=THUNDERHEAD_HEADERS,
                    allow_redirects=True
                )
                response.raise_for_status()
                return response.json().get('chart_data', [])
                
            # Use asyncio.to_thread instead of TaskGroup
            data = await asyncio.to_thread(request_task)
            return query, data
        except Exception as e:
            logger.error(f"Failed to fetch Thunderhead query {query}: {str(e)}")
            raise
    
    # Create tasks for each query
    tasks = [fetch_query(query) for query in THUNDERHEAD_QUERIES]
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch Thunderhead data: {str(result)}")
            continue
            
        query, data = result
        raw_thunder_data[query] = data
        logger.info(f"Successfully fetched Thunderhead {query}")
    
    # Cache the fetched data
    with open(thunder_cache_path, 'w') as f:
        json.dump(raw_thunder_data, f)
            
    return raw_thunder_data

def process_thunderhead_data(raw_thunder_data):
    """Process Thunderhead data with error handling."""
    logger.info("Processing Thunderhead data...")
    try:
        dfs = []
        for key, records in raw_thunder_data.items():
            if key == 'liquidity_by_coin':
                df = pd.DataFrame({
                    (entry['time'], coin): {**entry, 'time': 0}
                    for coin, entries in records.items()
                    for entry in entries
                }).T
                dfs.append(df)
            else:
                df = pd.DataFrame(records).set_index(['time', 'coin'])
                dfs.append(df)

        coin_time_df = pd.concat(dfs, axis=1).unstack(0)
        
        # Process futures data
        fut_data_df = coin_time_df.unstack().unstack(0)
        fut_data_df['avg_notional_oi'] = (fut_data_df['avg_oracle_px'] *
                                           fut_data_df['avg_open_interest'])

        # Safe processing of time series data
        try:
            fut_s_df = fut_data_df.unstack(1).sort_index().iloc[-30:].mean().unstack(0)
        except Exception as e:
            logger.warning(f"Failed to process time series data: {str(e)}")
            fut_s_df = pd.DataFrame()

        # Clean symbols safely and ensure unique indices
        try:
            fut_s_df.index = [clean_symbol(sym) for sym in fut_s_df.index]
            # Handle any duplicate indices after cleaning
            fut_s_df = fut_s_df.groupby(level=0).agg('mean')
        except Exception as e:
            logger.warning(f"Failed to clean symbols: {str(e)}")

        output_df = fut_s_df.fillna(0)
        
        # Calculate metrics with safe fallbacks
        try:
            # Round slippage values to basis points
            output_df['Slip. $3k'] = output_df['median_slippage_3000'] / 100_00
            output_df['Slip. $30k'] = output_df['median_slippage_30000'] / 100_00
        except Exception:
            output_df['Slip. $3k'] = 0
            output_df['Slip. $30k'] = 0

        # Safe metric calculations
        safe_metrics = {
            'OI $m': ('avg_notional_oi', 1e6),
            'Volume $m': ('total_volume', 1e6),
        }

        for metric, (source, factor) in safe_metrics.items():
            try:
                output_df[metric] = output_df[source] / factor
            except Exception:
                output_df[metric] = 0

        logger.info(f"Processed Thunderhead data for {len(output_df)} tokens")
        return output_df[['Slip. $3k', 'Slip. $30k', 'OI $m', 'Volume $m']]
    except Exception as e:
        logger.error(f"Error processing Thunderhead data: {str(e)}")
        raise

# --- Drift Data Fetching Functions ---
async def get_drift_data(drift_client, perp_map, user_map):
    """Fetches data from Drift Protocol using provided maps, excluding ignored symbols."""
    try:
        logger.info("==> Fetching Drift Protocol market data")

        # Initialize results list
        drift_data = []

        # Get all perp markets
        perp_markets = list(perp_map.values())
        markets_count = len(perp_markets)
        if not perp_markets:
            logger.error("==> No perp markets found in Drift data")
            return None

        logger.info(f"==> Found {markets_count} perp markets initially")

        # Track processed markets
        markets_processed = 0
        user_count = safe_count(user_map)
        logger.info(f"==> Processing {markets_count} markets across {user_count} users")

        # Track long and short positions separately for each market
        market_long_positions = {}  # Track sum of long positions
        market_short_positions = {}  # Track sum of short positions
        market_position_counts = {}

        # First pass - gather OI for all markets
        print(f"Aggregating positions from {user_count} users...")
        processed_users = 0
        for user_account in user_map.values():
            try:
                processed_users += 1
                if processed_users % 5000 == 0:
                    print(f"Processed {processed_users}/{user_count} users...")

                # Try different methods to access perp positions
                perp_positions = []

                # Method 1: get_active_perp_positions
                if hasattr(user_account, 'get_active_perp_positions'):
                    try:
                        perp_positions = user_account.get_active_perp_positions()
                    except Exception as e:
                        # Fallback to other methods
                        pass

                # Method 2: get_user_account.perp_positions
                if not perp_positions and hasattr(user_account, 'get_user_account'):
                    try:
                        user_data = user_account.get_user_account()
                        if hasattr(user_data, 'perp_positions'):
                            # Filter for active positions only
                            perp_positions = [pos for pos in user_data.perp_positions if pos.base_asset_amount != 0]
                    except Exception as e:
                        pass

                # Method 3: direct perp_positions attribute
                if not perp_positions and hasattr(user_account, 'perp_positions'):
                    try:
                        # Filter for active positions only
                        perp_positions = [pos for pos in user_account.perp_positions if pos.base_asset_amount != 0]
                    except Exception:
                        pass

                # Process each position
                for position in perp_positions:
                    if hasattr(position, 'market_index') and hasattr(position, 'base_asset_amount') and position.base_asset_amount != 0:
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

        print(f"Found positions in {len(market_long_positions)} markets")
        for market_idx in sorted(market_long_positions.keys()):
            print(
                f"Market {market_idx}: {market_position_counts[market_idx]} positions, "
                f"Long={market_long_positions[market_idx]}, Short={market_short_positions[market_idx]}"
            )

        # Collect symbols for volume calculation
        market_symbols = {}

        # Process each perp market
        ignored_count = 0
        for market in perp_markets:
            market_index = market.data.market_index

            try:
                # Get market config by index
                market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
                if not market_config:
                    logger.warning(f"==> No market config found for market index {market_index}")
                    continue

                # Get symbol
                symbol = market_config.symbol
                clean_sym = clean_symbol(symbol)

                # Skip if symbol is in the ignored list
                if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
                    logger.info(f"==> Skipping ignored market: {symbol} (Index: {market_index})")
                    ignored_count += 1
                    continue
                
                # Save symbol for volume calculation
                market_symbols[market_index] = symbol
                
                markets_processed += 1

                # Get max leverage (from initial margin ratio)
                initial_margin_ratio = market.data.margin_ratio_initial / 10000
                max_leverage = int(1 / initial_margin_ratio) if initial_margin_ratio > 0 else 0

                # Get oracle price
                oracle_price_data = drift_client.get_oracle_price_data_for_perp_market(market_index)
                oracle_price = oracle_price_data.price / 1e6  # Convert from UI price

                # Calculate OI as max(abs(long), abs(short))
                long_amount = market_long_positions.get(market_index, 0)
                short_amount = market_short_positions.get(market_index, 0)
                base_oi = max(long_amount, short_amount)
                positions_count = market_position_counts.get(market_index, 0)

                # Get base decimals - try market first, then known mappings, then fall back to constants
                base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)  # Use known mapping first

                try:
                    # Try to get decimals from market.data
                    if hasattr(market.data, 'base_decimals'):
                        base_decimals = market.data.base_decimals
                        logger.info(f"==> Using market.data.base_decimals={base_decimals} for market {market_index}")
                    # If not found, check if it's in market.data.amm
                    elif hasattr(market.data, 'amm') and hasattr(market.data.amm, 'base_asset_decimals'):
                        base_decimals = market.data.amm.base_asset_decimals
                        logger.info(f"==> Using market.data.amm.base_asset_decimals={base_decimals} for market {market_index}")
                    else:
                        logger.info(f"==> Using default base_decimals={base_decimals} for market {market_index}")
                except Exception as e:
                    # If any error occurs, use the known mapping or default value
                    logger.debug(f"==> Error getting base decimals for market {market_index}: {str(e)}")
                    logger.info(f"==> Falling back to default base_decimals={base_decimals} for market {market_index}")
                    pass

                # Convert to human readable base units and to full dollar amount (not millions)
                base_oi_readable = base_oi / (10 ** base_decimals)
                oi_usd = base_oi_readable * oracle_price  # Full dollars

                # Additional validation for known high-value markets like SOL-PERP
                if market_index == 0 or clean_sym == 'SOL':  # SOL-PERP
                    logger.warning(
                        f"==> SPECIAL VALIDATION FOR SOL-PERP (market_index={market_index}): "
                        f"OI=${oi_usd:,.2f}, positions={positions_count}, oracle_price=${oracle_price:,.2f}"
                    )

                    # Expected range verification (based on $90-100MM expected value)
                    if oi_usd < 80000000 or oi_usd > 120000000:
                        logger.warning(
                            f"==> POTENTIAL OI CALCULATION ISSUE FOR SOL-PERP: "
                            f"Calculated OI=${oi_usd:,.2f} outside expected range ($80M-$120M)"
                        )

                        # Double-check the base_decimals
                        logger.warning(f"==> Validating base_decimals={base_decimals} for SOL-PERP")

                        # Try alternative base_decimals values if suspect incorrect decimal scaling
                        for test_decimals in [6, 9, 10]:
                            test_base_oi = base_oi / (10 ** test_decimals)
                            test_oi_usd = test_base_oi * oracle_price
                            logger.warning(f"==> Test with base_decimals={test_decimals}: OI=${test_oi_usd:,.2f}")

                            # If a more reasonable value is found, consider using it
                            if 80000000 <= test_oi_usd <= 120000000:
                                logger.warning(f"==> Potentially better base_decimals={test_decimals} found for SOL-PERP")
                                if test_decimals != base_decimals:
                                    logger.warning(f"==> Overriding base_decimals from {base_decimals} to {test_decimals}")
                                    base_decimals = test_decimals
                                    base_oi_readable = base_oi / (10 ** base_decimals)
                                    oi_usd = base_oi_readable * oracle_price

                # Log raw values for debugging
                logger.info(
                    f"==> Raw OI Calculation for Market {market_index} ({clean_sym}): "
                    f"long_positions={long_amount}, short_positions={short_amount}, "
                    f"base_oi={base_oi}, base_decimals={base_decimals}, "
                    f"base_oi_readable={base_oi_readable}, oracle_price={oracle_price}, "
                    f"oi_usd_raw={oi_usd}"
                )

                # Get funding rate (hourly)
                funding_rate = market.data.amm.last_funding_rate / 1e6  # Convert to percentage
                hourly_funding = funding_rate * 100  # As percentage

                # We'll add volume data later from API
                drift_data.append({
                    'Symbol': clean_sym,
                    'Market Index': market_index,
                    'Max Lev. on Drift': max_leverage,
                    'OI on Drift': oi_usd,  # Use raw value without sig_figs
                    'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
                    'Oracle Price': oracle_price,
                    'Volume on Drift': 0  # Will be updated with actual data
                })
            except Exception as e:
                logger.error(f"==> Error processing market {market_index}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        logger.info(f"==> Successfully processed {markets_processed} markets (skipped {ignored_count} ignored markets)")

        if not drift_data:
            logger.error("==> No Drift markets data was processed after filtering")
            return None

        # Fetch actual trade volume data for all markets
        print("Fetching actual trade volume data from Drift API...")
        
        # Get volumes in bulk
        volumes_by_symbol = await batch_calculate_market_volumes(list(market_symbols.values()))
        
        # Update drift_data with actual volume values
        for item in drift_data:
            market_index = item['Market Index']
            symbol = market_symbols.get(market_index)
            if symbol in volumes_by_symbol:
                volume = volumes_by_symbol[symbol]
                item['Volume on Drift'] = volume
                logger.info(f"==> Updated market {market_index} ({symbol}) with actual volume: ${volume:,.2f}")
        
        return drift_data
    except Exception as e:
        logger.error(f"==> Error fetching Drift data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def build_scores(df):
    """Build scoring data with error handling."""
    logger.info("Building scores...")
    try:
        output = {}
        for score_category, category_details in SCORE_CUTOFFS.items():
            output[score_category] = pd.Series(0, index=df.index)
            for score_var, thresholds in category_details.items():
                if thresholds['kind'] == 'exp':
                    point_thresholds = {
                        thresholds['start'] 
                        * (thresholds['end']/thresholds['start']) 
                        ** (k/thresholds['steps']):
                            k for k in range(0, thresholds['steps']+1)}
                elif thresholds['kind'] == 'linear':
                    point_thresholds = {
                        thresholds['start']
                        + (thresholds['end'] - thresholds['start'])
                        * (k/thresholds['steps']):
                            k for k in range(0, thresholds['steps']+1)}
                else: 
                    raise ValueError(f"Invalid threshold kind: {thresholds['kind']}")
                    
                score_name = 'Partial_Score_'+score_var
                output[score_name] = pd.Series(0, index=df.index)
                for lb, value in sorted(point_thresholds.items()):
                    output[score_name].loc[df[score_var] >= lb] = value
                output[score_category] += output[score_name]

        output_df = pd.concat(output, axis=1)
        
        # Add Strict boost for specific tokens
        output_df['Strict'] = df.index.isin(STRICT_TOKENS)
        output_df['Score'] = output_df[[*SCORE_CUTOFFS]].sum(axis=1) + output_df['Strict']*STRICT_BOOST

        return output_df
    except Exception as e:
        logger.error(f"Error building scores: {str(e)}")
        raise

def build_drift_scores(df):
    """Calculates scores for Drift assets based on metrics."""
    logger.info("Building Drift-specific scores...")
    try:
        output_scores = {}

        # Calculate scores for each category
        for score_category, category_details in DRIFT_SCORE_CUTOFFS.items():
            output_scores[score_category] = pd.Series(0.0, index=df.index)

            for score_var, thresholds in category_details.items():
                if score_var not in df.columns:
                     logger.warning(f"==> Scoring variable '{score_var}' not found for category '{score_category}'. Skipping.")
                     continue

                # Generate threshold points
                point_thresholds = {}
                steps = thresholds['steps']
                start = thresholds['start']
                end = thresholds['end']

                if thresholds['kind'] == 'exp':
                    # Exponential spacing
                    if start <= 0:
                        # Fallback to linear if start <= 0
                        logger.warning(f"==> Exponential scale needs start > 0 for '{score_var}'. Using linear.")
                        for k in range(steps + 1):
                            point_thresholds[start + (end - start) * (k / steps)] = k
                    else:
                        ratio = end / start
                        for k in range(steps + 1):
                            point_thresholds[start * (ratio ** (k / steps))] = k
                elif thresholds['kind'] == 'linear':
                    # Linear spacing
                    for k in range(steps + 1):
                        point_thresholds[start + (end - start) * (k / steps)] = k

                # Calculate partial score
                score_name = f'Partial_Score_{score_var}'
                output_scores[score_name] = pd.Series(0.0, index=df.index)

                # Apply thresholds
                if thresholds['kind'] == 'exp' or (thresholds['kind'] == 'linear' and start <= end):
                    for threshold_val, points in sorted(point_thresholds.items()):
                        output_scores[score_name].loc[df[score_var] >= threshold_val] = points
                else: # Linear decreasing score
                    for threshold_val, points in sorted(point_thresholds.items(), reverse=True):
                        output_scores[score_name].loc[df[score_var] <= threshold_val] = points

                # Add to category score
                output_scores[score_category] += output_scores[score_name]

        # Convert to DataFrame
        output_df = pd.concat(output_scores, axis=1)

        # Calculate final score
        score_components = list(DRIFT_SCORE_CUTOFFS.keys())
        output_df['Drift Score'] = output_df[score_components].sum(axis=1)

        # Apply score boost for specific symbols
        output_df['Drift Score'] = output_df['Drift Score'].add(
            pd.Series(
                [DRIFT_SCORE_BOOST_AMOUNT if symbol in DRIFT_SCORE_BOOST_SYMBOLS else 0 
                 for symbol in output_df.index],
                index=output_df.index
            )
        )

        return output_df
    except Exception as e:
        logger.error(f"Error building drift scores: {str(e)}")
        return pd.DataFrame(index=df.index)

def generate_recommendation(row):
    """Generate recommendation with error handling."""
    try:
        # For list recommender, we recommend "List" if the score is high enough
        current_leverage = int(0 if pd.isna(row['Max Lev.']) else row['Max Lev.'])
        score = row['Score']
        
        # Find the largest key in SCORE_UB that is less than or equal to the current leverage
        upper_bound_key = 0
        for k in sorted(SCORE_UB.keys()):
            if k <= current_leverage:
                upper_bound_key = k
            else:
                break
                
        # Check if score is above upper bound
        is_above_upper_bound = score >= SCORE_UB[upper_bound_key]
        
        # Generate recommendation
        if current_leverage == 0 and is_above_upper_bound:
            return 'List'
        elif current_leverage > 0 and is_above_upper_bound:
            return 'Increase Leverage'
        else:
            return 'Monitor'  # No listing recommended yet
    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        return 'Error'

def generate_delist_recommendation(row):
    """Generates recommendation for delisting or changing leverage based on score and current leverage."""
    try:
        current_leverage = int(0 if pd.isna(row['Max Lev. on Drift']) else row['Max Lev. on Drift'])
        score = row['Score']

        # No recommendation needed if not listed on Drift
        if current_leverage == 0:
            return 'Not Listed'

        # Determine relevant score boundaries
        # Find the largest key in SCORE_LB that is less than or equal to the current leverage
        lower_bound_key = 0
        for k in sorted(SCORE_LB.keys()):
            if k <= current_leverage:
                lower_bound_key = k
            else:
                break

        # Check if score is below lower bound
        is_below_lower_bound = score < SCORE_LB[lower_bound_key]

        # Generate recommendation
        if is_below_lower_bound:
            if current_leverage > 5: # If leverage is higher than 5x and score is low, recommend decrease
                return 'Decrease Leverage'
            else: # If leverage is 5x or lower and score is low, recommend delist
                return 'Delist'
        else:
            return 'Keep'  # No change recommended
    except Exception as e:
        logger.error(f"Error generating delist recommendation: {str(e)}")
        return 'Error'

async def get_list_recommendations(request=None):
    """Main function to get listing recommendations.
    
    Args:
        request: Optional BackendRequest object to access Drift data
    """
    global last_recommendation_data, last_recommendation_time
    
    # Check if we have cached recommendations less than 24 hours old
    current_time = time.time()
    if last_recommendation_data and (current_time - last_recommendation_time < CACHE_DURATION):
        logger.info(f"Using cached recommendations (age: {(current_time - last_recommendation_time)/3600:.1f} hours)")
        return last_recommendation_data
    
    try:
        start_time = datetime.now()
        logger.info("==> Starting to generate listing recommendations")
        print("Starting to generate listing recommendations...")
        
        # Run data fetching in parallel where possible
        cmc_task = asyncio.create_task(dl_cmc_data_async())
        thunder_task = asyncio.create_task(dl_thunderhead_data_async())
        
        # Execute reference exchange data download (already parallelized internally)
        logger.info("==> Fetching reference exchange data")
        print("Fetching exchange data...")
        ref_data = await dl_reference_exch_data_async()
        
        # Wait for all other tasks to complete
        logger.info("==> Waiting for remaining data fetching tasks")
        print("Waiting for remaining data fetching tasks...")
        cmc_data = await cmc_task
        thunder_data = await thunder_task
        
        # Process all data (these are CPU-bound so we run them in the normal flow)
        logger.info("==> Processing collected data")
        print("Processing collected data...")
        
        # Process each data source
        cmc_df = process_cmc_data(cmc_data)
        exchange_df = process_reference_exch_data(ref_data)
        thunder_df = process_thunderhead_data(thunder_data)
        
        # Get all unique symbols across all DataFrames
        all_symbols = sorted(set(
            list(cmc_df.index) +
            list(exchange_df.index) +
            list(thunder_df.index)
        ))
        
        # Reindex all DataFrames to have the same index
        cmc_df = cmc_df.reindex(all_symbols, fill_value=0)
        exchange_df = exchange_df.reindex(all_symbols, fill_value=0)
        thunder_df = thunder_df.reindex(all_symbols, fill_value=0)
        
        # Now concatenate the aligned DataFrames
        df = pd.concat([cmc_df, exchange_df, thunder_df], axis=1)
        
        # Remove stablecoins
        df = df.loc[~df.index.isin(STABLE_COINS)]
        
        # Add symbol and default leverage
        df['Symbol'] = df.index
        df['Max Lev.'] = 0  # Default to 0 as this is for unlisted tokens
        
        # Try to get Drift data if available in the backend state
        drift_data = None
        try:
            # Access the backend state directly from the request object if it exists
            backend_state = getattr(request, 'state', None)
            if backend_state:
                backend_state = getattr(backend_state, 'backend_state', None)
            
            if backend_state and hasattr(backend_state, 'vat') and hasattr(backend_state, 'perp_map') and hasattr(backend_state, 'user_map') and hasattr(backend_state, 'dc'):
                logger.info("==> Fetching Drift protocol data")
                print("Fetching Drift protocol data...")
                
                drift_data = await get_drift_data(
                    backend_state.dc,
                    backend_state.perp_map,
                    backend_state.user_map
                )
                
                if drift_data:
                    # Create DataFrame from Drift data
                    drift_df = pd.DataFrame(drift_data).set_index('Symbol')
                    logger.info(f"==> Found {len(drift_df)} tokens on Drift platform")
                    print(f"Found {len(drift_df)} tokens on Drift platform")
                    
                    # Update the main DataFrame with info about currently listed tokens
                    for symbol in drift_df.index:
                        if symbol in df.index:
                            # Update leverage for tokens already in our dataset
                            df.loc[symbol, 'Max Lev.'] = drift_df.loc[symbol, 'Max Lev. on Drift']
                            # Copy other relevant fields (OI and Volume are stored as full dollars in drift_df)
                            df.loc[symbol, 'OI on Drift'] = drift_df.loc[symbol, 'OI on Drift']
                            df.loc[symbol, 'Volume on Drift'] = drift_df.loc[symbol, 'Volume on Drift']
                            df.loc[symbol, 'Funding Rate % (1h)'] = drift_df.loc[symbol, 'Funding Rate % (1h)']
                            
                            # Also set the OI $m and Volume $m fields for consistency (in millions)
                            if 'OI $m' in df.columns and df.loc[symbol, 'OI $m'] == 0:
                                df.loc[symbol, 'OI $m'] = drift_df.loc[symbol, 'OI on Drift'] / 1e6
                            if 'Volume $m' in df.columns and df.loc[symbol, 'Volume $m'] == 0:
                                df.loc[symbol, 'Volume $m'] = drift_df.loc[symbol, 'Volume on Drift'] / 1e6
                        else:
                            # Add new row for tokens only on Drift
                            new_row = pd.Series(0, index=df.columns)
                            new_row['Symbol'] = symbol
                            new_row['Max Lev.'] = drift_df.loc[symbol, 'Max Lev. on Drift']
                            new_row['OI on Drift'] = drift_df.loc[symbol, 'OI on Drift']
                            new_row['Volume on Drift'] = drift_df.loc[symbol, 'Volume on Drift']
                            new_row['Funding Rate % (1h)'] = drift_df.loc[symbol, 'Funding Rate % (1h)']
                            new_row['OI $m'] = drift_df.loc[symbol, 'OI on Drift'] / 1e6
                            new_row['Volume $m'] = drift_df.loc[symbol, 'Volume on Drift'] / 1e6
                            df = pd.concat([df, pd.DataFrame([new_row]).set_index('Symbol')])
                            
                    # Also prepare data for the drift-specific scoring
                    # Create MC column from MC $m for Drift scoring
                    df['MC'] = df['MC $m'] * 1e6
                    
                    # Create columns needed for Drift scoring - convert from $m to full dollars if they exist
                    vol_columns = {
                        'Spot Volume': 'Spot Volume $m',
                        'Spot Vol Geomean': 'Spot Vol Geomean $m',
                        'Fut Volume': 'Fut Volume $m',
                        'Fut Vol Geomean': 'Fut Vol Geomean $m'
                    }
                    
                    for full_col, m_col in vol_columns.items():
                        if m_col in df.columns:
                            df[full_col] = df[m_col] * 1e6
                    
                    logger.info("==> Successfully merged Drift data with other data sources")
                    print("Successfully merged Drift data with other data sources")
                else:
                    logger.warning("==> Failed to get Drift data or no markets found")
                    print("Failed to get Drift data or no markets found")
            else:
                logger.warning("==> Backend state not available, skipping Drift data")
                print("Backend state not available, skipping Drift data")
        except Exception as e:
            logger.error(f"==> Error fetching Drift data: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error fetching Drift data: {str(e)}")
        
        # Calculate scores and list recommendations
        listing_scores_df = build_scores(df)
        df = pd.concat([df, listing_scores_df], axis=1)
        df['Recommendation'] = df.apply(generate_recommendation, axis=1)
        
        # Calculate drift-specific scores and delist recommendations if drift data is available
        if drift_data:
            # Build drift-specific scores
            drift_scores_df = build_drift_scores(df)
            df = pd.concat([df, drift_scores_df], axis=1)
            
            # Generate delist recommendations using the Drift score
            # First create a temporary series with the Drift score in the 'Score' column for the recommendation function
            temp_series = df.copy()
            temp_series['Score'] = temp_series['Drift Score']
            df['Delist Recommendation'] = temp_series.apply(generate_delist_recommendation, axis=1)
        else:
            # If no drift data, mark all as "Not Listed"
            df['Delist Recommendation'] = 'Not Listed'
        
        # Filter to keep tokens with actionable recommendations
        actionable_df = df[
            (df['Recommendation'].isin(['List', 'Increase Leverage'])) |  # List recommender recommendations
            (df['Delist Recommendation'].isin(['Delist', 'Decrease Leverage']))  # Delist recommendations
        ].copy()
        
        # Sort by Score in descending order
        df_for_main_data = actionable_df.sort_values('Score', ascending=False)
        
        # Prepare columns for output
        output_cols = list(OUTPUT_COLS)  # Start with standard output columns
        
        # Add Drift-specific columns if available
        if 'Delist Recommendation' in df_for_main_data.columns:
            if 'Delist Recommendation' not in output_cols:
                output_cols.append('Delist Recommendation')
            if 'Funding Rate % (1h)' not in output_cols:
                output_cols.append('Funding Rate % (1h)')
            if 'Drift Score' not in output_cols:
                output_cols.append('Drift Score')
        
        # Filter columns to only those that exist in the DataFrame
        output_cols = [col for col in output_cols if col in df_for_main_data.columns]
        
        # Format values with sig_figs - convert columns to proper display format
        for c in df_for_main_data.columns:
            if str(df_for_main_data[c].dtype) in ['int64', 'float64']:
                df_for_main_data[c] = df_for_main_data[c].map(sig_figs)
        
        # Calculate summary stats
        total_tokens = len(df)
        # List recommendation stats
        list_tokens = len(df[df['Recommendation'] == 'List'])
        increase_lev_tokens = len(df[df['Recommendation'] == 'Increase Leverage'])
        monitor_tokens = len(df[df['Recommendation'] == 'Monitor'])
        
        # Delist recommendation stats
        delist_tokens = len(df[df['Delist Recommendation'] == 'Delist'])
        decrease_lev_tokens = len(df[df['Delist Recommendation'] == 'Decrease Leverage'])
        keep_tokens = len(df[df['Delist Recommendation'] == 'Keep'])
        not_listed_tokens = len(df[df['Delist Recommendation'] == 'Not Listed'])
        
        # Prepare results for API response
        drift_slot = None
        if 'backend_state' in locals() and backend_state and hasattr(backend_state, 'last_oracle_slot'):
            drift_slot = backend_state.last_oracle_slot
            
        results = {
            "slot": drift_slot,
            "results": df_for_main_data[output_cols].reset_index().to_dict(orient='records'),
            "summary": {
                "total_tokens": total_tokens,
                "list_tokens": list_tokens,
                "increase_leverage_tokens": increase_lev_tokens,
                "monitor_tokens": monitor_tokens,
                "delist_tokens": delist_tokens,
                "decrease_leverage_tokens": decrease_lev_tokens,
                "keep_tokens": keep_tokens,
                "not_listed_tokens": not_listed_tokens,
                "listed_on_drift": total_tokens - not_listed_tokens,
                "top_candidates": actionable_df.sort_values('Score', ascending=False).head(10).index.tolist()
            },
            "score_boundaries": {
                "upper": SCORE_UB,  # For listing/increasing leverage
                "lower": SCORE_LB   # For delisting/decreasing leverage
            }
        }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"==> Completed generating list/delist recommendations in {duration:.2f} seconds")
        print(f"Completed generating list/delist recommendations in {duration:.2f} seconds")
        
        # Cache the results
        last_recommendation_data = {
            "status": "success",
            "message": "Listing and delisting recommendations generated successfully",
            "data": results
        }
        last_recommendation_time = current_time
        
        return last_recommendation_data
    except Exception as e:
        logger.error(f"==> Error generating list recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error generating list recommendations: {str(e)}",
            "data": None
        }

# --- API Endpoints ---
@router.get("/recommendations")
async def get_recommendations(request: BackendRequest):
    """Get listing and delisting recommendations for markets."""
    logger.info("Received request for listing and delisting recommendations")
    print("Processing listing and delisting recommendations request")
    
    try:
        # Verify backend state for Drift data
        if hasattr(request, 'state') and hasattr(request.state, 'backend_state'):
            backend_state = request.state.backend_state
            if backend_state:
                drift_attrs_available = all([
                    hasattr(backend_state, attr) 
                    for attr in ['vat', 'dc', 'perp_map', 'user_map']
                ])
                if drift_attrs_available:
                    # Count perp markets and users using the safe_count helper
                    perp_markets_count = safe_count(backend_state.perp_map) if hasattr(backend_state, 'perp_map') else 0
                    user_count = safe_count(backend_state.user_map) if hasattr(backend_state, 'user_map') else 0
                    
                    logger.info(f"Drift data available, perp markets: {perp_markets_count}, users: {user_count}")
                    print(f"Drift data available, perp markets: {perp_markets_count}, users: {user_count}")
                else:
                    logger.warning("Drift data partially available or missing required attributes")
                    print("Drift data partially available or missing required attributes")
            else:
                logger.warning("Backend state is empty")
                print("Backend state is empty")
        else:
            logger.warning("Request does not have backend state attribute")
            print("Request does not have backend state attribute")
            
        # Generate recommendations with the request object for accessing Drift data
        result = await get_list_recommendations(request)
        return result
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"API error: {str(e)}",
            "data": None
        }

@router.get("/single_market_recommendation")
async def get_single_market_recommendation(
    request: BackendRequest,
    market_index: int = Query(..., description="Market index to analyze")
):
    """Get detailed recommendation for a single market, showing all intermediate calculations."""
    print(f"\n===> SINGLE MARKET RECOMMENDATION API CALL RECEIVED FOR MARKET INDEX {market_index} <===")
    logger.info(f"==> [api] Received single market recommendation request for market index {market_index}")

    try:
        # Check if backend state is available
        if not hasattr(request, 'state') or not hasattr(request.state, 'backend_state') or not request.state.backend_state:
            logger.error("==> Backend state not available for single market analysis")
            return {
                "status": "error",
                "message": "Backend state not available",
                "data": None
            }

        # Check if Drift client is available
        backend_state = request.state.backend_state
        if not hasattr(backend_state, 'vat') or not hasattr(backend_state, 'dc'):
            logger.error("==> Drift client not available in backend state")
            return {
                "status": "error",
                "message": "Drift client not available",
                "data": None
            }

        # Check if perp markets are available
        if not hasattr(backend_state.vat, 'perp_markets'):
            logger.error("==> Perp markets not available in vat")
            return {
                "status": "error",
                "message": "Perp markets not available",
                "data": None
            }

        # Check if this market exists
        perp_market = backend_state.vat.perp_markets.get(market_index)
        if not perp_market:
            logger.error(f"==> Market with index {market_index} not found")
            return {
                "status": "error",
                "message": f"Market with index {market_index} not found",
                "data": None
            }
        
        # Get market config with proper error handling
        try:
            market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
            if not market_config:
                logger.error(f"==> Market config for index {market_index} not found")
                return {
                    "status": "error",
                    "message": f"Market config for index {market_index} not found",
                    "data": None
                }
            
            # Get symbol
            symbol = market_config.symbol
            clean_sym = clean_symbol(symbol)
            
            logger.info(f"==> Processing single market recommendation for {clean_sym} (index: {market_index})")
        except Exception as e:
            logger.error(f"==> Error getting market config: {str(e)}")
            return {
                "status": "error",
                "message": f"Error getting market config: {str(e)}",
                "data": None
            }
        
        # Check if the market symbol is in the ignored list
        if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
            logger.error(f"==> Market {market_index} ({symbol}) is explicitly ignored")
            return {
                "status": "error",
                "message": f"Market {market_index} ({symbol}) is explicitly ignored and cannot be analyzed.",
                "data": None
            }
            
        # Get market recommendations data
        logger.info(f"==> Getting recommendations for {clean_sym}")
        try:
            # Get full recommendations data that includes this market
            recommendations_result = await get_list_recommendations(request)
            if recommendations_result["status"] != "success":
                return recommendations_result
                
            # Extract the data for this specific market
            market_data = None
            if "results" in recommendations_result.get("data", {}):
                for item in recommendations_result["data"]["results"]:
                    if item.get("Symbol") == clean_sym:
                        market_data = item
                        break
                    
            # If market not found in results, try to get its raw data
            if not market_data:
                logger.info(f"==> Market {clean_sym} not found in recommendations, fetching raw data")
                try:
                    if (hasattr(backend_state, 'perp_map') and 
                        hasattr(backend_state, 'user_map') and 
                        hasattr(backend_state, 'dc')):
                        
                        drift_data = await get_drift_data(
                            backend_state.dc,
                            backend_state.perp_map,
                            backend_state.user_map
                        )
                        
                        if drift_data:
                            for item in drift_data:
                                if item.get("Symbol") == clean_sym:
                                    market_data = item
                                    break
                    else:
                        logger.warning(f"==> Cannot fetch raw Drift data, missing required attributes")
                except Exception as e:
                    logger.error(f"==> Error fetching raw Drift data: {str(e)}")
        except Exception as e:
            logger.error(f"==> Error getting recommendations: {str(e)}")
            market_data = None
                        
        # Get detailed information from the market object with error handling
        try:
            market_details = {
                "symbol": clean_sym,
                "original_symbol": symbol,
                "market_index": market_index,
            }
            
            # Add attributes with proper error handling
            if hasattr(perp_market, 'data'):
                if hasattr(perp_market.data, 'margin_ratio_initial'):
                    market_details["max_leverage"] = perp_market.data.margin_ratio_initial / 10000
                
                market_details["base_decimals"] = getattr(perp_market.data, 'base_decimals', 
                                                         MARKET_BASE_DECIMALS.get(market_index, 9))
                
                if hasattr(perp_market.data, 'amm') and hasattr(perp_market.data.amm, 'last_funding_rate'):
                    market_details["funding_rate"] = perp_market.data.amm.last_funding_rate / 1e6
            
            # Get oracle price with error handling
            if hasattr(backend_state.dc, 'get_oracle_price_data_for_perp_market'):
                try:
                    oracle_price_data = backend_state.dc.get_oracle_price_data_for_perp_market(market_index)
                    if oracle_price_data and hasattr(oracle_price_data, 'price'):
                        market_details["oracle_price"] = oracle_price_data.price / 1e6
                except Exception as e:
                    logger.error(f"==> Error getting oracle price: {str(e)}")
                    market_details["oracle_price"] = None
            
            logger.info(f"==> Successfully collected market details for {clean_sym}")
        except Exception as e:
            logger.error(f"==> Error collecting market details: {str(e)}")
            market_details = {
                "symbol": clean_sym,
                "original_symbol": symbol,
                "market_index": market_index,
                "error": str(e)
            }
            
        return {
            "status": "success",
            "message": f"Single market data for {clean_sym}",
            "data": {
                "slot": getattr(backend_state, 'last_oracle_slot', None),
                "market_index": market_index,
                "symbol": clean_sym,
                "market_details": market_details,
                "recommendation_data": market_data
            }
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"==> [api] Error generating single market recommendation: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error generating single market recommendation: {str(e)}",
            "data": None
        }

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify routing is working."""
    logger.info("Test endpoint called")
    return {"status": "success", "message": "List recommender API is working"}

# Synchronous wrapper for script execution
def main():
    """Entry point for direct script execution."""
    asyncio.run(get_list_recommendations())

if __name__ == "__main__":
    main()

# --- Drift API Volume Calculation Functions ---
async def fetch_api_page(session, url: str, retries: int = 5):
    """
    Fetch a single page from the Drift API with rate limiting and retries.
    """
    global last_request_time
    
    for attempt in range(retries):
        # Apply rate limiting
        async with rate_limit_lock:
            current_time = time.time()
            wait_time = API_RATE_LIMIT_INTERVAL - (current_time - last_request_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            last_request_time = time.time()
        
        try:
            async with session.get(url, headers=DRIFT_DATA_API_HEADERS, timeout=10) as response:
                if response.status != 200:
                    logger.warning(f"API request failed: {url}, status: {response.status}")
                    if attempt < retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"success": False, "records": [], "meta": {"totalRecords": 0}}
                
                data = await response.json()
                return data
        except Exception as e:
            logger.warning(f"Error fetching {url}: {str(e)}")
            if attempt < retries - 1:
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                continue
            return {"success": False, "records": [], "meta": {"totalRecords": 0}}
    
    return {"success": False, "records": [], "meta": {"totalRecords": 0}}

async def fetch_market_trades(session, symbol: str, start_date: datetime, end_date: datetime = None):
    """
    Fetch all trades for a market within the specified date range.
    """
    if end_date is None:
        end_date = datetime.now()
    
    current_date = start_date
    all_trades = []
    
    while current_date <= end_date:
        year, month, day = current_date.year, current_date.month, current_date.day
        
        # Make sure DRIFT_DATA_API_BASE_URL is available
        if not DRIFT_DATA_API_BASE_URL:
            logger.error("DRIFT_DATA_API_BASE_URL environment variable not set")
            return []
            
        url = f"{DRIFT_DATA_API_BASE_URL}/market/{symbol}/trades/{year}/{month}/{day}?format=json"
        
        logger.debug(f"Fetching trades for {symbol} on {year}/{month}/{day}")
        
        # Fetch first page
        data = await fetch_api_page(session, url)
        
        if data["success"] and "records" in data:
            all_trades.extend(data["records"])
            
            # Handle pagination if needed
            page = 1
            total_pages = data.get("meta", {}).get("totalPages", 1)
            
            while page < total_pages and page < 10:  # Limit to 10 pages per day to avoid excessive requests
                page += 1
                paginated_url = f"{url}&page={page}"
                page_data = await fetch_api_page(session, paginated_url)
                
                if page_data["success"] and "records" in page_data:
                    all_trades.extend(page_data["records"])
                else:
                    break
        
        # Move to next day
        current_date += timedelta(days=1)
    
    return all_trades

async def calculate_market_volume(symbol: str):
    """
    Calculate the total trading volume for a market over the past 30 days.
    
    Args:
        symbol: Market symbol (e.g., "SOL-PERP")
        
    Returns:
        Total volume in USD (float)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_CONSIDER)
    
    try:
        async with aiohttp.ClientSession() as session:
            trades = await fetch_market_trades(session, symbol, start_date, end_date)
            
            # Sum up the quote asset amounts from all filled trades
            total_volume = 0.0
            for trade in trades:
                try:
                    quote_amount = float(trade.get("quoteAssetAmountFilled", 0))
                    total_volume += quote_amount
                except (ValueError, TypeError):
                    continue
            
            logger.info(f"Calculated {DAYS_TO_CONSIDER}-day volume for {symbol}: ${total_volume:,.2f}")
            return total_volume
    except Exception as e:
        logger.error(f"Error calculating volume for {symbol}: {str(e)}")
        logger.error(traceback.format_exc())
        return 0.0

async def batch_calculate_market_volumes(symbols: List[str]):
    """
    Calculate volumes for multiple markets in parallel with rate limiting.
    
    Args:
        symbols: List of market symbols
        
    Returns:
        Dictionary mapping symbols to their volumes
    """
    logger.info(f"Calculating actual trade volumes for {len(symbols)} markets")
    print(f"Fetching actual 30-day trade volumes from Drift API for {len(symbols)} markets...")
    
    volumes = {}
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    async def fetch_with_semaphore(symbol):
        async with semaphore:
            volume = await calculate_market_volume(symbol)
            volumes[symbol] = volume
            print(f"Fetched {symbol} volume: ${volume:,.2f}")
    
    tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
    await asyncio.gather(*tasks)
    
    return volumes