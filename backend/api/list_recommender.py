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
from typing import Dict, Optional, Any, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from backend.state import BackendRequest

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

async def get_list_recommendations():
    """Main function to get listing recommendations."""
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
        
        # Calculate scores and recommendations
        df = pd.concat([df, build_scores(df)], axis=1)
        df['Recommendation'] = df.apply(generate_recommendation, axis=1)
        
        # Filter to only keep tokens with 'List' or 'Increase Leverage' recommendation
        list_candidates = df[df['Recommendation'].isin(['List', 'Increase Leverage'])].copy()
        
        # Sort by Score in descending order
        df_for_main_data = list_candidates.sort_values('Score', ascending=False)[OUTPUT_COLS]
        
        # Format values with sig_figs
        for c in df_for_main_data.columns:
            if str(df_for_main_data[c].dtype) in ['int64', 'float64']:
                df_for_main_data[c] = df_for_main_data[c].map(sig_figs)
        
        # Calculate summary stats
        total_tokens = len(df)
        list_tokens = len(df[df['Recommendation'] == 'List'])
        increase_lev_tokens = len(df[df['Recommendation'] == 'Increase Leverage'])
        monitor_tokens = len(df[df['Recommendation'] == 'Monitor'])
        
        # Prepare results for API response
        results = {
            "slot": None,  # Not applicable for this API
            "results": df_for_main_data.reset_index().to_dict(orient='records'),
            "summary": {
                "total_tokens": total_tokens,
                "list_tokens": list_tokens,
                "increase_leverage_tokens": increase_lev_tokens,
                "monitor_tokens": monitor_tokens,
                "top_candidates": list_candidates.sort_values('Score', ascending=False).head(10).index.tolist()
            },
            "score_boundaries": SCORE_UB
        }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"==> Completed generating list recommendations in {duration:.2f} seconds")
        print(f"Completed generating list recommendations in {duration:.2f} seconds")
        
        # Cache the results
        last_recommendation_data = {
            "status": "success",
            "message": "Listing recommendations generated successfully",
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
    """Get listing recommendations for potential new markets."""
    logger.info("Received request for listing recommendations")
    
    try:
        result = await get_list_recommendations()
        return result
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"API error: {str(e)}",
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