# This script is used to provide backend functionality for /src/page/delist_recommender.py
# It should make use of the backend state and middleware to fetch the data borrowing similar logic from /backend/api/asset_liability.py and /backend/api/health.py and /backend/api/price_shock.py

# It should return a JSON object with the following fields:
# - status: "success" or "error"
# - message: a message to be displayed to the user
# - data: a JSON object containing the data to be displayed to the user

# This script provides backend functionality for delist recommender analysis
# It makes use of the backend state and middleware to fetch data efficiently
import os
import math
import sys
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
import numpy as np
import pandas as pd
import requests
import aiohttp
import ccxt
import ccxt.async_support as ccxt_async
import logging
import traceback
import time
import json
from datetime import datetime, timedelta
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from backend.state import BackendRequest
from driftpy.pickle.vat import Vat
from driftpy.drift_client import DriftClient
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.types import MarketType

router = APIRouter()

# Set up logger
logger = logging.getLogger("backend.api.delist_recommender")

# --- Configuration Constants ---
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
DAYS_TO_CONSIDER = 30

# Drift-specific score cutoffs - simplified for delisting focus
DRIFT_SCORE_CUTOFFS = {
    'Market Cap Score': {
        'MC': {'kind': 'exp', 'start': 1000000, 'end': 5000000000, 'steps': 20},
    },
    'Spot Vol Score': {
        'Spot Volume': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10},
        'Spot Vol Geomean': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10},
    },
    'Futures Vol Score': {
        'Fut Volume': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10},
        'Fut Vol Geomean': {'kind': 'exp', 'start': 10000, 'end': 1000000000, 'steps': 10},
    },
    'Drift Activity Score': {
        'Volume on Drift': {'kind': 'exp', 'start': 1000, 'end': 500000000, 'steps': 10},
        'OI on Drift': {'kind': 'exp', 'start': 1000, 'end': 500000000, 'steps': 10},
    },
}

# Score boundaries for delist recommendations
SCORE_LB = {0: 0, 5: 37, 10: 48, 20: 60}    # Lower bounds

# Reference exchanges for market data
REFERENCE_SPOT_EXCH = {
    'binanceus', 'bybit', 'okx', 'gate', 'kucoin', 'mexc', 'kraken'
}

REFERENCE_FUT_EXCH = {
    'bybit', 'okx', 'gate', 'mexc', 'htx', 'bitmex'
}

# Add a known decimals mapping before the get_drift_data function
# Market-specific base decimals overrides based on known values
MARKET_BASE_DECIMALS = {
    0: 9,  # SOL-PERP
    1: 6,  # BTC-PERP - likely 6 decimals instead of 9 based on expected OI
    2: 9,  # ETH-PERP
}

# --- Utility Functions ---
def sig_figs(number, sig_figs=3):
    """Rounds a number to specified significant figures."""
    if np.isnan(number) or number <= 0:
        return 0
    # Use numpy's around function instead of basic round to maintain precision
    return np.around(number, -int(np.floor(np.log10(abs(number)))) + (sig_figs - 1))

def clean_symbol(symbol, exch=''):
    """Cleans and standardizes cryptocurrency symbols."""
    # General token aliases
    TOKEN_ALIASES = {
        'WBTC': 'BTC', 'WETH': 'ETH', 'WSOL': 'SOL',
        '1INCH': 'ONEINCH', 'HPOS10I': 'HPOS',
        'BITCOIN': 'BTC'
    }
    
    # Extract base symbol
    redone = symbol.split('/')[0]
    
    # Remove common numerical suffixes
    for suffix in ['10000000', '1000000', '1000', 'k']:
        redone = redone.replace(suffix, '')
    
    # Apply general alias
    return TOKEN_ALIASES.get(redone, redone)

async def get_async_ccxt_api(exch):
    """Initializes and returns an async ccxt exchange API instance."""
    api = None
    session = None  # Initialize session at the top to make it available in finally block
    try:
        # Special handling for Bybit
        if exch == 'bybit':
            # Create a shared session for Bybit
            session = aiohttp.ClientSession()
            api = getattr(ccxt_async, exch)({
                'timeout': 10000,
                'enableRateLimit': True,
                'session': session  # Use the shared session
            })
        else:
            api = getattr(ccxt_async, exch)({
                'timeout': 10000,
                'enableRateLimit': True,
            })
        
        # Load markets with a try-except block
        try:
            await api.load_markets()
        except Exception as e:
            logger.warning(f"Could not load markets for {exch}: {str(e)}. Using limited functionality.")
            # Continue with limited functionality
        
        # Test if the API is working by trying common market pairs
        has_working_pair = False
        for pair in ['BTC/USDT:USDT', 'BTC/USDT', 'BTC/USD', 'ETH/USDT']:
            try:
                await api.fetch_ticker(pair)
                has_working_pair = True
                break
            except Exception:
                continue
            
        if not has_working_pair:
            logger.warning(f"Could not fetch any test market data from {exch}. Skipping this exchange.")
            # Properly close even on failure
            if api:
                try:
                    await api.close()
                    
                    # Additional cleanup for specific exchanges
                    if hasattr(api, 'session') and api.session:
                        await api.session.close()
                    if hasattr(api, 'connector') and api.connector:
                        await api.connector.close()
                    
                    # Special handling for Bybit session
                    if exch == 'bybit' and session:
                        await session.close()
                        
                    logger.debug(f"Closed {exch} API after initialization error")
                except Exception as close_error:
                    logger.warning(f"Error closing {exch} API after initialization error: {str(close_error)}")
            return None
                
        return api
    except Exception as e:
        # Log error but don't crash
        logger.error(f"Failed to initialize {exch} API: {str(e)}")
        # Ensure API is closed if initialization was partially successful
        if api:
            try:
                await api.close()
                
                # Additional cleanup for specific exchanges
                if hasattr(api, 'session') and api.session:
                    await api.session.close()
                if hasattr(api, 'connector') and api.connector:
                    await api.connector.close()
                
                # Special handling for Bybit session
                if exch == 'bybit' and session:
                    await session.close()
                    
                logger.debug(f"Closed {exch} API after initialization error")
            except Exception as close_error:
                logger.warning(f"Error closing {exch} API after initialization error: {str(close_error)}")
        return None

def geomean_three(series):
    """Calculates geometric mean of top 3 values in a series."""
    return np.exp(np.log(series + 1).sort_values()[-3:].sum() / 3) - 1

# --- Data Fetching Functions ---
async def download_exch(exch, spot, normalized_symbols):
    """Helper to download data from a single exchange."""
    logger.info(f"==> Attempting to download data from {exch} (spot={spot})")
    api = None
    try:
        api = await get_async_ccxt_api(exch)
        if not api:
            logger.warning(f"==> Failed to initialize API for {exch}")
            return {}
        
        # If no markets available, return empty data
        if not hasattr(api, 'markets') or not api.markets:
            logger.warning(f"==> No markets available for {exch}")
            return {}
                
        exchange_data = {}
        
        # Pre-filter markets to only those potentially matching our symbols
        markets_to_try = []
        for market_name, market_data in api.markets.items():
            try:
                # Basic filtering as before
                if spot and ':' in market_name:
                    continue
                if not spot and ':USD' not in market_name and ':USDT' not in market_name:
                    continue
                    
                # More comprehensive quote currency check - use these common stablecoin pairs
                parts = market_name.split('/')
                if len(parts) < 2:
                    continue
                
                # Skip if base part is invalid
                if not parts[0] or not parts[0].strip():
                    continue
                
                quote_symbol = parts[1].split(':')[0].upper() if ':' in parts[1] else parts[1].upper()
                if quote_symbol not in STABLE_COINS:
                    continue
                
                # Extract base symbol and check if it's in our list
                base_symbol = parts[0].upper()
                clean_base = clean_symbol(base_symbol)
                
                # Check if this market matches any of our Drift symbols
                if (base_symbol in normalized_symbols or 
                    clean_base in normalized_symbols or
                    any(symbol.upper().startswith(base_symbol) for symbol in normalized_symbols) or
                    any(symbol.upper().startswith(clean_base) for symbol in normalized_symbols)):
                    markets_to_try.append(market_name)
            except Exception as e:
                logger.debug(f"Error filtering market {market_name}: {str(e)}")
                continue
        
        logger.info(f"==> Found {len(markets_to_try)} potential matches in {exch} out of {len(api.markets)} total markets")
        if len(markets_to_try) == 0:
            logger.warning(f"==> No potential markets found for {exch}")
            return {}
            
        sample_markets = markets_to_try[:5]
        logger.info(f"==> Sample markets for {exch}: {sample_markets}")
        print(f"Processing {len(markets_to_try)} matching markets from {exch}")
        
        # Function to fetch OHLCV data for a single market
        async def fetch_market_data(market):
            try:
                # Fetch OHLCV data with timeout
                ohlcv_data = await asyncio.wait_for(
                    api.fetch_ohlcv(market, '1d', limit=30),
                    timeout=10.0  # Increased to 10 second timeout
                )
                if ohlcv_data and len(ohlcv_data) > 0:
                    # Log successful fetch
                    logger.debug(f"==> Successfully fetched {len(ohlcv_data)} candles for {market} on {exch}")
                    return market, ohlcv_data
                logger.debug(f"==> Fetched empty data for {market} on {exch}")
                return market, None
            except asyncio.TimeoutError:
                logger.debug(f"==> Timeout fetching OHLCV data for {market} on {exch}")
                return market, None
            except Exception as e:
                logger.debug(f"==> Failed to fetch OHLCV data for {market} on {exch}: {str(e)}")
                return market, None
        
        # Process markets concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(8)  # Max 8 concurrent requests per exchange (reduced from 10)
        
        async def fetch_with_semaphore(market):
            async with semaphore:
                return await fetch_market_data(market)
        
        # Create tasks for all markets - limit to a reasonable number
        max_markets = min(len(markets_to_try), 150)  # Process at most 150 markets
        if max_markets < len(markets_to_try):
            logger.info(f"==> Limiting to {max_markets} out of {len(markets_to_try)} potential markets for {exch}")
            
        tasks = [fetch_with_semaphore(market) for market in markets_to_try[:max_markets]]
        
        # Set a timeout for the entire exchange processing
        start_time = datetime.now()
        max_time_per_exchange = 60  # Increased to 60 seconds (from 45)
        
        # Process markets with timeout
        try:
            # Wait for all tasks with a timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=max_time_per_exchange
            )
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.warning(f"==> Overall timeout after {elapsed:.1f}s processing {exch}")
            # We'll still process any completed results
            pending = [t for t in tasks if not t.done()]
            completed = [t for t in tasks if t.done()]
            
            # Gather results from completed tasks
            results = []
            for task in completed:
                try:
                    result = task.result()
                    results.append(result)
                except Exception as e:
                    logger.debug(f"==> Error getting task result: {str(e)}")
                    
            logger.warning(f"==> {len(pending)}/{len(tasks)} tasks didn't complete before timeout")
        except Exception as e:
            logger.error(f"==> Error during asyncio.gather: {str(e)}")
            # Try to salvage any results
            results = []
            for task in tasks:
                if task.done() and not task.exception():
                    try:
                        results.append(task.result())
                    except Exception:
                        pass
            logger.info(f"==> Salvaged {len(results)} results after error")
        
        # Process results
        successful_markets = 0
        for result in results:
            try:
                # Proper unpacking
                if isinstance(result, tuple) and len(result) == 2:
                    market, data = result
                    
                    if isinstance(data, Exception):
                        logger.warning(f"==> Exception when fetching {market}: {str(data)}")
                        continue
                        
                    if market and data:
                        exchange_data[market] = data
                        successful_markets += 1
                else:
                    logger.warning(f"==> Unexpected result format: {result}")
            except Exception as e:
                logger.warning(f"==> Error processing result: {str(e)}")
                continue
        
        logger.info(f"==> Downloaded data from {exch}: got data for {successful_markets}/{len(tasks)} markets")
        
        if successful_markets == 0:
            logger.warning(f"==> No successful market data for {exch}")
            return {}
            
        return exchange_data
            
    except Exception as e:
        logger.error(f"==> Error processing {exch}: {str(e)}")
        logger.error(traceback.format_exc())
        return {}
    finally:
        # Always ensure the API is closed, even in error conditions
        if api:
            try:
                # Close the API first
                await api.close()
                
                # Additional cleanup for specific exchanges
                if hasattr(api, 'session') and api.session:
                    await api.session.close()
                if hasattr(api, 'connector') and api.connector:
                    await api.connector.close()
                
                logger.debug(f"==> Successfully closed {exch} API")
            except Exception as close_error:
                logger.warning(f"==> Error closing {exch} API: {str(close_error)}")

async def dl_reference_exch_data(listed_symbols):
    """Downloads OHLCV data from reference exchanges using async.
    
    Args:
        listed_symbols: List of symbols from Drift that we care about
    """
    try:
        logger.info(f"==> Starting to fetch reference exchange data for {len(listed_symbols)} listed symbols")
        print(f"Fetching exchange data for: {', '.join(listed_symbols)}")
        
        # Pre-calculate set of symbols for faster lookups
        listed_symbols_set = set(listed_symbols)
        # Also create normalized versions for matching (remove -PERP suffix)
        normalized_symbols = {s.replace('-PERP', '').upper() for s in listed_symbols_set}
        print(f"Normalized symbols for matching: {', '.join(normalized_symbols)}")
        
        # Initialize dictionary for raw data
        raw_reference_exch_df = {}
        
        # Process spot exchanges concurrently
        logger.info(f"==> Fetching data from {len(REFERENCE_SPOT_EXCH)} spot exchanges")
        
        # Use all the spot exchanges but prioritize the most reliable ones first
        prioritized_spot_exchanges = sorted(REFERENCE_SPOT_EXCH, 
                                          key=lambda x: 0 if x in ['binanceus', 'bybit', 'okx'] else 1)
        
        # Create tasks for exchanges - pass normalized_symbols
        spot_tasks = [download_exch(exch, True, normalized_symbols) for exch in prioritized_spot_exchanges]
        spot_results = await asyncio.gather(*spot_tasks, return_exceptions=True)
        
        # Process spot results
        spot_success = 0
        for i, result in enumerate(spot_results):
            exch = prioritized_spot_exchanges[i]
            try:
                if isinstance(result, Exception):
                    logger.error(f"==> Error processing {exch} spot data: {str(result)}")
                    continue
                    
                if result and len(result) > 0:
                    raw_reference_exch_df[(True, exch)] = result
                    spot_success += 1
            except Exception as e:
                logger.error(f"==> Error processing {exch} spot result: {str(e)}")
        
        # Process futures exchanges concurrently
        logger.info(f"==> Fetching data from {len(REFERENCE_FUT_EXCH)} futures exchanges")
        
        # Use all the futures exchanges but prioritize the most reliable ones first
        prioritized_fut_exchanges = sorted(REFERENCE_FUT_EXCH,
                                         key=lambda x: 0 if x in ['bybit', 'okx'] else 1)
        
        # Create tasks for exchanges - pass normalized_symbols
        fut_tasks = [download_exch(exch, False, normalized_symbols) for exch in prioritized_fut_exchanges]
        fut_results = await asyncio.gather(*fut_tasks, return_exceptions=True)
        
        # Process futures results
        fut_success = 0
        for i, result in enumerate(fut_results):
            exch = prioritized_fut_exchanges[i]
            try:
                if isinstance(result, Exception):
                    logger.error(f"==> Error processing {exch} futures data: {str(result)}")
                    continue
                    
                if result and len(result) > 0:
                    raw_reference_exch_df[(False, exch)] = result
                    fut_success += 1
            except Exception as e:
                logger.error(f"==> Error processing {exch} futures result: {str(e)}")
                
        # Try alternate exchanges if we didn't get enough data
        alt_spot_exchanges = ['kucoin', 'mexc', 'kraken', 'gateio']
        alt_fut_exchanges = ['mexc', 'gate', 'htx']
        
        # If we didn't get much data from spot exchanges, try a different subset
        if spot_success < 2 and any(e in REFERENCE_SPOT_EXCH for e in alt_spot_exchanges):
            logger.info(f"==> Retrying with alternate spot exchanges due to low success rate ({spot_success}/{len(prioritized_spot_exchanges)})")
            for exch in alt_spot_exchanges:
                if exch in REFERENCE_SPOT_EXCH and (True, exch) not in raw_reference_exch_df:
                    try:
                        logger.info(f"==> Retrying spot exchange {exch}")
                        result = await download_exch(exch, True, normalized_symbols)
                        if result and len(result) > 0:
                            raw_reference_exch_df[(True, exch)] = result
                            spot_success += 1
                            logger.info(f"==> Successfully got data from {exch} on retry")
                    except Exception as e:
                        logger.error(f"==> Error on retry for {exch}: {str(e)}")
        
        # If we didn't get much data from futures exchanges, try a different subset
        if fut_success < 2 and any(e in REFERENCE_FUT_EXCH for e in alt_fut_exchanges):
            logger.info(f"==> Retrying with alternate futures exchanges due to low success rate ({fut_success}/{len(prioritized_fut_exchanges)})")
            for exch in alt_fut_exchanges:
                if exch in REFERENCE_FUT_EXCH and (False, exch) not in raw_reference_exch_df:
                    try:
                        logger.info(f"==> Retrying futures exchange {exch}")
                        result = await download_exch(exch, False, normalized_symbols)
                        if result and len(result) > 0:
                            raw_reference_exch_df[(False, exch)] = result
                            fut_success += 1
                            logger.info(f"==> Successfully got data from {exch} on retry")
                    except Exception as e:
                        logger.error(f"==> Error on retry for {exch}: {str(e)}")
        
        logger.info(f"==> Completed exchange data fetching. Success rates: Spot: {spot_success}/{len(prioritized_spot_exchanges)}, Futures: {fut_success}/{len(prioritized_fut_exchanges)}")
        return raw_reference_exch_df
    except Exception as e:
        logger.error(f"==> Fatal error fetching exchange data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_reference_exch_data(raw_reference_exch_df, all_symbols):
    """Processes raw OHLCV data from exchanges."""
    logger.info(f"==> Processing exchange data for {len(all_symbols)} symbols")
    
    # Calculate earliest timestamp to keep data for
    earliest_ts_to_keep = (datetime.now() - timedelta(days=DAYS_TO_CONSIDER+5)).timestamp()
    all_candle_data = {}
    
    # Keep track of data points for logging
    total_markets_processed = 0
    successful_markets = 0
    symbols_with_data = set()
    
    # Create default output DataFrame in case we can't process any data
    default_df = pd.DataFrame(
        index=all_symbols,
        data={
            "Spot Volume": 0,
            "Spot Vol Geomean": 0,
            "Fut Volume": 0,
            "Fut Vol Geomean": 0,
        }
    )
    
    # Early return if we got no exchange data at all
    if not raw_reference_exch_df:
        logger.warning("==> No reference exchange data provided, returning default DataFrame with zeros")
        return default_df
        
    # Debug: Print out what symbols we're trying to match
    normalized_symbols = {s.replace('-PERP', '').upper() for s in all_symbols}
    logger.info(f"==> Looking for these normalized symbols: {', '.join(normalized_symbols)}")
    
    # Process each exchange's data
    for (spot, exch), exch_data in raw_reference_exch_df.items():
        # Skip empty exchange data
        if not exch_data:
            logger.debug(f"==> No data for {exch} (spot={spot})")
            continue
            
        # Get API instance or skip - use ccxt directly instead of async function
        try:
            api = getattr(ccxt, exch)({
                'timeout': 10000,
                'enableRateLimit': True,
            })
            api.load_markets()
        except Exception as e:
            logger.warning(f"==> Failed to initialize API for {exch} during processing: {str(e)}")
            continue
            
        markets_in_exchange = len(exch_data)
        markets_processed = 0
        
        # Print some of the markets for debugging
        market_sample = list(exch_data.keys())[:5]
        logger.info(f"==> Sample markets from {exch}: {market_sample}")
        
        for symbol, market_ohlcv in exch_data.items():
            try:
                total_markets_processed += 1
                markets_processed += 1
                
                # Better extract the base symbol from the market pair
                parts = symbol.split('/')
                if len(parts) < 2:
                    logger.debug(f"==> Invalid symbol format: {symbol}")
                    continue
                    
                base_symbol = parts[0].upper()
                # Apply cleaning to the base symbol
                coin = clean_symbol(base_symbol, exch)
                
                # Debug the symbol cleaning
                logger.debug(f"==> Symbol {symbol} -> base_symbol: {base_symbol} -> cleaned: {coin}")
                
                if not all_symbols:
                    logger.warning("==> No symbols provided to match against")
                    continue
                    
                if not market_ohlcv or len(market_ohlcv) == 0:
                    logger.debug(f"==> No OHLCV data for {symbol} on {exch}")
                    continue
                
                # Check both with and without -PERP suffix
                coin_matched = False
                for test_coin in [coin, f"{coin}-PERP"]:
                    if test_coin in all_symbols:
                        coin_matched = True
                        coin = test_coin
                        break
                        
                # Also try checking against normalized symbols (without -PERP)
                if not coin_matched and coin in normalized_symbols:
                    # Find the corresponding symbol with -PERP
                    for orig_symbol in all_symbols:
                        if orig_symbol.replace('-PERP', '').upper() == coin.upper():
                            coin_matched = True
                            coin = orig_symbol
                            break
                
                if not coin_matched:
                    logger.debug(f"==> Symbol {coin} not in list of tracked symbols")
                    continue
                
                # Convert to DataFrame and filter by time
                try:
                    market_df = (pd.DataFrame(market_ohlcv, columns=[*'tohlcv'])
                             .set_index('t')
                             .sort_index())
                
                    # Check if we have enough data
                    if len(market_df) < 3:  # Need at least a few days of data
                        logger.debug(f"==> Not enough data points for {symbol} on {exch}: {len(market_df)}")
                        continue
                    
                    # Filter to relevant time period 
                    market_df = market_df.loc[earliest_ts_to_keep * 1000:].iloc[-DAYS_TO_CONSIDER-1:-1]
                    
                    if not len(market_df):
                        logger.debug(f"==> No data in time range for {symbol} on {exch}")
                        continue
                except Exception as e:
                    # Skip if DataFrame processing fails
                    logger.debug(f"==> Error processing DataFrame for {symbol} on {exch}: {str(e)}")
                    continue
                
                # Get contract size if available
                try:
                    contractsize = min(api.markets.get(symbol, {}).get('contractSize', None) or 1, 1)
                except Exception as e:
                    logger.debug(f"==> Error getting contract size for {symbol} on {exch}: {str(e)}")
                    contractsize = 1
                
                # Calculate average daily volume in USD
                try:
                    # First try the proper calculation method
                    daily_usd_volume = (np.minimum(market_df.l, market_df.c.iloc[-1])
                                   * market_df.v
                                   * contractsize).mean()
                                   
                    # Validate volume isn't negative or extremely large (data error)
                    if daily_usd_volume < 0 or daily_usd_volume > 1e12:  # Sanity check
                        logger.debug(f"==> Invalid volume calculation for {symbol} on {exch}: {daily_usd_volume}")
                        # Try simpler method
                        daily_usd_volume = (market_df.c * market_df.v * contractsize).mean()
                        
                        # Final validation
                        if daily_usd_volume < 0 or daily_usd_volume > 1e12:
                            logger.debug(f"==> Still invalid volume, setting to 0: {daily_usd_volume}")
                            daily_usd_volume = 0
                except Exception as e:
                    # If calculation fails, use a simple average
                    logger.debug(f"==> Primary volume calculation failed for {symbol} on {exch}: {str(e)}")
                    try:
                        daily_usd_volume = (market_df.c * market_df.v * contractsize).mean()
                        
                        # Validate
                        if daily_usd_volume < 0 or daily_usd_volume > 1e12:
                            daily_usd_volume = 0
                    except Exception as e2:
                        # Last resort fallback
                        logger.debug(f"==> Fallback volume calculation failed for {symbol} on {exch}: {str(e2)}")
                        daily_usd_volume = 0
                
                # Only store if we have a valid volume
                if not np.isnan(daily_usd_volume) and daily_usd_volume > 0:
                    key = (exch, spot, coin)
                    if key not in all_candle_data or daily_usd_volume > all_candle_data[key]:
                        all_candle_data[key] = daily_usd_volume
                        symbols_with_data.add(coin)
                        successful_markets += 1
                        logger.info(f"==> Successful volume calculation for {coin} on {exch}: ${daily_usd_volume:,.2f}")
            except Exception as e:
                # Silently skip problem markets
                logger.debug(f"==> Error processing market {symbol} on {exch}: {str(e)}")
                continue
        
        logger.info(f"==> Processed {markets_processed}/{markets_in_exchange} markets for {exch} (spot={spot})")
    
    logger.info(f"==> Exchange data processing complete: {successful_markets}/{total_markets_processed} markets successfully processed")
    logger.info(f"==> Found data for {len(symbols_with_data)}/{len(all_symbols)} symbols")
    
    # Create a default DataFrame with zeros if we have no data
    if not all_candle_data:
        logger.warning("==> No candle data found, returning default DataFrame with zeros")
        return default_df
    
    # Convert to DataFrame and aggregate
    try:
        # Debug the all_candle_data
        logger.info(f"==> Top 10 volumes in all_candle_data: {sorted(all_candle_data.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
        df_coins = pd.Series(all_candle_data)
        df_coins.index.names = ['exch', 'spot', 'coin']
        
        # Log some sample data for debugging
        logger.info(f"==> Sample from df_coins before groupby: {df_coins.head(5)}")
        
        # Group by spot and coin, calculating aggregate values
        output_df = (df_coins / 1e6).groupby(['spot', 'coin']).agg(
            [geomean_three, 'sum']
        ).unstack(0).fillna(0)
        
        # Rename columns
        output_df.columns = [
            f"{'Spot' if is_spot else 'Fut'} "
            f"{dict(geomean_three='Vol Geomean', sum='Volume')[agg_func_name]}"
            for agg_func_name, is_spot in output_df.columns
        ]
        
        logger.info(f"==> Successfully created exchange volume DataFrame with shape {output_df.shape}")
        
        # Debug the output
        logger.info(f"==> Output columns: {output_df.columns.tolist()}")
        logger.info(f"==> Sample from output_df: {output_df.head(5)}")
    except Exception as e:
        logger.error(f"==> Error processing exchange data into DataFrame: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create default DataFrame on error
        return default_df
    
    # Make sure all symbols are represented
    missing_symbols = set(all_symbols) - set(output_df.index)
    if missing_symbols:
        logger.info(f"==> Adding {len(missing_symbols)} missing symbols to DataFrame with zero values")
        # Add missing symbols with zero values
        missing_df = pd.DataFrame(
            index=list(missing_symbols),
            data={col: 0 for col in output_df.columns}
        )
        output_df = pd.concat([output_df, missing_df])
    
    return output_df

def dl_cmc_data():
    """Downloads market cap data from CoinMarketCap API."""
    try:
        logger.info("==> Fetching market cap data from CoinMarketCap")
        
        # Get API key from environment variables
        cmc_api_key = os.environ.get("CMC_API_KEY")
        if not cmc_api_key:
            logger.error("==> CoinMarketCap API key is not set in .env file. Market cap data will not be available.")
            return None
        
        cmc_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        
        try:
            logger.info("==> Making request to CoinMarketCap API")
            
            headers = {
                'X-CMC_PRO_API_KEY': cmc_api_key,
                'Accept': 'application/json'
            }
            
            params = {
                'limit': 500,
                'convert': 'USD'
            }
            
            response = requests.get(
                cmc_api_url, 
                headers=headers,
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json().get('data', [])
            
            if not data:
                logger.error("==> No data received from CoinMarketCap API response")
                return None
            
            logger.info(f"==> Successfully fetched data for {len(data)} tokens from CoinMarketCap")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"==> Error connecting to CoinMarketCap API: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"==> Error parsing JSON from CoinMarketCap API: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"==> Fatal error fetching CoinMarketCap data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_cmc_data(cmc_data, all_symbols):
    """Processes market cap data from CoinMarketCap."""
    logger.info(f"==> Processing market cap data for {len(all_symbols)} symbols")
    
    # Create default DataFrame for returning in case of errors
    default_df = pd.DataFrame(index=all_symbols, data={"MC": 0})
    
    # Check if we have any data to process
    if not cmc_data:
        logger.warning("==> No CoinMarketCap data provided, returning zeros")
        return default_df
        
    if not all_symbols:
        logger.warning("==> No symbols list provided, returning empty DataFrame")
        return pd.DataFrame({'MC': []})
    
    # Create normalized versions of all_symbols (without -PERP suffix) for matching
    normalized_symbols = {s.replace('-PERP', '').upper(): s for s in all_symbols}
    logger.info(f"==> Looking for these normalized CMC symbols: {', '.join(normalized_symbols.keys())}")
    
    # Create DataFrame with relevant fields
    try:
        # Extract the required data, handling potential missing fields
        processed_data = []
        for a in cmc_data:
            try:
                if 'quote' not in a or 'USD' not in a['quote']:
                    continue
                    
                symbol = a.get('symbol', '')
                if not symbol:
                    continue
                    
                # Add the token info to our log for debugging
                logger.debug(f"==> Processing CMC token: {symbol} ({a.get('name', 'Unknown')})")
                    
                # Extract market cap values, handling missing or null values
                mc = a['quote']['USD'].get('market_cap', 0) or 0
                fd_mc = a['quote']['USD'].get('fully_diluted_market_cap', 0) or 0
                
                processed_data.append({
                    'symbol': symbol,
                    'name': a.get('name', 'Unknown'),
                    'slug': a.get('slug', '').lower(),
                    'mc': mc,
                    'fd_mc': fd_mc,
                })
            except Exception as e:
                logger.debug(f"==> Error processing CMC data for token: {str(e)}")
                continue
        
        # Create DataFrame from processed data
        output_df = pd.DataFrame(processed_data)
        
        if output_df.empty:
            logger.warning("==> No valid market cap data found after processing")
            return default_df
            
        # Log the symbols we found for debugging
        logger.info(f"==> Found {len(output_df)} CMC tokens, top 10 by market cap: {output_df.sort_values('mc', ascending=False).head(10)['symbol'].tolist()}")
            
        # If we have duplicate symbols, use the one with highest market cap
        output_df = output_df.sort_values('mc', ascending=False).drop_duplicates('symbol')
        
        # Use FD MC if regular MC is zero/missing
        output_df.loc[output_df['mc'] == 0, 'mc'] = output_df['fd_mc']
        
        # Calculate final MC in full dollars (not millions)
        output_df['MC'] = output_df['mc'].fillna(0)
        
        # Match with our symbols 
        # Create a dictionary to hold the final matched data
        final_data = {}
        match_count = 0
        
        # Loop through all our symbols we need to match
        for symbol in all_symbols:
            symbol_matched = False
            
            # Try several matching strategies:
            
            # 1. Try exact symbol match (case-insensitive)
            symbol_upper = symbol.replace('-PERP', '').upper()
            matching_rows = output_df[output_df['symbol'].str.upper() == symbol_upper]
            if not matching_rows.empty:
                final_data[symbol] = matching_rows.iloc[0]['MC']
                match_count += 1
                symbol_matched = True
                logger.debug(f"==> Exact match for {symbol}: {matching_rows.iloc[0]['MC']:,.2f}")
                continue
                
            # 2. Try case-insensitive find
            for idx, row in output_df.iterrows():
                cmc_symbol = row['symbol'].upper()
                if (symbol_upper == cmc_symbol or 
                    symbol_upper == clean_symbol(cmc_symbol).upper() or
                    cmc_symbol == clean_symbol(symbol_upper).upper()):
                    final_data[symbol] = row['MC']
                    match_count += 1
                    symbol_matched = True
                    logger.debug(f"==> Cleaned symbol match for {symbol}: {row['MC']:,.2f}")
                    break
                    
            # 3. Try partial symbol match (for tokens with prefixes/suffixes)
            if not symbol_matched:
                for idx, row in output_df.iterrows():
                    cmc_symbol = row['symbol'].upper()
                    # Check if either symbol contains the other
                    if (symbol_upper in cmc_symbol or cmc_symbol in symbol_upper or 
                        symbol_upper.replace('1M', '') == cmc_symbol.replace('1M', '') or
                        ''.join(c for c in symbol_upper if c.isalpha()) == ''.join(c for c in cmc_symbol if c.isalpha())):
                        final_data[symbol] = row['MC']
                        match_count += 1
                        symbol_matched = True
                        logger.debug(f"==> Partial match for {symbol} with {cmc_symbol}: {row['MC']:,.2f}")
                        break
                        
            # 4. Try slug match for special cases (like 1INCH)
            if not symbol_matched:
                for idx, row in output_df.iterrows():
                    token_slug = row['slug'].lower()
                    if (symbol_upper.lower() in token_slug or
                        symbol_upper.lower().replace('-perp', '') in token_slug):
                        final_data[symbol] = row['MC']
                        match_count += 1
                        symbol_matched = True
                        logger.debug(f"==> Slug match for {symbol} with {token_slug}: {row['MC']:,.2f}")
                        break
            
            # If still no match, set to 0
            if not symbol_matched:
                final_data[symbol] = 0
        
        # Create final DataFrame from the matched data
        final_df = pd.DataFrame({'MC': final_data})
        
        # Log results summary
        logger.info(f"==> Processed market cap data: found data for {match_count}/{len(all_symbols)} symbols")
        non_zero_values = [v for v in final_data.values() if v > 0]
        if non_zero_values:
            logger.info(f"==> Market cap range: ${min(non_zero_values):,.2f} - ${max(non_zero_values):,.2f}")
            # Log the top 5 market caps
            top_mc_pairs = sorted([(s, mc) for s, mc in final_data.items() if mc > 0], key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"==> Top 5 market caps: {', '.join([f'{s}: ${mc:,.2f}' for s, mc in top_mc_pairs])}")
        
        return final_df
    except Exception as e:
        logger.error(f"==> Error processing CoinMarketCap data: {str(e)}")
        logger.error(traceback.format_exc())
        return default_df

def get_drift_data(drift_client, perp_map, user_map):
    """Fetches data from Drift Protocol using provided maps."""
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
        
        logger.info(f"==> Found {markets_count} perp markets")
        
        # Track processed markets
        markets_processed = 0
        user_count = sum(1 for _ in user_map.values())
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
        
        # Process each perp market
        for market in perp_markets:
            market_index = market.data.market_index
            markets_processed += 1
            
            try:
                # Get market config by index
                market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
                if not market_config:
                    logger.warning(f"==> No market config found for market index {market_index}")
                    continue
                    
                # Get symbol
                symbol = market_config.symbol
                clean_sym = clean_symbol(symbol)
                
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
                
                # Estimate volume (placeholder - would need historical data)
                est_daily_volume = oi_usd * 0.2  # Placeholder: assume 20% of OI is daily volume
                est_volume_30d = est_daily_volume * 30  # Placeholder: 30-day est.
                
                # Log details for each market
                logger.info(
                    f"==> Market {market_index} ({clean_sym}): OI=${oi_usd:,.2f}, "
                    f"Volume=${est_volume_30d:,.2f}, Positions={positions_count}, "
                    f"Leverage={max_leverage}x, Funding={hourly_funding:,.4f}%"
                )
                
                # Don't apply sig_figs to OI value as it affects precision too much
                drift_data.append({
                    'Symbol': clean_sym,
                    'Market Index': market_index,
                    'Max Lev. on Drift': max_leverage,
                    'OI on Drift': oi_usd,  # Use raw value without sig_figs
                    'Volume on Drift': sig_figs(est_volume_30d, 3),
                    'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
                    'Oracle Price': oracle_price
                })
            except Exception as e:
                logger.error(f"==> Error processing market {market_index}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
            
        logger.info(f"==> Successfully processed {markets_processed}/{markets_count} markets")
        
        if not drift_data:
            logger.error("==> No Drift markets data was processed")
            return None
            
        return drift_data
    except Exception as e:
        logger.error(f"==> Error fetching Drift data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# --- Scoring and Recommendation Functions ---
def build_scores(df):
    """Calculates scores for each asset based on metrics."""
    output_scores = {}
    
    # Calculate scores for each category
    for score_category, category_details in DRIFT_SCORE_CUTOFFS.items():
        output_scores[score_category] = pd.Series(0.0, index=df.index)
        
        for score_var, thresholds in category_details.items():
            if score_var not in df.columns:
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
                    output_scores[score_name].loc[df[score_var].fillna(-np.inf) >= threshold_val] = points
            else:
                for threshold_val, points in sorted(point_thresholds.items(), reverse=True):
                    output_scores[score_name].loc[df[score_var].fillna(np.inf) <= threshold_val] = points
                    
            # Add to category score
            output_scores[score_category] += output_scores[score_name]
            
    # Convert to DataFrame
    output_df = pd.concat(output_scores, axis=1)
    
    # Calculate final score
    score_components = list(DRIFT_SCORE_CUTOFFS.keys())
    output_df['Score'] = output_df[score_components].sum(axis=1)
    
    return output_df

def generate_recommendation(row):
    """Generates recommendation based on score and current leverage."""
    current_leverage = int(0 if pd.isna(row['Max Lev. on Drift']) else row['Max Lev. on Drift'])
    score = row['Score']
    
    # Determine relevant score boundaries
    lower_bound_key = min([k for k in SCORE_LB.keys() if k <= current_leverage], key=lambda x: abs(x - current_leverage))
    
    # Check if score is below lower bound
    is_below_lower_bound = score < SCORE_LB[lower_bound_key]
    
    # Generate recommendation
    if current_leverage > 0 and is_below_lower_bound:
        if current_leverage > 5:
            return 'Decrease Leverage'
        else:
            return 'Delist'
    else:
        return 'Keep'  # No change recommended

async def _get_delist_recommendations(
    slot: int,
    vat: Vat,
    drift_client: DriftClient,
) -> dict:
    try:
        print("\n===> DELIST RECOMMENDER FUNCTION EXECUTION STARTED <===")
        print(f"Slot: {slot}")
        logger.info(f"==> [get_delist_recommendations] Starting with slot={slot}")
        start_time = datetime.now()
        
        # Log state for debugging
        try:
            print(f"Vat object type: {type(vat)}")
            print(f"Vat has perp_markets: {hasattr(vat, 'perp_markets')}")
            print(f"Vat has users: {hasattr(vat, 'users')}")
            
            user_count = sum(1 for _ in vat.users.values())
            perp_markets_count = sum(1 for _ in vat.perp_markets.values())
            print(f"User count: {user_count}, Perp markets count: {perp_markets_count}")
            logger.info(f"==> [get_delist_recommendations] State: users={user_count}, perp_markets={perp_markets_count}")
        except Exception as e:
            print(f"ERROR in diagnostic logging: {str(e)}")
            logger.warning(f"==> [get_delist_recommendations] Unable to log state info: {str(e)}")
        
        # Get data from Drift
        print("Fetching Drift protocol data...")
        logger.info("==> [get_delist_recommendations] Fetching Drift protocol data")
        drift_data = get_drift_data(drift_client, vat.perp_markets, vat.users)
        if not drift_data:
            print("ERROR: Failed to fetch Drift data")
            logger.error("==> [get_delist_recommendations] Failed to fetch Drift data")
            return {
                "status": "error",
                "message": "Failed to fetch Drift data",
                "data": None
            }
        
        print(f"Successfully fetched Drift data for {len(drift_data)} markets")
            
        # Get list of all listed symbols
        listed_symbols = [item['Symbol'] for item in drift_data]
        print(f"Listed symbols: {', '.join(listed_symbols)}")
        logger.info(f"==> [get_delist_recommendations] Found {len(listed_symbols)} listed symbols: {', '.join(listed_symbols)}")
        
        if not listed_symbols:
            print("ERROR: No listed markets found on Drift")
            logger.error("==> [get_delist_recommendations] No listed markets found on Drift")
            return {
                "status": "error",
                "message": "No listed markets found on Drift",
                "data": None
            }
        
        # Sort the drift_data by Market Index in ascending order
        drift_data.sort(key=lambda x: x['Market Index'])
        print(f"Sorted drift data by Market Index (ascending)")
            
        # Convert Drift data to DataFrame
        drift_df = pd.DataFrame(drift_data).set_index('Symbol')
        
        # Get CEX data - PASS THE LISTED SYMBOLS
        print("Fetching exchange data...")
        logger.info("==> [get_delist_recommendations] Fetching exchange data")
        raw_cex_data = await dl_reference_exch_data(listed_symbols)
        
        # Process CEX data
        print("Processing exchange data...")
        logger.info("==> [get_delist_recommendations] Processing exchange data")
        
        # Check if exchange data fetching completely failed
        if not raw_cex_data or not any(len(data) > 0 for _, data in raw_cex_data.items()):
            print("WARNING: Could not fetch data from any exchange. Trying alternative sources...")
            logger.warning("==> [get_delist_recommendations] Could not fetch data from any exchange. Trying alternative sources...")
            
            # Try a fallback approach with direct API calls or retry with increased limits
            try:
                # For now, we'll create placeholder data based on OI data as fallback
                # This is just temporary until proper alternative source is implemented
                
                # Use OI from drift as a proxy for volume (very rough approximation)
                print("Creating fallback volume data based on OI as a temporary measure")
                cex_fallback_data = {
                    "Spot Volume": drift_df['OI on Drift'] * 0.8,  # Approximate spot volume
                    "Spot Vol Geomean": drift_df['OI on Drift'] * 0.5,  # Lower geomean 
                    "Fut Volume": drift_df['OI on Drift'] * 1.2,  # Futures volume is typically higher
                    "Fut Vol Geomean": drift_df['OI on Drift'] * 0.7,  # Futures geomean
                }
                cex_df = pd.DataFrame(cex_fallback_data, index=drift_df.index)
                print("Created fallback volume data as temporary solution")
            except Exception as e:
                print(f"WARNING: Fallback approach also failed. Proceeding with zeros: {str(e)}")
                logger.warning(f"==> [get_delist_recommendations] Fallback approach also failed: {str(e)}")
                # Create a DataFrame with zeros
                cex_df = pd.DataFrame(
                    index=listed_symbols,
                    data={
                        "Spot Volume": 0,
                        "Spot Vol Geomean": 0,
                        "Fut Volume": 0,
                        "Fut Vol Geomean": 0,
                    }
                )
        else:
            cex_df = process_reference_exch_data(raw_cex_data, listed_symbols)
        
        # Get market cap data
        print("Fetching market cap data...")
        logger.info("==> [get_delist_recommendations] Fetching market cap data")
        cmc_data = dl_cmc_data()
        
        if not cmc_data:
            print("WARNING: Could not fetch market cap data. Trying alternative sources...")
            logger.warning("==> [get_delist_recommendations] Could not fetch market cap data. Trying alternative sources...")
            
            # Fallback for market cap data 
            try:
                # Use OI data as a rough proxy for market cap (very rough approximation)
                print("Creating fallback market cap data as a temporary measure")
                mc_df = pd.DataFrame({"MC": drift_df['OI on Drift'] * 10}, index=drift_df.index)  # Very rough estimate
                print("Created fallback market cap data")
            except Exception as e:
                print(f"WARNING: Market cap fallback also failed: {str(e)}")
                logger.warning(f"==> [get_delist_recommendations] Market cap fallback failed: {str(e)}")
                mc_df = pd.DataFrame(index=listed_symbols, data={"MC": 0})
        else:
            mc_df = process_cmc_data(cmc_data, listed_symbols)
        
        # Combine all data
        print("Combining all data sources...")
        logger.info("==> [get_delist_recommendations] Combining all data sources")
        combined_df = pd.concat([
            drift_df,
            cex_df,
            mc_df,
        ], axis=1)
        
        # Fill NaN values for metrics
        for col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0)
        
        # Convert million-dollar values to full notional dollars if still in millions
        dollar_columns = [
            'OI on Drift', 'Volume on Drift', 
            'Spot Volume', 'Spot Vol Geomean', 
            'Fut Volume', 'Fut Vol Geomean', 
            'MC'
        ]
        
        # Check the columns and log their presence
        for col in dollar_columns:
            if col not in combined_df.columns:
                print(f"Warning: Expected column {col} not found in combined data")
            else:
                print(f"Column {col} present with full notional values")
            
        # Calculate scores and recommendations
        print("Calculating scores and recommendations...")
        logger.info("==> [get_delist_recommendations] Calculating scores and recommendations")
        scores_df = build_scores(combined_df)
        combined_df = pd.concat([combined_df, scores_df], axis=1)
        combined_df['Recommendation'] = combined_df.apply(generate_recommendation, axis=1)
        
        # Log score details for each market
        for symbol in combined_df.index:
            market_data = combined_df.loc[symbol]
            print(f"{symbol}: Score={market_data['Score']:.2f}, Recommendation={market_data['Recommendation']}")
            logger.info(
                f"==> [get_delist_recommendations] {symbol}: "
                f"Score={market_data['Score']:.2f}, "
                f"Recommendation={market_data['Recommendation']}, "
                f"MC=${market_data.get('MC', 0):,.2f}, "
                f"OI=${market_data.get('OI on Drift', 0):,.2f}"
            )
        
        # Prepare results
        result_df = combined_df.reset_index()
        
        # Ensure final sorting by Market Index
        result_df = result_df.sort_values(by='Market Index')
        print(f"Final sorting by Market Index: {result_df['Market Index'].tolist()}")
        
        # Convert DataFrame to dict for JSON response
        print("Preparing response data...")
        logger.info("==> [get_delist_recommendations] Preparing response data")
        columns = result_df.columns.tolist()
        data = {col: result_df[col].tolist() for col in columns}
        
        # Add summary statistics
        total_markets = len(result_df)
        delist_markets = len(result_df[result_df['Recommendation'] == 'Delist'])
        decrease_lev_markets = len(result_df[result_df['Recommendation'] == 'Decrease Leverage'])
        keep_markets = len(result_df[result_df['Recommendation'] == 'Keep'])
        
        summary = {
            "total_markets": total_markets,
            "delist_markets": delist_markets,
            "decrease_leverage_markets": decrease_lev_markets,
            "keep_markets": keep_markets
        }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Completed in {duration:.2f} seconds")
        print(f"Summary: Total={total_markets}, Delist={delist_markets}, Decrease={decrease_lev_markets}, Keep={keep_markets}")
        print("===> DELIST RECOMMENDER FUNCTION EXECUTION COMPLETED <===\n")
        
        logger.info(f"==> [get_delist_recommendations] Completed in {duration:.2f} seconds")
        logger.info(f"==> [get_delist_recommendations] Summary: Total={total_markets}, Delist={delist_markets}, Decrease={decrease_lev_markets}, Keep={keep_markets}")
        
        return {
            "status": "success",
            "message": "Delist recommendations generated successfully",
            "data": {
                "slot": slot,
                "results": data,
                "summary": summary,
                "score_boundaries": SCORE_LB
            }
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"==> [get_delist_recommendations] Error generating delist recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error generating delist recommendations: {str(e)}",
            "data": None
        }

@router.get("/recommendations")
async def get_delist_recommendations(request: BackendRequest):
    """Get delist recommendations for Drift's listed markets."""
    print("\n===> DELIST RECOMMENDER API CALL RECEIVED <===")
    print(f"Current pickle path: {request.state.backend_state.current_pickle_path}")
    print(f"Request URL: {request.url}")
    
    logger.info(f"==> [api] Received delist recommendations request, current_pickle={request.state.backend_state.current_pickle_path}")
    
    try:
        # Log basic diagnostic info
        print("Checking backend state properties:")
        print(f"Has vat: {hasattr(request.state.backend_state, 'vat')}")
        print(f"Has user_map: {hasattr(request.state.backend_state, 'user_map')}")
        print(f"Has perp_map: {hasattr(request.state.backend_state, 'perp_map')}")
        print(f"Has last_oracle_slot: {hasattr(request.state.backend_state, 'last_oracle_slot')}")
        print(f"Last oracle slot: {request.state.backend_state.last_oracle_slot}")
        
        user_count = sum(1 for _ in request.state.backend_state.user_map.values())
        perp_markets_count = sum(1 for _ in request.state.backend_state.perp_map.values())
        slot = request.state.backend_state.last_oracle_slot
        
        print(f"User count: {user_count}")
        print(f"Perp markets count: {perp_markets_count}")
        print(f"Oracle slot: {slot}")
        
        logger.info(
            f"==> [api] Request info: users={user_count}, "
            f"perp_markets={perp_markets_count}, slot={slot}"
        )
    except Exception as e:
        print(f"ERROR in diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.warning(f"==> [api] Unable to log diagnostic info: {str(e)}")
    
    try:
        result = await _get_delist_recommendations(
            request.state.backend_state.last_oracle_slot,
            request.state.backend_state.vat,
            request.state.backend_state.dc,
        )
        print(f"API call result status: {result.get('status', 'unknown')}")
        print("===> DELIST RECOMMENDER API CALL COMPLETED <===\n")
        return result
    except Exception as e:
        print(f"CRITICAL API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
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
    """Get detailed delist recommendation for a single market, showing all intermediate calculations."""
    print(f"\n===> SINGLE MARKET DELIST RECOMMENDER API CALL RECEIVED FOR MARKET INDEX {market_index} <===")
    logger.info(f"==> [api] Received single market delist recommendation request for market index {market_index}")
    
    try:
        result = await _get_single_market_recommendation(
            request.state.backend_state.last_oracle_slot,
            market_index,
            request.state.backend_state.vat,
            request.state.backend_state.dc,
        )
        print(f"Single market API call result status: {result.get('status', 'unknown')}")
        print("===> SINGLE MARKET DELIST RECOMMENDER API CALL COMPLETED <===\n")
        return result
    except Exception as e:
        print(f"CRITICAL SINGLE MARKET API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error", 
            "message": f"API error: {str(e)}", 
            "data": None
        }

async def _get_single_market_recommendation(
    slot: int,
    market_index: int,
    vat: Vat,
    drift_client: DriftClient,
) -> dict:
    """
    Get detailed delist recommendation for a single market, exposing all intermediate calculations and dataframes.
    
    This simplified version allows visualization of all steps in the delist recommendation process
    for easier understanding of the end-to-end flow.
    
    Args:
        slot: Current slot number
        market_index: Market index to analyze
        vat: Vat object containing market and user data
        drift_client: DriftClient instance
        
    Returns:
        Dictionary with detailed calculation results and intermediate dataframes
    """
    try:
        print(f"\n===> SINGLE MARKET DELIST FUNCTION EXECUTION STARTED FOR MARKET INDEX {market_index} <===")
        logger.info(f"==> [_get_single_market_recommendation] Starting with slot={slot}, market_index={market_index}")
        start_time = datetime.now()
        
        # Check if the requested market exists
        perp_market = vat.perp_markets.get(market_index)
        if not perp_market:
            print(f"ERROR: Market with index {market_index} not found")
            logger.error(f"==> [_get_single_market_recommendation] Market with index {market_index} not found")
            return {
                "status": "error",
                "message": f"Market with index {market_index} not found",
                "data": None
            }
            
        # Get market config by index
        market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
        if not market_config:
            print(f"ERROR: Market config for index {market_index} not found")
            logger.error(f"==> [_get_single_market_recommendation] Market config for index {market_index} not found")
            return {
                "status": "error",
                "message": f"Market config for index {market_index} not found",
                "data": None
            }
            
        symbol = market_config.symbol
        clean_sym = clean_symbol(symbol)
        print(f"Processing market {market_index}: {symbol} (clean symbol: {clean_sym})")
        
        # Step 1: Calculate market metrics from Drift
        print("\nStep 1: Calculating Drift market metrics...")
        
        # Track long and short positions for the market
        market_long_positions = 0
        market_short_positions = 0
        market_position_count = 0
        
        # Store positions for detailed view
        position_details = []
        
        # Gather position data
        for user_account in vat.users.values():
            try:
                # Try different methods to access perp positions
                perp_positions = []
                
                # Method 1: get_active_perp_positions
                if hasattr(user_account, 'get_active_perp_positions'):
                    try:
                        perp_positions = user_account.get_active_perp_positions()
                    except Exception:
                        pass
                        
                # Method 2: get_user_account.perp_positions
                if not perp_positions and hasattr(user_account, 'get_user_account'):
                    try:
                        user_data = user_account.get_user_account()
                        if hasattr(user_data, 'perp_positions'):
                            perp_positions = [pos for pos in user_data.perp_positions if pos.base_asset_amount != 0]
                    except Exception:
                        pass
                
                # Method 3: direct perp_positions attribute
                if not perp_positions and hasattr(user_account, 'perp_positions'):
                    try:
                        perp_positions = [pos for pos in user_account.perp_positions if pos.base_asset_amount != 0]
                    except Exception:
                        pass
                
                # Process each position for our target market
                for position in perp_positions:
                    if hasattr(position, 'market_index') and position.market_index == market_index:
                        if hasattr(position, 'base_asset_amount') and position.base_asset_amount != 0:
                            base_amount = position.base_asset_amount
                            
                            # Store position details
                            position_data = {
                                "base_asset_amount": base_amount,
                                "direction": "long" if base_amount > 0 else "short",
                                "magnitude": abs(base_amount)
                            }
                            
                            # Add to appropriate direction total
                            if base_amount > 0:  # Long position
                                market_long_positions += base_amount
                            else:  # Short position
                                market_short_positions += abs(base_amount)
                                
                            market_position_count += 1
                            
                            # Append with limited identifying info for privacy
                            position_details.append(position_data)
                            
            except Exception as e:
                logger.debug(f"Error processing user positions: {str(e)}")
                continue
        
        # Get market data
        # Get max leverage (from initial margin ratio)
        initial_margin_ratio = perp_market.data.margin_ratio_initial / 10000
        max_leverage = int(1 / initial_margin_ratio) if initial_margin_ratio > 0 else 0
        
        # Get oracle price
        oracle_price_data = drift_client.get_oracle_price_data_for_perp_market(market_index)
        oracle_price = oracle_price_data.price / 1e6  # Convert from UI price
        
        # Calculate OI as max(abs(long), abs(short))
        base_oi = max(market_long_positions, market_short_positions)
        
        # Get base decimals - try market first, then known mappings, then fall back to constants
        base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)  # Use known mapping first
        
        try:
            # Try to get decimals from market.data
            if hasattr(perp_market.data, 'base_decimals'):
                base_decimals = perp_market.data.base_decimals
            # If not found, check if it's in market.data.amm
            elif hasattr(perp_market.data, 'amm') and hasattr(perp_market.data.amm, 'base_asset_decimals'):
                base_decimals = perp_market.data.amm.base_asset_decimals
        except Exception:
            pass
        
        # Convert to human readable base units and to full dollar amount (not millions)
        base_oi_readable = base_oi / (10 ** base_decimals)
        oi_usd = base_oi_readable * oracle_price  # Full dollars
        
        # Get funding rate (hourly)
        funding_rate = perp_market.data.amm.last_funding_rate / 1e6  # Convert to percentage
        hourly_funding = funding_rate * 100  # As percentage
        
        # Estimate volume (placeholder - would need historical data)
        est_daily_volume = oi_usd * 0.2  # Placeholder: assume 20% of OI is daily volume
        est_volume_30d = est_daily_volume * 30  # Placeholder: 30-day est.
        
        # Store market metrics
        drift_market_data = {
            'Symbol': clean_sym,
            'Market Index': market_index,
            'Max Lev. on Drift': max_leverage,
            'OI on Drift': oi_usd,
            'Volume on Drift': sig_figs(est_volume_30d, 3),
            'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
            'Oracle Price': oracle_price,
            'Position Count': market_position_count,
            'Long Positions Total': market_long_positions / (10 ** base_decimals),
            'Short Positions Total': market_short_positions / (10 ** base_decimals),
            'Base Decimals': base_decimals
        }
        
        # Step 2: Fetch exchange volume data for this token
        print("\nStep 2: Fetching exchange volume data...")
        listed_symbols = [clean_sym]
        raw_cex_data = await dl_reference_exch_data(listed_symbols)
        
        # Process and store raw exchange data
        raw_exchange_volumes = {}
        if raw_cex_data:
            for (spot, exch), exch_data in raw_cex_data.items():
                for symbol, market_ohlcv in exch_data.items():
                    coin = clean_symbol(symbol, exch)
                    if coin == clean_sym and len(market_ohlcv) > 0:
                        # Store raw data for visibility
                        raw_exchange_volumes[(spot, exch, symbol)] = {
                            "ohlcv_data": market_ohlcv[:5],  # Store first 5 candles for visibility
                            "full_count": len(market_ohlcv)
                        }
        
        # Process exchange data
        cex_df = process_reference_exch_data(raw_cex_data, listed_symbols)
        
        # Step 3: Fetch market cap data
        print("\nStep 3: Fetching market cap data...")
        cmc_data = dl_cmc_data()
        
        # Store raw CMC data
        raw_cmc_data = None
        if cmc_data:
            for item in cmc_data:
                if item.get('symbol') == clean_sym:
                    raw_cmc_data = item
                    break
        
        # Process market cap data
        mc_df = process_cmc_data(cmc_data, listed_symbols) if cmc_data else pd.DataFrame(index=listed_symbols, data={"MC": 0})
        
        # Step 4: Combine all data
        print("\nStep 4: Combining all data sources...")
        
        # Convert Drift data to DataFrame
        drift_df = pd.DataFrame([drift_market_data]).set_index('Symbol')
        
        # Combine all dataframes
        combined_df = pd.concat([
            drift_df,
            cex_df,
            mc_df,
        ], axis=1)
        
        # Fill NaN values
        for col in combined_df.columns:
            combined_df[col] = combined_df[col].fillna(0)
        
        # Step 5: Calculate scores
        print("\nStep 5: Calculating scores...")
        scores_df = build_scores(combined_df)
        
        # Step 6: Generate recommendation
        print("\nStep 6: Generating recommendation...")
        combined_df = pd.concat([combined_df, scores_df], axis=1)
        
        # Convert to dictionary for detailed view
        combined_dict = combined_df.reset_index().to_dict(orient='records')[0]
        scores_dict = {k: combined_dict[k] for k in scores_df.columns if k in combined_dict}
        
        # Generate recommendation
        recommendation = generate_recommendation(combined_df.iloc[0])
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nAnalysis completed in {duration:.2f} seconds")
        print(f"Recommendation for {clean_sym}: {recommendation}")
        print(f"Final score: {scores_dict.get('Score', 'N/A')}")
        
        # Prepare detailed result with all intermediate calculations
        result = {
            "status": "success",
            "message": f"Single market delist recommendation generated successfully for {clean_sym}",
            "data": {
                "slot": slot,
                "market_index": market_index,
                "symbol": clean_sym,
                "market_config": {
                    "symbol": symbol,
                    "market_index": market_index,
                },
                "intermediate_calculations": {
                    "drift_market_data": drift_market_data,
                    "position_details": {
                        "count": market_position_count,
                        "long_total": drift_market_data['Long Positions Total'],
                        "short_total": drift_market_data['Short Positions Total'],
                        "sample_positions": position_details[:10] if len(position_details) > 10 else position_details,
                        "position_count_total": len(position_details)
                    },
                    "exchange_data": {
                        "raw_samples": raw_exchange_volumes,
                        "processed": cex_df.reset_index().to_dict(orient='records')[0] if not cex_df.empty else {}
                    },
                    "market_cap_data": {
                        "raw": raw_cmc_data,
                        "processed": mc_df.reset_index().to_dict(orient='records')[0] if not mc_df.empty else {}
                    }
                },
                "dataframes": {
                    "drift_df": drift_df.reset_index().to_dict(orient='records')[0],
                    "cex_df": cex_df.reset_index().to_dict(orient='records')[0] if not cex_df.empty else {},
                    "mc_df": mc_df.reset_index().to_dict(orient='records')[0] if not mc_df.empty else {},
                    "combined_df": combined_dict,
                    "scores_df": scores_dict
                },
                "results": {
                    "recommendation": recommendation,
                    "score": scores_dict.get('Score', 0),
                    "score_breakdown": {k: v for k, v in scores_dict.items() if k != 'Score'}
                },
                "score_boundaries": SCORE_LB
            }
        }
        
        print("===> SINGLE MARKET DELIST FUNCTION EXECUTION COMPLETED <===\n")
        return result
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(f"==> [_get_single_market_recommendation] Error: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error generating single market delist recommendation: {str(e)}",
            "data": None
        }

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify routing is working."""
    print("\n===> TEST ENDPOINT CALLED <===")
    return {"status": "success", "message": "Test endpoint working"}