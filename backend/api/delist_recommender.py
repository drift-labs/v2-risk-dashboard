# This script is used to provide backend functionality for /src/page/delist_recommender.py

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

# Drift API configuration
DRIFT_DATA_API_BASE_URL = "https://y7n4m4tnpb.execute-api.eu-west-1.amazonaws.com"
DRIFT_DATA_API_HEADERS = {"X-Origin-Verify": "AolCE35uXby9TJHHgoz6"}
API_RATE_LIMIT_INTERVAL = 0.1  # seconds between requests

# Drift Score Boost - These are symbols that get a score boost in the delist recommender
DRIFT_SCORE_BOOST_SYMBOLS = {
    "DRIFT-PERP",
}

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

# Score boundaries for delist recommendations
SCORE_UB = {0: 62, 3: 75, 5: 85, 10: 101} # Upper Bound: If score >= this, consider increasing leverage
SCORE_LB = {0: 0, 5: 37, 10: 48, 20: 60}  # Lower Bound: If score < this, consider decreasing leverage/delisting

# Reference exchanges for market data
REFERENCE_SPOT_EXCH = {
    'coinbase', 'okx', 'gate', 'kucoin', 'mexc', 'kraken', 'htx'
}

REFERENCE_FUT_EXCH = {
    'okx', 'gate', 'mexc', 'htx', 'bitmex', 'bingx', 'xt'
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
    # Ensure series contains only positive values for log calculation
    positive_series = series[series > 0]
    if len(positive_series) < 3:
        # Not enough data points for top 3, return 0 or handle as appropriate
        return 0
    # Use log1p and expm1 for numerical stability if values can be close to zero
    return np.expm1(np.log1p(positive_series).nlargest(3).sum() / 3)

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

                # Check if this market matches any of our Drift symbols (ignore IGNORED_SYMBOLS)
                if (base_symbol not in IGNORED_SYMBOLS and clean_base not in IGNORED_SYMBOLS and
                   (base_symbol in normalized_symbols or
                    clean_base in normalized_symbols or
                    any(symbol.upper().startswith(base_symbol) for symbol in normalized_symbols) or
                    any(symbol.upper().startswith(clean_base) for symbol in normalized_symbols))):
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
        listed_symbols: List of symbols from Drift that we care about (excluding ignored ones)
    """
    try:
        logger.info(f"==> Starting to fetch reference exchange data for {len(listed_symbols)} listed symbols (after filtering)")
        print(f"Fetching exchange data for: {', '.join(listed_symbols)}")

        # Pre-calculate set of symbols for faster lookups
        listed_symbols_set = set(listed_symbols)
        # Also create normalized versions for matching (remove -PERP suffix)
        # Filter out ignored symbols here as well
        normalized_symbols = {s.replace('-PERP', '').upper() for s in listed_symbols_set if s not in IGNORED_SYMBOLS}
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
    """Processes raw OHLCV data from exchanges to calculate average daily volumes (in full dollars).

    Args:
        raw_reference_exch_df: Dictionary of raw OHLCV data.
        all_symbols: List of symbols to include (already filtered for ignored ones).
    """
    logger.info(f"==> Processing exchange data for {len(all_symbols)} symbols")

    # Calculate earliest timestamp to keep data for
    earliest_ts_to_keep = (datetime.now() - timedelta(days=DAYS_TO_CONSIDER+5)).timestamp()
    all_candle_data = {} # Stores average daily volume for each (exch, spot, coin)

    # Keep track of data points for logging
    total_markets_processed = 0
    successful_markets = 0
    symbols_with_data = set()

    # Create default columns for output DataFrame
    default_columns = {
        "Spot Volume": 0,      # Sum of avg daily spot volumes across exchanges (in $m)
        "Spot Vol Geomean": 0, # Geomean of top 3 avg daily spot volumes (in $m)
        "Fut Volume": 0,       # Sum of avg daily futures volumes across exchanges (in $m)
        "Fut Vol Geomean": 0,  # Geomean of top 3 avg daily futures volumes (in $m)
    }

    # Create default output DataFrame in case we can't process any data
    default_df = pd.DataFrame(index=all_symbols, data=default_columns)

    # Early return if we got no exchange data at all
    if not raw_reference_exch_df:
        logger.warning("==> No reference exchange data provided, returning default DataFrame with zeros")
        return default_df

    # Debug: Print out what symbols we're trying to match
    normalized_symbols = {s.replace('-PERP', '').upper() for s in all_symbols if s not in IGNORED_SYMBOLS} # Ensure ignored are excluded
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

                # Skip if the cleaned base symbol is in the ignored list
                if coin in IGNORED_SYMBOLS:
                    logger.debug(f"==> Skipping ignored symbol: {coin} (from {symbol})")
                    continue

                # Debug the symbol cleaning
                logger.debug(f"==> Symbol {symbol} -> base_symbol: {base_symbol} -> cleaned: {coin}")

                if not all_symbols:
                    logger.warning("==> No symbols provided to match against")
                    continue

                if not market_ohlcv or len(market_ohlcv) == 0:
                    logger.debug(f"==> No OHLCV data for {symbol} on {exch}")
                    continue

                # Check both with and without -PERP suffix against the filtered all_symbols list
                coin_matched = False
                final_coin_symbol = None # Store the symbol from all_symbols that matched
                for test_coin in [coin, f"{coin}-PERP"]:
                    if test_coin in all_symbols: # all_symbols is already filtered
                        coin_matched = True
                        final_coin_symbol = test_coin
                        break

                # Also try checking against normalized symbols (without -PERP)
                if not coin_matched and coin in normalized_symbols: # normalized_symbols is filtered
                    # Find the corresponding symbol with -PERP from the filtered list
                    for orig_symbol in all_symbols:
                        if orig_symbol.replace('-PERP', '').upper() == coin.upper():
                            coin_matched = True
                            final_coin_symbol = orig_symbol
                            break

                if not coin_matched or final_coin_symbol is None:
                    logger.debug(f"==> Symbol {coin} not in list of tracked symbols (post-filter)")
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

                # Calculate average daily volume in USD (full dollar amount)
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
                    key = (exch, spot, final_coin_symbol) # Use the matched symbol from all_symbols
                    if key not in all_candle_data or daily_usd_volume > all_candle_data.get(key, 0):
                         all_candle_data[key] = daily_usd_volume # Store the average daily volume in dollars
                         symbols_with_data.add(final_coin_symbol)
                         successful_markets += 1
                         logger.info(f"==> Successful volume calculation for {final_coin_symbol} on {exch}: ${daily_usd_volume:,.2f}")
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
        # Debug the all_candle_data (average daily volumes in dollars)
        logger.info(f"==> Top 10 avg daily volumes in all_candle_data: {sorted(all_candle_data.items(), key=lambda x: x[1], reverse=True)[:10]}")

        df_coins = pd.Series(all_candle_data) # Values are average daily USD volume
        df_coins.index.names = ['exch', 'spot', 'coin']

        # Log some sample data for debugging
        logger.info(f"==> Sample from df_coins before groupby: {df_coins.head(5)}")

        # Group by spot and coin, calculating aggregate values (sum of avg daily vols, geomean of top 3 avg daily vols)
        # Perform aggregation on raw dollar values FIRST
        aggregated_df_raw = (df_coins.groupby(['spot', 'coin']).agg(
            [geomean_three, 'sum']
        ).unstack(0).fillna(0))

        # NOW divide by 1e6 to convert to millions for final output
        output_df = aggregated_df_raw / 1e6

        # Rename columns for clarity (indicating values are in $m)
        output_df.columns = [
            f"{'Spot' if is_spot else 'Fut'} "
            f"{dict(geomean_three='Vol Geomean $m', sum='Volume $m')[agg_func_name]}"
            for agg_func_name, is_spot in output_df.columns
        ]

        logger.info(f"==> Successfully created exchange volume DataFrame with shape {output_df.shape}")

        # Debug the output
        logger.info(f"==> Output columns: {output_df.columns.tolist()}")
        logger.info(f"==> Sample from output_df (in $m): {output_df.head(5)}")
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

    # Rename columns to match expected input for scoring (without $m)
    # Scoring function expects full dollar amounts based on cutoffs
    output_df = output_df.rename(columns={
        'Spot Volume $m': 'Spot Volume',
        'Spot Vol Geomean $m': 'Spot Vol Geomean',
        'Fut Volume $m': 'Fut Volume',
        'Fut Vol Geomean $m': 'Fut Vol Geomean',
    })
    # Multiply columns back by 1e6 to pass full dollar amounts to scoring function
    scoring_cols = ['Spot Volume', 'Spot Vol Geomean', 'Fut Volume', 'Fut Vol Geomean']
    for col in scoring_cols:
        if col in output_df.columns:
            output_df[col] = output_df[col] * 1e6

    logger.info(f"==> Final CEX volume columns for scoring: {output_df.columns.tolist()}")
    logger.info(f"==> Sample CEX volume data for scoring (full dollars): {output_df.head()}")

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
    """Processes market cap data from CoinMarketCap.

    Args:
        cmc_data: List of data from CMC API.
        all_symbols: List of symbols to include (already filtered for ignored ones).
    """
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
    # all_symbols is already filtered for ignored symbols
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

                # Skip if the symbol is in the ignored list
                if symbol in IGNORED_SYMBOLS or clean_symbol(symbol) in IGNORED_SYMBOLS:
                    logger.debug(f"==> Skipping ignored CMC token: {symbol}")
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

        # Match with our symbols (which are already filtered)
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
        
        # Get volumes in bulk - use await instead of asyncio.run()
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

# --- Scoring and Recommendation Functions ---
def build_scores(df):
    """Calculates scores for each asset based on metrics."""
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
                    output_scores[score_name].loc[df[score_var].fillna(-np.inf) >= threshold_val] = points
            else: # Linear decreasing score
                for threshold_val, points in sorted(point_thresholds.items(), reverse=True):
                    output_scores[score_name].loc[df[score_var].fillna(np.inf) <= threshold_val] = points

            # Add to category score
            output_scores[score_category] += output_scores[score_name]

    # Convert to DataFrame
    output_df = pd.concat(output_scores, axis=1)

    # Calculate final score
    score_components = list(DRIFT_SCORE_CUTOFFS.keys())
    output_df['Score'] = output_df[score_components].sum(axis=1)

    # Apply score boost for specific symbols
    output_df['Score'] = output_df['Score'].add(
        pd.Series(
            [DRIFT_SCORE_BOOST_AMOUNT if symbol in DRIFT_SCORE_BOOST_SYMBOLS else 0 
             for symbol in output_df.index],
            index=output_df.index
        )
    )

    return output_df

def generate_recommendation(row):
    """Generates recommendation based on score and current leverage."""
    current_leverage = int(0 if pd.isna(row['Max Lev. on Drift']) else row['Max Lev. on Drift'])
    score = row['Score']

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
    if current_leverage > 0 and is_below_lower_bound:
        if current_leverage > 5: # If leverage is higher than 5x and score is low, recommend decrease
            return 'Decrease Leverage'
        else: # If leverage is 5x or lower and score is low, recommend delist
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

        # Get data from Drift (this function now filters ignored symbols)
        print("Fetching Drift protocol data...")
        logger.info("==> [get_delist_recommendations] Fetching Drift protocol data")
        drift_data = await get_drift_data(drift_client, vat.perp_markets, vat.users)
        if not drift_data:
            print("ERROR: Failed to fetch Drift data or no valid markets remain after filtering")
            logger.error("==> [get_delist_recommendations] Failed to fetch Drift data or no valid markets remain after filtering")
            return {
                "status": "error",
                "message": "Failed to fetch Drift data or no valid markets remain after filtering",
                "data": None
            }

        print(f"Successfully fetched Drift data for {len(drift_data)} markets (after filtering ignored)")

        # Get list of all listed symbols (already filtered)
        listed_symbols = [item['Symbol'] for item in drift_data]
        print(f"Listed symbols for analysis: {', '.join(listed_symbols)}")
        logger.info(f"==> [get_delist_recommendations] Found {len(listed_symbols)} listed symbols for analysis: {', '.join(listed_symbols)}")

        if not listed_symbols:
            print("ERROR: No listed markets left to analyze after filtering")
            logger.error("==> [get_delist_recommendations] No listed markets left to analyze after filtering")
            return {
                "status": "error",
                "message": "No listed markets left to analyze after filtering",
                "data": None
            }

        # Sort the drift_data by Market Index in ascending order
        drift_data.sort(key=lambda x: x['Market Index'])
        print(f"Sorted drift data by Market Index (ascending)")

        # Convert Drift data to DataFrame
        drift_df = pd.DataFrame(drift_data).set_index('Symbol')

        # Get CEX data - PASS THE FILTERED LISTED SYMBOLS
        print("Fetching exchange data...")
        logger.info("==> [get_delist_recommendations] Fetching exchange data")
        raw_cex_data = await dl_reference_exch_data(listed_symbols) # Pass filtered list

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
                    # These are now full dollar amounts expected by scoring
                    "Spot Volume": drift_df['OI on Drift'] * 0.8,  # Approximate spot volume sum
                    "Spot Vol Geomean": drift_df['OI on Drift'] * 0.5,  # Approximate spot geomean
                    "Fut Volume": drift_df['OI on Drift'] * 1.2,  # Futures volume sum is typically higher
                    "Fut Vol Geomean": drift_df['OI on Drift'] * 0.7,  # Approximate futures geomean
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
                        # Expect full dollar amounts
                        "Spot Volume": 0,
                        "Spot Vol Geomean": 0,
                        "Fut Volume": 0,
                        "Fut Vol Geomean": 0,
                    }
                )
        else:
            # process_reference_exch_data now returns full dollar amounts, expects filtered list
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
                mc_df = pd.DataFrame({"MC": drift_df['OI on Drift'] * 10}, index=drift_df.index)  # Very rough estimate (full dollars)
                print("Created fallback market cap data")
            except Exception as e:
                print(f"WARNING: Market cap fallback also failed: {str(e)}")
                logger.warning(f"==> [get_delist_recommendations] Market cap fallback failed: {str(e)}")
                mc_df = pd.DataFrame(index=listed_symbols, data={"MC": 0})
        else:
            # process_cmc_data returns full dollar amounts, expects filtered list
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

        # Verify columns needed for scoring exist and log sample data (should be full dollars)
        scoring_input_cols = ['MC', 'Spot Volume', 'Spot Vol Geomean', 'Fut Volume', 'Fut Vol Geomean', 'Volume on Drift', 'OI on Drift']
        logger.info("==> Sample data being passed to scoring function (expecting full dollars):")
        for col in scoring_input_cols:
            if col not in combined_df.columns:
                logger.warning(f"==> Column '{col}' missing for scoring!")
            else:
                 logger.info(f"Sample data for {col}: {combined_df[col].head().tolist()}")

        # Calculate scores and recommendations
        print("Calculating scores and recommendations...")
        logger.info("==> [get_delist_recommendations] Calculating scores and recommendations")
        scores_df = build_scores(combined_df) # Expects full dollar inputs
        combined_df = pd.concat([combined_df, scores_df], axis=1)
        combined_df['Recommendation'] = combined_df.apply(generate_recommendation, axis=1)

        # Log score details for each market
        for symbol in combined_df.index:
            market_data = combined_df.loc[symbol]
            # Format volume/MC/OI columns back to millions for readable logging/output if needed
            mc_display = market_data.get('MC', 0) / 1e6
            oi_display = market_data.get('OI on Drift', 0) / 1e6
            print(f"{symbol}: Score={market_data['Score']:.2f}, Recommendation={market_data['Recommendation']}")
            logger.info(
                f"==> [get_delist_recommendations] {symbol}: "
                f"Score={market_data['Score']:.2f}, "
                f"Recommendation={market_data['Recommendation']}, "
                f"MC=${mc_display:,.2f}m, " # Display in millions
                f"OI=${oi_display:,.2f}m" # Display in millions
            )

        # Prepare results
        result_df = combined_df.reset_index()

        # Ensure final sorting by Market Index
        result_df = result_df.sort_values(by='Market Index')
        print(f"Final sorting by Market Index: {result_df['Market Index'].tolist()}")

        # Convert DataFrame to dict for JSON response
        # Convert relevant columns back to millions ($m) for the final JSON output if desired
        # Or keep as full dollars if the frontend expects that
        # For now, keeping full dollars as calculated, frontend can format
        print("Preparing response data...")
        logger.info("==> [get_delist_recommendations] Preparing response data")
        # Convert NaN to None for JSON compatibility
        result_df = result_df.replace({np.nan: None})
        columns = result_df.columns.tolist()
        # Convert numpy types to native Python types
        data = {}
        for col in columns:
             # Convert pandas numeric types to float/int, handle potential NaNs converted to None
            if pd.api.types.is_numeric_dtype(result_df[col].dtype):
                 data[col] = [x if x is None else (int(x) if x == int(x) else float(x)) for x in result_df[col]]
            else:
                 data[col] = result_df[col].tolist()

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

        # Check if the market symbol is in the ignored list
        if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
             print(f"ERROR: Market {market_index} ({symbol}) is in the ignored list.")
             logger.error(f"==> [_get_single_market_recommendation] Market {market_index} ({symbol}) is ignored.")
             return {
                 "status": "error",
                 "message": f"Market {market_index} ({symbol}) is explicitly ignored and cannot be analyzed.",
                 "data": None
             }

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

        # Fetch actual volume from the Drift API
        print(f"\nStep 2: Fetching actual trade volume data for {symbol} from Drift API...")
        
        # Fetch volume data from API
        volume_usd = await calculate_market_volume(symbol)
        print(f"Retrieved actual 30-day volume for {symbol}: ${volume_usd:,.2f}")

        # Store market metrics (using full dollar amounts for calculations)
        drift_market_data = {
            'Symbol': clean_sym,
            'Market Index': market_index,
            'Max Lev. on Drift': max_leverage,
            'OI on Drift': oi_usd, # Full dollars
            'Volume on Drift': volume_usd, # Actual volume from API
            'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
            'Oracle Price': oracle_price,
            'Position Count': market_position_count,
            'Long Positions Total': market_long_positions / (10 ** base_decimals),
            'Short Positions Total': market_short_positions / (10 ** base_decimals),
            'Base Decimals': base_decimals
        }

        # Step 3: Fetch exchange volume data for this token
        print("\nStep 3: Fetching exchange volume data...")
        listed_symbols = [clean_sym] # List with just the one symbol (already checked not ignored)
        raw_cex_data = await dl_reference_exch_data(listed_symbols)

        # Process and store raw exchange data
        raw_exchange_volumes = {}
        if raw_cex_data:
            for (spot, exch), exch_data in raw_cex_data.items():
                for symbol, market_ohlcv in exch_data.items():
                    # Use the cleaned base symbol for matching
                    base_symbol_raw = symbol.split('/')[0].upper()
                    coin = clean_symbol(base_symbol_raw, exch)
                    if coin == clean_sym and len(market_ohlcv) > 0:
                        # Store raw data for visibility
                        raw_exchange_volumes[(spot, exch, symbol)] = {
                            "ohlcv_data": market_ohlcv[:5],  # Store first 5 candles for visibility
                            "full_count": len(market_ohlcv)
                        }

        # Process exchange data (returns full dollar amounts)
        cex_df = process_reference_exch_data(raw_cex_data, listed_symbols)

        # Step 4: Fetch market cap data
        print("\nStep 4: Fetching market cap data...")
        cmc_data = dl_cmc_data()

        # Store raw CMC data
        raw_cmc_data = None
        if cmc_data:
            for item in cmc_data:
                # Also check against ignored symbols here
                item_sym = item.get('symbol')
                if item_sym and (item_sym not in IGNORED_SYMBOLS and clean_symbol(item_sym) not in IGNORED_SYMBOLS):
                    if item.get('symbol') == clean_sym:
                        raw_cmc_data = item
                        break

        # Process market cap data (returns full dollar amounts)
        mc_df = process_cmc_data(cmc_data, listed_symbols) if cmc_data else pd.DataFrame(index=listed_symbols, data={"MC": 0})

        # Step 5: Combine all data
        print("\nStep 5: Combining all data sources...")

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

        # Step 6: Calculate scores
        print("\nStep 6: Calculating scores...")
        scores_df = build_scores(combined_df) # build_scores expects full dollar amounts

        # Step 7: Generate recommendation
        print("\nStep 7: Generating recommendation...")
        combined_df = pd.concat([combined_df, scores_df], axis=1)

        # Convert to dictionary for detailed view
        # Handle potential numpy types before converting to dict
        combined_df_native = combined_df.astype(object) # Convert to object dtype first
        combined_df_native = combined_df_native.where(pd.notnull(combined_df_native), None) # Replace NaN with None
        combined_dict = combined_df_native.reset_index().to_dict(orient='records')[0]

        scores_dict_native = scores_df.astype(object)
        scores_dict_native = scores_dict_native.where(pd.notnull(scores_dict_native), None)
        scores_dict = scores_dict_native.reset_index().to_dict(orient='records')[0]

        # Generate recommendation
        recommendation = generate_recommendation(combined_df.iloc[0])

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\nAnalysis completed in {duration:.2f} seconds")
        print(f"Recommendation for {clean_sym}: {recommendation}")
        print(f"Final score: {scores_dict.get('Score', 'N/A')}")

        # Function to convert DataFrames to JSON-serializable dicts
        def df_to_dict_safe(df):
            if df.empty:
                return {}
            # Convert numpy types to native python types
            df_obj = df.astype(object).where(pd.notnull(df), None)
            return df_obj.reset_index().to_dict(orient='records')[0]


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
                        "raw_samples": {str(k): v for k, v in raw_exchange_volumes.items()}, # Convert tuple key to string
                        "processed": df_to_dict_safe(cex_df)
                    },
                    "market_cap_data": {
                        "raw": raw_cmc_data,
                        "processed": df_to_dict_safe(mc_df)
                    },
                    "drift_volume_data": {
                        "volume_30d": volume_usd
                    }
                },
                "dataframes": {
                    "drift_df": df_to_dict_safe(drift_df),
                    "cex_df": df_to_dict_safe(cex_df),
                    "mc_df": df_to_dict_safe(mc_df),
                    "combined_df": combined_dict,
                    "scores_df": scores_dict
                },
                "results": {
                    "recommendation": recommendation,
                    "score": scores_dict.get('Score', 0),
                    "score_breakdown": {k: v for k, v in scores_dict.items() if k != 'Score' and k != 'index'}
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

# --- Drift API Data Fetching Functions ---

async def fetch_api_page(session, url: str, retries: int = 5):
    """
    Fetch a single page from the Drift API with rate limiting and retries.
    
    Args:
        session: aiohttp ClientSession
        url: API endpoint to fetch
        retries: Number of retry attempts for failed requests
        
    Returns:
        API response data as JSON
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
    
    Args:
        session: aiohttp ClientSession
        symbol: Market symbol (e.g., "SOL-PERP")
        start_date: Start date for fetching trades
        end_date: End date for fetching trades (defaults to current date)
        
    Returns:
        List of trade records
    """
    if end_date is None:
        end_date = datetime.now()
    
    current_date = start_date
    all_trades = []
    
    while current_date <= end_date:
        year, month, day = current_date.year, current_date.month, current_date.day
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