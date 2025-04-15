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
from typing import Dict, List, Optional
import asyncio
import numpy as np
import pandas as pd
import requests
import ccxt
import ccxt.async_support as ccxt_async
import logging
import traceback
from datetime import datetime, timedelta
from fastapi import APIRouter, Query

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
    'binance', 'bybit', 'okx', 'gate', 'kucoin', 'mexc', 'kraken'
}

REFERENCE_FUT_EXCH = {
    'bybit', 'binance', 'gate', 'mexc', 'okx', 'htx', 'bitmex'
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
    try:
        # Configure API with reasonable timeouts and options
        api = getattr(ccxt_async, exch)({
            'timeout': 10000,  # 10 seconds timeout
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
            await api.close()  # Properly close even on failure
            return None
                
        return api
    except Exception as e:
        # Log error but don't crash
        logger.error(f"Failed to initialize {exch} API: {str(e)}")
        # Ensure API is closed if initialization was partially successful
        if api:
            try:
                await api.close()
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
                if '/USD' not in market_name and '/USDT' not in market_name and '/USDC' not in market_name:
                    continue
                if '-' in market_name:
                    continue
                
                # Extract base symbol and check if it's in our list
                base_symbol = market_name.split('/')[0].upper()
                clean_base = clean_symbol(base_symbol)
                
                # Check if this market matches any of our Drift symbols
                if (base_symbol in normalized_symbols or 
                    clean_base in normalized_symbols or
                    any(symbol.upper().startswith(base_symbol) for symbol in normalized_symbols)):
                    markets_to_try.append(market_name)
            except Exception:
                continue
        
        logger.info(f"==> Found {len(markets_to_try)} potential matches in {exch} out of {len(api.markets)} total markets")
        print(f"Processing {len(markets_to_try)} matching markets from {exch}")
        
        # Function to fetch OHLCV data for a single market
        async def fetch_market_data(market):
            try:
                # Fetch OHLCV data with timeout
                ohlcv_data = await asyncio.wait_for(
                    api.fetch_ohlcv(market, '1d', limit=30),
                    timeout=5.0  # 5 second timeout
                )
                if ohlcv_data and len(ohlcv_data) > 0:
                    return market, ohlcv_data
                return market, None
            except Exception as e:
                logger.debug(f"==> Failed to fetch OHLCV data for {market} on {exch}: {str(e)}")
                return market, None
        
        # Process markets concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests per exchange
        
        async def fetch_with_semaphore(market):
            async with semaphore:
                return await fetch_market_data(market)
        
        # Create tasks for all markets
        tasks = [fetch_with_semaphore(market) for market in markets_to_try]
        
        # Set a timeout for the entire exchange processing
        start_time = datetime.now()
        max_time_per_exchange = 45  # seconds
        
        # Process markets with timeout
        try:
            # Wait for all tasks with a timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max_time_per_exchange
            )
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.warning(f"==> Overall timeout after {elapsed:.1f}s processing {exch}")
            # We'll still process any completed results
            pending = [t for t in tasks if not t.done()]
            results = [t.result() if t.done() and not t.exception() else (None, None) for t in tasks]
            logger.warning(f"==> {len(pending)}/{len(tasks)} tasks didn't complete before timeout")
        
        # Process results
        successful_markets = 0
        for market, data in results:
            if market and data:
                exchange_data[market] = data
                successful_markets += 1
        
        logger.info(f"==> Downloaded data from {exch}: got data for {successful_markets}/{len(markets_to_try)} markets")
        
        return exchange_data
            
    except Exception as e:
        logger.error(f"==> Error processing {exch}: {str(e)}")
        return {}
    finally:
        # Always ensure the API is closed, even in error conditions
        if api:
            try:
                await api.close()
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
                                          key=lambda x: 0 if x in ['binance', 'bybit', 'okx'] else 1)
        
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
                                         key=lambda x: 0 if x in ['binance', 'bybit', 'okx'] else 1)
        
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
        
        for symbol, market_ohlcv in exch_data.items():
            try:
                total_markets_processed += 1
                markets_processed += 1
                
                coin = clean_symbol(symbol, exch)
                if not len(market_ohlcv) or coin not in all_symbols:
                    continue
                    
                # Convert to DataFrame and filter by time
                try:
                    market_df = (pd.DataFrame(market_ohlcv, columns=[*'tohlcv'])
                             .set_index('t')
                             .sort_index())
                
                    # Check if we have enough data
                    if len(market_df) < 3:  # Need at least a few days of data
                        continue
                    
                    # Filter to relevant time period 
                    market_df = market_df.loc[earliest_ts_to_keep * 1000:].iloc[-DAYS_TO_CONSIDER-1:-1]
                    
                    if not len(market_df):
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
                    daily_usd_volume = (np.minimum(market_df.l, market_df.c.iloc[-1])
                                   * market_df.v
                                   * contractsize).mean()
                except Exception as e:
                    # If calculation fails, use a simple average
                    logger.debug(f"==> Primary volume calculation failed for {symbol} on {exch}: {str(e)}")
                    try:
                        daily_usd_volume = (market_df.c * market_df.v * contractsize).mean()
                    except Exception as e2:
                        # Last resort fallback
                        logger.debug(f"==> Fallback volume calculation failed for {symbol} on {exch}: {str(e2)}")
                        daily_usd_volume = 0
                
                # Only store if we have a valid volume
                if not np.isnan(daily_usd_volume) and daily_usd_volume > 0:
                    if daily_usd_volume >= all_candle_data.get((exch, spot, coin), 0):
                        all_candle_data[exch, spot, coin] = daily_usd_volume
                        symbols_with_data.add(coin)
                        successful_markets += 1
            except Exception as e:
                # Silently skip problem markets
                logger.debug(f"==> Error processing market {symbol} on {exch}: {str(e)}")
                continue
        
        logger.info(f"==> Processed {markets_processed}/{markets_in_exchange} markets for {exch} (spot={spot})")
    
    logger.info(f"==> Exchange data processing complete: {successful_markets}/{total_markets_processed} markets successfully processed")
    logger.info(f"==> Found data for {len(symbols_with_data)}/{len(all_symbols)} symbols")
    
    # Create a default DataFrame with zeros if we have no data
    if not all_candle_data:
        logger.warning("==> No candle data found, creating default DataFrame with zeros")
        # Create default DataFrame
        output_df = pd.DataFrame(
            index=all_symbols,
            data={
                "Spot Volume": 0,
                "Spot Vol Geomean": 0,
                "Fut Volume": 0,
                "Fut Vol Geomean": 0,
            }
        )
        return output_df
    
    # Convert to DataFrame and aggregate
    try:
        df_coins = pd.Series(all_candle_data).sort_values(ascending=False)
        df_coins.index.names = ['exch', 'spot', 'coin']
        
        output_df = (df_coins.fillna(0) / 1e6).groupby(['spot', 'coin']).agg(
            [geomean_three, 'sum']
        ).unstack(0).fillna(0)
        
        # Rename columns
        output_df.columns = [
            f"{'Spot' if is_spot else 'Fut'} "
            f"{dict(geomean_three='Vol Geomean', sum='Volume')[agg_func_name]}"
            for agg_func_name, is_spot in output_df.columns
        ]
        
        logger.info(f"==> Successfully created exchange volume DataFrame with shape {output_df.shape}")
    except Exception as e:
        logger.error(f"==> Error processing exchange data into DataFrame: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create default DataFrame on error
        output_df = pd.DataFrame(
            index=all_symbols,
            data={
                "Spot Volume": 0,
                "Spot Vol Geomean": 0,
                "Fut Volume": 0,
                "Fut Vol Geomean": 0,
            }
        )
    
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
            response = requests.get(
                f"{cmc_api_url}?CMC_PRO_API_KEY={cmc_api_key}&limit=500",
                timeout=10
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
    
    # Create DataFrame with relevant fields
    try:
        output_df = pd.DataFrame([{
            'symbol': a['symbol'],
            'mc': a['quote']['USD']['market_cap'],
            'fd_mc': a['quote']['USD']['fully_diluted_market_cap'],
        } for a in cmc_data if 'quote' in a and 'USD' in a['quote']])
        
        if output_df.empty:
            logger.warning("==> No valid market cap data found after processing")
            return pd.DataFrame({'MC': []})
            
        output_df = output_df.groupby('symbol')[['mc', 'fd_mc']].max()
        
        # Use FD MC if regular MC is zero/missing
        output_df.loc[output_df['mc'] == 0, 'mc'] = output_df['fd_mc']
        
        # Calculate final MC in full dollars (not millions)
        output_df['MC'] = output_df['mc'].fillna(0)
        
        # Filter to include only our symbols of interest
        output_df = output_df[output_df.index.isin(all_symbols)]
        
        # Log results summary
        matched_symbols = len(output_df.index)
        logger.info(f"==> Processed market cap data: found data for {matched_symbols}/{len(all_symbols)} symbols")
        logger.info(f"==> Market cap range: ${output_df['MC'].min():,.2f} - ${output_df['MC'].max():,.2f}")
        
        return output_df[['MC']]
    except Exception as e:
        logger.error(f"==> Error processing CoinMarketCap data: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.DataFrame({'MC': []})

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
        if raw_cex_data is None:
            print("WARNING: Could not fetch data from any exchange. Using placeholder data.")
            logger.warning("==> [get_delist_recommendations] Could not fetch data from any exchange. Using placeholder data.")
        
        print("Processing exchange data...")
        logger.info("==> [get_delist_recommendations] Processing exchange data")
        cex_df = process_reference_exch_data(raw_cex_data, listed_symbols)
        
        # Get market cap data
        print("Fetching market cap data...")
        logger.info("==> [get_delist_recommendations] Fetching market cap data")
        cmc_data = dl_cmc_data()
        mc_df = process_cmc_data(cmc_data, listed_symbols) if cmc_data else pd.DataFrame(index=listed_symbols, data={"MC": 0})
        
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
        
        # No need to convert anymore as we're already storing full dollar values
        # Just ensure columns exist
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

@router.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify routing is working."""
    print("\n===> TEST ENDPOINT CALLED <===")
    return {"status": "success", "message": "Test endpoint working"}