# src/page/list_delist_recommender.py
import json
import asyncio
import math
import os
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import requests
import ccxt
from datetime import datetime, timedelta
import time
import pickle
from pathlib import Path
import shutil

from driftpy.drift_client import DriftClient
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.config import configs
from anchorpy import Provider, Wallet
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair  # type: ignore
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig, WebsocketConfig
from driftpy.user_map.user_map import UserMap
from driftpy.user_map.user_map_config import UserMapConfig
from driftpy.user_map.userstats_map import UserStatsMap
from driftpy.user_map.user_map_config import UserStatsMapConfig
from driftpy.pickle.vat import Vat
from driftpy.types import MarketType

# --- Configuration Constants ---
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
DAYS_TO_CONSIDER = 30

# Drift-specific score cutoffs
DRIFT_SCORE_CUTOFFS = {
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
    'Solana Spot Vol Score': {
        'Solana Spot Vol $m': {'kind': 'exp', 'start': 0.01, 'end': 500, 'steps': 10},
    },
    'Drift Activity Score': {
        'Volume on Drift $m': {'kind': 'exp', 'start': 0.001, 'end': 500, 'steps': 10},
        'OI on Drift $m': {'kind': 'exp', 'start': 0.001, 'end': 500, 'steps': 10},
    },
    'Drift Liquidity Score': {
        'Drift Slip. $5k (bps)': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5},
        'Drift Slip. $50k (bps)': {'kind': 'linear', 'start': 200, 'end': 0, 'steps': 5},
    }
}

# Score boundaries for leverage recommendations
SCORE_UB = {0: 62, 5: 75, 10: 85, 20: 101}  # Upper bounds
SCORE_LB = {0: 0, 5: 37, 10: 48, 20: 60}    # Lower bounds

# Reference exchanges for market data
REFERENCE_SPOT_EXCH = {
    'binance', 'bybit', 'okx', 'gate', 'kucoin', 'mexc',
    'cryptocom', 'coinbase', 'kraken'
}

REFERENCE_FUT_EXCH = {
    'bybit', 'binance', 'gate', 'mexc', 'okx',
    'htx', 'krakenfutures', 'cryptocom', 'bitmex'
}

# Calculate earliest timestamp to keep data for
earliest_ts_to_keep = (datetime.now() - timedelta(days=DAYS_TO_CONSIDER+5)).timestamp()

# --- Utility Functions ---
@st.cache_data(ttl=3600)
def sig_figs(number, sig_figs=3):
    """Rounds a number to specified significant figures."""
    if np.isnan(number) or number <= 0:
        return 0
    return round(number, int(sig_figs - 1 - math.log10(number)))

@st.cache_data(ttl=3600)
def clean_symbol(symbol, exch=''):
    """Cleans and standardizes cryptocurrency symbols."""
    # General token aliases
    TOKEN_ALIASES = {
        'WBTC': 'BTC', 'WETH': 'ETH', 'WSOL': 'SOL',
        '1INCH': 'ONEINCH', 'HPOS10I': 'HPOS',
        'BITCOIN': 'BTC'
    }
    
    # Exchange-specific token aliases
    EXCH_TOKEN_ALIASES = {
        ('NEIRO', 'bybit'): 'NEIROETH',
    }
    
    # Extract base symbol
    redone = symbol.split('/')[0]
    
    # Remove common numerical suffixes
    for suffix in ['10000000', '1000000', '1000', 'k']:
        redone = redone.replace(suffix, '')
    
    # Apply exchange-specific alias if applicable
    redone = EXCH_TOKEN_ALIASES.get((redone, exch), redone)
    
    # Apply general alias
    return TOKEN_ALIASES.get(redone, redone)

@st.cache_data(ttl=3600)
def get_hot_ccxt_api(exch):
    """Initializes and returns a ccxt exchange API instance."""
    try:
        # Configure API with reasonable timeouts and options
        api = getattr(ccxt, exch)({
            'timeout': 10000,  # 10 seconds timeout
            'enableRateLimit': True,
            'options': {
                # Use a simpler method without lambda for pickle compatibility
                'timeout': 10000
            }
        })
        
        # Load markets with a try-except block
        try:
            # Special handling for Kraken
            if exch.lower() == 'kraken':
                # For Kraken, we'll manually test if the API works without requiring load_markets
                try:
                    # Try a simple ticker fetch for BTC/USD
                    api.fetch_ticker('BTC/USD')
                    # If successful, we can work with limited market data
                    st.info(f"Using limited functionality for Kraken (API structure differences). This is normal.")
                    return api
                except Exception as kraken_err:
                    st.warning(f"Could not connect to Kraken API: {str(kraken_err)}. Skipping this exchange.")
                    return None
            # Special handling for Crypto.com
            elif exch.lower() == 'cryptocom':
                # For crypto.com, we'll manually test if the API works without requiring load_markets
                try:
                    # Try a simple ticker fetch for BTC/USDT
                    api.fetch_ticker('BTC/USDT')
                    # If successful, we can work with limited market data
                    st.info(f"Using limited functionality for Crypto.com (API structure differences). This is normal.")
                    return api
                except Exception as cryptocom_err:
                    st.warning(f"Could not connect to Crypto.com API: {str(cryptocom_err)}. Skipping this exchange.")
                    return None
            else:
                # Standard market loading for other exchanges
                api.load_markets()
        except Exception as e:
            st.warning(f"Could not load markets for {exch}: {str(e)}. Using limited functionality.")
            # Continue with limited functionality
        
        # Test if the API is working by trying common market pairs
        has_working_pair = False
        for pair in ['BTC/USDT:USDT', 'BTC/USDT', 'BTC/USD', 'ETH/USDT']:
            try:
                api.fetch_ticker(pair)
                has_working_pair = True
                break
            except Exception:
                continue
        
        if not has_working_pair:
            st.warning(f"Could not fetch any test market data from {exch}. Skipping this exchange.")
            return None
            
        return api
    except Exception as e:
        # Log error but don't crash
        st.warning(f"Failed to initialize {exch} API: {str(e)}")
        return None

def geomean_three(series):
    """Calculates geometric mean of top 3 values in a series."""
    return np.exp(np.log(series + 1).sort_values()[-3:].sum() / 3) - 1

# --- Data Fetching Functions ---

@st.cache_data(ttl=7200)
def dl_reference_exch_data():
    """Downloads OHLCV data from reference exchanges."""
    try:
        def download_exch(exch, spot):
            """Helper to download data for a single exchange."""
            filename = f'{exch}_{"spot" if spot else "fut"}_data.json'
            
            api = get_hot_ccxt_api(exch)
            if not api:
                return {}
                
            # If no markets available, return empty data
            if not hasattr(api, 'markets') or not api.markets:
                return {}
                
            exchange_data = {}
            rate_limit_delay = 2 if exch.lower() == 'bybit' else 1
            
            # Use a placeholder for progress to show in Streamlit
            progress_text = f"Downloading {exch} {'spot' if spot else 'futures'} data..."
            
            # Limit to first 20 markets for demonstration
            market_count = 0
            
            # Make a copy of markets to avoid concurrent modification issues
            markets_to_try = list(api.markets.keys())[:100]  # Limit to first 100 markets
            
            for market in markets_to_try:
                if market_count >= 20:  # Limit markets for MVP
                    break
                    
                try:
                    # Market filtering logic
                    if spot and ':' in market:
                        continue
                    if not spot and ':USD' not in market and ':USDT' not in market:
                        continue
                    if '/USD' not in market and '/USDT' not in market and '/USDC' not in market:
                        continue
                    if '-' in market:
                        continue
                        
                    # Fetch OHLCV data with timeout and error handling
                    try:
                        ohlcv_data = api.fetch_ohlcv(market, '1d', limit=30)
                        if ohlcv_data and len(ohlcv_data) > 0:
                            exchange_data[market] = ohlcv_data
                            market_count += 1
                    except Exception as e:
                        # Silently continue on individual market failures
                        continue
                        
                except Exception as e:
                    # Silently continue on market filtering errors
                    continue
                    
            return exchange_data
            
        # Main download process with Streamlit progress indicator
        progress_bar = st.progress(0, "Fetching exchange data...")
        
        # Initialize dictionary for raw data
        raw_reference_exch_df = {}
        
        # Download data for spot exchanges
        total_exchanges = len(REFERENCE_SPOT_EXCH) + len(REFERENCE_FUT_EXCH)
        exchange_count = 0
        
        for exch in REFERENCE_SPOT_EXCH:
            try:
                exchange_data = download_exch(exch, True)
                if exchange_data and len(exchange_data) > 0:
                    raw_reference_exch_df[(True, exch)] = exchange_data
            except Exception as e:
                st.warning(f"Error processing {exch} spot data: {str(e)}")
            
            exchange_count += 1
            progress_bar.progress(exchange_count / total_exchanges, f"Processed {exch} spot")
        
        # Download data for futures exchanges
        for exch in REFERENCE_FUT_EXCH:
            try:
                exchange_data = download_exch(exch, False)
                if exchange_data and len(exchange_data) > 0:
                    raw_reference_exch_df[(False, exch)] = exchange_data
            except Exception as e:
                st.warning(f"Error processing {exch} futures data: {str(e)}")
                
            exchange_count += 1
            progress_bar.progress(exchange_count / total_exchanges, f"Processed {exch} futures")
        
        progress_bar.empty()
        
        # If we got no data at all, create some placeholder data
        if not raw_reference_exch_df:
            st.warning("Could not fetch data from any exchange. Using placeholder data.")
            # Create placeholder data for BTC and ETH
            placeholder_data = {}
            for symbol in ['BTC', 'ETH']:
                ohlcv = [[int(datetime.now().timestamp() * 1000), 1000, 1100, 900, 1050, 1000] for _ in range(30)]
                placeholder_data[f"{symbol}/USDT"] = ohlcv
            
            # Add placeholder data for a mock exchange
            raw_reference_exch_df[(True, 'mock_exchange')] = placeholder_data
        
        return raw_reference_exch_df
    except Exception as e:
        st.error(f"Fatal error fetching exchange data: {str(e)}")
        return None

@st.cache_data(ttl=7200)
def process_reference_exch_data(raw_reference_exch_df, all_symbols):
    """Processes raw OHLCV data from exchanges."""
    all_candle_data = {}
    
    # Process each exchange's data
    for (spot, exch), exch_data in raw_reference_exch_df.items():
        # Skip empty exchange data
        if not exch_data:
            continue
            
        # Get API instance or skip
        api = get_hot_ccxt_api(exch)
        if not api:
            continue
            
        for symbol, market_ohlcv in exch_data.items():
            try:
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
                    continue
                
                # Get contract size if available
                try:
                    contractsize = min(api.markets.get(symbol, {}).get('contractSize', None) or 1, 1)
                except:
                    contractsize = 1
                
                # Calculate average daily volume in USD
                try:
                    daily_usd_volume = (np.minimum(market_df.l, market_df.c.iloc[-1])
                                   * market_df.v
                                   * contractsize).mean()
                except:
                    # If calculation fails, use a simple average
                    try:
                        daily_usd_volume = (market_df.c * market_df.v * contractsize).mean()
                    except:
                        # Last resort fallback
                        daily_usd_volume = 0
                
                # Only store if we have a valid volume
                if not np.isnan(daily_usd_volume) and daily_usd_volume > 0:
                    if daily_usd_volume >= all_candle_data.get((exch, spot, coin), 0):
                        all_candle_data[exch, spot, coin] = daily_usd_volume
            except Exception as e:
                # Silently skip problem markets
                continue
    
    # Create a default DataFrame with zeros if we have no data
    if not all_candle_data:
        default_data = {}
        for sym in all_symbols:
            default_data[(True, sym)] = 0  # Spot volume
            default_data[(False, sym)] = 0  # Futures volume
        
        # Convert to DataFrame
        output_df = pd.DataFrame(
            index=all_symbols,
            data={
                "Spot Volume $m": 0,
                "Spot Vol Geomean $m": 0,
                "Fut Volume $m": 0,
                "Fut Vol Geomean $m": 0,
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
            f"{dict(geomean_three='Vol Geomean $m', sum='Volume $m')[agg_func_name]}"
            for agg_func_name, is_spot in output_df.columns
        ]
    except Exception as e:
        st.warning(f"Error processing exchange data: {str(e)}. Using default values.")
        
        # Create default DataFrame on error
        output_df = pd.DataFrame(
            index=all_symbols,
            data={
                "Spot Volume $m": 0,
                "Spot Vol Geomean $m": 0,
                "Fut Volume $m": 0,
                "Fut Vol Geomean $m": 0,
            }
        )
    
    # Make sure all symbols are represented
    missing_symbols = set(all_symbols) - set(output_df.index)
    if missing_symbols:
        # Add missing symbols with zero values
        missing_df = pd.DataFrame(
            index=list(missing_symbols),
            data={col: 0 for col in output_df.columns}
        )
        output_df = pd.concat([output_df, missing_df])
    
    return output_df

@st.cache_data(ttl=7200)
def dl_cmc_data():
    """Downloads market cap data from CoinMarketCap API."""
    try:
        # Get API key from environment variables
        cmc_api_key = os.environ.get("CMC_API_KEY")
        if not cmc_api_key:
            st.error("CoinMarketCap API key is not set in .env file. Market cap data will not be available.")
            return None
        
        cmc_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        
        # Symbol overrides for inconsistent naming
        cmc_symbol_overrides = {
            'Neiro Ethereum': 'NEIROETH',
            'HarryPotterObamaSonic10Inu (ERC-20)': 'HPOS',
            'Wrapped Bitcoin': 'BTC',
            'Wrapped Ethereum': 'ETH',
            'Wrapped Solana': 'SOL'
        }
        
        try:
            response = requests.get(
                f"{cmc_api_url}?CMC_PRO_API_KEY={cmc_api_key}&limit=500",
                timeout=10
            )
            response.raise_for_status()
            data = response.json().get('data', [])
            
            if not data:
                st.error("No data received from CoinMarketCap API.")
                return None
            
            # Apply symbol overrides
            for item in data:
                item['symbol'] = cmc_symbol_overrides.get(item['name'], item['symbol'])
                
            return data
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to CoinMarketCap API: {str(e)}")
            return None
        except ValueError as e:
            st.error(f"Error parsing JSON from CoinMarketCap API: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Fatal error fetching CoinMarketCap data: {str(e)}")
        return None

@st.cache_data(ttl=7200)
def process_cmc_data(cmc_data, all_symbols):
    """Processes market cap data from CoinMarketCap."""
    # Create DataFrame with relevant fields
    try:
        output_df = pd.DataFrame([{
            'symbol': a['symbol'],
            'mc': a['quote']['USD']['market_cap'],
            'fd_mc': a['quote']['USD']['fully_diluted_market_cap'],
        } for a in cmc_data if 'quote' in a and 'USD' in a['quote']])
        
        if output_df.empty:
            return pd.DataFrame({'MC $m': []})
            
        output_df = output_df.groupby('symbol')[['mc', 'fd_mc']].max()
        
        # Use FD MC if regular MC is zero/missing
        output_df.loc[output_df['mc'] == 0, 'mc'] = output_df['fd_mc']
        
        # Calculate final MC in millions
        output_df['MC $m'] = output_df['mc'].fillna(0) / 1e6
        
        # Filter to include only our symbols of interest
        output_df = output_df[output_df.index.isin(all_symbols)]
        
        return output_df[['MC $m']]
    except Exception as e:
        st.error(f"Error processing CoinMarketCap data: {str(e)}")
        return pd.DataFrame({'MC $m': []})

DEFAULT_PICKLE_DIR = "pickles"
CACHE_MAX_AGE_SECONDS = 3600  # 1 hour

def get_newest_pickle_set(directory: str) -> tuple[Optional[Dict[str, str]], Optional[float]]:
    """
    Find the newest set of pickle files in the given directory.
    Returns a tuple of (file_dict, timestamp_in_seconds) or (None, None) if no valid files found.
    """
    if not os.path.exists(directory):
        return None, None
    
    # Look for pickle files that start with vat- prefix
    subdirs = [d for d in os.listdir(directory) if d.startswith("vat-") and os.path.isdir(os.path.join(directory, d))]
    if not subdirs:
        return None, None
    
    # Sort by timestamp to find the newest
    subdirs.sort(reverse=True)
    subdir = subdirs[0]
    path = os.path.join(directory, subdir)
    
    # Check if this is a complete pickle set with all required files
    required_prefixes = ["perp_", "spot_", "usermap_", "userstats_", "perporacles_", "spotoracles_"]
    files_present = []
    
    for f in os.listdir(path):
        if f.endswith(".pkl") and any(f.startswith(prefix) for prefix in required_prefixes):
            files_present.append(f)
            
    # Check if all required file types are present
    if all(any(f.startswith(prefix) for f in files_present) for prefix in required_prefixes):
        # Extract datetime from directory name
        try:
            # Format is vat-%Y-%m-%d-%H-%M-%S
            date_part = subdir[4:]  # Remove 'vat-' prefix
            dt = datetime.strptime(date_part, "%Y-%m-%d-%H-%M-%S")
            timestamp = dt.timestamp()
            
            # Create a map of file types to full paths
            file_map = {}
            for prefix in required_prefixes:
                prefix_base = prefix.rstrip("_")
                matching_files = [f for f in files_present if f.startswith(prefix)]
                if matching_files:
                    # Sort by slot number to get the newest file
                    matching_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
                    file_map[prefix_base] = os.path.join(path, matching_files[0])
            
            # Ensure all required file types have a corresponding file
            if all(prefix_base in file_map for prefix_base in [p.rstrip("_") for p in required_prefixes]):
                return file_map, timestamp
            else:
                st.warning(f"Incomplete pickle set in {subdir}. Missing some required files.")
                return None, None
        except Exception as e:
            st.warning(f"Error parsing timestamp from directory {subdir}: {str(e)}")
    
    return None, None

def is_pickle_fresh(timestamp: float, max_age_seconds: int = CACHE_MAX_AGE_SECONDS) -> bool:
    """Check if a pickle is fresh enough"""
    current_time = time.time()
    age = current_time - timestamp
    return age < max_age_seconds

def delete_old_vat_dirs(pickle_dir, except_dir=None):
    """Delete all VAT directories except the one specified"""
    if not os.path.exists(pickle_dir):
        return
            
    for item in os.listdir(pickle_dir):
        item_path = os.path.join(pickle_dir, item)
        if os.path.isdir(item_path) and item.startswith("vat-") and item != except_dir:
            try:
                shutil.rmtree(item_path)
                st.info(f"Deleted old VAT directory: {item}")
            except Exception as e:
                st.warning(f"Warning: Failed to delete old VAT directory {item}: {e}")

@st.cache_data(ttl=3600)
def cached_fetch_drift_data(_all_symbols):
    """Cached wrapper for fetching Drift Protocol data."""
    try:
        return asyncio.run(_fetch_drift_data(_all_symbols))
    except Exception as e:
        st.error(f"Error in Drift data fetching: {str(e)}")
        return None

async def _fetch_drift_data(_all_symbols):
    """Fetches data from Drift Protocol using driftpy."""
    try:
        # RPC endpoint from environment variables
        rpc_endpoint = os.environ.get("RPC_URL", "https://api.mainnet-beta.solana.com")
        if not rpc_endpoint:
            st.error("Solana RPC endpoint is not set in .env file.")
            return None
        
        # Initialize Solana client and provider with a dummy wallet (read-only access)
        connection = AsyncClient(rpc_endpoint, commitment="confirmed")
        keypair = Keypair()
        wallet = Wallet(keypair)
        pickle_dir = DEFAULT_PICKLE_DIR
        
        # Create pickle directory if it doesn't exist
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        
        # Check if we have fresh pickle data available
        pickle_files, timestamp = get_newest_pickle_set(pickle_dir)
        using_pickled_data = False
        
        # Check if pickle is fresh before initializing drift client to avoid unnecessary subscriptions
        if pickle_files and timestamp and is_pickle_fresh(timestamp):
            # Initialize with cached subscription mode
            try:
                # Only initialize basic drift client without subscription
                drift_client = DriftClient(
                    connection=connection,
                    wallet=wallet,
                    account_subscription=AccountSubscriptionConfig("cached"),
                    env="mainnet"
                )
            
                # Important: Force connection to have explicit commitment
                connection._commitment = "confirmed"
                
                # We absolutely need program to proceed
                program = drift_client.program
                if program is None:
                    raise ValueError("Drift client program is None, cannot proceed with cached data")
                
                # Force program to have connection with commitment
                if program.provider is None:
                    raise ValueError("Drift client program provider is None")
                
                program.provider.connection = connection
                
                st.info(f"Using cached Drift data from {datetime.fromtimestamp(timestamp)}")
                using_pickled_data = True
                
                # Create maps avoiding any None references
                spot_map = MarketMap(
                    MarketMapConfig(
                        program,
                        MarketType.Spot(),
                        None,
                        connection=connection,
                    )
                )
                
                # Force connection on maps
                spot_map.connection = connection
                
                perp_map = MarketMap(
                    MarketMapConfig(
                        program,
                        MarketType.Perp(),
                        None,
                        connection=connection,
                    )
                )
                perp_map.connection = connection
                
                user_map = UserMap(
                    UserMapConfig(
                        drift_client,
                        None,
                    )
                )
                
                stats_map = UserStatsMap(UserStatsMapConfig(drift_client))
                
                # Initialize VAT with connection already set in all maps
                vat = Vat(
                    drift_client,
                    user_map,
                    stats_map,
                    spot_map,
                    perp_map,
                )
                
                # Force connection on VAT
                vat.connection = connection
                
                # Verify all required pickle files exist before trying to unpickle
                required_keys = ['usermap', 'userstats', 'spot', 'perp', 'spotoracles', 'perporacles']
                if not all(k in pickle_files for k in required_keys):
                    missing = [k for k in required_keys if k not in pickle_files]
                    st.warning(f"Missing required pickle files: {missing}. Fetching fresh data instead.")
                    using_pickled_data = False
                else:
                    # Load from pickle with robust error handling
                    try:
                        # Double-check all connections before unpickling
                        drift_client.connection = connection
                        
                        await vat.unpickle(
                            users_filename=pickle_files.get('usermap'),
                            user_stats_filename=pickle_files.get('userstats'),
                            spot_markets_filename=pickle_files.get('spot'),
                            perp_markets_filename=pickle_files.get('perp'),
                            spot_oracles_filename=pickle_files.get('spotoracles'),
                            perp_oracles_filename=pickle_files.get('perporacles'),
                        )
                        
                        # Validate unpickled data
                        if len(perp_map.values()) == 0:
                            st.warning("No perp markets found after unpickling. Fetching fresh data.")
                            using_pickled_data = False
                        else:
                            # Success - set all connections on all objects again for safety
                            drift_client.connection = connection
                            vat.connection = connection
                            spot_map.connection = connection
                            perp_map.connection = connection
                    except Exception as inner_e:
                        st.warning(f"Error during unpickling process: {str(inner_e)}")
                        using_pickled_data = False
            except Exception as e:
                st.warning(f"Error setting up for pickle loading: {str(e)}")
                using_pickled_data = False
        
        # Fetch fresh data if needed
        if not using_pickled_data:
            st.info("Fetching fresh Drift data from RPC...")
            
            try:
                # We need a fresh client for reliable data
                drift_client = DriftClient(
                    connection=connection,
                    wallet=wallet,
                    account_subscription=AccountSubscriptionConfig("cached"),
                    env="mainnet"
                )
                
                # Subscribe to the drift client - only do this when fresh data is needed
                await drift_client.subscribe()
                
                # Initialize all maps and subscribe to them
                spot_map = MarketMap(
                    MarketMapConfig(
                        drift_client.program,
                        MarketType.Spot(),
                        WebsocketConfig(),
                        connection,
                    )
                )
                perp_map = MarketMap(
                    MarketMapConfig(
                        drift_client.program,
                        MarketType.Perp(),
                        WebsocketConfig(),
                        connection,
                    )
                )
                user_map = UserMap(
                    UserMapConfig(
                        drift_client,
                        WebsocketConfig(),
                    )
                )
                stats_map = UserStatsMap(UserStatsMapConfig(drift_client))
                
                # Initialize VAT
                vat = Vat(
                    drift_client,
                    user_map,
                    stats_map,
                    spot_map,
                    perp_map,
                )
                
                # Explicitly set the connection
                vat.connection = connection 
                
                # Subscribe to all maps
                await asyncio.gather(
                    spot_map.subscribe(),
                    perp_map.subscribe(),
                    user_map.subscribe(),
                    stats_map.subscribe(),
                )
                
                # Save to pickle for future use
                # Create timestamped directory for the new pickle set
                now = datetime.now()
                folder_name = now.strftime("vat-%Y-%m-%d-%H-%M-%S")
                path = os.path.join(pickle_dir, folder_name, "")
                
                os.makedirs(path, exist_ok=True)
                await vat.pickle(path)
                
                # Delete old VAT directories
                delete_old_vat_dirs(pickle_dir, except_dir=folder_name)
            except Exception as e:
                st.error(f"Failed to fetch fresh Drift data: {str(e)}")
                return None

        # Initialize results dictionary
        drift_data = []
        
        # Get all perp markets
        perp_markets = list(perp_map.values())
        if not perp_markets:
            st.error("No perp markets found in Drift data")
            return None
        
        # Process each perp market
        for market in perp_markets:
            market_index = market.data.market_index
            
            # Get market config by index
            market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
            if not market_config:
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
            
            # Calculate OI in USD by summing all user positions
            base_oi = 0
            # Iterate through all users to calculate total open interest
            for user in user_map.values():
                try:
                    perp_positions = user.get_perp_positions()
                    for position in perp_positions:
                        if position.market_index == market_index and position.base_asset_amount != 0:
                            # Only add positive positions to prevent double counting
                            base_oi += abs(position.base_asset_amount)
                except Exception as e:
                    continue
            
            # Get base decimals - try market first, then fall back to constants
            base_decimals = 6  # Default to 6 decimals
            try:
                # Try to get decimals from market.data
                if hasattr(market.data, 'base_decimals'):
                    base_decimals = market.data.base_decimals
                # If not found, check if it's in market.data.amm
                elif hasattr(market.data, 'amm') and hasattr(market.data.amm, 'base_asset_decimals'):
                    base_decimals = market.data.amm.base_asset_decimals
            except Exception:
                # If any error occurs, keep the default value
                pass
            
            # Convert to human readable base units and then to millions of USD
            base_oi_readable = base_oi / (10 ** base_decimals)
            oi_usd = base_oi_readable * oracle_price / 1e6  # In millions USD
            
            # Get funding rate (hourly)
            funding_rate = market.data.amm.last_funding_rate / 1e6  # Convert to percentage
            hourly_funding = funding_rate * 100  # As percentage
            
            # Estimate slippage (simplified for MVP)
            # In a real implementation, would need to query orderbook depth or use simulation
            est_slippage_5k = 10  # Placeholder: 10 basis points for $5k order
            est_slippage_50k = 50  # Placeholder: 50 basis points for $50k order
            
            # Estimate volume (placeholder - would need historical data)
            # In a real implementation, would query historical trading data or API
            est_daily_volume = oi_usd * 0.2  # Placeholder: assume 20% of OI is daily volume
            est_volume_30d = est_daily_volume * 30  # Placeholder: 30-day est.
            
            drift_data.append({
                'Symbol': clean_sym,
                'Market Index': market_index,
                'Max Lev. on Drift': max_leverage,
                'OI on Drift $m': sig_figs(oi_usd, 3),
                'Volume on Drift $m': sig_figs(est_volume_30d, 3),
                'Drift Slip. $5k (bps)': est_slippage_5k,
                'Drift Slip. $50k (bps)': est_slippage_50k,
                'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
                'Is Listed on Drift': True,
                'Oracle Price': oracle_price
            })
        
        # Clean up resources
        if not using_pickled_data:
            try:
                # Only attempt to unsubscribe if the object has the method
                if hasattr(spot_map, 'unsubscribe') and callable(spot_map.unsubscribe):
                    await spot_map.unsubscribe()
                if hasattr(perp_map, 'unsubscribe') and callable(perp_map.unsubscribe):
                    await perp_map.unsubscribe()
                if hasattr(user_map, 'unsubscribe') and callable(user_map.unsubscribe):
                    await user_map.unsubscribe()
                if hasattr(stats_map, 'unsubscribe') and callable(stats_map.unsubscribe):
                    await stats_map.unsubscribe()
                if hasattr(drift_client, 'unsubscribe') and callable(drift_client.unsubscribe):
                    await drift_client.unsubscribe()
            except Exception as e:
                st.warning(f"Error during cleanup: {str(e)}")
                # Continue execution even if cleanup fails
        
        if not drift_data:
            st.error("No Drift markets data was processed")
            return None
            
        return drift_data
    except Exception as e:
        st.error(f"Error fetching Drift data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def dl_solana_dex_data(symbols):
    """Fetches Solana DEX volume data (placeholder for MVP)."""
    try:
        # In a real implementation, would fetch from Jupiter API, Birdeye, etc.
        # For MVP, generate placeholder data
        solana_dex_data = {}
        
        for symbol in symbols:
            # Skip stablecoins
            if symbol in STABLE_COINS:
                continue
                
            # Generate placeholder volume (higher for major coins)
            base_vol = 0.1  # Default for most tokens
            if symbol in ['BTC', 'ETH', 'SOL']:
                base_vol = 100  # Higher for major tokens
            elif symbol in ['BONK', 'JTO', 'RNDR', 'JUP']:
                base_vol = 30  # Medium for popular tokens
                
            # Add some randomness
            random_factor = 0.5 + np.random.random()
            solana_dex_data[symbol] = base_vol * random_factor
            
        return pd.DataFrame({'Solana Spot Vol $m': solana_dex_data})
    except Exception as e:
        st.error(f"Error generating Solana DEX data: {str(e)}")
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
    
    # Adjustments for non-listed assets
    drift_score_cols = [c for c in output_df.columns if 'Drift ' in c or c.startswith('Partial_Score_Drift')]
    output_df.loc[~df['Is Listed on Drift'].fillna(False), drift_score_cols] = 0
    
    # Boost for potentially listable assets
    non_drift_categories = ['Market Cap Score', 'Spot Vol Score', 'Futures Vol Score', 'Solana Spot Vol Score']
    output_df['NON_DRIFT_SCORE_BOOST'] = (
        0.5  # Boost factor
        * (~df['Is Listed on Drift'].fillna(False))  # Only for non-listed assets
        * output_df[non_drift_categories].sum(axis=1)  # Sum of non-Drift scores
    ).astype(float)
    
    # Calculate final score
    score_components = [*DRIFT_SCORE_CUTOFFS.keys(), 'NON_DRIFT_SCORE_BOOST']
    output_df['Score'] = output_df[score_components].sum(axis=1)
    
    return output_df

def generate_recommendation(row):
    """Generates recommendation based on score and current leverage."""
    is_listed = row['Is Listed on Drift']
    current_leverage = int(0 if pd.isna(row['Max Lev. on Drift']) else row['Max Lev. on Drift'])
    score = row['Score']
    
    # For non-listed assets
    if not is_listed:
        # Check if score meets minimum listing threshold
        if score >= SCORE_UB[0]:
            # Determine recommended listing leverage
            if score >= SCORE_UB[20]:
                return 'List (20x)'
            elif score >= SCORE_UB[10]:
                return 'List (10x)'
            elif score >= SCORE_UB[5]:
                return 'List (5x)'
            else:
                return 'List (5x)'
        else:
            return 'Do Not List'
            
    # For listed assets
    else:
        # Determine relevant score boundaries
        lower_bound_key = min([k for k in SCORE_LB.keys() if k <= current_leverage], key=lambda x: abs(x - current_leverage))
        upper_bound_key = min([k for k in SCORE_UB.keys() if k <= current_leverage], key=lambda x: abs(x - current_leverage))
        
        # Check if score is below lower bound
        is_below_lower_bound = score < SCORE_LB[lower_bound_key]
        
        # Check if score is above upper bound
        is_above_upper_bound = score >= SCORE_UB[upper_bound_key]
        
        # Generate recommendation
        if current_leverage > 0 and is_below_lower_bound:
            if current_leverage > 5:
                return 'Dec. Lev.'
            else:
                return 'Delist'
        elif current_leverage > 0 and is_above_upper_bound:
            # Check if higher leverage tier is available
            next_tier = next((tier for tier in sorted(SCORE_UB.keys()) if tier > current_leverage), None)
            if next_tier and score >= SCORE_UB[next_tier]:
                return f'Inc. Lev. ({next_tier}x)'
            else:
                return ''  # No change recommended
        else:
            return ''  # No change recommended

# --- Main Streamlit Page ---

def list_delist_recommender_page():
    st.title("Drift Protocol List/Delist Recommender")
    
    st.markdown("""
    This tool evaluates assets for listing or delisting on Drift Protocol based on various market metrics.
    It analyzes market cap, volume across exchanges, on-chain activity, and Drift-specific data to generate recommendations.
    """)
    
    # Validate environment variables early
    rpc_endpoint = os.environ.get("RPC_URL")
    if not rpc_endpoint:
        st.error("❌ ERROR: Solana RPC endpoint is not set in .env file. Please add RPC_URL to your .env file.")
        st.stop()
    
    cmc_api_key = os.environ.get("CMC_API_KEY")
    if not cmc_api_key:
        st.error("❌ ERROR: CoinMarketCap API key is not set in .env file. Please add CMC_API_KEY to your .env file.")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        st.write("Days to consider for historical data:")
        days = st.slider("Days", min_value=7, max_value=90, value=30, step=1)
        
        st.write("Add candidate symbols for analysis:")
        candidate_symbols = st.text_area(
            "Enter symbols for evaluation (comma separated)",
            value="JUP,BONK,WIF,RNDR,PYTH,JTO",
            help="Add potential assets to evaluate for listing"
        )
        
        show_filter = st.radio(
            "Filter assets:",
            options=["Show All", "Show Listed Only", "Show Candidates Only"]
        )
        
        st.write("Debug Options:")
        debug_mode = st.checkbox("Show detailed scores", value=False)
    
    # Process candidate symbols
    candidate_symbols_list = [s.strip().upper() for s in candidate_symbols.split(",") if s.strip()]
    
    # Check if any symbols have been entered
    if not candidate_symbols_list:
        st.error("❌ ERROR: No symbols to analyze. Please add candidate symbols in the sidebar.")
        st.stop()
    
    # Wait for async Drift data
    with st.spinner("Fetching Drift data..."):
        try:
            drift_data = cached_fetch_drift_data(candidate_symbols_list)
            if drift_data is None:
                st.error("❌ ERROR: Failed to fetch Drift data. Please check your RPC endpoint configuration.")
                st.stop()
        except Exception as e:
            st.error(f"❌ ERROR: Failed to fetch Drift data: {str(e)}")
            st.stop()
    
    # Create a master list of all symbols to analyze
    listed_symbols = [item['Symbol'] for item in drift_data] if drift_data else []
    all_symbols = list(set(listed_symbols + candidate_symbols_list))
    
    # Fetch and process data
    with st.spinner("Fetching and processing market data..."):
        # Create a mapping for listed status
        is_listed_map = {sym: sym in listed_symbols for sym in all_symbols}
        
        # Convert Drift data to DataFrame
        if not drift_data:
            st.error("❌ ERROR: No Drift data available. Make sure your .env file is set up correctly with RPC_URL.")
            st.stop()
            
        drift_df = pd.DataFrame(drift_data).set_index('Symbol')
        
        # Get CEX data
        try:
            raw_cex_data = dl_reference_exch_data()
            if raw_cex_data is None:
                st.warning("⚠️ Warning: Could not fetch data from any exchange. Using placeholder data.")
            
            cex_df = process_reference_exch_data(raw_cex_data, all_symbols)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to fetch exchange data: {str(e)}")
            st.stop()
        
        # Get market cap data
        try:
            cmc_data = dl_cmc_data()
            if cmc_data is None:
                st.error("❌ ERROR: Failed to fetch market cap data. Please check your CMC_API_KEY.")
                st.stop()
                
            mc_df = process_cmc_data(cmc_data, all_symbols)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to process market cap data: {str(e)}")
            st.stop()
        
        # Get Solana DEX data
        try:
            solana_dex_df = dl_solana_dex_data(all_symbols)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to process Solana DEX data: {str(e)}")
            st.stop()
        
        # Combine all data
        try:
            combined_df = pd.concat([
                drift_df,
                cex_df,
                mc_df,
                solana_dex_df
            ], axis=1)
            
            # Fill in missing values
            combined_df['Is Listed on Drift'] = combined_df.index.map(lambda x: is_listed_map.get(x, False))
            combined_df['Max Lev. on Drift'] = combined_df['Max Lev. on Drift'].fillna(0)
            
            # Fill NaN values for metrics
            for col in combined_df.columns:
                if col not in ['Is Listed on Drift', 'Symbol']:
                    combined_df[col] = combined_df[col].fillna(0)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to combine data: {str(e)}")
            st.stop()
    
    # Calculate scores and recommendations
    with st.spinner("Calculating scores and recommendations..."):
        try:
            scores_df = build_scores(combined_df)
            combined_df = pd.concat([combined_df, scores_df], axis=1)
            combined_df['Recommendation'] = combined_df.apply(generate_recommendation, axis=1)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to calculate scores: {str(e)}")
            st.stop()
    
    # Filter based on user selection
    if show_filter == "Show Listed Only":
        display_df = combined_df[combined_df['Is Listed on Drift'] == True]
    elif show_filter == "Show Candidates Only":
        display_df = combined_df[combined_df['Is Listed on Drift'] == False]
    else:
        display_df = combined_df
    
    # Check if there's any data to display after filtering
    if display_df.empty:
        st.warning("⚠️ No data to display after applying filters. Try changing your filter settings.")
        st.stop()
    
    # Sort by score descending
    display_df = display_df.sort_values('Score', ascending=False)
    
    # Select columns for display
    display_columns = [
        'Is Listed on Drift', 'Max Lev. on Drift', 'Recommendation',
        'Score', 'MC $m', 'Spot Volume $m', 'Fut Volume $m',
        'Solana Spot Vol $m', 'OI on Drift $m', 'Volume on Drift $m',
        'Drift Slip. $5k (bps)', 'Drift Slip. $50k (bps)', 'Funding Rate % (1h)'
    ]
    
    # Add score breakdown columns if debug mode is enabled
    if debug_mode:
        score_columns = [col for col in scores_df.columns if 'Score' in col and not col.startswith('Partial_')]
        display_columns = display_columns[:3] + score_columns + display_columns[3:]
    
    # Display the results table
    st.subheader("Listing/Delisting Recommendations")
    st.dataframe(display_df[display_columns], use_container_width=True)
    
    # Visualization
    st.subheader("Visualization of Recommendations")
    
    # Prepare data for visualization
    viz_df = display_df.reset_index().rename(columns={'index': 'Symbol'})
    
    # Create bar chart for scores
    fig = px.bar(
        viz_df,
        x='Symbol',
        y='Score',
        color='Recommendation',
        hover_data=['Max Lev. on Drift', 'MC $m', 'OI on Drift $m'],
        title="Asset Scores and Recommendations",
        color_discrete_map={
            'List (5x)': 'green', 
            'List (10x)': 'darkgreen',
            'List (20x)': 'forestgreen',
            'Inc. Lev. (10x)': 'lightgreen',
            'Inc. Lev. (20x)': 'lime',
            'Dec. Lev.': 'orange',
            'Delist': 'red',
            'Do Not List': 'gray',
            '': 'lightgray'
        }
    )
    
    # Add horizontal lines for score boundaries
    for lev, bound in SCORE_UB.items():
        if lev > 0:  # Skip level 0
            fig.add_hline(
                y=bound,
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"Min score for {lev}x lev",
                annotation_position="right"
            )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Score breakdown visualization
    if debug_mode:
        st.subheader("Score Composition")
        
        # Prepare data for stacked bar chart
        score_components = [col for col in scores_df.columns if col in DRIFT_SCORE_CUTOFFS.keys()]
        score_components.append('NON_DRIFT_SCORE_BOOST')
        
        score_breakdown = viz_df.sort_values('Score', ascending=False)[['Symbol'] + score_components]
        
        # Create stacked bar chart
        fig2 = px.bar(
            score_breakdown,
            x='Symbol',
            y=score_components,
            title="Score Composition",
            labels={'value': 'Score', 'variable': 'Component'},
            hover_data=['Symbol'],
        )
        
        fig2.update_layout(xaxis_tickangle=-45, barmode='stack')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market Cap vs Volume")
        
        fig3 = px.scatter(
            viz_df,
            x='MC $m',
            y='Spot Volume $m',
            color='Is Listed on Drift',
            size='Score',
            hover_name='Symbol',
            log_x=True,
            log_y=True,
            title="Market Cap vs Spot Volume"
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("CEX vs Solana DEX Volume")
        
        fig4 = px.scatter(
            viz_df,
            x='Spot Volume $m',
            y='Solana Spot Vol $m',
            color='Is Listed on Drift',
            size='Score',
            hover_name='Symbol',
            log_x=True,
            log_y=True,
            title="CEX vs Solana DEX Volume"
        )
        
        st.plotly_chart(fig4, use_container_width=True)