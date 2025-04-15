# src/page/delist_recommender.py
import asyncio
import math
import os
from typing import Dict, List, Optional
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

from driftpy.drift_client import DriftClient
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from anchorpy import Wallet
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair  # type: ignore
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig, WebsocketConfig
from driftpy.user_map.user_map import UserMap
from driftpy.user_map.user_map_config import UserMapConfig
from driftpy.types import MarketType

# --- Configuration Constants ---
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
DAYS_TO_CONSIDER = 30

# Drift-specific score cutoffs - simplified for delisting focus
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
    'Drift Activity Score': {
        'Volume on Drift $m': {'kind': 'exp', 'start': 0.001, 'end': 500, 'steps': 10},
        'OI on Drift $m': {'kind': 'exp', 'start': 0.001, 'end': 500, 'steps': 10},
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

# Calculate earliest timestamp to keep data for
earliest_ts_to_keep = (datetime.now() - timedelta(days=DAYS_TO_CONSIDER+5)).timestamp()

# --- Utility Functions ---
def sig_figs(number, sig_figs=3):
    """Rounds a number to specified significant figures."""
    if np.isnan(number) or number <= 0:
        return 0
    return round(number, int(sig_figs - 1 - math.log10(number)))

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

def get_hot_ccxt_api(exch):
    """Initializes and returns a ccxt exchange API instance."""
    try:
        # Configure API with reasonable timeouts and options
        api = getattr(ccxt, exch)({
            'timeout': 10000,  # 10 seconds timeout
            'enableRateLimit': True,
        })
        
        # Load markets with a try-except block
        try:
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

def dl_reference_exch_data():
    """Downloads OHLCV data from reference exchanges."""
    try:
        def download_exch(exch, spot):
            """Helper to download data for a single exchange."""
            api = get_hot_ccxt_api(exch)
            if not api:
                return {}
                
            # If no markets available, return empty data
            if not hasattr(api, 'markets') or not api.markets:
                return {}
                
            exchange_data = {}
            
            # Make a copy of markets to avoid concurrent modification issues
            markets_to_try = list(api.markets.keys())[:100]  # Limit to first 100 markets
            
            for market in markets_to_try:
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
                    except Exception:
                        # Silently continue on individual market failures
                        continue
                        
                except Exception:
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
        
        return raw_reference_exch_df
    except Exception as e:
        st.error(f"Fatal error fetching exchange data: {str(e)}")
        return None

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
                except Exception:
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
            except Exception:
                # Silently skip problem markets
                continue
    
    # Create a default DataFrame with zeros if we have no data
    if not all_candle_data:
        # Create default DataFrame
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

def dl_cmc_data():
    """Downloads market cap data from CoinMarketCap API."""
    try:
        # Get API key from environment variables
        cmc_api_key = os.environ.get("CMC_API_KEY")
        if not cmc_api_key:
            st.error("CoinMarketCap API key is not set in .env file. Market cap data will not be available.")
            return None
        
        cmc_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
        
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

async def fetch_drift_data():
    """Fetches basic data from Drift Protocol."""
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
        
        # Initialize drift client and subscribe
        drift_client = DriftClient(
            connection=connection,
            wallet=wallet,
            account_subscription=AccountSubscriptionConfig("cached"),
            env="mainnet"
        )
        
        await drift_client.subscribe()
        
        # Initialize maps and subscribe to them
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
        
        # Subscribe to maps
        await asyncio.gather(
            perp_map.subscribe(),
            user_map.subscribe(),
        )

        # Initialize results list
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
                except Exception:
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
            
            # Estimate volume (placeholder - would need historical data)
            est_daily_volume = oi_usd * 0.2  # Placeholder: assume 20% of OI is daily volume
            est_volume_30d = est_daily_volume * 30  # Placeholder: 30-day est.
            
            drift_data.append({
                'Symbol': clean_sym,
                'Market Index': market_index,
                'Max Lev. on Drift': max_leverage,
                'OI on Drift $m': sig_figs(oi_usd, 3),
                'Volume on Drift $m': sig_figs(est_volume_30d, 3),
                'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
                'Oracle Price': oracle_price
            })
        
        # Clean up resources
        try:
            if hasattr(perp_map, 'unsubscribe') and callable(perp_map.unsubscribe):
                await perp_map.unsubscribe()
            if hasattr(user_map, 'unsubscribe') and callable(user_map.unsubscribe):
                await user_map.unsubscribe()
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

def cached_fetch_drift_data():
    """Cached wrapper for fetching Drift Protocol data."""
    try:
        return asyncio.run(fetch_drift_data())
    except Exception as e:
        st.error(f"Error in Drift data fetching: {str(e)}")
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

# --- Main Streamlit Page ---

def delist_recommender_page():
    st.title("Drift Protocol Delist Recommender")
    
    st.markdown("""
    This tool evaluates currently listed assets on Drift Protocol and recommends markets for delisting or leverage reduction.
    It analyzes market cap, volume across exchanges, and on-chain activity to identify underperforming markets.
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
        
        st.write("Filtering Options:")
        show_filter = st.radio(
            "Show:",
            options=["All Markets", "Only Delist Recommendations", "Only Low Activity Markets"]
        )
        
        st.write("Display Options:")
        debug_mode = st.checkbox("Show detailed scores", value=False)
        sort_by = st.radio("Sort by:", ["Score (Ascending)", "Open Interest", "Volume"])
    
    # Fetch Drift data
    with st.spinner("Fetching Drift data..."):
        try:
            drift_data = cached_fetch_drift_data()
            if drift_data is None:
                st.error("❌ ERROR: Failed to fetch Drift data. Please check your RPC endpoint configuration.")
                st.stop()
        except Exception as e:
            st.error(f"❌ ERROR: Failed to fetch Drift data: {str(e)}")
            st.stop()
    
    # Get list of all listed symbols
    listed_symbols = [item['Symbol'] for item in drift_data] if drift_data else []
    if not listed_symbols:
        st.error("❌ ERROR: No listed markets found on Drift.")
        st.stop()
    
    # Fetch and process data
    with st.spinner("Fetching and processing market data..."):
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
            
            cex_df = process_reference_exch_data(raw_cex_data, listed_symbols)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to fetch exchange data: {str(e)}")
            st.stop()
        
        # Get market cap data
        try:
            cmc_data = dl_cmc_data()
            if cmc_data is None:
                st.error("❌ ERROR: Failed to fetch market cap data. Please check your CMC_API_KEY.")
                st.stop()
                
            mc_df = process_cmc_data(cmc_data, listed_symbols)
        except Exception as e:
            st.error(f"❌ ERROR: Failed to process market cap data: {str(e)}")
            st.stop()
        
        # Combine all data
        try:
            combined_df = pd.concat([
                drift_df,
                cex_df,
                mc_df,
            ], axis=1)
            
            # Fill NaN values for metrics
            for col in combined_df.columns:
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
    if show_filter == "Only Delist Recommendations":
        display_df = combined_df[combined_df['Recommendation'].isin(['Delist', 'Decrease Leverage'])]
    elif show_filter == "Only Low Activity Markets":
        display_df = combined_df[combined_df['OI on Drift $m'] < 0.5]  # Markets with less than $500k OI
    else:
        display_df = combined_df
    
    # Check if there's any data to display after filtering
    if display_df.empty:
        st.warning("⚠️ No markets match the current filter criteria.")
        st.stop()
    
    # Sort based on user preference
    if sort_by == "Score (Ascending)":
        display_df = display_df.sort_values('Score', ascending=True)  # Lower scores first for delist focus
    elif sort_by == "Open Interest":
        display_df = display_df.sort_values('OI on Drift $m', ascending=True)  # Lower OI first
    else:  # Volume
        display_df = display_df.sort_values('Volume on Drift $m', ascending=True)  # Lower volume first
    
    # Select columns for display
    display_columns = [
        'Recommendation', 'Score', 'Max Lev. on Drift', 
        'MC $m', 'Spot Volume $m', 'Fut Volume $m',
        'OI on Drift $m', 'Volume on Drift $m', 'Funding Rate % (1h)'
    ]
    
    # Add score breakdown columns if debug mode is enabled
    if debug_mode:
        score_columns = [col for col in scores_df.columns if col in DRIFT_SCORE_CUTOFFS.keys()]
        display_columns = display_columns[:2] + score_columns + display_columns[2:]
    
    # Display the results table
    st.subheader("Delisting Recommendations")
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
            'Delist': 'red',
            'Decrease Leverage': 'orange',
            'Keep': 'green'
        }
    )
    
    # Add horizontal lines for score boundaries
    for lev, bound in SCORE_LB.items():
        if lev > 0:  # Skip level 0
            fig.add_hline(
                y=bound,
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Min score for {lev}x lev",
                annotation_position="right"
            )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volume vs Open Interest")
        
        fig2 = px.scatter(
            viz_df,
            x='Volume on Drift $m',
            y='OI on Drift $m',
            color='Recommendation',
            size='Score',
            hover_name='Symbol',
            log_x=True,
            log_y=True,
            title="Volume vs OI (Log Scale)",
            color_discrete_map={
                'Delist': 'red',
                'Decrease Leverage': 'orange',
                'Keep': 'green'
            }
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("Market Cap vs Score")
        
        fig3 = px.scatter(
            viz_df,
            x='MC $m',
            y='Score',
            color='Recommendation',
            size='OI on Drift $m',
            hover_name='Symbol',
            log_x=True,
            title="Market Cap vs Score",
            color_discrete_map={
                'Delist': 'red',
                'Decrease Leverage': 'orange',
                'Keep': 'green'
            }
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary")
    
    total_markets = len(display_df)
    delist_markets = len(display_df[display_df['Recommendation'] == 'Delist'])
    decrease_lev_markets = len(display_df[display_df['Recommendation'] == 'Decrease Leverage'])
    keep_markets = len(display_df[display_df['Recommendation'] == 'Keep'])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Markets to Delist", delist_markets)
    with col2:
        st.metric("Markets to Decrease Leverage", decrease_lev_markets)
    with col3:
        st.metric("Markets to Keep", keep_markets)
    
    st.info(f"Total markets analyzed: {total_markets}") 