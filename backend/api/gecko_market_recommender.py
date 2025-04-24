import os
import json
import asyncio
import aiohttp
import aiofiles
import logging
import time
import traceback
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
# Assuming DriftPy and related types are available
from driftpy.constants.config import mainnet_perp_market_configs
from driftpy.types import MarketType, OraclePriceData
from driftpy.drift_client import DriftClient
from driftpy.constants.numeric_constants import * 
from driftpy.constants import BASE_PRECISION, PRICE_PRECISION, SPOT_BALANCE_PRECISION
from driftpy.pickle.vat import Vat
from driftpy.types import is_variant
from solders.pubkey import Pubkey
from anchorpy import Provider

from fastapi import APIRouter
from backend.state import BackendRequest

# Constants for market data
IGNORED_SYMBOLS = set()  # Add any symbols to ignore
MARKET_BASE_DECIMALS = {
    0: 9,  # SOL-PERP
    1: 6,  # BTC-PERP
    2: 6,  # ETH-PERP
    # Add other known market decimals
}

# Add after existing constants
DRIFT_PROGRAM_ID = Pubkey.from_string(os.getenv("DRIFT_PROGRAM_ID", "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH"))

# Time window for data analysis
DAYS_TO_CONSIDER = 30  # Number of days to look back for volume data

def clean_symbol(symbol: str) -> str:
    """Clean and standardize market symbol."""
    if not symbol:
        return ""
    # Remove common suffixes and convert to uppercase
    clean = symbol.upper().replace("-PERP", "").replace("/USD", "").replace("USD", "")
    return clean.strip()

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Add FileHandler if needed
)
logger = logging.getLogger("market_data_engine")

MARKET_DATA_FILE = Path("cache/market_data.json")  # Store in project root's cache directory

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
            "coingecko_total_volume_24h": None, "coingecko_30d_volume_total": None,
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
    """Helper to fetch 30d volume data for a single coin.
    
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
    """Fetches 30d volume data from CG /coins/{id}/market_chart.
    
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


async def get_drift_data(vat: Vat):
    """Fetches data from Drift Protocol using Vat."""
    try:
        logger.info("==> Fetching Drift Protocol market data")

        # Initialize results list
        drift_data = []

        # Track long and short positions separately for each market
        market_long_positions = {}   # Track sum of long positions
        market_short_positions = {}  # Track sum of short positions
        market_position_counts = {}  # Track number of positions

        # Process each user's positions
        processed_users = 0
        logger.info("==> Processing positions from users via Vat...")
        
        for user in vat.users.values():
            try:
                processed_users += 1
                if processed_users % 5000 == 0:
                    logger.info(f"Processed {processed_users} users...")

                # Get perp positions directly from user account
                for position in user.get_user_account().perp_positions:
                    if position.base_asset_amount != 0:
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

        logger.info(f"Found positions in {len(market_long_positions)} markets")

        # Process each market
        for market_index in market_long_positions.keys():
            try:
                # Get market config
                market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
                if not market_config:
                    continue

                # Get symbol and clean it
                symbol = market_config.symbol
                clean_sym = clean_symbol(symbol)

                # Skip ignored symbols
                if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
                    continue

                # Get oracle price directly from vat
                market_price = vat.perp_oracles.get(market_index)
                if market_price is None:
                    continue

                oracle_price = market_price.price / PRICE_PRECISION

                # Get base decimals
                base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)

                # Calculate OI
                long_amount = market_long_positions[market_index]
                short_amount = market_short_positions[market_index]
                base_oi = max(long_amount, short_amount)
                base_oi_readable = base_oi / (10 ** base_decimals)
                oi_usd = base_oi_readable * oracle_price

                # Get market from vat
                market = vat.perp_markets.get(market_index)
                if not market:
                    continue

                # Get max leverage
                initial_margin_ratio = market.data.margin_ratio_initial / 10000
                max_leverage = int(1 / initial_margin_ratio) if initial_margin_ratio > 0 else 0

                # Get funding rate
                funding_rate = market.data.amm.last_funding_rate / 1e6
                hourly_funding = funding_rate * 100

                drift_data.append({
                    'Symbol': clean_sym,
                    'Market Index': market_index,
                    'Max Lev. on Drift': max_leverage,
                    'OI on Drift': oi_usd,
                    'Funding Rate % (1h)': sig_figs(hourly_funding, 3),
                    'Oracle Price': oracle_price,
                    'Volume on Drift': 0  # Will be updated with actual data
                })

            except Exception as e:
                logger.error(f"==> Error processing market {market_index}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if not drift_data:
            logger.error("==> No Drift markets data was processed")
            return None

        # Fetch actual trade volume data for all markets
        symbols = [item['Symbol'] for item in drift_data]
        volumes_by_symbol = await batch_calculate_market_volumes(symbols)
        
        # Update drift_data with actual volume values
        for item in drift_data:
            symbol = item['Symbol']
            if symbol in volumes_by_symbol:
                item['Volume on Drift'] = volumes_by_symbol[symbol]
        
        return drift_data

    except Exception as e:
        logger.error(f"==> Error fetching Drift data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def fetch_drift_sdk_data(drift_client) -> dict:
    """Fetches current on-chain data using DriftPy SDK.
    
    Args:
        drift_client: Initialized Drift client instance
        
    Returns:
        dict: Mapping of symbol to current market data from SDK
    """
    drift_sdk_metrics = {}  # Dict: symbol -> sdk_data_dict
    logger.info("Fetching data from Drift SDK...")
    
    if not drift_client:
        logger.error("Drift client not available or initialized.")
        return {}

    try:
        # Get all perp markets
        perp_markets = drift_client.get_perp_markets()
        markets_count = len(perp_markets)
        if not perp_markets:
            logger.error("No perp markets found in Drift data")
            return {}

        logger.info(f"Found {markets_count} perp markets initially")

        # Process each perp market
        for market in perp_markets:
            try:
                market_index = market.data.market_index

                # Get market config by index
                market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
                if not market_config:
                    logger.warning(f"No market config found for market index {market_index}")
                    continue

                # Get symbol and clean it
                symbol = market_config.symbol
                clean_sym = clean_symbol(symbol)

                # Skip if symbol is in ignored list
                if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
                    logger.info(f"Skipping ignored market: {symbol} (Index: {market_index})")
                    continue

                # Get max leverage (from initial margin ratio)
                initial_margin_ratio = market.data.margin_ratio_initial / 10000
                max_leverage = int(1 / initial_margin_ratio) if initial_margin_ratio > 0 else 0

                # Get oracle price
                oracle_price_data = drift_client.get_oracle_price_data_for_perp_market(market_index)
                oracle_price = oracle_price_data.price / 1e6  # Convert from UI price

                # Get funding rate (hourly)
                funding_rate = market.data.amm.last_funding_rate / 1e6  # Convert to percentage
                hourly_funding = funding_rate * 100  # As percentage

                # Store market data
                drift_sdk_metrics[clean_sym] = {
            'is_listed_perp': True, 
                    'perp_market': f"{clean_sym}-PERP",
                    'oracle_price': oracle_price,
                    'max_leverage': max_leverage,
                    'funding_rate_1h': hourly_funding,
                    'is_listed_spot': False,  # Would need spot market lookup logic
                    'spot_market': None,
                    'market_index': market_index
                }

                logger.info(f"Processed market data for {clean_sym} (Index: {market_index})")

            except Exception as e:
                logger.error(f"Error processing market {market_index}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

    except Exception as e:
        logger.error(f"Error fetching Drift SDK data: {e}", exc_info=True)
    
    return drift_sdk_metrics


async def get_drift_client_and_vat():
    """Initialize and return a Drift client instance and Vat."""
    try:
        # Get RPC endpoint from environment or use default
        rpc_endpoint = os.getenv("RPC_ENDPOINT", "https://api.mainnet-beta.solana.com")
        
        # Initialize provider
        provider = Provider.local(rpc_endpoint)
        
        # Initialize Drift client with the correct parameters
        drift_client = DriftClient(
            connection=provider.connection,
            wallet=provider.wallet,
            program_id=DRIFT_PROGRAM_ID,
            opts=provider.opts
        )
        
        # Initialize the client
        await drift_client.subscribe()
        
        # Initialize Vat
        vat = Vat(drift_client.program)
        await vat.update()
        
        return drift_client, vat
    except Exception as e:
        logger.error(f"Error initializing Drift client and Vat: {e}")
        return None, None

async def fetch_drift_api_data(session: aiohttp.ClientSession, drift_client: DriftClient, vat: Vat) -> dict:
    """Fetches historical/aggregated data like 30d vol and OI from Drift Data API.
    
    Args:
        session: aiohttp client session
        drift_client: Initialized Drift client instance
        vat: Initialized Vat instance for efficient data access
        
    Returns:
        dict: Mapping of symbol to volume and OI data
    """
    drift_api_metrics = {}  # Dict: symbol -> api_data_dict
    logger.info("Fetching volume and OI data from Drift...")

    try:
        # Track long and short positions separately for each market
        market_long_positions = {}   # Track sum of long positions
        market_short_positions = {}  # Track sum of short positions
        market_position_counts = {}  # Track number of positions

        # Get user accounts from Vat
        user_accounts = vat.get_user_accounts()
        user_count = len(user_accounts)
        logger.info(f"Processing positions from {user_count} users via Vat...")
        
        processed_users = 0
        for user_account in user_accounts:
            try:
                processed_users += 1
                if processed_users % 5000 == 0:
                    logger.info(f"Processed {processed_users}/{user_count} users...")

                # Get active perp positions from user account
                perp_positions = user_account.get_active_perp_positions()

                # Process each position
                for position in perp_positions:
                    if position.base_asset_amount != 0:
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

        logger.info(f"Found positions in {len(market_long_positions)} markets")

        # Get all perp markets for symbol mapping
        perp_markets = drift_client.get_perp_markets()
        
        # Process each market's OI
        for market in perp_markets:
            try:
                market_index = market.data.market_index
                
                # Skip if no positions found for this market
                if market_index not in market_long_positions and market_index not in market_short_positions:
                    continue

                # Get market config and symbol
                market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
                if not market_config:
                    continue
                
                symbol = clean_symbol(market_config.symbol)
                
                # Skip ignored symbols
                if symbol in IGNORED_SYMBOLS:
                    continue

                # Get oracle price
                oracle_price_data = drift_client.get_oracle_price_data_for_perp_market(market_index)
                oracle_price = oracle_price_data.price / 1e6  # Convert from UI price

                # Calculate OI as max(abs(long), abs(short))
                long_amount = market_long_positions.get(market_index, 0)
                short_amount = market_short_positions.get(market_index, 0)
                base_oi = max(long_amount, short_amount)
                
                # Get base decimals
                base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)
                
                try:
                    if hasattr(market.data, 'base_decimals'):
                        base_decimals = market.data.base_decimals
                    elif hasattr(market.data, 'amm') and hasattr(market.data.amm, 'base_asset_decimals'):
                        base_decimals = market.data.amm.base_asset_decimals
                except Exception:
                    pass

                # Convert to human readable units
                base_oi_readable = base_oi / (10 ** base_decimals)
                oi_usd = base_oi_readable * oracle_price

                # Fetch 30-day volume from API
                volume_30d = await batch_calculate_market_volumes([symbol])
                volume = volume_30d.get(symbol, 0) if volume_30d else 0

                drift_api_metrics[symbol] = {
                    'volume_30d': volume,
                    'open_interest': oi_usd
                }

                logger.info(
                    f"Processed market {symbol}: OI=${oi_usd:,.2f}, "
                    f"Volume=${volume:,.2f}, Positions={market_position_counts.get(market_index, 0)}"
                )

            except Exception as e:
                logger.error(f"Error processing market metrics: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error fetching Drift metrics: {e}", exc_info=True)

    return drift_api_metrics


# --- Data Update/Merge Function ---

async def update_market_data(market_data: Dict[str, Any], vat: Vat) -> Dict[str, Any]:
    """
    Update existing market data with fetched data from various sources.
    """
    try:
        market_index = market_data.get('market_index')
        if market_index is None:
            return market_data

        # Get oracle price directly from vat
        market_price = vat.perp_oracles.get(market_index)
        if market_price is None:
            return market_data

        # Update market data with oracle price
        market_data['oracle_price'] = market_price.price / PRICE_PRECISION
        
        return market_data
    except Exception as e:
        logger.error(f"Error updating market data: {e}", exc_info=True)
        return market_data


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
            vol_30d_total = cg_data.get('coingecko_30d_volume_total') or 0
            # Convert total to daily average for scoring
            vol_30d_avg = vol_30d_total / 30 if vol_30d_total > 0 else 0
            vol_config = SCORE_CUTOFFS.get('Global Vol Score', {}).get('coingecko_global_volume_30d_avg', {})
            if vol_config and vol_30d_avg > 0:
                partial_global_vol_score = calculate_partial_score(vol_30d_avg, vol_config)
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

def sig_figs(number: float, sig_figs: int = 3) -> float:
    """Round a number to specified significant figures."""
    if number == 0:
        return 0
    try:
        return round(number, sig_figs - 1 - int(math.log10(abs(number))))
    except (ValueError, TypeError):
        return 0

# --- Main Orchestration Function (Async) ---

async def main():
    """Main function to orchestrate loading, fetching, updating, scoring, and saving."""
    start_time = time.time()
    logger.info("=== Starting Market Data Engine Run ===")

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
        
    # Step 3: Calculate scores
    logger.info("Step 3: Calculating scores and recommendations...")
    market_data = calculate_scores(market_data)

    # Step 4: Save updated data
    logger.info(f"Step 4: Saving {len(market_data)} records to {MARKET_DATA_FILE}...")
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

# Create FastAPI router
router = APIRouter()

@router.get("/market_recommendations")
async def get_market_recommendations(request: BackendRequest):
    """
    Get market recommendations based on CoinGecko data and Drift protocol data.
    
    This endpoint combines data from CoinGecko (market caps, volumes) with
    Drift protocol data (positions, oracle prices) to generate market recommendations.
    """
    logger.info("Received request for market recommendations")
    
    try:
        start_time = time.time()
        
        # Get Vat from backend state
        vat: Vat = request.state.backend_state.vat
        if not vat:
            return {
                "status": "error",
                "message": "Vat not available in backend state",
                "data": None
            }
            
        # Load existing data
        market_data = await load_market_data(MARKET_DATA_FILE)
        symbols_in_file = {item['symbol'] for item in market_data}
        logger.info(f"Found {len(symbols_in_file)} symbols in existing file.")

        # Fetch CoinGecko data
        async with aiohttp.ClientSession() as session:
            # Step 1: Fetch CoinGecko market data
            logger.info("Step 1: Fetching CoinGecko market data...")
            fetched_cg_market = await fetch_all_coingecko_market_data(session)
            
            if not fetched_cg_market:
                logger.error("Failed to fetch CoinGecko market data")
                return {
                    "status": "error",
                    "message": "Failed to fetch CoinGecko market data",
                    "data": None
                }
                
            logger.info(f"Successfully fetched data for {len(fetched_cg_market)} markets from CoinGecko")
            
            # Step 2: Fetch CoinGecko volume data
            coingecko_ids = list(fetched_cg_market.keys())
            logger.info(f"Step 2: Fetching volumes for {len(coingecko_ids)} CoinGecko IDs...")
            fetched_cg_volume = await fetch_all_coingecko_volumes(session, coingecko_ids)

        # Step 3: Fetch Drift data using Vat
        logger.info("Step 3: Fetching Drift protocol data...")
        drift_data = await get_drift_data(vat)
        
        # Step 4: Update market data
        logger.info("Step 4: Updating market data...")
        for item in market_data:
            item = await update_market_data(item, vat)

        # Step 5: Calculate scores
        logger.info("Step 5: Calculating scores and recommendations...")
        market_data = calculate_scores(market_data)

        # Save updated data
        await save_market_data(market_data, MARKET_DATA_FILE)

        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "status": "success",
            "message": "Market recommendations generated successfully",
            "data": {
                "recommendations": market_data,
                "metadata": {
                    "processing_time": duration,
                    "coingecko_markets": len(fetched_cg_market),
                    "drift_markets": len(drift_data) if drift_data else 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating market recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error generating market recommendations: {str(e)}",
            "data": None
        }

@router.get("/single_market")
async def get_single_market_data(request: BackendRequest, market_index: int):
    """
    Get detailed data for a single market.
    
    This endpoint provides detailed analysis of a single market, combining
    CoinGecko data with Drift protocol data.
    """
    logger.info(f"Received request for market {market_index}")
    
    try:
        # Get Vat from backend state
        vat: Vat = request.state.backend_state.vat
        if not vat:
            return {
                "status": "error",
                "message": "Vat not available in backend state",
                "data": None
            }
            
        # Get market config
        market_config = next((cfg for cfg in mainnet_perp_market_configs if cfg and cfg.market_index == market_index), None)
        if not market_config:
            return {
                "status": "error",
                "message": f"No market config found for index {market_index}",
                "data": None
            }
            
        # Get symbol
        symbol = market_config.symbol
        clean_sym = clean_symbol(symbol)
        
        # Skip ignored symbols
        if symbol in IGNORED_SYMBOLS or clean_sym in IGNORED_SYMBOLS:
            return {
                "status": "error",
                "message": f"Market {symbol} is in the ignored list",
                "data": None
            }
            
        # Get market data from Vat
        market = vat.perp_markets.get(market_index)
        if not market:
            return {
                "status": "error",
                "message": f"Market {market_index} not found in Vat",
                "data": None
            }
            
        # Get oracle price
        market_price = vat.perp_oracles.get(market_index)
        if market_price is None:
            return {
                "status": "error",
                "message": f"Oracle price not found for market {market_index}",
                "data": None
            }
            
        oracle_price = market_price.price / PRICE_PRECISION
        
        # Fetch CoinGecko data for this symbol
        async with aiohttp.ClientSession() as session:
            cg_market = await fetch_all_coingecko_market_data(session)
            cg_volume = await fetch_all_coingecko_volumes(session, [clean_sym])
            
        return {
            "status": "success",
            "message": f"Data retrieved for market {clean_sym}",
            "data": {
                "symbol": clean_sym,
                "market_index": market_index,
                "oracle_price": oracle_price,
                "drift_data": {
                    "max_leverage": int(1 / (market.data.margin_ratio_initial / 10000)) if market.data.margin_ratio_initial > 0 else 0,
                    "funding_rate": market.data.amm.last_funding_rate / 1e6 * 100,
                    "base_decimals": MARKET_BASE_DECIMALS.get(market_index, 9)
                },
                "coingecko_data": cg_market.get(clean_sym, {}),
                "volume_data": cg_volume.get(clean_sym, 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching single market data: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Error fetching single market data: {str(e)}",
            "data": None
        }

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