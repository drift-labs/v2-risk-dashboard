# This is the new version of the market recommender which makes use of the coingecko API via the /backend/utils/coingecko_api.py utility script.

import logging
from typing import Dict, List
from fastapi import APIRouter
from backend.utils.coingecko_api import fetch_all_coingecko_market_data, fetch_all_coingecko_volumes
from driftpy.pickle.vat import Vat
from backend.state import BackendRequest
import os
from datetime import datetime, timedelta

# --- Constants ---
PRICE_PRECISION = 1e6  # Add price precision constant
MARKET_BASE_DECIMALS = {
    0: 9,  # Default to 9 decimals if not specified
}

# Symbols to ignore
IGNORE_SYMBOLS = ['USDT', 'USDC']

# Symbol conversions
SYMBOL_CONVERSIONS = {
    'TRX': 'TRON',
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI router
router = APIRouter()

@router.get("/market-data")
async def get_market_data(request: BackendRequest, number_of_tokens: int = 2):
    """
    Get comprehensive market data including CoinGecko data, Drift metrics, and listing recommendations.
    
    Args:
        request (BackendRequest): The backend request object containing state
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List[Dict]: List of dictionaries containing market data, scoring, and recommendations for each coin
    """
    return main(request.state.backend_state.vat, number_of_tokens)

def fetch_coingecko_market_data(number_of_tokens: int = 2) -> List[Dict]:
    """
    Fetches market data for top N coins from CoinGecko API.
    
    Args:
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List of dictionaries containing market data for each coin
    """
    logger.info(f"Fetching market data for top {number_of_tokens} coins from CoinGecko...")
    
    try:
        # Fetch market data
        market_data = fetch_all_coingecko_market_data(number_of_tokens)
        
        if not market_data:
            logger.error("Failed to fetch market data from CoinGecko")
            return []
        
        # Get list of coin IDs for volume fetching
        coin_ids = list(market_data.keys())
        
        # Fetch 30-day volume data
        volume_data = fetch_all_coingecko_volumes(coin_ids)
        
        # Format response data
        formatted_data = []
        for coin_id, data in market_data.items():
            try:
                formatted_entry = {
                    "symbol": data.get('symbol', '').upper(),
                    "coingecko_data": {
                        "coingecko_name": data.get('name'),
                        "coingecko_id": coin_id,
                        "coingecko_image_url": data.get('image_url'),
                        "coingecko_current_price": data.get('current_price'),
                        "coingecko_market_cap_rank": data.get('market_cap_rank'),
                        "coingecko_market_cap": data.get('market_cap'),
                        "coingecko_fully_diluted_valuation": data.get('fully_diluted_valuation'),
                        "coingecko_total_volume_24h": data.get('total_volume_24h'),
                        "coingecko_mc_derived": data.get('market_cap'),  # Same as market_cap for now
                        "coingecko_circulating": data.get('circulating_supply'),
                        "coingecko_total_supply": data.get('total_supply'),
                        "coingecko_max_supply": data.get('max_supply'),
                        "coingecko_ath_price": data.get('ath_price'),
                        "coingecko_30d_volume": volume_data.get(coin_id, 0)
                    }
                }
                formatted_data.append(formatted_entry)
                logger.info(f"Processed market data for {formatted_entry['symbol']}")
            except Exception as e:
                logger.error(f"Error formatting data for coin {coin_id}: {e}")
                continue
        
        logger.info(f"Successfully processed market data for {len(formatted_data)} coins")
        return formatted_data
    
    except Exception as e:
        logger.error(f"Error in fetch_coingecko_market_data: {e}", exc_info=True)
        return []
    
def fetch_driftpy_data(vat: Vat) -> Dict:
    """
    Fetches market data from DriftPy to check for perp and spot market listings.
    
    Args:
        vat: Vat instance containing market data
    
    Returns:
        Dict: Dictionary containing Drift market data with nested perp and spot markets
    """
    logger.info("Fetching DriftPy data to check perp and spot market listings...")
    
    try:
        # Track long and short positions separately for each market
        market_long_positions = {}   # Track sum of long positions
        market_short_positions = {}  # Track sum of short positions
        market_position_counts = {}  # Track number of positions
        
        # Process each user's positions
        processed_users = 0
        logger.info("Processing positions from users via Vat...")
        
        # Iterate through all users to aggregate positions
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
        
        # Initialize drift markets dictionary
        drift_markets = {}
        
        # Process perpetual markets
        # Use values() instead of items() since MarketMap is an object with specific methods
        for market in vat.perp_markets.values():
            try:
                # Get market name and clean it
                market_name = bytes(market.data.name).decode('utf-8').strip('\x00').strip()
                # Preserve original case for API calls
                original_market_name = market_name
                
                # Extract symbol from market name (e.g., "BTC-PERP" -> "BTC")
                # Keep original case for API calls, but use uppercase for dictionary keys
                symbol_parts = market_name.split('-')
                if len(symbol_parts) > 0:
                    # Use uppercase for normalized symbol (dictionary keys)
                    normalized_symbol = symbol_parts[0].upper().strip()
                else:
                    normalized_symbol = market_name.upper().strip()
                
                # Get oracle price from perp_oracles
                market_price = vat.perp_oracles.get(market.data.market_index)
                if market_price is None:
                    logger.warning(f"Oracle price not found for market {market.data.market_index}")
                    continue
                
                # Get oracle price from the OraclePriceData object and apply precision
                oracle_price = float(market_price.price) / PRICE_PRECISION
                
                # Calculate OI
                market_index = market.data.market_index
                long_amount = market_long_positions.get(market_index, 0)
                short_amount = market_short_positions.get(market_index, 0)
                base_oi = max(long_amount, short_amount)
                
                # Get base decimals
                base_decimals = MARKET_BASE_DECIMALS.get(market_index, 9)
                
                # Calculate OI in USD
                base_oi_readable = base_oi / (10 ** base_decimals)
                oi_usd = base_oi_readable * oracle_price
                
                # Calculate max leverage from margin ratio initial
                initial_margin_ratio = market.data.margin_ratio_initial / 10000
                max_leverage = int(1 / initial_margin_ratio) if initial_margin_ratio > 0 else 0
                
                # Initialize symbol entry if not exists
                if normalized_symbol not in drift_markets:
                    drift_markets[normalized_symbol] = {
                        "drift_is_listed_perp": "true",
                        "drift_is_listed_spot": "false",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_quote_volume_30d": 0.0,
                        "drift_total_base_volume_30d": 0.0,
                        "drift_max_leverage": max_leverage,
                        "drift_open_interest": oi_usd,
                        "drift_funding_rate_1h": float(market.data.amm.last_funding_rate) / 1e6 * 100
                    }
                
                # Add perp market data - use original market name with preserved case for API calls
                drift_markets[normalized_symbol]["drift_perp_markets"][original_market_name] = {
                    "drift_perp_oracle_price": oracle_price,
                    "drift_perp_quote_volume_30d": 0.0,
                    "drift_perp_base_volume_30d": 0.0
                }
                
            except Exception as e:
                logger.error(f"Error processing perp market: {e}")
                continue
        
        logger.info(f"Found {len(drift_markets)} perpetual markets on Drift")
        
        # Process spot markets
        logger.info("Processing spot markets from Vat...")
        spot_market_count = sum(1 for _ in vat.spot_markets.values())
        logger.info(f"Found {spot_market_count} spot markets in vat.spot_markets")
        
        # Process each spot market
        for market in vat.spot_markets.values():
            try:
                # Get market name and clean it
                market_name = bytes(market.data.name).decode('utf-8').strip('\x00').strip()
                logger.info(f"Processing spot market: {market_name} (market_index: {market.data.market_index})")
                
                # Extract symbol with special handling for BTC variants
                # IMPORTANT: Preserve original case for API calls
                raw_symbol = market_name.strip()  # Preserve case for API calls
                symbol_key = raw_symbol.upper()  # Use uppercase for dictionary keys for consistent lookups
                logger.info(f"Raw symbol before BTC check: {raw_symbol}")
                
                if symbol_key in ['WBTC', 'CBBTC']:
                    normalized_symbol = 'BTC'
                    logger.info(f"Found BTC variant: {raw_symbol}, mapping to {normalized_symbol}")
                else:
                    normalized_symbol = symbol_key.replace('W', '').strip()
                
                # Get spot oracle price
                market_price = vat.spot_oracles.get(market.data.market_index)
                if market_price is None:
                    logger.warning(f"No oracle price found for market {raw_symbol} (index: {market.data.market_index})")
                    continue
                
                spot_oracle_price = float(market_price.price) / PRICE_PRECISION
                logger.info(f"Got oracle price for {raw_symbol}: {spot_oracle_price}")
                
                # Initialize or update symbol entry
                if normalized_symbol not in drift_markets:
                    logger.info(f"Initializing new market entry for symbol {normalized_symbol}")
                    drift_markets[normalized_symbol] = {
                        "drift_is_listed_perp": "false",
                        "drift_is_listed_spot": "true",
                        "drift_perp_markets": {},
                        "drift_spot_markets": {},
                        "drift_total_quote_volume_30d": 0.0,
                        "drift_total_base_volume_30d": 0.0,
                        "drift_max_leverage": 0.0,
                        "drift_open_interest": 0.0,
                        "drift_funding_rate_1h": 0.0
                    }
                else:
                    logger.info(f"Updating existing market entry for symbol {normalized_symbol}")
                    drift_markets[normalized_symbol]["drift_is_listed_spot"] = "true"
                
                # Add spot market data - use the original market name with preserved case for API calls
                logger.info(f"Adding spot market {raw_symbol} to {normalized_symbol}'s drift_spot_markets")
                drift_markets[normalized_symbol]["drift_spot_markets"][raw_symbol] = {
                    "drift_spot_oracle_price": spot_oracle_price,
                    "drift_spot_quote_volume_30d": 0.0,
                    "drift_spot_base_volume_30d": 0.0
                }
                logger.info(f"Current spot markets for {normalized_symbol}: {list(drift_markets[normalized_symbol]['drift_spot_markets'].keys())}")
                
            except Exception as e:
                logger.error(f"Error processing spot market: {e}")
                continue
        
        logger.info(f"Completed market processing. Final market count: {len(drift_markets)}")
        return drift_markets
        
    except Exception as e:
        logger.error(f"Error in fetch_driftpy_data: {e}")
        return {}

def fetch_drift_data_api_data(discovered_markets: Dict = None) -> Dict:
    """
    Fetches market data from Drift API including historical trade data.
    Synchronous version of the trade fetching functionality.
    
    Args:
        discovered_markets (Dict): Dictionary of markets discovered from fetch_driftpy_data
                                   Contains market structure with perp and spot listings
    
    Returns:
        Dict: Dictionary containing Drift API data for each coin with nested market data
    """
    logger.info("Fetching Drift API data...")
    
    # Initialize result dictionary
    drift_markets = {}
    
    # Constants for trade fetching
    DAYS_TO_CONSIDER = 30
    DRIFT_DATA_API_BASE = os.getenv("DRIFT_DATA_API_BASE_URL", "https://data.api.drift.trade")
    DRIFT_DATA_API_AUTH = os.getenv("DRIFT_DATA_API_AUTH", "")  # Get auth token from env
    DRIFT_DATA_API_HEADERS = {
        "accept": "application/json",
        "x-origin-verify": DRIFT_DATA_API_AUTH  # Use x-origin-verify header
    }
    DRIFT_API_RATE_LIMIT_INTERVAL = 0.2  # seconds between requests
    
    def fetch_api_page(url: str, retries: int = 5) -> Dict:
        """
        Synchronous version of API page fetching with retries and rate limiting.
        """
        import requests
        import time
        
        if not DRIFT_DATA_API_AUTH:
            logger.error("DRIFT_DATA_API_AUTH environment variable not set")
        
        last_request_time = getattr(fetch_api_page, 'last_request_time', 0)
        
        for attempt in range(retries):
            # Apply rate limiting
            current_time = time.time()
            wait_time = DRIFT_API_RATE_LIMIT_INTERVAL - (current_time - last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            
            try:
                logger.info(f"Fetching data from: {url}")
                response = requests.get(url, headers=DRIFT_DATA_API_HEADERS, timeout=10)
                fetch_api_page.last_request_time = time.time()
                
                if response.status_code != 200:
                    logger.warning(f"API request failed: {url}, status: {response.status_code}, response: {response.text[:200]}")
                    if attempt < retries - 1:
                        logger.info(f"Retrying request (attempt {attempt + 2}/{retries})")
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return {"success": False, "records": [], "meta": {"totalRecords": 0}}
                
                data = response.json()
                logger.info(f"Successfully fetched data from {url} - got {len(data.get('records', []))} records")
                return data
                
            except Exception as e:
                logger.warning(f"Error fetching {url}: {str(e)}")
                if attempt < retries - 1:
                    logger.info(f"Retrying request (attempt {attempt + 2}/{retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {"success": False, "records": [], "meta": {"totalRecords": 0}}
        
        return {"success": False, "records": [], "meta": {"totalRecords": 0}}

    def fetch_market_trades(symbol: str, start_date: datetime, end_date: datetime = None) -> List[Dict]:
        """
        Fetch market candle data for a symbol using the candles endpoint.
        This replaces the inefficient day-by-day trade fetching with a single API call.
        
        Args:
            symbol: Market symbol (e.g., "SOL-PERP")
            start_date: Start date for candle data
            end_date: End date for candle data (defaults to current time)
            
        Returns:
            List[Dict]: List of candle data records
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Log the start of fetching for this symbol
        logger.info(f"Starting to fetch candle data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates to Unix timestamps (seconds)
        # NOTE: For the Drift API, startTs should be the more recent timestamp (end_date)
        # and endTs should be the older timestamp (start_date), which is counter-intuitive
        end_ts = int(start_date.timestamp())     # Earlier date - goes in endTs
        start_ts = int(end_date.timestamp())     # Later date - goes in startTs
        
        # Calculate number of days to request (plus 1 to include both start and end)
        days = (end_date - start_date).days + 1
        
        if not DRIFT_DATA_API_BASE:
            logger.error("DRIFT_DATA_API_BASE environment variable not set")
            return []
        
        # Use the candles endpoint with daily resolution (D)
        # The API expects: startTs = most recent, endTs = oldest (reverse of what might be expected)
        url = f"{DRIFT_DATA_API_BASE}/market/{symbol}/candles/D?startTs={start_ts}&endTs={end_ts}&limit={min(days, 31)}"
        
        logger.info(f"Requesting candles with startTs={start_ts} (now), endTs={end_ts} (past)")
        
        # Fetch candle data
        data = fetch_api_page(url)
        
        if data.get("success") and "records" in data:
            records_count = len(data["records"])
            logger.info(f"Successfully fetched {records_count} candles for {symbol}")
            return data["records"]
        else:
            logger.warning(f"Failed to fetch candle data for {symbol}")
            return []
    
    def calculate_market_volume(symbol: str) -> dict:
        """
        Calculate total trading volume for a market over the past 30 days using candle data.
        Returns both quote volume (in USD) and base volume (in token units).
        
        Args:
            symbol: Market symbol (e.g., "SOL-PERP")
            
        Returns:
            dict: Dictionary with quote_volume and base_volume
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_TO_CONSIDER)
        
        try:
            candles = fetch_market_trades(symbol, start_date, end_date)
            
            total_quote_volume = 0.0
            total_base_volume = 0.0
            
            for candle in candles:
                try:
                    # Extract both quote and base volumes
                    quote_volume = float(candle.get("quoteVolume", 0))
                    base_volume = float(candle.get("baseVolume", 0))
                    
                    total_quote_volume += quote_volume
                    total_base_volume += base_volume
                except (ValueError, TypeError):
                    continue
            
            logger.info(f"Calculated {DAYS_TO_CONSIDER}-day volumes for {symbol}: " 
                        f"${total_quote_volume:,.2f} (quote), {total_base_volume:,.2f} (base)")
            
            return {
                "quote_volume": total_quote_volume,
                "base_volume": total_base_volume
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume for {symbol}: {str(e)}")
            return {
                "quote_volume": 0.0,
                "base_volume": 0.0
            }
    
    # Process each market
    try:
        if not discovered_markets:
            logger.warning("No discovered markets provided, skipping market processing")
            return {}

        logger.info(f"Processing {len(discovered_markets)} discovered markets")
        
        for symbol, market_info in discovered_markets.items():
            drift_markets[symbol] = {
                "drift_is_listed_spot": market_info.get("drift_is_listed_spot", "false"),
                "drift_is_listed_perp": market_info.get("drift_is_listed_perp", "false"),
                "drift_perp_markets": {},
                "drift_spot_markets": {},
                "drift_total_quote_volume_30d": 0.0,
                "drift_total_base_volume_30d": 0.0,
                "drift_max_leverage": market_info.get("drift_max_leverage", 0.0),
                "drift_open_interest": market_info.get("drift_open_interest", 0.0),
                "drift_funding_rate_1h": market_info.get("drift_funding_rate_1h", 0.0)
            }
            
            # Process perp markets
            total_quote_volume = 0.0
            total_base_volume = 0.0
            for perp_market_name in market_info.get("drift_perp_markets", {}):
                # Use original market name (case-preserved) for API call
                logger.info(f"Calculating volume for perp market: {perp_market_name} (preserving case)")
                volume_data = calculate_market_volume(perp_market_name)  # Returns dict with quote and base volumes
                
                drift_markets[symbol]["drift_perp_markets"][perp_market_name] = {
                    "drift_perp_oracle_price": market_info["drift_perp_markets"][perp_market_name].get("drift_perp_oracle_price", 0.0),
                    "drift_perp_quote_volume_30d": volume_data["quote_volume"],
                    "drift_perp_base_volume_30d": volume_data["base_volume"],
                    "drift_is_listed_perp": True,
                    "drift_perp_oi": volume_data["quote_volume"] * 0.5  # Example calculation
                }
                total_quote_volume += volume_data["quote_volume"]
                total_base_volume += volume_data["base_volume"]
            
            # Process spot markets
            for spot_market_name in market_info.get("drift_spot_markets", {}):
                # Use original market name (case-preserved) for API call
                logger.info(f"Calculating volume for spot market: {spot_market_name} (preserving case)")
                volume_data = calculate_market_volume(spot_market_name)  # Returns dict with quote and base volumes
                
                drift_markets[symbol]["drift_spot_markets"][spot_market_name] = {
                    "drift_spot_oracle_price": market_info["drift_spot_markets"][spot_market_name].get("drift_spot_oracle_price", 0.0),
                    "drift_spot_quote_volume_30d": volume_data["quote_volume"],
                    "drift_spot_base_volume_30d": volume_data["base_volume"],
                    "drift_is_listed_spot": True
                }
                total_quote_volume += volume_data["quote_volume"]
                total_base_volume += volume_data["base_volume"]
            
            # Update total volumes
            drift_markets[symbol]["drift_total_quote_volume_30d"] = total_quote_volume
            drift_markets[symbol]["drift_total_base_volume_30d"] = total_base_volume
            
            # Update OI and funding rate if perp markets exist
            if market_info.get("drift_is_listed_perp") == "true":
                drift_markets[symbol]["drift_open_interest"] = total_quote_volume * 0.75  # Example calculation
                drift_markets[symbol]["drift_funding_rate_1h"] = market_info.get("drift_funding_rate_1h", 0.0)
    
    except Exception as e:
        logger.error(f"Error in fetch_drift_data_api_data: {e}")
        return {}
    
    return drift_markets

def calculate_drift_volume_score(volume_30d: float) -> float:
    """
    Calculate a score based on Drift Protocol 30-day trading volume.
    
    Args:
        volume_30d (float): 30-day trading volume in USD
        
    Returns:
        float: Volume score between 0 and 25
    """
    # Score based on volume tiers according to the scoring breakdown
    if volume_30d >= 500_000_000:  # $500M+
        return 25.0
    elif volume_30d >= 100_000_000:  # $100M - $499M
        return 20.0
    elif volume_30d >= 25_000_000:  # $25M - $99M
        return 15.0
    elif volume_30d >= 1_000_000:  # $1M - $24M
        return 10.0
    elif volume_30d >= 100_000:  # $100K - $999K
        return 5.0
    else:  # < $100K
        return 0.0

def calculate_open_interest_score(open_interest: float) -> float:
    """
    Calculate a score based on Drift Protocol open interest.
    
    Args:
        open_interest (float): Open interest in USD
        
    Returns:
        float: Open interest score between 0 and 25
    """
    # Score based on OI tiers according to the scoring breakdown
    if open_interest >= 5_000_000:  # $5M+
        return 25.0
    elif open_interest >= 1_000_000:  # $1M - $4.9M
        return 20.0
    elif open_interest >= 250_000:  # $250K - $999K
        return 15.0
    elif open_interest >= 50_000:  # $50K - $249K
        return 10.0
    elif open_interest >= 5_000:  # $5K - $49K
        return 5.0
    else:  # < $5K
        return 0.0

def calculate_global_volume_score(daily_volume: float) -> float:
    """
    Calculate a score based on global trading volume from CoinGecko.
    
    Args:
        daily_volume (float): Daily trading volume in USD
        
    Returns:
        float: Global volume score between 0 and 40
    """
    # Score based on global volume tiers according to the scoring breakdown
    if daily_volume >= 15_000_000_000:  # $500M+
        return 40.0
    elif daily_volume >= 7_500_000_000:  # $250M - $499M
        return 30.0
    elif daily_volume >= 3_000_000_000:  # $100M - $249M
        return 20.0
    elif daily_volume >= 750_000_000:  # $25M - $99M
        return 10.0
    elif daily_volume >= 150_000_000:  # $5M - $24M
        return 5.0
    else:  # < $5M
        return 0.0

def calculate_fdv_score(fdv: float) -> float:
    """
    Calculate a score based on Fully Diluted Valuation (FDV).
    
    Args:
        fdv (float): Fully Diluted Valuation in USD
        
    Returns:
        float: FDV score between 0 and 10
    """
    # Score based on FDV tiers according to the scoring breakdown
    if fdv >= 10_000_000_000:  # $10B+
        return 10.0
    elif fdv >= 1_000_000_000:  # $1B - $9.9B
        return 8.0
    elif fdv >= 500_000_000:  # $500M - $999M
        return 6.0
    elif fdv >= 100_000_000:  # $100M - $499M
        return 2.0
    else:  # < $100M
        return 0.0

def get_market_recommendation(total_score: float, current_leverage: float) -> str:
    """
    Determine market recommendation based on total score and current leverage.
    
    Args:
        total_score (float): Total score from all criteria (0-100)
        current_leverage (float): Current maximum leverage offered
        
    Returns:
        str: Recommendation (List, Increase Leverage, Decrease Leverage, Delist, Keep Unlisted, or Maintain Leverage)
    """
    # Define upper and lower bound thresholds for different leverage levels
    SCORE_UB = {
        0: 45,    # Threshold to list (if score >= 45)
        2: float('inf'),  # No upper bound for 2x (cannot increase further via this logic)
        4: 75,    # Threshold to increase from 4x (if score >= 75)
        5: 80,    # Threshold to increase from 5x (if score >= 80)
        10: 90,   # Threshold to increase from 10x (if score >= 90)
        20: 95    # Threshold to increase from 20x (if score >= 95) - Usually max
    }
    
    SCORE_LB = {
        0: 0,     # Not applicable for unlisted
        2: 40,    # Delist threshold for 2x (if score <= 40)
        4: 50,    # Threshold to decrease from 4x (if score <= 50)
        5: 60,    # Threshold to decrease from 5x (if score <= 60)
        10: 70,   # Threshold to decrease from 10x (if score <= 70)
        20: 75    # Threshold to decrease from 20x (if score <= 75)
    }
    
    # Find the closest applicable leverage level for thresholds
    # This logic primarily finds the *current* leverage level within the defined thresholds
    leverage_levels = sorted(SCORE_UB.keys())
    closest_leverage = leverage_levels[0] # Default to 0 (unlisted)
    
    for level in leverage_levels:
        if level <= current_leverage:
            closest_leverage = level
    
    # Apply decision logic
    if current_leverage == 0:  # Unlisted market
        if total_score >= SCORE_UB[0]:
            return "List"
        else:
            return "Keep Unlisted" # Score is too low to list
            
    elif current_leverage == 2: # Currently listed at 2x leverage
        if total_score <= SCORE_LB[2]:
            return "Delist"
        # Note: Increase leverage from 2x might require manual review or different logic
        # This logic currently doesn't recommend increasing from 2x
        else:
            return "Maintain Leverage" 
            
    else: # Listed market with leverage > 2x
        # Check if score is high enough to increase leverage (only if current leverage is not max)
        if current_leverage < 20 and total_score >= SCORE_UB.get(closest_leverage, float('inf')):
             return "Increase Leverage"
        # Check if score is low enough to decrease leverage
        elif total_score <= SCORE_LB.get(closest_leverage, 0):
             # Ensure we don't decrease below 2x (handled by Delist logic for 2x)
             if closest_leverage > 2: 
                  return "Decrease Leverage"
             else: # If logic somehow reaches here for 2x, delist check already happened
                  return "Maintain Leverage" 
        # Score is within the bounds for the current leverage level
        else:
             return "Maintain Leverage"

def score_assets(assets: List[Dict], drift_data: Dict) -> List[Dict]:
    """
    Score assets based on Drift market data and other metrics.
    
    Args:
        assets (List[Dict]): List of asset dictionaries with market data
        drift_data (Dict): Nested dictionary containing Drift market data for perp and spot markets
        
    Returns:
        List[Dict]: List of scored assets with additional metrics
    """
    scored_assets = []
    
    for asset in assets:
        symbol = asset.get('symbol', '').upper()
        
        # Get Drift market data if available
        market_info = drift_data.get(symbol, {})
        
        # Get raw metrics
        drift_volume_30d = market_info.get('drift_total_quote_volume_30d', 0.0)
        drift_open_interest = market_info.get('drift_open_interest', 0.0)
        global_30d_volume = asset.get('coingecko_data', {}).get('coingecko_30d_volume', 0.0)
        global_24h_volume = asset.get('coingecko_data', {}).get('coingecko_total_volume_24h', 0.0)
        fdv = asset.get('coingecko_data', {}).get('coingecko_fully_diluted_valuation', 0.0)
        # Handle case where FDV is None or 0, fall back to market cap
        if not fdv or fdv == 0:
            fdv = asset.get('coingecko_data', {}).get('coingecko_market_cap', 0.0)
            
        current_leverage = market_info.get('drift_max_leverage', 0.0)
        
        # Calculate component scores
        drift_volume_score = calculate_drift_volume_score(drift_volume_30d)
        open_interest_score = calculate_open_interest_score(drift_open_interest)
        global_volume_score = calculate_global_volume_score(global_30d_volume) # Use 30d volume for scoring
        fdv_score = calculate_fdv_score(fdv or 0.0) # Ensure fdv is not None
        
        # Calculate total score
        total_score = drift_volume_score + open_interest_score + global_volume_score + fdv_score
        
        # Get recommendation
        recommendation = get_market_recommendation(total_score, current_leverage)
        
        # Create scored asset dictionary with nested drift_data
        scored_asset = {
            **asset,  # Include all original asset data
            'drift_volume_score': drift_volume_score,
            'open_interest_score': open_interest_score,
            'global_volume_score': global_volume_score,
            'fdv_score': fdv_score,
            'total_score': total_score,
            'recommendation': recommendation,
            'raw_metrics': {
                'drift_volume_30d': drift_volume_30d,
                'drift_open_interest': drift_open_interest,
                'coingecko_30d_volume': global_30d_volume,
                'coingecko_total_volume_24h': global_24h_volume, # Keep 24h for info
                'fdv': fdv,
                'current_max_leverage': current_leverage
            }
        }
        
        # If there's no drift data, initialize with default structure
        if symbol not in drift_data:
            scored_asset['drift_data'] = {
                "drift_is_listed_spot": "false",
                "drift_is_listed_perp": "false",
                "drift_perp_markets": {},
                "drift_spot_markets": {},
                "drift_total_quote_volume_30d": 0.0,
                "drift_total_base_volume_30d": 0.0,
                "drift_max_leverage": 0.0,
                "drift_open_interest": 0.0,
                "drift_funding_rate_1h": 0.0
            }
        else:
            scored_asset['drift_data'] = drift_data[symbol]
        
        # Ensure both 24h and 30d volumes are present in the response coingecko_data
        if 'coingecko_data' in scored_asset:
            scored_asset['coingecko_data']['coingecko_30d_volume'] = global_30d_volume
            scored_asset['coingecko_data']['coingecko_total_volume_24h'] = global_24h_volume
        
        scored_assets.append(scored_asset)
    
    # Sort assets by total score in descending order
    scored_assets.sort(key=lambda x: x['total_score'], reverse=True)
    
    return scored_assets

def process_drift_markets(scored_data: List[Dict], drift_data: Dict) -> Dict:
    """
    Process Drift markets data and update drift_data dictionary with the new nested structure.
    This function primarily reformats and aggregates volumes already fetched.
    
    Args:
        scored_data (List[Dict]): List of dictionaries containing scored market data (unused in current logic but kept for potential future use)
        drift_data (Dict): Dictionary containing Drift market data with nested perp and spot markets
        
    Returns:
        Dict: Updated drift_data dictionary ready for integration
    """
    processed_drift_data = {}
    
    # Iterate through the symbols present in the original drift_data
    for symbol, symbol_drift_data in drift_data.items():
        if not symbol:
            continue
            
        # Initialize default structure for the symbol
        processed_drift_data[symbol] = {
            "drift_is_listed_perp": symbol_drift_data.get('drift_is_listed_perp', 'false'),
            "drift_is_listed_spot": symbol_drift_data.get('drift_is_listed_spot', 'false'),
            "drift_perp_markets": symbol_drift_data.get('drift_perp_markets', {}),
            "drift_spot_markets": symbol_drift_data.get('drift_spot_markets', {}),
            "drift_total_quote_volume_30d": 0.0,  # Will be calculated from market volumes
            "drift_total_base_volume_30d": 0.0,   # Will be calculated from market volumes
            "drift_max_leverage": symbol_drift_data.get('drift_max_leverage', 0.0),
            "drift_open_interest": symbol_drift_data.get('drift_open_interest', 0.0),
            "drift_funding_rate_1h": symbol_drift_data.get('drift_funding_rate_1h', 0.0)
        }
        
        # Calculate total volumes by summing across all markets for the symbol
        total_quote_volume = 0.0
        total_base_volume = 0.0
        
        # Sum volumes from perp markets
        for market_name, market_data in processed_drift_data[symbol].get("drift_perp_markets", {}).items():
            total_quote_volume += market_data.get("drift_perp_quote_volume_30d", 0.0)
            total_base_volume += market_data.get("drift_perp_base_volume_30d", 0.0)
            # Clean up redundant/old fields if they exist
            if "drift_perp_volume_30d" in market_data:
                del market_data["drift_perp_volume_30d"]
            if "drift_perp_oi" in market_data: # Example OI was added here earlier, remove if final OI is at symbol level
                 del market_data["drift_perp_oi"]

        # Sum volumes from spot markets
        for market_name, market_data in processed_drift_data[symbol].get("drift_spot_markets", {}).items():
            total_quote_volume += market_data.get("drift_spot_quote_volume_30d", 0.0)
            total_base_volume += market_data.get("drift_spot_base_volume_30d", 0.0)
             # Clean up redundant/old fields if they exist
            if "drift_spot_volume_30d" in market_data:
                del market_data["drift_spot_volume_30d"]
                
        # Update total volumes at the symbol level
        processed_drift_data[symbol]["drift_total_quote_volume_30d"] = total_quote_volume
        processed_drift_data[symbol]["drift_total_base_volume_30d"] = total_base_volume
        
        # Ensure OI and funding rate are correctly placed (usually associated with perps)
        # If no perps listed, these should likely be 0 or handled appropriately
        if processed_drift_data[symbol]["drift_is_listed_perp"] == 'false':
             processed_drift_data[symbol]["drift_open_interest"] = 0.0
             processed_drift_data[symbol]["drift_funding_rate_1h"] = 0.0
             processed_drift_data[symbol]["drift_max_leverage"] = 0.0 # Max leverage applies to perps


    return processed_drift_data

def main(vat: Vat, number_of_tokens: int = 2) -> List[Dict]:
    """
    Main function to fetch, aggregate, and score market data from multiple sources.
    Provides recommendations for Drift protocol listing decisions.
    
    Args:
        vat: Vat instance containing market data
        number_of_tokens (int, optional): Number of tokens to fetch data for. Defaults to 2.
    
    Returns:
        List[Dict]: List of dictionaries containing comprehensive market data and Drift-specific recommendations
    """
    logger.info("Starting market data aggregation process...")
    
    try:
        # Fetch CoinGecko data first to get the list of tokens we're interested in
        coingecko_data = fetch_coingecko_market_data(number_of_tokens)
        if not coingecko_data:
            logger.error("Failed to fetch CoinGecko data")
            return []
            
        # Get the list of symbols we're processing
        processed_symbols = {asset["symbol"].upper() for asset in coingecko_data}
        logger.info(f"Processing {len(processed_symbols)} symbols from CoinGecko: {', '.join(processed_symbols)}")
        
        # Fetch DriftPy data for market discovery (OI, leverage, funding, prices)
        driftpy_discovered_data = fetch_driftpy_data(vat)
        
        # Filter drift_data to only include symbols we got from CoinGecko
        filtered_driftpy_data = {
            symbol: data for symbol, data in driftpy_discovered_data.items()
            if symbol in processed_symbols
        }
        logger.info(f"Filtered Drift markets from Vat to {len(filtered_driftpy_data)} matching symbols")
        
        # Fetch Drift API data (primarily for volumes) using the discovered and filtered markets
        drift_api_data = fetch_drift_data_api_data(filtered_driftpy_data)
        
        # Merge drift_api_data volumes into driftpy_discovered_data structure
        # This combined structure will hold the most complete Drift-specific info
        combined_drift_data = filtered_driftpy_data.copy() # Start with Vat data (OI, leverage etc)

        for symbol, api_market_data in drift_api_data.items():
            if symbol in combined_drift_data:
                # Update perp market volumes
                for market_name, perp_api_data in api_market_data.get('drift_perp_markets', {}).items():
                    if market_name in combined_drift_data[symbol].get('drift_perp_markets', {}):
                        # Update with both quote and base volumes from API
                        combined_drift_data[symbol]['drift_perp_markets'][market_name]['drift_perp_quote_volume_30d'] = perp_api_data.get('drift_perp_quote_volume_30d', 0.0)
                        combined_drift_data[symbol]['drift_perp_markets'][market_name]['drift_perp_base_volume_30d'] = perp_api_data.get('drift_perp_base_volume_30d', 0.0)
                
                # Update spot market volumes
                for market_name, spot_api_data in api_market_data.get('drift_spot_markets', {}).items():
                    if market_name in combined_drift_data[symbol].get('drift_spot_markets', {}):
                         # Update with both quote and base volumes from API
                        combined_drift_data[symbol]['drift_spot_markets'][market_name]['drift_spot_quote_volume_30d'] = spot_api_data.get('drift_spot_quote_volume_30d', 0.0)
                        combined_drift_data[symbol]['drift_spot_markets'][market_name]['drift_spot_base_volume_30d'] = spot_api_data.get('drift_spot_base_volume_30d', 0.0)
                
                # Update total volumes at symbol level from API data
                combined_drift_data[symbol]['drift_total_quote_volume_30d'] = api_market_data.get('drift_total_quote_volume_30d', 0.0)
                combined_drift_data[symbol]['drift_total_base_volume_30d'] = api_market_data.get('drift_total_base_volume_30d', 0.0)
                
                # Note: OI, leverage, funding rate should already be present from fetch_driftpy_data

        # Process the combined drift data (calculate totals, clean structure)
        final_drift_data = process_drift_markets([], combined_drift_data) # Pass combined data

        # Score the assets using CoinGecko data and the final processed Drift data
        scored_data = score_assets(coingecko_data, final_drift_data)
        
        # The 'drift_data' key within each scored_asset item now holds the final processed data
        # The structure is already integrated during score_assets
        
        logger.info(f"Successfully aggregated and scored data for {len(scored_data)} assets")
        return scored_data
        
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return []

if __name__ == "__main__":
    # Example of how to run locally if needed (requires setting up Vat)
    # from driftpy.pickle.vat import Vat
    # vat = Vat(...) # Initialize Vat somehow
    # results = main(vat, 25)
    # print(results)
    pass # Cannot run directly without Vat initialization
