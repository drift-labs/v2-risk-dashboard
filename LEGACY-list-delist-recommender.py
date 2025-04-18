# This page offers recommendations on listing and delisting policies for the hyperliquid exchange.
# stalequant 2025-04-02

# Import necessary libraries
import json # For handling JSON data (reading/writing files, parsing responses)
import math # For mathematical operations like logarithm
import requests # For making HTTP requests to fetch data from APIs
import time # For time-related functions, like getting the current time

import ccxt # A library for interacting with various cryptocurrency exchanges
import numpy as np # For numerical operations, especially with arrays and matrices
import pandas as pd # For data manipulation and analysis using DataFrames

# --- Configuration Constants ---

# Set of stablecoin symbols to exclude from analysis
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
# Number of past days to consider for historical data analysis
DAYS_TO_CONSIDER = 30

# Define the columns to be included in the final output JSON file
OUTPUT_COLS = [
    'Symbol', 'Max Lev. on HL', 'Strict','Recommendation',
    'Market Cap Score', 'Spot Vol Score', 'Futures Vol Score',
    'HL Activity Score', 'HL Liquidity Score', 'Score',
    'MC $m', 'Spot Volume $m', 'Spot Vol Geomean $m',
    'Fut Volume $m', 'Fut Vol Geomean $m', 'OI on HL $m',
    'Volume on HL $m', 'HLP Vol Share %', 'HLP OI Share %',
    'HL Slip. $3k', 'HL Slip. $30k'
]
# Filename for the output JSON containing the correlation matrix
OUTPUT_CORR_MAT_JSON = 'hl_screen_corr.json'
# Filename for the output JSON containing the main processed data and recommendations
OUTPUT_RAW_DATA_JSON = 'hl_screen_main.json'

# Define score boundaries (Lower Bound, Upper Bound) for different leverage levels (0, 3, 5, 10)
# Used in the recommendation logic to determine if leverage should be increased/decreased or if a coin should be listed/delisted
SCORE_UB = {0: 62, 3: 75, 5: 85, 10: 101} # Upper Bound: If score >= this, consider increasing leverage/listing
SCORE_LB = {0: 0, 3: 37, 5: 48, 10: 60}  # Lower Bound: If score < this, consider decreasing leverage/delisting

# Set of reference exchanges for spot market data
REFERENCE_SPOT_EXCH = {
    'binance', 'bybit', 'okx', 'gate', 'kucoin', 'mexc',
    'cryptocom', 'coinbase', 'kraken', 'hyperliquid'
}
# Set of reference exchanges for futures market data
REFERENCE_FUT_EXCH = {
    'bybit', 'binance', 'gate', 'mexc', 'okx',
    'htx', 'krakenfutures', 'cryptocom', 'bitmex',
    'hyperliquid'
}
# Set of symbols on Hyperliquid considered 'strict' - they get a score boost
HL_STRICT = {'PURR','CATBAL','HFUN','PIP','JEFF','VAPOR','SOLV',
             'FARM','ATEHUN','SCHIZO','OMNIX','POINTS','RAGE'}

# Define the structure for calculating scores based on various metrics
# Each category (e.g., 'Market Cap Score') has sub-metrics (e.g., 'MC $m')
# 'kind' specifies the scaling ('exp' for exponential, 'linear' for linear)
# 'start', 'end', 'steps' define the range and granularity for score calculation
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
    'HL Activity Score': {
        'Volume on HL $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10},
        'OI on HL $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10},
    },
    'HL Liquidity Score': {
        'HLP Vol Share %': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5}, # Higher % is worse -> lower score
        'HLP OI Share %': {'kind': 'linear', 'start': 10, 'end': 0, 'steps': 5},  # Higher % is worse -> lower score
        'HL Slip. $3k': {'kind': 'linear', 'start': 5, 'end': 0, 'steps': 5},    # Higher slippage is worse -> lower score
        'HL Slip. $30k': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5},   # Higher slippage is worse -> lower score
    }
}
# Score boost applied to symbols in the HL_STRICT set
HL_STRICT_BOOST = 5

# Calculate the earliest timestamp (in seconds since epoch) to keep data for
# Adds 5 extra days buffer to ensure enough data is fetched
earliest_ts_to_keep = time.time()-(DAYS_TO_CONSIDER+5)*24*60*60

# --- Utility Functions ---

def sig_figs(number, sig_figs=3):
    """
    Rounds a number to a specified number of significant figures.
    Handles NaN and non-positive numbers by returning 0.

    Args:
        number (float or int): The number to round.
        sig_figs (int): The desired number of significant figures.

    Returns:
        float or int: The rounded number, or 0 if input is NaN or <= 0.
    """
    if np.isnan(number) or number <= 0:
        return 0 # Return 0 for NaN or non-positive numbers
    # Calculate the rounding precision based on the number's magnitude and desired sig figs
    return round(number, int(sig_figs - 1 - math.log10(number)))


def clean_symbol(symbol, exch=''):
    """
    Cleans and standardizes cryptocurrency symbols.
    Removes common suffixes (like '1000', 'k'), applies general aliases,
    and applies exchange-specific aliases.

    Args:
        symbol (str): The raw symbol string (e.g., 'BTC/USDT:USDT', '1000PEPE/USD').
        exch (str, optional): The exchange name for applying specific aliases. Defaults to ''.

    Returns:
        str: The cleaned, standardized base coin symbol (e.g., 'BTC', 'PEPE').
    """
    # General token aliases (maps variations or specific names to a standard symbol)
    TOKEN_ALIASES = {
        'HPOS10I': 'BITCOIN', 'HPOS': 'HPOS', 'HPO': 'HPOS',
        'BITCOIN': 'HPOS', 'NEIROCTO': 'NEIRO', '1MCHEEMS': 'CHEEMS',
        '1MBABYDOGE': 'BABYDOGE', 'JELLYJELLY': 'JELLY'
    }
    # Exchange-specific token aliases (handles cases where a coin has a different ticker on certain exchanges)
    EXCH_TOKEN_ALIASES = {
        ('NEIRO', 'bybit'): 'NEIROETH',
        ('NEIRO', 'gate'): 'NEIROETH',
        ('NEIRO', 'kucoin'): 'NEIROETH'
    }
    # Extract the base part of the symbol (before '/')
    redone = symbol.split('/')[0]
    # Remove common numerical/magnitude suffixes
    for suffix in ['10000000', '1000000', '1000', 'k']:
        redone = redone.replace(suffix, '')
    # Apply exchange-specific alias if applicable
    redone = EXCH_TOKEN_ALIASES.get((redone, exch), redone)
    # Apply general alias if applicable
    return TOKEN_ALIASES.get(redone, redone)

def get_hot_ccxt_api(exch):
    """
    Initializes and returns a 'hot' ccxt exchange API instance.
    'Hot' means it attempts a test API call to ensure connectivity/authentication
    (though the test call here might not be strictly necessary for public data).

    Args:
        exch (str): The name of the exchange (compatible with ccxt library).

    Returns:
        ccxt.Exchange: An initialized ccxt exchange object.
    """
    # Get the exchange class from ccxt based on the name
    api = getattr(ccxt, exch)()
    
    # Ensure markets are loaded
    try:
        api.load_markets()
    except Exception as e:
        print(f"Error loading markets for {exch}: {e}")
    
    try:
        # Attempt a simple API call to 'warm up' or test the connection
        # Fetches ticker info for BTC/USDT perpetual swap
        api.fetch_ticker('BTC/USDT:USDT')
    except Exception as e:
        # Test call might fail if market doesn't exist or rate limits - log but continue
        print(f"Test API call for {exch} failed: {e}")
        # Try alternative common market pairs if the first one fails
        try:
            api.fetch_ticker('BTC/USDT')
        except Exception:
            try:
                api.fetch_ticker('BTC/USD')
            except Exception:
                # Log warning but continue - we've tried multiple pairs
                print(f"Warning: Could not verify API connectivity for {exch}")
    
    # Final check if markets were loaded
    if api.markets is None or len(api.markets) == 0:
        print(f"Warning: No markets were loaded for {exch}")
    
    return api

# Fetch market data specifically for Hyperliquid once to use later
# Stores details about available markets (symbols, precision, limits, etc.)
hl_markets = get_hot_ccxt_api('hyperliquid').markets

# %% --- Data Download and Processing Sections ---

# --- Reference Exchange Data (Spot & Futures) ---

def dl_reference_exch_data():
    """
    Downloads or loads cached OHLCV (Open, High, Low, Close, Volume) data
    for specified reference spot and futures exchanges. Saves data to JSON files.

    Returns:
        dict: A dictionary where keys are tuples (is_spot, exchange_name)
              and values are dictionaries of {symbol: ohlcv_data}.
    """
    def download_exch(exch, spot):
        """Helper function to download/load data for a single exchange."""
        # Define filename based on exchange and market type (spot/futures)
        filename = f'exch_candles_{exch}_{"s" if spot else "f"}.json'
        try:
            # Try to open the file - if it exists, skip download
            with open(filename) as f:
                pass # File exists, do nothing
            print(f'SKIPPING {exch} AS DOWNLOADED')
            return # Exit the function for this exchange
        except Exception:
            # File doesn't exist or other error, proceed with download
            print(f'DOWNLOADING {exch} TO TEMP')

        # Initialize the ccxt API for the exchange
        api = get_hot_ccxt_api(exch)
        # Dictionary to store OHLCV data for the exchange's markets
        exchange_data = {}
        
        # Check if markets is None before iterating
        if api.markets is None:
            print(f"Error: Could not retrieve markets for {exch}. Skipping.")
            # Save empty data to avoid repeated failure attempts
            with open(filename, 'w') as f:
                json.dump({}, f)
            return
        
        # Add exchange-specific rate limiting
        rate_limit_delay = 1  # Default 1 second between requests
        if exch.lower() == 'bybit':
            rate_limit_delay = 2  # Bybit needs more conservative rate limiting
            
        # Iterate through all markets available on the exchange
        for market in api.markets:
            try:
                # --- Market Filtering Logic ---
                # If downloading spot data, skip markets with ':' (usually futures/swaps)
                if spot and ':' in market:
                    continue
                # If downloading futures data, only keep perpetual swaps marked with ':USD' or ':USDT'
                if not spot and ':USD' not in market and ':USDT' not in market : # Adjusted to include USDT perps
                    continue
                 # Ensure the quote currency is USD-based (common standard)
                if '/USD' not in market and '/USDT' not in market and '/USDC' not in market: # Include common USD stablecoins
                     continue
                # Skip markets with '-' (often expiring futures, not perpetuals)
                if '-' in market:
                    continue
                # --- Fetch OHLCV Data ---
                print(f"Fetching {exch} {market}") # Log progress
                
                # Add retry mechanism for potentially transient errors
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        # Fetch daily ('1d') OHLCV data for the market
                        # Returns a list of lists: [timestamp, open, high, low, close, volume]
                        ohlcv_data = api.fetch_ohlcv(market, '1d')
                        # Store the fetched data if successful
                        if ohlcv_data:
                            exchange_data[market] = ohlcv_data
                        break  # Exit retry loop on success
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"Error fetching {exch} {market} after {max_retries} attempts: {e}")
                        else:
                            print(f"Retry {retry_count}/{max_retries} for {exch} {market}: {e}")
                            time.sleep(rate_limit_delay * 2)  # Longer delay on retries
                
                # Rate limiting to avoid API bans
                time.sleep(rate_limit_delay)
                
            except Exception as e:
                # Print error if fetching fails for a specific market and continue
                print(f"Error processing {exch} {market}: {e}")
                time.sleep(rate_limit_delay)

        # Save the fetched data for the exchange to a JSON file
        # Save even if partial data was collected to avoid repeated failures
        print(f"Saving data for {exch} with {len(exchange_data)} markets")
        with open(filename, 'w') as f:
            json.dump(exchange_data, f)

    # --- Main Download Loop ---
    # Download data for all specified spot exchanges
    for exch in REFERENCE_SPOT_EXCH:
        download_exch(exch, True) # True indicates spot market

    # Download data for all specified futures exchanges
    for exch in REFERENCE_FUT_EXCH:
        download_exch(exch, False) # False indicates futures market

    # --- Load Data from Files ---
    # Dictionary to hold the raw data loaded from JSON files
    raw_reference_exch_df = {}
    # Iterate through spot and futures exchange lists
    for spot, exchs in {True: REFERENCE_SPOT_EXCH, False: REFERENCE_FUT_EXCH}.items():
        for exch in exchs:
            # Construct filename
            filename = f'exch_candles_{exch}_{"fs"[spot]}.json' # Uses 's' for spot, 'f' for futures
            try:
                # Open and load the JSON data
                with open(filename) as f:
                    loaded_json = json.load(f)
                # If the file contained data, add it to the dictionary
                if loaded_json:
                    raw_reference_exch_df[(spot, exch)] = loaded_json
                else:
                    # If the JSON file is empty, raise an exception (or handle as needed)
                    raise Exception('Empty JSON file')
            except Exception as e:
                # Print error if loading fails for an exchange
                print(f"Failed to load {filename}: {e}")
                pass # Continue to the next exchange

    # Return the dictionary containing all loaded raw OHLCV data
    return raw_reference_exch_df

def geomean_three(series):
    """
    Calculates the geometric mean of the top 3 values in a pandas Series.
    Adds 1 before taking log, subtracts 1 after exponentiating to handle zeros.

    Args:
        series (pd.Series): Input series of numbers (e.g., daily volumes).

    Returns:
        float: The geometric mean of the top 3 values.
    """
    # Take log(x+1), sort, get top 3, sum logs, divide by 3, then exp(result)-1
    return np.exp(np.log(series + 1).sort_values()[-3:].sum() / 3) - 1

def process_reference_exch_data(raw_reference_exch_df):
    """
    Processes the raw OHLCV data downloaded from reference exchanges.
    Calculates average daily volume ($) for each coin across exchanges.
    Aggregates data per coin, calculating total volume and geometric mean volume.

    Args:
        raw_reference_exch_df (dict): Dictionary from dl_reference_exch_data.

    Returns:
        pd.DataFrame: DataFrame indexed by cleaned coin symbol, with columns for
                      Spot/Futures Volume ($m) and Geometric Mean Volume ($m).
    """
    # Dictionary to store calculated average daily volume for each (exchange, spot/fut, coin)
    all_candle_data = {}

    # Iterate through the raw data dictionary (key=(spot, exch), value=exch_data)
    for (spot, exch), exch_data in raw_reference_exch_df.items():
        print(f'PROCESSING {exch} {"spot" if spot else "futures"}')
        # Initialize ccxt API for the exchange (needed for market info like contract size)
        api = get_hot_ccxt_api(exch)
        # Iterate through symbols (markets) and their OHLCV data within the exchange data
        for symbol, market_ohlcv in exch_data.items():
            # Clean the symbol to get the base coin name
            coin = clean_symbol(symbol, exch)
            # Skip if no OHLCV data was fetched for this market
            if not len(market_ohlcv):
                continue
            # Convert OHLCV list of lists into a pandas DataFrame
            market_df = (pd.DataFrame(market_ohlcv, columns=[*'tohlcv']) # timestamp, open, high, low, close, volume
                           .set_index('t') # Set timestamp as index
                           .sort_index() # Ensure data is sorted by time
                           # Filter data to keep only timestamps within the desired range
                           .loc[earliest_ts_to_keep * 1000:] # Convert seconds to milliseconds for index
                           # Keep only the last DAYS_TO_CONSIDER days (iloc[-N-1:-1] includes N days)
                           .iloc[-DAYS_TO_CONSIDER-1:-1])
            # Skip if no data remains after filtering
            if not len(market_df):
                continue

            # Get contract size from market info (defaults to 1 if not found or None)
            # Use min(..., 1) to handle potential non-perpetual inverse contracts (though filtering should handle most)
            contractsize = min(api.markets.get(symbol, {}).get('contractSize', None) or 1, 1)

            # Calculate average daily volume in USD
            # Use minimum of low price and last close price as a conservative price estimate for the day's volume
            # Multiply price by volume ('v') and contract size
            # Take the mean over the considered days
            # Note: Using market_df.c.iloc[-1] (last close) assumes it's a decent proxy for recent price.
            # Could also use (open+high+low+close)/4 or similar.
            daily_usd_volume = (np.minimum(market_df.l, market_df.c.iloc[-1]) # Conservative price for the day
                                * market_df.v # Volume in base currency or contracts
                                * contractsize # Adjustment for contract value (often 1 for linear contracts)
                               ).mean() # Average daily USD volume over the period

            # Store the highest calculated volume if multiple markets map to the same coin on the same exchange
            # (e.g., BTC/USDT, BTC/USDC) - keeps the most liquid one
            if daily_usd_volume >= all_candle_data.get((exch, spot, coin), 0):
                all_candle_data[exch, spot, coin] = daily_usd_volume

    # Convert the aggregated volume data into a pandas Series
    df_coins = pd.Series(all_candle_data).sort_values(ascending=False)
    # Name the index levels
    df_coins.index.names = ['exch', 'spot', 'coin']

    # Group by 'spot' (True/False) and 'coin', then aggregate:
    # - Calculate geometric mean of top 3 volumes per coin (using geomean_three)
    # - Calculate sum of volumes across all exchanges per coin
    # Convert volumes to millions ($m) by dividing by 1e6
    # Unstack the 'spot' level to create separate columns for spot and futures
    # Fill any resulting NaN values with 0
    output_df = (df_coins.fillna(0) / 1e6).groupby(['spot', 'coin']).agg(
        [geomean_three, 'sum'] # Apply both aggregation functions
    ).unstack(0).fillna(0) # Unstack 'spot' level, fill NaNs

    # Rename columns for clarity
    output_df.columns = [
        f"{'Spot' if is_spot else 'Fut'} " # Prefix with Spot or Fut
        f"{dict(geomean_three='Vol Geomean $m', sum='Volume $m')[agg_func_name]}" # Append metric name
        for agg_func_name, is_spot in output_df.columns # Iterate through multi-index columns
    ]

    return output_df


# %% --- Hyperliquid Specific Data ---

def dl_hl_data():
    """
    Downloads metadata and asset context information directly from the Hyperliquid API.

    Returns:
        list: A list containing two elements:
              [0] Universe data (general market info)
              [1] Asset contexts (specific asset details like leverage, OI)
              as returned by the HL API endpoint.
    """
    # Define the Hyperliquid API endpoint URL
    HL_API_URL = "https://api.hyperliquid.xyz/info"
    # Define the request payload to get metadata and asset contexts
    payload = {"type": "metaAndAssetCtxs"}
    # Define headers for the request
    headers = {"Content-Type": "application/json"}

    # Make a POST request to the Hyperliquid API
    response = requests.post(HL_API_URL, headers=headers, json=payload)
    # Raise an exception if the request was not successful (e.g., status code 4xx or 5xx)
    response.raise_for_status()
    # Parse the JSON response and return it
    return response.json()


def process_hl_data(raw_hl_data):
    """
    Processes the raw data fetched from the Hyperliquid API.
    Extracts relevant fields like max leverage and filters out delisted assets.

    Args:
        raw_hl_data (list): The raw list returned by dl_hl_data.

    Returns:
        pd.DataFrame: DataFrame indexed by asset name (cleaned), containing
                      'Max Lev. on HL' and other metadata from the API.
    """
    # Extract the universe data and asset context data from the response list
    universe, asset_ctxs = raw_hl_data[0]['universe'], raw_hl_data[1]
    # Merge the corresponding elements from universe and asset_ctxs lists
    # Assumes they are ordered correctly and correspond element-wise
    merged_data = [u | a for u, a in zip(universe, asset_ctxs)] # Python 3.9+ dictionary merge
    # Create a pandas DataFrame from the merged data
    output_df = pd.DataFrame(merged_data)
    # Filter out assets that are marked as delisted
    output_df = output_df[output_df.isDelisted != True]
    # Set the DataFrame index to the asset name
    # Cleans names starting with 'k' (common in some internal representations?)
    output_df.index = [name[1:] if name.startswith('k') else name for name in output_df.name]
    # Extract the 'maxLeverage' field and rename the column for clarity
    output_df['Max Lev. on HL'] = output_df['maxLeverage']
    # Return the processed DataFrame (contains 'Max Lev. on HL' and other raw fields)
    return output_df

# %% --- Thunderhead Data (Hyperliquid Analytics) ---

def dl_thunderhead_data():
    """
    Downloads various analytics data points from the Thunderhead public API
    (assumed to be a Hyperliquid analytics data source).

    Returns:
        dict: A dictionary where keys are query names (e.g., 'daily_usd_volume_by_coin')
              and values are the JSON data (usually list of records) returned by the API for that query.
    """
    # Base URL for the Thunderhead API
    THUNDERHEAD_URL = "https://d2v1fiwobg9w6.cloudfront.net"
    # Standard headers for the request
    THUNDERHEAD_HEADERS = {"accept": "*/*"}
    # Set of specific data endpoints/queries to fetch from Thunderhead
    THUNDERHEAD_QUERIES = {'daily_usd_volume_by_coin',
                           'total_volume',
                           'asset_ctxs', # Note: Might overlap with dl_hl_data, check usage
                           'hlp_positions',
                           'liquidity_by_coin'}

    # Dictionary to store the raw data fetched for each query
    raw_thunder_data = {}
    # Loop through each query name
    for query in THUNDERHEAD_QUERIES:
        # Construct the full URL for the query
        url = f"{THUNDERHEAD_URL}/{query}"
        print(f"Fetching Thunderhead data: {query}") # Log progress
        # Make a GET request to the Thunderhead endpoint
        response = requests.get(url, headers=THUNDERHEAD_HEADERS, allow_redirects=True)
        # Check if the request was successful
        response.raise_for_status()
        # Parse the JSON response and store the 'chart_data' part (or empty list if key not found)
        # Assumes the relevant data is nested under 'chart_data' key for most endpoints
        raw_thunder_data[query] = response.json().get('chart_data', [])
    # Return the dictionary of fetched data
    return raw_thunder_data

def process_thunderhead_data(raw_thunder_data):
    """
    Processes the raw data fetched from the Thunderhead API.
    Combines data from different queries, calculates relevant metrics like
    HLP volume/OI share, average OI, average volume, and slippage.

    Args:
        raw_thunder_data (dict): Dictionary returned by dl_thunderhead_data.

    Returns:
        pd.DataFrame: DataFrame indexed by cleaned coin symbol, containing calculated
                      metrics like 'HLP Vol Share %', 'OI on HL $m', 'HL Slip. $3k', etc.
    """
    # List to hold DataFrames created from each query's data
    dfs = []

    # Process each query's data
    for key, records in raw_thunder_data.items():
        if not records: # Skip if no data was returned for this query
            print(f"Warning: No data found for Thunderhead query '{key}'")
            continue
        # Special handling for 'liquidity_by_coin' which has a different structure (dict of lists)
        if key == 'liquidity_by_coin':
            # Reshape the data: create a multi-index (time, coin) DataFrame
            restructured_data = {}
            for coin, entries in records.items():
                for entry in entries:
                    # Use (time, coin) as the key for the outer dictionary
                    # Store the entry data, potentially setting 'time' field to 0 or removing it if redundant
                    restructured_data[(entry['time'], coin)] = {**entry, 'time': 0} # Keep other fields
            # Create DataFrame from the restructured dictionary, transpose to get time/coin as index
            dfs.append(pd.DataFrame(restructured_data).T)
        else:
            # For other queries, assume they are lists of records with 'time' and 'coin' fields
            # Create DataFrame and set multi-index ('time', 'coin')
            dfs.append(pd.DataFrame(records).set_index(['time', 'coin']))

    # Concatenate all individual DataFrames along columns, aligning by the multi-index (time, coin)
    # Unstack the 'time' level to make time periods into columns
    coin_time_df = pd.concat(dfs, axis=1).unstack(0)

    # --- Separate Spot and Futures Data ---
    # Create a mapping from Hyperliquid spot market names (e.g., 'BTC') to the base symbol ('BTC')
    # Uses the hl_markets data fetched earlier
    spot_mapping = {d['info']['name']: symbol.split('/')[0]
                    for symbol, d in hl_markets.items() if ':' not in symbol} # Filter for spot markets

    # Select rows corresponding to spot markets using the mapping keys
    # Rename the index using the mapping to get standard coin symbols
    # Unstack/reshape data (might need adjustment depending on exact structure)
    spot_data_df = (coin_time_df.loc[coin_time_df.index.isin(spot_mapping)]
                    .rename(spot_mapping).unstack().unstack(0))

    # Select rows that are *not* in the spot mapping (assumed to be futures)
    fut_data_df = (coin_time_df.loc[~coin_time_df.index.isin(spot_mapping)]
                   .unstack().unstack(0))

    # --- Calculate Futures Metrics ---
    # Calculate average notional Open Interest (OI) = avg price * avg OI quantity
    fut_data_df['avg_notional_oi'] = (fut_data_df['avg_oracle_px'] *
                                      fut_data_df['avg_open_interest'])

    # --- Aggregate Data Over Time ---
    # Calculate the mean of the last 30 days for each metric for futures
    # Unstack level 1 (original time columns), sort by time, take last 30 rows, calculate mean
    # Unstack level 0 (metric names) to get coins as index and metrics as columns
    fut_s_df = fut_data_df.unstack(1).sort_index().iloc[-DAYS_TO_CONSIDER:].mean().unstack(0)
    # Do the same for spot data (though fewer metrics might be relevant)
    spot_s_df = spot_data_df.unstack(1).sort_index().iloc[-DAYS_TO_CONSIDER:].mean().unstack(0)

    # Clean the index (coin symbols) for both DataFrames
    spot_s_df.index = [clean_symbol(sym) for sym in spot_s_df.index]
    fut_s_df.index = [clean_symbol(sym) for sym in fut_s_df.index]

    # --- Combine and Calculate Final Metrics ---
    # Focus on futures data for the final output metrics
    output_df = fut_s_df

    # Calculate HLP (Hyperliquid Pool) Volume Share percentage
    # Assumes 'total_volume' is taker volume and 'daily_usd_volume' needs halving? Check logic.
    # Logic: (Total Volume - (Taker Volume / 2)) / Total Volume * 100 ? -> This seems complex, verify calculation intent
    # Simpler interpretation: (HLP Volume) / Total Volume. Need HLP Volume source.
    # Assuming 'daily_usd_volume' relates to taker/maker flows in a way that this calc works for HLP share. Needs verification.
    output_df['HLP Vol Share %'] = ((output_df['total_volume']
                                     - output_df['daily_usd_volume']/2) # Verify this calculation
                                    / output_df['total_volume'] * 100).fillna(0) # Fill NaN with 0

    # Calculate HLP Open Interest (OI) Share percentage
    # Assumes 'daily_ntl_abs' is related to HLP notional position size
    output_df['HLP OI Share %'] = (output_df['daily_ntl_abs'] # HLP absolute notional value
                                   / output_df['avg_notional_oi'] * 100).fillna(0) # Fill NaN with 0

    # Calculate average notional OI in millions of dollars
    output_df['OI on HL $m'] = output_df['avg_notional_oi'] / 1e6

    # Calculate average total volume in millions of dollars
    output_df['Volume on HL $m'] = output_df['total_volume'] / 1e6

    # Calculate Slippage metrics
    # Assumes slippage values from Thunderhead are basis points (bps), convert to percentage * 100 ?
    # Multiplies by 100_00 -> This converts a raw fraction (e.g., 0.001) into basis points * 100 (e.g., 10) ? Or % points? Verify units.
    # Assuming the input is a decimal slippage (e.g., 0.0005 for 0.05%), multiplying by 10000 gives 5.
    output_df['HL Slip. $3k'] = output_df['median_slippage_3000'] * 100_00
    output_df['HL Slip. $30k'] = output_df['median_slippage_30000'] * 100_00

    # Return the DataFrame with calculated Hyperliquid-specific metrics
    return output_df


# %% --- CoinMarketCap Data ---

def dl_cmc_data():
    """
    Downloads market capitalization data from the CoinMarketCap (CMC) API.
    Requires an API key stored using the 'keyring' library.

    Returns:
        list: A list of dictionaries, where each dictionary represents a cryptocurrency
              and contains data fetched from the CMC API (including name, symbol, quote data).
    """
    # Import keyring only when needed to avoid dependency if not used elsewhere
    import keyring # Used to securely retrieve the API key

    # CMC API endpoint for latest listings
    CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    # Retrieve the CMC API key stored under service='cmc', username='cmc'
    CMC_API_KEY = keyring.get_password('cmc', 'cmc')
    # Check if API key was retrieved successfully
    if not CMC_API_KEY:
        raise ValueError("CoinMarketCap API key not found in keyring. Please store it using keyring.set_password('cmc', 'cmc', 'YOUR_API_KEY').")

    # Define overrides for CMC symbols/names that don't match the desired standard symbol
    CMC_SYMBOL_OVERRIDES = {
        'Neiro Ethereum': 'NEIROETH', # Maps CMC name to standard symbol
        'HarryPotterObamaSonic10Inu (ERC-20)': 'HPOS' # Maps CMC name to standard symbol
    }

    # Make the GET request to the CMC API
    try:
        response = requests.get(
            f"{CMC_API_URL}?CMC_PRO_API_KEY={CMC_API_KEY}&limit=5000", # Request up to 5000 coins
            timeout=10 # Set a timeout for the request
        )
        # Check for HTTP errors
        response.raise_for_status()
        # Parse the JSON response
        data = response.json().get('data', []) # Extract the list of coins under the 'data' key

        # Apply symbol overrides based on the 'name' field from CMC
        for item in data:
            # If the CMC name is in the overrides map, update the 'symbol' field in the item
            item['symbol'] = CMC_SYMBOL_OVERRIDES.get(item['name'], item['symbol'])

        # Return the list of coin data
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CoinMarketCap API: {e}")
        # Depending on requirements, could return empty list or re-raise
        return [] # Return empty list on error to allow partial processing downstream


def process_cmc_data(cmc_data):
    """
    Processes the raw data fetched from the CoinMarketCap API.
    Extracts market cap (MC) and fully diluted market cap (FD MC).
    Calculates MC in millions ($m) and combines with estimated MC from Hyperliquid prices.

    Args:
        cmc_data (list): List of coin data dictionaries from dl_cmc_data.

    Returns:
        pd.DataFrame: DataFrame indexed by symbol, containing 'MC $m'.
    """
    # Create a DataFrame with relevant fields from the CMC data
    output_df = pd.DataFrame([{
        'symbol': a['symbol'], # Use the (potentially overridden) symbol
        'mc': a['quote']['USD']['market_cap'], # Market Cap
        'fd_mc': a['quote']['USD']['fully_diluted_market_cap'], # Fully Diluted Market Cap
    } for a in cmc_data if 'quote' in a and 'USD' in a['quote'] ] # Ensure quote data exists
    ).groupby('symbol')[['mc', 'fd_mc']].max() # Group by symbol and take max (handles potential duplicates)

    # Use fully diluted MC if regular MC is zero or missing
    output_df.loc[output_df['mc'] == 0, 'mc'] = output_df['fd_mc']

    # --- Estimate Market Cap using Hyperliquid Data as Fallback/Reference ---
    # Calculate estimated MC based on Hyperliquid circulating supply and mark price
    # This provides a fallback if CMC data is missing or unreliable for certain assets
    hl_market_caps = {}
    for symbol, data in hl_markets.items():
        # Only consider spot markets (no ':')
        if ':' not in symbol:
            base_symbol = symbol.split('/')[0] # Get base symbol (e.g., 'BTC' from 'BTC/USD')
            try:
                # Get circulating supply and mark price from HL market info
                supply = float(data['info'].get('circulatingSupply', 0))
                price = float(data['info'].get('markPx', 0))
                # Calculate estimated market cap
                if supply > 0 and price > 0:
                    hl_market_caps[base_symbol] = supply * price
            except (ValueError, TypeError, KeyError) as e:
                # Handle cases where data might be missing or not convertible to float
                # print(f"Warning: Could not calculate HL market cap for {base_symbol}: {e}") # Optional warning
                pass # Ignore if calculation fails

    # Combine CMC MC with the estimated HL MC
    output_df = pd.concat([
        output_df,
        pd.Series(hl_market_caps, name='hl_mc') # Add HL estimated MC as a new column
    ], axis=1) # Concatenate along columns, aligning by index (symbol)

    # Calculate the final 'MC $m' (Market Cap in millions)
    # Use the maximum of the CMC-derived MC and the Hyperliquid-estimated MC
    # Fill NaN values with 0 before taking max to avoid issues
    output_df['MC $m'] = output_df[['mc', 'hl_mc']].fillna(0).max(axis=1) / 1e6

    # Return the DataFrame with the calculated 'MC $m' column
    return output_df[['MC $m']] # Return only the final column


# %% --- Scoring Logic ---

def build_scores(df):
    """
    Calculates various scores for each coin based on the metrics in the input DataFrame
    using the predefined SCORE_CUTOFFS. Also applies boosts and adjustments.

    Args:
        df (pd.DataFrame): Combined DataFrame containing all metrics (MC, Volume, HL data, etc.).

    Returns:
        pd.DataFrame: DataFrame containing the calculated partial and final scores,
                      and the 'Strict' flag.
    """
    # Dictionary to store the calculated score components
    output_scores = {}
    # Iterate through each main score category defined in SCORE_CUTOFFS
    for score_category, category_details in SCORE_CUTOFFS.items():
        # Initialize the main category score to zero for all coins
        output_scores[score_category] = pd.Series(0.0, index=df.index) # Use float for scores
        # Iterate through each variable contributing to this score category
        for score_var, thresholds in category_details.items():
            # Check if the required variable exists in the input DataFrame
            if score_var not in df.columns:
                print(f"Warning: Scoring variable '{score_var}' not found in DataFrame. Skipping for category '{score_category}'.")
                continue

            # --- Generate Threshold Points ---
            # These points map metric values to score points (0 to steps)
            point_thresholds = {}
            steps = thresholds['steps']
            start = thresholds['start']
            end = thresholds['end']

            if thresholds['kind'] == 'exp':
                # Calculate thresholds using exponential spacing
                # Handle start=0 case or log(0) potential
                if start <= 0: # Need to handle non-positive start for log scale
                   print(f"Warning: Exponential scale requires start > 0 for '{score_var}'. Using linear instead or adjust config.")
                   # Fallback or specific handling needed here - using linear as placeholder
                   for k in range(steps + 1):
                       point_thresholds[start + (end - start) * (k / steps)] = k
                else:
                   ratio = end / start
                   for k in range(steps + 1):
                       point_thresholds[start * (ratio ** (k / steps))] = k

            elif thresholds['kind'] == 'linear':
                # Calculate thresholds using linear spacing
                for k in range(steps + 1):
                    point_thresholds[start + (end - start) * (k / steps)] = k
            else:
                # Raise error for unknown threshold kind
                raise ValueError(f"Unknown threshold kind: {thresholds['kind']} for {score_var}")

            # --- Calculate Partial Score ---
            # Create a temporary Series to store the score for this specific variable
            score_name = 'Partial_Score_' + score_var
            output_scores[score_name] = pd.Series(0.0, index=df.index) # Initialize partial score

            # Apply thresholds to assign score points based on the variable's value
            # Sort thresholds by value (metric value) to apply correctly
            # For variables where higher is better (exp scale, or linear start < end)
            if thresholds['kind'] == 'exp' or (thresholds['kind'] == 'linear' and start <= end):
                for threshold_val, points in sorted(point_thresholds.items()):
                    # If the coin's metric value is >= threshold, assign the corresponding points
                    # This overwrites lower scores, effectively finding the highest threshold met
                    output_scores[score_name].loc[df[score_var].fillna(-np.inf) >= threshold_val] = points
            # For variables where lower is better (linear start > end, e.g., slippage, HLP share)
            else: # Linear decreasing score (start > end)
                # Sort by score points (k) to ensure correct application
                # We assign score k if value <= threshold_for_k
                # Easier to iterate thresholds from highest value (lowest score) to lowest value (highest score)
                 for threshold_val, points in sorted(point_thresholds.items(), reverse=True): # Iterate high metric value (low score) first
                    # If the coin's metric value is <= threshold, assign the points
                    # This finds the lowest threshold the metric is *below*, assigning that score.
                    output_scores[score_name].loc[df[score_var].fillna(np.inf) <= threshold_val] = points


            # Add the partial score (for this variable) to the main category score
            output_scores[score_category] += output_scores[score_name]

    # Convert the dictionary of score Series into a DataFrame
    output_df = pd.concat(output_scores, axis=1)

    # --- Adjustments and Final Score ---
    # Zero out Hyperliquid-specific scores for coins not listed on HL (Max Lev < 1)
    # Identify columns related to HL scores
    hl_score_cols = [c for c in output_df.columns if 'HL ' in c or c.startswith('Partial_Score_HL')] # Include partials too? Check category names
    # Set these scores to 0 where Max Lev on HL is 0 (or less than 1, treating 0.x as not listed effectively)
    output_df.loc[df['Max Lev. on HL'].fillna(0) < 1, hl_score_cols] = 0

    # Calculate a boost for assets not listed on HL based on their non-HL scores
    # This helps identify potentially listable assets
    non_hl_categories = ['Market Cap Score', 'Spot Vol Score', 'Futures Vol Score']
    output_df['NON_HL_SCORE_BOOST'] = (
        0.5 # Scaling factor for the boost
        * (df['Max Lev. on HL'].fillna(0) < 1) # Condition: Only apply if not listed on HL (Max Lev < 1)
        * output_df[non_hl_categories].sum(axis=1) # Sum of non-HL score categories
    ).astype(float) # Keep as float

    # Add a flag indicating if the coin is in the 'strict' list
    output_df['Strict'] = output_df.index.isin(HL_STRICT)

    # Calculate the final total score
    # Sum of all main score categories + the non-HL boost + the strict boost
    score_components = [*SCORE_CUTOFFS.keys(), 'NON_HL_SCORE_BOOST'] # Get main category names dynamically
    output_df['Score'] = (
        output_df[score_components].sum(axis=1) # Sum main categories and boost
        + output_df['Strict'] * HL_STRICT_BOOST # Add strict boost
    )

    return output_df

# %% --- Recommendation Logic ---

def generate_recommendation(row):
    """
    Generates a recommendation ('List', 'Delist', 'Inc. Lev.', 'Dec. Lev.', '')
    based on the coin's final score and its current maximum leverage on Hyperliquid.
    Uses the predefined SCORE_LB and SCORE_UB boundaries.

    Args:
        row (pd.Series): A row from the combined DataFrame containing 'Score' and 'Max Lev. on HL'.

    Returns:
        str: The recommendation string.
    """
    # Get the current max leverage, default to 0 if NaN
    current_leverage = int(0 if pd.isna(row['Max Lev. on HL']) else row['Max Lev. on HL'])
    score = row['Score']

    # Determine the relevant score boundaries for the *current* leverage level
    # Use min(max(SCORE_LB), current_leverage) to handle leverage levels not explicitly in the dict keys (e.g., leverage 20 uses level 10 bounds)
    # Ensure current_leverage is used as key after clamping it to the available keys in SCORE_LB/SCORE_UB
    lower_bound_key = min(max(SCORE_LB.keys()), current_leverage)
    upper_bound_key = min(max(SCORE_UB.keys()), current_leverage)

    # Check if score is below the lower bound for the *current* leverage tier
    is_below_lower_bound = score < SCORE_LB[lower_bound_key]
    # Check if score is above or equal to the upper bound for the *current* leverage tier
    is_above_upper_bound = score >= SCORE_UB[upper_bound_key]

    # --- Recommendation Rules ---
    # 1. High Leverage & Low Score -> Decrease Leverage
    if current_leverage > 3 and is_below_lower_bound:
        return 'Dec. Lev.'
    # 2. Low Leverage (3x) & Low Score -> Delist
    if current_leverage == 3 and is_below_lower_bound:
        return 'Delist'
    # 3. Not Listed (Lev=0) & High Score -> List (at lowest tier, e.g., 3x)
    if current_leverage == 0 and is_above_upper_bound:
         # Need to check UB for level 0 specifically. If score > UB[0], recommend list.
         if score >= SCORE_UB[0]: # Check against level 0 upper bound explicitly
            return 'List'
    # 4. Listed & High Score -> Increase Leverage
    if current_leverage > 0 and is_above_upper_bound:
         # Check if current leverage is already max possible defined in bounds
         if current_leverage < max(SCORE_UB.keys()):
             return 'Inc. Lev.'
         else:
             return '' # Already at max leverage defined in bounds
    # 5. Otherwise, no change recommended
    return ''

# %% --- Main Execution ---

# Download and process data from all sources
print("Starting data download and processing...")
processed_data = [
    process_cmc_data(dl_cmc_data()), # Fetch and process CMC market cap
    process_reference_exch_data(dl_reference_exch_data()), # Fetch/load and process reference exchange volumes
    process_hl_data(dl_hl_data()), # Fetch and process Hyperliquid metadata
    process_thunderhead_data(dl_thunderhead_data()) # Fetch and process Hyperliquid analytics
]
print("Data processing complete.")

# %% Combine all processed data into a single DataFrame
# Concatenate along columns, aligning by index (coin symbol)
print("Combining data...")
df = pd.concat(processed_data, axis=1)

# Filter out stablecoins
df = df.loc[~df.index.isin(STABLE_COINS)]

# Add 'Symbol' column from the index for output convenience
df['Symbol'] = df.index
# Fill NaN values in 'Max Lev. on HL' with 0 (assuming NaN means not listed)
df['Max Lev. on HL'] = df['Max Lev. on HL'].fillna(0)
print("Data combined.")

# Build scores based on the combined data
print("Building scores...")
df_scores = build_scores(df)
df = pd.concat([df, df_scores], axis=1) # Add scores to the main DataFrame
print("Scores built.")

# Generate recommendations for each coin
print("Generating recommendations...")
df['Recommendation'] = df.apply(generate_recommendation, axis=1)
print("Recommendations generated.")

# --- Prepare and Save Output ---

# Create DataFrame for the main output JSON
# Select and order columns as defined in OUTPUT_COLS
# Sort by final 'Score' in descending order
df_for_main_data = df[OUTPUT_COLS].sort_values('Score', ascending=False).copy()

# Format numerical columns to significant figures for cleaner output
print("Formatting output data...")
for c in df_for_main_data.columns:
    # Check if the column data type is numeric (integer or float)
    if pd.api.types.is_numeric_dtype(df_for_main_data[c]):
        # Apply the sig_figs function to each element in the column
        # Use .map() for applying element-wise function on Series
        df_for_main_data[c] = df_for_main_data[c].map(lambda x: sig_figs(x, 3)) # Apply sig figs function
print("Formatting complete.")

# Save the main data to the specified JSON file
# orient='records' creates a list of dictionaries, suitable for web/JS consumption
print(f"Saving main data to {OUTPUT_RAW_DATA_JSON}...")
with open(OUTPUT_RAW_DATA_JSON, 'w') as f:
    # Convert DataFrame to list of dictionaries and dump as JSON
    json.dump(df_for_main_data.to_dict(orient='records'), f, indent=4) # Added indent for readability
print("Main data saved.")


# Create and save the correlation matrix
print("Calculating and saving correlation matrix...")
# Select columns relevant for correlation (Leverage + Score components)
corr_cols = ['Max Lev. on HL'] + [c for c in df.columns if "Score" in c or c == 'Strict'] # Include Strict?
df_for_corr = df[corr_cols].copy()

# Set HL-specific scores to NaN for coins not listed on HL (Max Lev < 1) before calculating correlation
hl_cols_for_corr = [c for c in df_for_corr.columns if 'HL ' in c or c.startswith('Partial_Score_HL')]
df_for_corr.loc[df_for_corr['Max Lev. on HL'] < 1, hl_cols_for_corr] = np.nan # Set to NaN so they don't affect corr for unlisted coins

# Rename columns slightly for better readability in the matrix (e.g., 'Partial_Score_MC $m' -> '..MC $m')
df_for_corr = df_for_corr.rename(
    {c: c.replace('Partial_Score_', '..') for c in df_for_corr.columns}, axis=1)

# Calculate the correlation matrix
# Multiply by 100 and convert to integer for percentage representation
corr_mat = (df_for_corr.corr() * 100).round().astype(int) # Round before converting to int

# Reset index to make 'Symbol' (original index/metric name) a column
corr_mat = corr_mat.reset_index().rename({'index': 'Symbol'}, axis=1)

# Save the correlation matrix to the specified JSON file
print(f"Saving correlation matrix to {OUTPUT_CORR_MAT_JSON}...")
with open(OUTPUT_CORR_MAT_JSON, 'w') as f:
    json.dump(corr_mat.to_dict(orient='records'), f, indent=4) # Added indent for readability
print("Correlation matrix saved.")
print("Script finished successfully.")