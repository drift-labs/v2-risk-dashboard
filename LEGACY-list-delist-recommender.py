# %% Import necessary libraries
import json  # For handling JSON data (reading/writing files)
import math  # For mathematical operations (like log10, needed for sig_figs)
import requests  # For making HTTP requests (downloading data from APIs)
import time  # For time-related functions (like getting current time, pausing execution)

import ccxt  # A library to connect and trade with cryptocurrency exchanges
import numpy as np  # For numerical operations, especially with arrays and matrices
import pandas as pd  # For data manipulation and analysis, primarily using DataFrames

# %% Configuration and Constants

# Define a set of stablecoin symbols to exclude from analysis
STABLE_COINS = {
            "USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE',
            'TUSD', 'USR', 'USDY', 'PYUSD', 'USDe', 'USDS'
                }

# Define the number of past days of data to consider for calculations
DAYS_TO_CONSIDER = 30

# Define the columns to include in the final output JSON file, and their order
OUTPUT_COLS = [
    'Symbol', 'Max Lev. on HL', 'Strict','Recommendation',
    'Market Cap Score', 'Spot Vol Score', 'Futures Vol Score',
    'HL Activity Score', 'HL Liquidity Score', 'Score',
    'MC $m', 'Spot Volume $m', 'Spot Vol Geomean $m',
    'Fut Volume $m', 'Fut Vol Geomean $m', 'OI on HL $m',
    'Volume on HL $m', 'HLP Vol Share %', 'HLP OI Share %',
    'HL Slip. $3k', 'HL Slip. $30k'
]
# Define filenames for the output JSON files
OUTPUT_CORR_MAT_JSON = 'hl_screen_corr.json' # File for correlation matrix output
OUTPUT_RAW_DATA_JSON = 'hl_screen_main.json' # File for main screening data output

# Define score boundaries (upper and lower bounds) based on maximum leverage levels on Hyperliquid (HL)
# Used in the recommendation logic to determine if leverage should be increased/decreased or asset listed/delisted
SCORE_UB = {0: 62, 3: 75, 5: 85, 10: 101} # Upper bound score thresholds
SCORE_LB = {0: 0, 3: 37, 5: 48, 10: 60}   # Lower bound score thresholds

# Define sets of reference exchanges for fetching spot and futures market data
REFERENCE_SPOT_EXCH = {
    'binance', 'bybit', 'okx', 'gate', 'kucoin', 'mexc',
    'cryptocom', 'coinbase', 'kraken', 'hyperliquid'
}
REFERENCE_FUT_EXCH = {
    'bybit', 'binance', 'gate', 'mexc', 'okx',
    'htx', 'krakenfutures', 'cryptocom', 'bitmex',
    'hyperliquid'
}
# Define a set of specific Hyperliquid coins marked as 'Strict', potentially receiving a score boost
HL_STRICT = {'PURR','CATBAL','HFUN','PIP','JEFF','VAPOR','SOLV',
             'FARM','ATEHUN','SCHIZO','OMNIX','POINTS','RAGE'}

# Define the parameters for calculating scores based on different metrics
# Each score category maps to metrics and specifies how thresholds are generated ('exp' or 'linear')
SCORE_CUTOFFS = {
    'Market Cap Score': {
        'MC $m': {'kind': 'exp', 'start': 1, 'end': 5000, 'steps': 20}, # Market Cap scoring (exponential thresholds)
    },
    'Spot Vol Score': {
        'Spot Volume $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10}, # Spot Volume scoring (exponential)
        'Spot Vol Geomean $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10}, # Spot GeoMean Volume scoring (exponential)
    },
    'Futures Vol Score': {
        'Fut Volume $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10}, # Futures Volume scoring (exponential)
        'Fut Vol Geomean $m': {'kind': 'exp', 'start': 0.01, 'end': 1000, 'steps': 10}, # Futures GeoMean Volume scoring (exponential)
    },
    'HL Activity Score': {
        'Volume on HL $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10}, # Hyperliquid Volume scoring (exponential)
        'OI on HL $m': {'kind': 'exp', 'start': 0.001, 'end': 1000, 'steps': 10}, # Hyperliquid Open Interest scoring (exponential)
    },
    'HL Liquidity Score': {
        'HLP Vol Share %': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5}, # HLP Volume Share scoring (linear, lower is better)
        'HLP OI Share %': {'kind': 'linear', 'start': 10, 'end': 0, 'steps': 5}, # HLP Open Interest Share scoring (linear, lower is better)
        'HL Slip. $3k': {'kind': 'linear', 'start': 5, 'end': 0, 'steps': 5}, # Hyperliquid Slippage ($3k) scoring (linear, lower is better)
        'HL Slip. $30k': {'kind': 'linear', 'start': 50, 'end': 0, 'steps': 5}, # Hyperliquid Slippage ($30k) scoring (linear, lower is better)
    }
}
# Define the score boost applied to coins in the HL_STRICT set
HL_STRICT_BOOST = 5

# Calculate the earliest timestamp (in seconds since epoch) to keep data based on DAYS_TO_CONSIDER
# Adds 5 extra days buffer when fetching data before filtering
earliest_ts_to_keep = time.time()-(DAYS_TO_CONSIDER+5)*24*60*60

# %% Helper Functions

def sig_figs(number, sig_figs=3):
    """
    Rounds a number to a specified number of significant figures.
    Handles NaN and non-positive numbers by returning 0.
    """
    if np.isnan(number) or number <= 0:
        return 0
    # Calculate the rounding decimal place based on the number's magnitude and desired sig figs
    return round(number, int(sig_figs - 1 - math.log10(number)))


def clean_symbol(symbol, exch=''):
    """
    Cleans and standardizes cryptocurrency symbols.
    Removes common suffixes (like '1000'), applies general aliases,
    and applies exchange-specific aliases.
    """
    # Define general token symbol aliases
    TOKEN_ALIASES = {
        'HPOS10I': 'BITCOIN', 'HPOS': 'HPOS', 'HPO': 'HPOS',
        'BITCOIN': 'HPOS', 'NEIROCTO': 'NEIRO', '1MCHEEMS': 'CHEEMS',
        '1MBABYDOGE': 'BABYDOGE', 'JELLYJELLY': 'JELLY'
    }
    # Define exchange-specific token symbol aliases
    EXCH_TOKEN_ALIASES = {
        ('NEIRO', 'bybit'): 'NEIROETH',
        ('NEIRO', 'gate'): 'NEIROETH',
        ('NEIRO', 'kucoin'): 'NEIROETH'
    }
    # Extract the base symbol (part before '/')
    redone = symbol.split('/')[0]
    # Remove common numerical/size suffixes
    for suffix in ['10000000', '1000000', '1000', 'k']:
        redone = redone.replace(suffix, '')
    # Apply exchange-specific alias if applicable
    redone = EXCH_TOKEN_ALIASES.get((redone, exch), redone)
    # Apply general alias if applicable
    return TOKEN_ALIASES.get(redone, redone)

def get_hot_ccxt_api(exch):
    """
    Initializes a ccxt exchange API object for the given exchange name.
    Performs a test fetch_ticker call to ensure the API is responsive ("hot").
    Includes basic error handling for the test call.
    """
    # Initialize the ccxt exchange object dynamically based on the exchange name string
    api = getattr(ccxt, exch)()
    try:
        # Attempt a test API call (fetching ticker for BTC/USDT perpetual swap)
        api.fetch_ticker('BTC/USDT:USDT')
    except Exception as e:
        # Ignore exceptions during the test call (the API might still work for other calls)
        pass
    return api

# Fetch market data from Hyperliquid once using the helper function
hl_markets = get_hot_ccxt_api('hyperliquid').markets

# %% Data Download Functions

def dl_reference_exch_data():
    """
    Downloads historical OHLCV data from reference spot and futures exchanges.
    Uses a nested function to handle individual exchange downloads and caching.
    Returns a dictionary containing the raw downloaded data.
    """
    def download_exch(exch, spot):
        """
        Downloads and saves 1-day OHLCV data for a single exchange.
        Skips download if a corresponding file already exists.
        Filters for relevant markets (spot or USD-margined futures).
        """
        try:
            # Check if data file already exists to potentially skip download
            with open(f'exch_candles_{exch}_{"s" if spot else "f"}.json') as f:
                pass # File exists
            print(f'SKIPPING {exch} AS DOWNLOADED')
            return # Skip download
        except Exception:
            # File doesn't exist or is inaccessible, proceed with download
            print(f'DOWNLOADING {exch} TO TEMP')

        # Initialize the CCXT API for the exchange
        api = get_hot_ccxt_api(exch)

        exchange_data = {}
        # Iterate through all markets available on the exchange
        for market in api.markets:
            try:
                # --- Market Filtering Logic ---
                # Skip swap markets if downloading spot data
                if spot and ':' in market:
                    continue
                # Skip non-USD margined futures if downloading futures data
                if not spot and ':USD' not in market:
                    continue
                # Ensure the market is a USD pair (spot or futures)
                if '/USD' not in market:
                    continue
                # Skip markets with hyphens (often indices or non-tradable)
                if '-' in market:
                    continue
                # --- End Filtering ---

                print(exch, market) # Log the market being fetched
                # Fetch 1-day OHLCV data for the market
                exchange_data[market] = api.fetch_ohlcv(market, '1d')
            except Exception as e:
                # Handle errors during fetching (e.g., rate limits, invalid market)
                print(e)
                time.sleep(1) # Pause briefly after an error

        # Save the fetched data to a JSON file
        with open(f'exch_candles_{exch}_{"fs"[spot]}.json', 'w') as f:
            json.dump(exchange_data, f)

    # --- Main Download Logic ---
    # Download data for all specified spot exchanges
    for exch in REFERENCE_SPOT_EXCH:
        download_exch(exch, True) # True indicates spot data

    # Download data for all specified futures exchanges
    for exch in REFERENCE_FUT_EXCH:
        download_exch(exch, False) # False indicates futures data

    # --- Load Downloaded Data ---
    raw_reference_exch_df = {}
    # Iterate through spot and futures exchange lists
    for spot, exchs in {True: REFERENCE_SPOT_EXCH,
                        False: REFERENCE_FUT_EXCH}.items():
        for exch in exchs:
            try:
                # Load data from the JSON file
                with open(f'exch_candles_{exch}_{"fs"[spot]}.json') as f:
                    loaded_json = json.load(f)
                    if loaded_json: # Check if the file contains data
                        raw_reference_exch_df[spot, exch] = loaded_json
                    else:
                        raise Exception('Missing file or empty file') # Raise error if file is empty
            except Exception as e:
                # Handle errors during file loading (e.g., file not found)
                print(exch, spot, e, 'fail')
                pass # Continue even if one file fails to load
    return raw_reference_exch_df


def dl_hl_data():
    """
    Downloads metadata and asset context information from the Hyperliquid API.
    Makes a POST request to the specified endpoint.
    """
    # Hyperliquid API endpoint for metadata and asset contexts
    url = "https://api.hyperliquid.xyz/info"
    headers = {"Content-Type": "application/json"}
    payload = {"type": "metaAndAssetCtxs"}

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload)
    # Raise an exception if the request was unsuccessful (e.g., 4xx or 5xx status code)
    response.raise_for_status()
    # Return the parsed JSON response
    return response.json()


def dl_thunderhead_data():
    """
    Downloads various analytical data points (volume, OI, HLP positions, liquidity)
    from the Thunderhead API (appears to be a Hyperliquid analytics provider).
    """
    # Base URL for the Thunderhead API
    THUNDERHEAD_URL = "https://d2v1fiwobg9w6.cloudfront.net"
    # Standard headers for the requests
    THUNDERHEAD_HEADERS = {"accept": "*/*", }
    # Specific endpoints/queries to fetch data from
    THUNDERHEAD_QUERIES = {'daily_usd_volume_by_coin',
                           'total_volume',
                           'asset_ctxs',
                           'hlp_positions',
                           'liquidity_by_coin'}

    raw_thunder_data = {}
    # Loop through each query endpoint
    for query in THUNDERHEAD_QUERIES:
        # Construct the full URL
        url = f"{THUNDERHEAD_URL}/{query}"
        # Make the GET request
        response = requests.get(url, headers=THUNDERHEAD_HEADERS, allow_redirects=True)
        # Check for request errors
        response.raise_for_status()
        # Store the 'chart_data' part of the JSON response
        # Uses .get() to avoid errors if 'chart_data' key is missing
        raw_thunder_data[query] = response.json().get('chart_data', [])
    return raw_thunder_data


def dl_cmc_data():
    """
    Downloads cryptocurrency listing data, including market cap,
    from the CoinMarketCap (CMC) Pro API.
    Requires a CMC API key stored securely (using the 'keyring' library).
    """
    # Import keyring library locally, as it's only needed here for API key access
    import keyring
    # CMC API endpoint for latest listings
    CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    # Retrieve the CMC API key securely using keyring
    # Assumes the key is stored under service='cmc', username='cmc'
    CMC_API_KEY = keyring.get_password('cmc', 'cmc')
    # Define overrides for CMC symbols that don't match standard usage elsewhere
    CMC_SYMBOL_OVERRIDES = {
        'Neiro Ethereum': 'NEIROETH',
        'HarryPotterObamaSonic10Inu (ERC-20)': 'HPOS'
    }

    # Construct the full API request URL with the API key and limit parameters
    url = f"{CMC_API_URL}?CMC_PRO_API_KEY={CMC_API_KEY}&limit=5000"
    # Make the GET request with a timeout
    response = requests.get(url, timeout=10)
    # Check for request errors
    response.raise_for_status()
    # Extract the 'data' list from the JSON response
    data = response.json().get('data', [])

    # Apply symbol overrides based on the 'name' field from CMC data
    for item in data:
        item['symbol'] = CMC_SYMBOL_OVERRIDES.get(item['name'], item['symbol'])

    # Return the raw list of cryptocurrency data from CMC
    return data


# %% Data Processing Functions

def geomean_three(series):
    """
    Calculates the geometric mean of the top 3 highest values in a pandas Series.
    Adds 1 before taking log and subtracts 1 after exponentiation to handle potential zeros.
    """
    # Sort the series, take the log of (value+1), sum the top 3, divide by 3, exponentiate, subtract 1
    return np.exp(np.log(series + 1).sort_values()[-3:].sum() / 3) - 1


def process_reference_exch_data(raw_reference_exch_df):
    """
    Processes the raw OHLCV data downloaded from reference exchanges.
    Calculates average daily volume in USD for each coin across exchanges.
    Aggregates data, calculating total volume and geomean of top 3 exchange volumes.
    Returns a DataFrame indexed by coin with aggregated spot and futures volumes.
    """
    all_candle_data = {} # Dictionary to store calculated daily volume per exchange/coin

    # Iterate through the raw data dictionary (key = (spot_boolean, exch_name))
    for (spot, exch), exch_data in raw_reference_exch_df.items():
        print('PROCESSING '+exch + ' ' + ('spot' if spot else 'futures'))
        # Get the CCXT API object for contract size information
        api = get_hot_ccxt_api(exch)
        # Iterate through symbols (markets) and their OHLCV data within an exchange
        for symbol, market_data in exch_data.items():
            # Clean the symbol using the helper function
            coin = clean_symbol(symbol, exch)
            # Skip if market data is empty
            if not len(market_data):
                continue

            # Convert list of lists to pandas DataFrame
            market_df = (pd.DataFrame(market_data, columns=[*'tohlcv']) # time, open, high, low, close, volume
                           .set_index('t') # Set timestamp as index
                           .sort_index() # Ensure data is sorted by time
                           # Filter data to keep only relevant time period (ms timestamp)
                           .loc[earliest_ts_to_keep*1000:]
                           # Keep only the last DAYS_TO_CONSIDER days of data (excluding the last, possibly incomplete, day)
                           .iloc[-DAYS_TO_CONSIDER-1:-1])

            # Skip if the filtered DataFrame is empty
            if not len(market_df):
                continue

            # Get contract size from CCXT market info (default to 1 if not found or None)
            # Use min(..., 1) to handle inverse contracts where size might be > 1 (treat as 1 for volume calcs)
            contractsize = min(api.markets.get(
                symbol, {}).get('contractSize', None) or 1, 1)

            # Calculate average daily USD volume
            # Use min(low_price, last_close_price) as a conservative price estimate for the day's volume
            # Multiply volume (in base currency) by price and contract size
            my_val = (np.minimum( market_df.l, market_df.c.iloc[-1])
                      * market_df.v).mean() * contractsize

            # Store the calculated volume, keeping the highest volume if a coin appears multiple times on an exchange (e.g., /USDT and /USDC)
            if my_val >= all_candle_data.get((exch, spot, coin), 0):
                all_candle_data[exch, spot, coin] = my_val

    # Convert the dictionary of volumes into a pandas Series
    df_coins = pd.Series(all_candle_data).sort_values(ascending=False)
    df_coins.index.names = ['exch', 'spot', 'coin'] # Name the multi-index levels

    # Aggregate volumes per coin across all exchanges
    output_df = (df_coins.fillna(0)/1e6 # Convert volume to millions USD, fill NaNs with 0
                 .groupby(['spot', 'coin']) # Group by spot/futures and coin
                 .agg([geomean_three, 'sum']) # Calculate geomean of top 3 and total sum
                 .unstack(0) # Pivot the 'spot' level to become columns (spot/futures)
                 .fillna(0)) # Fill any remaining NaNs with 0

    # Rename columns for clarity
    output_df.columns = [
        f"{'Spot' if b else 'Fut'} " # Prepend 'Spot' or 'Fut' based on the unstacked level
        f"{dict(geomean_three='Vol Geomean $m', sum='Volume $m')[a]}" # Append metric name
        for a, b in output_df.columns] # Iterate through the multi-level column index

    return output_df


def process_hl_data(raw_hl_data):
    """
    Processes the raw metadata and asset context downloaded from Hyperliquid.
    Merges the two data sources, filters out delisted assets,
    cleans index names (removes 'k' prefix), and extracts max leverage.
    Returns a DataFrame indexed by cleaned asset name.
    """
    # Extract universe data (list of dicts) and asset context data (list of dicts)
    universe, asset_ctxs = raw_hl_data[0]['universe'], raw_hl_data[1]
    # Merge corresponding dictionaries from the two lists
    merged_data = [u | a for u, a in zip(universe, asset_ctxs)]
    # Create a DataFrame from the merged data
    output_df = pd.DataFrame(merged_data)
    # Filter out assets marked as delisted
    output_df = output_df[output_df.isDelisted != True]
    # Set the index to the asset name, removing the leading 'k' if present (spot markets)
    output_df.index = [name[1:] if name.startswith('k') else name for name in output_df.name]
    # Extract the maximum leverage and rename the column
    output_df['Max Lev. on HL'] = output_df['maxLeverage']
    return output_df


def process_thunderhead_data(raw_thunder_data):
    """
    Processes the raw analytical data downloaded from Thunderhead.
    Combines data from different queries, separates spot and futures data,
    calculates 30-day averages, derives additional metrics like HLP shares and slippage.
    Returns a DataFrame indexed by cleaned futures coin symbol with calculated metrics.
    """
    dfs = [] # List to hold DataFrames from each query

    # Process each query's data
    for key, records in raw_thunder_data.items():
        if key == 'liquidity_by_coin':
            # Liquidity data has a nested structure (coin -> list of time entries)
            # Transform it into a DataFrame with multi-index (time, coin)
            dfs.append(pd.DataFrame({
                (entry['time'], coin): {**entry, 'time': 0} # Create multi-index tuple
                for coin, entries in records.items()
                for entry in entries
            }).T) # Transpose to get time, coin as index
        else:
            # Other queries are simpler lists of records
            # Create DataFrame and set multi-index (time, coin)
            dfs.append(pd.DataFrame(records).set_index(['time', 'coin']))

    # Concatenate all DataFrames along columns, filling missing values based on index alignment
    # Unstack the 'time' level to columns for easier processing
    coin_time_df = pd.concat(dfs, axis=1).unstack(0)

    # Create a mapping from Hyperliquid spot market internal names (like 'kSOL') to base coin symbols ('SOL')
    # Uses the previously fetched hl_markets data
    spot_mapping = {d['info']['name']: symbol.split('/')[0]
                    for symbol, d in hl_markets.items() if ':' not in symbol} # Filter for spot markets

    # Separate spot and futures data based on whether the coin name is in the spot_mapping
    spot_data_df = (coin_time_df.loc[coin_time_df.index.isin(spot_mapping)]
                    .rename(spot_mapping) # Rename index using the spot mapping
                    .unstack().unstack(0)) # Restructure for averaging
    fut_data_df = (coin_time_df.loc[~coin_time_df.index.isin(spot_mapping)]
                   .unstack().unstack(0)) # Restructure for averaging

    # Calculate average notional open interest for futures
    fut_data_df['avg_notional_oi'] = (fut_data_df['avg_oracle_px'] *
                                       fut_data_df['avg_open_interest'])

    # Calculate the 30-day average for all metrics for futures
    # Unstack 'coin' level, sort by time, take last 30 days, calculate mean, unstack back
    fut_s_df = fut_data_df.unstack(1).sort_index().iloc[-30:].mean().unstack(0)
    # Calculate the 30-day average for all metrics for spot (though not used in the final return)
    spot_s_df = spot_data_df.unstack(1).sort_index().iloc[-30:].mean().unstack(0)

    # Clean the index (coin symbols) for both spot and futures DataFrames
    spot_s_df.index = [clean_symbol(sym) for sym in spot_s_df.index]
    fut_s_df.index = [clean_symbol(sym) for sym in fut_s_df.index]

    # --- Prepare final output DataFrame (using only futures data) ---
    output_df = fut_s_df

    # Calculate derived metrics based on the averaged Thunderhead data
    # HLP Volume Share: (Total Vol - User Vol) / Total Vol. User Vol is half of daily_usd_volume.
    output_df['HLP Vol Share %'] = ((output_df['total_volume']
                                     - output_df['daily_usd_volume']/2)
                                    / output_df['total_volume'] * 100)
    # HLP OI Share: Absolute HLP Notional Position / Average Total Notional OI
    output_df['HLP OI Share %'] = (output_df['daily_ntl_abs']
                                   / output_df['avg_notional_oi'] * 100)
    # Format key metrics: OI and Volume in millions USD, Slippage in basis points (bps)
    output_df['OI on HL $m'] = output_df['avg_notional_oi'] / 1e6
    output_df['Volume on HL $m'] = output_df['total_volume'] / 1e6
    # Slippage is given as a fraction, convert to bps (multiply by 100 * 100)
    output_df['HL Slip. $3k'] = output_df['median_slippage_3000'] * 100_00
    output_df['HL Slip. $30k'] = output_df['median_slippage_30000'] * 100_00

    # Return the processed DataFrame containing averaged futures metrics
    return output_df


def process_cmc_data(cmc_data):
    """
    Processes the raw CoinMarketCap listing data.
    Extracts market cap (MC) and fully diluted market cap (FD MC).
    Calculates an estimated market cap based on Hyperliquid data as a fallback/comparison.
    Selects the maximum market cap from available sources.
    Returns a DataFrame indexed by cleaned symbol with the final market cap in millions USD.
    """
    # Extract relevant fields (symbol, mc, fd_mc) from the raw CMC data
    output_df = pd.DataFrame([{
        'symbol':a['symbol'],
        'mc':a['quote']['USD']['market_cap'],
        'fd_mc':a['quote']['USD']['fully_diluted_market_cap'],}
    for a in cmc_data])
    # Group by symbol and take the max MC/FD MC in case CMC lists duplicates (e.g., different chains)
    output_df = output_df.groupby('symbol')[['mc', 'fd_mc',]].max()

    # If market cap is 0 or null, use the fully diluted market cap as a fallback
    output_df.loc[output_df['mc']==0,'mc'] = output_df['fd_mc']

    # Calculate an estimated market cap using Hyperliquid spot market data (Circulating Supply * Mark Price)
    # This serves as another data point, especially if CMC data is missing/inaccurate
    hl_mc_series = pd.Series({symbol.split('/')[0]: # Use base symbol
                               float(data['info'].get('circulatingSupply',0)) # Get circulating supply
                               *float(data['info'].get('markPx',0)) # Get mark price
                               for symbol,data in hl_markets.items()
                               # Only consider HL spot markets (no ':' in symbol)
                                   if ':' not in symbol}, name='hl_mc') # Name the series

    # Concatenate the CMC MC and HL estimated MC
    output_df = pd.concat([output_df, hl_mc_series], axis = 1)

    # Calculate the final 'MC $m' by taking the maximum of CMC MC and HL MC, converting to millions
    output_df['MC $m'] = output_df[['mc','hl_mc']].max(axis=1)/1e6

    # Return the DataFrame with the calculated market cap
    return output_df


# %% Scoring Logic

def build_scores(df):
    """
    Calculates various score components and a final aggregated score for each asset.
    Uses the SCORE_CUTOFFS configuration to define metrics and scoring thresholds.
    Applies adjustments based on whether the asset is on HL and if it's in the HL_STRICT list.
    Returns a DataFrame containing all calculated scores.
    """
    output = {} # Dictionary to store calculated score Series

    # Iterate through each defined score category (e.g., 'Market Cap Score')
    for score_category, category_details in SCORE_CUTOFFS.items():
        # Initialize the category score Series with zeros
        output[score_category] = pd.Series(0, index=df.index)
        # Iterate through each metric contributing to this category (e.g., 'MC $m')
        for score_var, thresholds in category_details.items():
            # --- Generate Score Thresholds ---
            if thresholds['kind'] == 'exp':
                # Generate exponentially spaced thresholds
                point_thresholds = {
                    thresholds['start']
                    * (thresholds['end']/thresholds['start'])
                    ** (k/thresholds['steps']): # Calculate threshold value
                       k for k in range(0, thresholds['steps']+1)} # Map threshold to score points (k)
            elif thresholds['kind'] == 'linear':
                # Generate linearly spaced thresholds
                 point_thresholds = {
                    thresholds['start']
                    + (thresholds['end']  - thresholds['start'] )
                    * (k/thresholds['steps'] ): # Calculate threshold value
                       k for k in range(0, thresholds['steps'] +1)} # Map threshold to score points (k)
            else:
                raise ValueError(f"Unknown threshold kind: {thresholds['kind']}")

            # --- Calculate Partial Score for the Metric ---
            score_name = 'Partial_Score_'+score_var # Name for the partial score column
            # Initialize the partial score Series with zeros
            output[score_name] = pd.Series(0, index=df.index)
            # Iterate through sorted thresholds and assign points
            # Assets get the score 'value' if their metric 'score_var' is >= the threshold 'lb'
            for lb, value in sorted(point_thresholds.items()):
                output[score_name].loc[df[score_var].fillna(0) >= lb] = value # Use fillna(0) for safety

            # Add this metric's partial score to the overall category score
            output[score_category] += output[score_name]

    # --- Apply Score Adjustments ---
    # Convert intermediate scores dictionary to a DataFrame
    output_df = pd.concat(output, axis=1)

    # Zero out scores related to Hyperliquid for assets not listed there (Max Lev < 1)
    hl_score_cols = [c for c in output_df if 'HL' in c] # Identify HL-related score columns
    output_df.loc[df['Max Lev. on HL'] < 1, hl_score_cols] = 0

    # Calculate a score boost for assets NOT listed on HL (Max Lev < 1)
    # Boost is 50% of their non-HL scores (MC, Spot Vol, Fut Vol)
    # This helps identify potentially listable assets
    output_df['NON_HL_SCORE_BOOST'] = (
        0.5
        * (df['Max Lev. on HL'] < 1) # Boolean Series (True if not on HL)
        * output_df[['Market Cap Score', 'Spot Vol Score', 'Futures Vol Score']]
        .sum(axis=1) # Sum of non-HL scores
    ).astype(int) # Convert boost to integer

    # --- Final Score Calculation ---
    # Add a boolean column indicating if the asset is in the HL_STRICT list
    output_df['Strict'] = output_df.index.isin(HL_STRICT)

    # Calculate the final aggregated score
    output_df['Score'] = (
        # Sum of all main score categories + the non-HL boost
        output_df[[*SCORE_CUTOFFS, 'NON_HL_SCORE_BOOST'] ].sum(axis=1)
        # Add the strict boost if applicable
        + output_df['Strict']*HL_STRICT_BOOST
    )

    return output_df


# %% Recommendation Logic

def generate_recommendation(row):
    """
    Generates a recommendation ('Inc. Lev.', 'Dec. Lev.', 'List', 'Delist', '')
    based on the asset's final 'Score' and its current 'Max Lev. on HL'.
    Uses the SCORE_LB and SCORE_UB dictionaries for thresholds.
    """
    # Determine the relevant leverage level for score thresholds (clamped between min/max keys of LB/UB)
    current_lev_key = min(max(SCORE_LB), int(row['Max Lev. on HL']))

    # Check if score is below the lower bound for the current leverage level
    is_below_lb = row['Score'] < SCORE_LB[current_lev_key]
    # Check if score is at or above the upper bound for the current leverage level
    is_above_ub = row['Score'] >= SCORE_UB[current_lev_key]

    # --- Recommendation Rules ---
    # High Leverage (5x, 10x) and score is too low -> Decrease Leverage
    if row['Max Lev. on HL'] > 3 and is_below_lb:
        return 'Dec. Lev.'
    # Medium Leverage (3x) and score is too low -> Delist
    if row['Max Lev. on HL'] == 3 and is_below_lb:
        return 'Delist'
    # Not Listed (0x Leverage) and score is high enough -> List (at 3x)
    if row['Max Lev. on HL'] == 0 and is_above_ub:
        return 'List'
    # Listed (3x+) and score is high enough -> Increase Leverage
    if row['Max Lev. on HL'] > 0 and is_above_ub:
        return 'Inc. Lev.'
    # Otherwise, no recommendation (score is within acceptable range for current leverage)
    return ''


# %% Main Execution - Data Pipeline and Output Generation

# --- Data Acquisition and Processing ---
# Execute all download and processing functions in sequence
# Store the resulting DataFrames in a list
processed_data = [
    process_cmc_data(dl_cmc_data()),                   # Process CoinMarketCap data
    process_reference_exch_data(dl_reference_exch_data()), # Process reference exchange data
    process_hl_data(dl_hl_data()),                     # Process Hyperliquid metadata
    process_thunderhead_data(dl_thunderhead_data())    # Process Thunderhead analytics data
]

# --- Data Merging and Final Calculations ---
# Concatenate all processed DataFrames into a single master DataFrame
# Alignment is based on the index (cleaned coin symbol)
df = pd.concat(processed_data, axis=1)

# Filter out stablecoins
df = df.loc[~df.index.isin(STABLE_COINS)]

# Add 'Symbol' column from the index
df['Symbol'] = df.index
# Fill missing 'Max Lev. on HL' values with 0 (for assets not found in HL data)
df['Max Lev. on HL'] = df['Max Lev. on HL'].fillna(0)

# Calculate scores by applying the build_scores function
df = pd.concat([df, build_scores(df)], axis=1)

# Generate recommendations by applying the generate_recommendation function row-wise
df['Recommendation'] = df.apply(generate_recommendation, axis=1)

# --- Output Generation ---

# 1. Main Data Output (JSON for UI/Analysis)
# Select and order columns as defined in OUTPUT_COLS
df_for_main_data = df[OUTPUT_COLS].sort_values('Score', ascending=False).copy()

# Format numerical columns to significant figures for cleaner display
for c in df_for_main_data.columns:
    if str(df_for_main_data[c].dtype) in ['int64', 'float64',]:
        df_for_main_data[c] = df_for_main_data[c].map(sig_figs)

# Save the main data to the specified JSON file in 'records' orientation
with open(OUTPUT_RAW_DATA_JSON, 'w') as f:
    json.dump(df_for_main_data.to_dict(orient='records'),f)


# 2. Correlation Matrix Output (JSON for UI/Analysis)
# Select columns relevant for correlation analysis (Leverage + all score columns)
df_for_corr = df[['Max Lev. on HL'] + [c for c in df if "Score" in c]].copy()

# Set HL-related scores to NaN for assets not listed on HL, so they don't affect HL correlations
hl_cols_in_corr = [c for c in df_for_corr if 'HL' in c]
df_for_corr.loc[df_for_corr['Max Lev. on HL'] < 1, hl_cols_in_corr] = np.nan

# Shorten partial score column names for better readability in the matrix
df_for_corr = df_for_corr.rename(
    {c: c.replace('Partial_Score_', '..') for c in df_for_corr}, axis=1)

# Calculate the Pearson correlation matrix, scale to percentage, convert to integer
corr_mat = (df_for_corr.corr()*100).astype(int)

# Reset index to turn the 'Symbol' index into a column, matching expected output format
corr_mat = corr_mat.reset_index().rename({'index': 'Symbol'}, axis=1)

# Save the correlation matrix to the specified JSON file in 'records' orientation
with open(OUTPUT_CORR_MAT_JSON, 'w') as f:
    json.dump(corr_mat.to_dict(orient='records'),f)

print(f"Script finished. Main data saved to {OUTPUT_RAW_DATA_JSON}, correlation matrix saved to {OUTPUT_CORR_MAT_JSON}")