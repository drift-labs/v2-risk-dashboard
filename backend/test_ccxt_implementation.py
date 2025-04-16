#!/usr/bin/env python
"""
Test script for troubleshooting CCXT implementation in delist_recommender.py

This script isolates and tests the exchange data retrieval functionality
to identify issues with fetching volume data for market symbols.
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
import logging
import ccxt
import ccxt.async_support as ccxt_async
import aiohttp
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_ccxt")

# Copy key constants from delist_recommender.py
STABLE_COINS = {"USDC", 'FDUSD', "USDT", 'DAI', 'USDB', 'USDE', 'TUSD', 'USR'}
DAYS_TO_CONSIDER = 30

REFERENCE_SPOT_EXCH = {
    'binanceus', 'bybit', 'okx', 'gate', 'kucoin', 'mexc', 'kraken'
}

REFERENCE_FUT_EXCH = {
    'bybit', 'binanceus', 'gate', 'mexc', 'okx', 'htx', 'bitmex'
}

console = Console()

# --- Copy key utility functions from delist_recommender.py ---

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

@asynccontextmanager
async def get_exchange_api(exch):
    """Context manager for handling exchange API lifecycle."""
    api = None
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
        
        try:
            await api.load_markets()
        except Exception as e:
            logger.warning(f"Could not load markets for {exch}: {str(e)}. Using limited functionality.")
        
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
            if api:
                await api.close()
                if exch == 'bybit' and hasattr(api, 'session') and api.session:
                    await api.session.close()
            yield None
            return
                
        yield api
    except Exception as e:
        logger.error(f"Failed to initialize {exch} API: {str(e)}")
        yield None
    finally:
        if api:
            try:
                # Close the API first
                await api.close()
                
                # Additional cleanup for specific exchanges
                if hasattr(api, 'session') and api.session:
                    await api.session.close()
                if hasattr(api, 'connector') and api.connector:
                    await api.connector.close()
                
                # Special handling for Bybit session
                if exch == 'bybit' and session:
                    await session.close()
                
                logger.debug(f"Successfully closed {exch} API and cleaned up resources")
            except Exception as close_error:
                logger.warning(f"Error closing {exch} API: {str(close_error)}")

def geomean_three(series):
    """Calculates geometric mean of top 3 values in a series."""
    return np.exp(np.log(series + 1).sort_values()[-3:].sum() / 3) - 1

# --- Modified versions of delist_recommender functions for testing ---

async def test_download_exch(exch, spot, symbol, verbose=False):
    """Test helper to download data from a single exchange for a specific symbol."""
    console.print(f"[bold cyan]Testing download from {exch} (spot={spot}) for symbol {symbol}[/bold cyan]")
    
    clean_sym = clean_symbol(symbol).upper()
    normalized_symbols = {clean_sym}
    
    async with get_exchange_api(exch) as api:
        if not api:
            console.print(f"[bold red]Failed to initialize API for {exch}[/bold red]")
            return {}
        
        # If no markets available, return empty data
        if not hasattr(api, 'markets') or not api.markets:
            console.print(f"[bold red]No markets available for {exch}[/bold red]")
            return {}
                
        exchange_data = {}
        
        # Show available markets if verbose
        if verbose:
            console.print(f"[cyan]Exchange {exch} has {len(api.markets)} markets[/cyan]")
            
        # Pre-filter markets to only those potentially matching our symbols
        markets_to_try = []
        matched_symbols = []
        
        # Detailed logging for market matching
        all_markets = [m for m in api.markets.keys()]
        
        # First pass - show all markets containing our symbol
        potential_matches = []
        for market_name in all_markets:
            try:
                # Extract base symbol and check if it contains our symbol
                parts = market_name.split('/')
                if len(parts) < 2:
                    continue
                
                base_symbol = parts[0].upper()
                quote_symbol = parts[1].split(':')[0].upper() if ':' in parts[1] else parts[1].upper()
                
                # Verbose output of all markets
                if verbose and (base_symbol == clean_sym or clean_sym in base_symbol):
                    potential_matches.append(market_name)
                    
                # Filter for stablecoin pairs and market type
                if spot and ':' in market_name:
                    continue
                if not spot and ':USD' not in market_name and ':USDT' not in market_name:
                    continue
                    
                # Verify quote currency is a stablecoin
                if quote_symbol not in STABLE_COINS:
                    continue
                
                # Check if base symbol matches
                clean_base = clean_symbol(base_symbol)
                if (base_symbol == clean_sym or clean_base == clean_sym):
                    markets_to_try.append(market_name)
                    matched_symbols.append(f"{base_symbol}/{quote_symbol}")
                
            except Exception as e:
                if verbose:
                    console.print(f"[dim red]Error processing market {market_name}: {str(e)}[/dim red]")
                continue
        
        if verbose:
            console.print(f"[cyan]Potential symbol matches found (any quote pair): {len(potential_matches)}[/cyan]")
            for m in potential_matches[:10]:  # Show first 10 to avoid flooding output
                console.print(f"  - {m}")
            if len(potential_matches) > 10:
                console.print(f"  ... and {len(potential_matches) - 10} more")
        
        console.print(f"[green]Found {len(markets_to_try)} valid market pairs for {symbol}:[/green]")
        for m in markets_to_try:
            console.print(f"  - {m}")
        
        if not markets_to_try:
            console.print(f"[bold red]No valid markets found for {symbol} on {exch}[/bold red]")
            return {}
        
        # Function to fetch OHLCV data for a single market
        async def fetch_market_data(market):
            try:
                console.print(f"[cyan]Fetching OHLCV data for {market} on {exch}...[/cyan]")
                # Fetch OHLCV data with timeout
                ohlcv_data = await asyncio.wait_for(
                    api.fetch_ohlcv(market, '1d', limit=30),
                    timeout=10.0  # 10 second timeout
                )
                if ohlcv_data and len(ohlcv_data) > 0:
                    console.print(f"[green]Successfully fetched {len(ohlcv_data)} candles for {market}[/green]")
                    return market, ohlcv_data
                console.print(f"[yellow]No data returned for {market}[/yellow]")
                return market, None
            except Exception as e:
                console.print(f"[red]Failed to fetch OHLCV data for {market} on {exch}: {str(e)}[/red]")
                return market, None
        
        # Process markets concurrently with a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests for testing
        
        async def fetch_with_semaphore(market):
            async with semaphore:
                return await fetch_market_data(market)
        
        # Create tasks for all markets
        tasks = [fetch_with_semaphore(market) for market in markets_to_try]
        
        # Process markets with timeout
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            console.print(f"[bold red]Error gathering market data: {str(e)}[/bold red]")
            results = []
            
        # Process results
        successful_markets = 0
        for market, data in results:
            if isinstance(data, Exception):
                console.print(f"[red]Exception when fetching {market}: {str(data)}[/red]")
            elif market and data:
                exchange_data[market] = data
                successful_markets += 1
        
        console.print(f"[bold green]Downloaded data from {exch}: got data for {successful_markets}/{len(markets_to_try)} markets[/bold green]")
        
        # Process some sample data to show volumes
        if exchange_data:
            console.print("[bold cyan]Sample volume calculations:[/bold cyan]")
            for market, ohlcv_data in exchange_data.items():
                try:
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv_data, columns=[*'tohlcv']).set_index('t')
                    
                    # Get contract size if available
                    try:
                        contractsize = min(api.markets.get(market, {}).get('contractSize', None) or 1, 1)
                    except Exception:
                        contractsize = 1
                        
                    # Calculate average daily volume in USD
                    daily_usd_volume = (df.c * df.v * contractsize).mean()
                    
                    console.print(f"[green]Market {market}:[/green]")
                    console.print(f"  - Average Daily Volume: ${daily_usd_volume:,.2f}")
                    console.print(f"  - Last Close Price: ${df.c.iloc[-1]:,.6f}")
                    console.print(f"  - First 3 candles (of {len(df)}):")
                    
                    # Print first 3 candles
                    for i, (ts, row) in enumerate(df.iloc[:3].iterrows()):
                        dt = datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d')
                        console.print(f"    {dt}: Open=${row.o:,.6f}, High=${row.h:,.6f}, Low=${row.l:,.6f}, Close=${row.c:,.6f}, Volume={row.v:,.2f}")
                        
                except Exception as e:
                    console.print(f"[red]Error processing data for {market}: {str(e)}[/red]")
        
        return exchange_data

async def test_market_data(symbol, verbose=False):
    """Test getting market data for a specific symbol."""
    
    console.print(f"[bold green]==== Testing Market Data Retrieval for {symbol} ====[/bold green]")
    
    # Normalize symbol - remove -PERP suffix if present
    base_symbol = symbol.replace('-PERP', '')
    console.print(f"[cyan]Normalized symbol for exchange matching: {base_symbol}[/cyan]")
    
    # Priority exchanges to test
    spot_exchanges = ['binanceus', 'bybit', 'okx']  # Spot exchanges
    fut_exchanges = ['bybit', 'okx']   # Futures exchanges (Binance US removed as it doesn't support futures)
    
    # Test spot exchanges
    spot_results = {}
    console.print("[bold magenta]\nTesting Spot Exchanges:[/bold magenta]")
    for exch in spot_exchanges:
        exch_data = await test_download_exch(exch, True, base_symbol, verbose)
        if exch_data:
            spot_results[exch] = exch_data
    
    # Test futures exchanges
    futures_results = {}
    console.print("[bold magenta]\nTesting Futures Exchanges:[/bold magenta]")
    for exch in fut_exchanges:
        exch_data = await test_download_exch(exch, False, base_symbol, verbose)
        if exch_data:
            futures_results[exch] = exch_data
    
    # Summarize results
    console.print("\n[bold green]===== Results Summary =====[/bold green]")
    
    # Create a table for the summary
    table = Table(title=f"Market Data Summary for {symbol}")
    table.add_column("Exchange", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Markets Found", style="green")
    table.add_column("Markets with Data", style="yellow")
    table.add_column("Best Market", style="blue")
    table.add_column("Avg Daily Volume", style="green")
    
    # Calculate total markets found
    total_spot_markets = sum(len(data) for data in spot_results.values())
    total_fut_markets = sum(len(data) for data in futures_results.values())
    
    # Process spot results
    for exch, data in spot_results.items():
        if not data:
            table.add_row(exch, "Spot", "0", "0", "-", "$0")
            continue
            
        markets_count = len(data)
        best_market = None
        best_volume = 0
        
        for market, ohlcv in data.items():
            try:
                # Calculate average volume
                df = pd.DataFrame(ohlcv, columns=[*'tohlcv']).set_index('t')
                vol = (df.c * df.v).mean()
                if vol > best_volume:
                    best_volume = vol
                    best_market = market
            except:
                continue
                
        table.add_row(
            exch, 
            "Spot", 
            str(markets_count), 
            str(len(data)), 
            best_market or "-", 
            f"${best_volume:,.2f}" if best_volume > 0 else "$0"
        )
    
    # Process futures results
    for exch, data in futures_results.items():
        if not data:
            table.add_row(exch, "Futures", "0", "0", "-", "$0")
            continue
            
        markets_count = len(data)
        best_market = None
        best_volume = 0
        
        for market, ohlcv in data.items():
            try:
                # Calculate average volume
                df = pd.DataFrame(ohlcv, columns=[*'tohlcv']).set_index('t')
                vol = (df.c * df.v).mean()
                if vol > best_volume:
                    best_volume = vol
                    best_market = market
            except:
                continue
                
        table.add_row(
            exch, 
            "Futures", 
            str(markets_count), 
            str(len(data)), 
            best_market or "-", 
            f"${best_volume:,.2f}" if best_volume > 0 else "$0"
        )
    
    # Add totals row
    table.add_row(
        "TOTAL", 
        "All", 
        str(total_spot_markets + total_fut_markets), 
        str(sum(len(data) for data in spot_results.values()) + sum(len(data) for data in futures_results.values())), 
        "-", 
        "-"
    )
    
    console.print(table)
    
    return {
        "spot": spot_results,
        "futures": futures_results
    }

async def test_cmc_data(symbol, api_key=None):
    """Test retrieving market cap data from CoinMarketCap."""
    console.print(f"[bold green]==== Testing Market Cap Data Retrieval for {symbol} ====[/bold green]")
    
    # Normalize symbol - remove -PERP suffix if present
    base_symbol = symbol.replace('-PERP', '')
    console.print(f"[cyan]Normalized symbol for CMC: {base_symbol}[/cyan]")
    
    # Use API key from environment variable if not provided
    cmc_api_key = api_key or os.environ.get("CMC_API_KEY")
    if not cmc_api_key:
        console.print("[bold red]CoinMarketCap API key is not set. Please provide via API_KEY env var.[/bold red]")
        console.print("You can get a free API key from https://coinmarketcap.com/api/")
        return None
    
    cmc_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    
    async with aiohttp.ClientSession() as session:
        try:
            console.print("[cyan]Making request to CoinMarketCap API...[/cyan]")
            
            headers = {
                'X-CMC_PRO_API_KEY': cmc_api_key,
                'Accept': 'application/json'
            }
            
            params = {
                'limit': 500,
                'convert': 'USD'
            }
            
            async with session.get(cmc_api_url, headers=headers, params=params, timeout=15) as response:
                response.raise_for_status()
                data = (await response.json()).get('data', [])
            
            if not data:
                console.print("[bold red]No data received from CoinMarketCap API response[/bold red]")
                return None
            
            console.print(f"[green]Successfully fetched data for {len(data)} tokens from CoinMarketCap[/green]")
            
            # Find our symbol
            found_tokens = []
            for token in data:
                if token.get('symbol', '') == base_symbol:
                    found_tokens.append(token)
            
            if not found_tokens:
                console.print(f"[bold yellow]Symbol {base_symbol} not found in CoinMarketCap data[/bold yellow]")
                
                # Try to find similar symbols
                similar_tokens = []
                for token in data:
                    symbol = token.get('symbol', '')
                    if base_symbol in symbol or symbol in base_symbol:
                        similar_tokens.append(token)
                
                if similar_tokens:
                    console.print(f"[yellow]Found {len(similar_tokens)} similar symbols:[/yellow]")
                    for token in similar_tokens:
                        console.print(f"  - {token.get('symbol')}: {token.get('name')}")
            else:
                console.print(f"[bold green]Found {len(found_tokens)} matching tokens for {base_symbol}[/bold green]")
                
                # Create a table to display the results
                table = Table(title=f"Market Cap Data for {base_symbol}")
                table.add_column("Name", style="cyan")
                table.add_column("Symbol", style="magenta")
                table.add_column("Market Cap", style="green")
                table.add_column("Fully Diluted MC", style="blue")
                table.add_column("Price", style="yellow")
                table.add_column("Volume (24h)", style="green")
                
                for token in found_tokens:
                    usd_data = token.get('quote', {}).get('USD', {})
                    table.add_row(
                        token.get('name', 'Unknown'),
                        token.get('symbol', 'Unknown'),
                        f"${usd_data.get('market_cap', 0):,.2f}",
                        f"${usd_data.get('fully_diluted_market_cap', 0):,.2f}",
                        f"${usd_data.get('price', 0):,.6f}",
                        f"${usd_data.get('volume_24h', 0):,.2f}"
                    )
                    
                console.print(table)
            
            return data
            
        except Exception as e:
            console.print(f"[bold red]Error connecting to CoinMarketCap API: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
            return None

async def main():
    """Main test function."""
    console.print("[bold]===============================================[/bold]")
    console.print("[bold cyan]CCXT Implementation Test for Delist Recommender[/bold cyan]")
    console.print("[bold]===============================================[/bold]")
    
    import argparse
    parser = argparse.ArgumentParser(description='Test CCXT implementation for a symbol')
    parser.add_argument('symbol', type=str, help='Symbol to test (e.g., SOL-PERP or SOL)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--cmc-only', action='store_true', help='Only test CoinMarketCap data')
    parser.add_argument('--ccxt-only', action='store_true', help='Only test CCXT exchange data')
    args = parser.parse_args()
    
    symbol = args.symbol
    
    console.print(f"Python version: {sys.version}")
    console.print(f"CCXT version: {ccxt.__version__}")
    console.print(f"Testing symbol: {symbol}")
    
    try:
        if not args.cmc_only:
            await test_market_data(symbol, args.verbose)
        
        if not args.ccxt_only:
            await test_cmc_data(symbol)
        
        console.print("[bold green]Testing complete![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Error in main execution: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
    finally:
        # Ensure all pending tasks are complete
        pending = asyncio.all_tasks()
        for task in pending:
            if not task.done() and task != asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Additional cleanup for any remaining sessions/connectors
        for task in pending:
            if hasattr(task, '_coro'):
                coro = task._coro
                if hasattr(coro, 'cr_frame'):
                    frame = coro.cr_frame
                    if frame is not None and hasattr(frame, 'f_locals'):
                        for obj in frame.f_locals.values():
                            # Clean up any remaining client sessions
                            if isinstance(obj, aiohttp.ClientSession):
                                await obj.close()
                            # Clean up any remaining connectors
                            if isinstance(obj, aiohttp.TCPConnector):
                                await obj.close()
        
        # Final sleep to allow cleanup to complete
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Set up asyncio policy for Windows compatibility if needed
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run with proper cleanup
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        # Ensure the loop is closed properly
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close() 