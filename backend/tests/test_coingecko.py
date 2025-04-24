"""
Simple manual testing script for CoinGecko API functions.
Run this directly to see the output in your terminal.

Usage:
    python test_coingecko.py --symbol BTC --coin-id bitcoin --num-tokens 10
    python test_coingecko.py -s ETH -c ethereum -n 5
"""

import asyncio
import aiohttp
import argparse
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import the utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.utils.coingecko_api import (
    fetch_coingecko_data,
    fetch_all_coingecko_market_data,
    fetch_all_coingecko_volumes,
    fetch_coingecko_historical_volume
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test CoinGecko API functions for a specific token')
    parser.add_argument('-s', '--symbol', type=str, default='BTC',
                      help='Token symbol (e.g., BTC, ETH)')
    parser.add_argument('-c', '--coin-id', type=str, default='bitcoin',
                      help='CoinGecko coin ID (e.g., bitcoin, ethereum)')
    parser.add_argument('-n', '--num-tokens', type=int, default=10,
                      help='Number of top tokens to fetch (default: 10, max: 250)')
    return parser.parse_args()

async def run_tests(symbol: str, coin_id: str, num_tokens: int):
    """Run tests for CoinGecko API functions."""
    print("\n=== Starting CoinGecko API Tests ===\n")
    
    async with aiohttp.ClientSession() as session:
        print(f"Fetching data for {num_tokens} top tokens by market cap...")
        market_data = await fetch_all_coingecko_market_data(session, num_tokens)
        
        if not market_data:
            print("Error: No market data received")
            return
        
        total_tokens = len(market_data)
        print(f"\nReceived data for {total_tokens} tokens")
        if total_tokens < num_tokens:
            print(f"Note: Requested {num_tokens} tokens but only received {total_tokens}.")
            print("This is normal if you requested more tokens than are available on CoinGecko.")
        
        print("\nTop 10 tokens by market cap (or all if less than 10):")
        tokens_to_show = sorted(
            market_data.items(),
            key=lambda x: x[1].get('market_cap_rank', float('inf'))
        )[:min(10, total_tokens)]
        
        for coin_id, data in tokens_to_show:
            symbol = data.get('symbol', '???')
            rank = data.get('market_cap_rank', 'N/A')
            price = data.get('current_price', 'N/A')
            print(f"#{rank}: {symbol} @ ${price}")
        
        # Test specific coin data if provided
        if coin_id and symbol:
            print(f"\nDetailed data for {symbol}:")
            coin_data = market_data.get(coin_id)
            
            if not coin_data:
                print(f"Error: Could not find data for {symbol} ({coin_id})")
                return
            
            # Display detailed coin information
            print(f"Name: {coin_data.get('name')}")
            print(f"Current Price: ${coin_data.get('current_price')}")
            print(f"Market Cap Rank: #{coin_data.get('market_cap_rank')}")
            print(f"Market Cap: ${coin_data.get('market_cap'):,}")
            print(f"Circulating Supply: {coin_data.get('circulating_supply'):,.2f} {symbol}")
            print(f"All-Time High: ${coin_data.get('ath_price')}")
            
            print("\nFetching historical volume data...")
            volume_data = await fetch_coingecko_historical_volume(coin_id)
            if volume_data:
                print(f"30-day Volume: ${volume_data:,.2f}")
            else:
                print("Error: Could not fetch volume data")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting CoinGecko API tests for {args.symbol} ({args.coin_id})...")
    print(f"Will fetch top {args.num_tokens} tokens by market cap...")
    asyncio.run(run_tests(args.symbol, args.coin_id, args.num_tokens))
    print("\nTests completed!") 