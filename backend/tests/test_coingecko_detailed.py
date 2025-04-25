"""
Detailed testing script for CoinGecko API that aggregates data from multiple endpoints.
Provides comprehensive token information including market data and historical volumes.

Usage:
    python test_coingecko_detailed.py --num-tokens 10
    python test_coingecko_detailed.py -n 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path so we can import the utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.utils.coingecko_api import (
    fetch_all_coingecko_market_data,
    fetch_coingecko_historical_volume,
    fetch_all_coingecko_volumes
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test CoinGecko API with detailed output')
    parser.add_argument('-n', '--num-tokens', type=int, default=10,
                      help='Number of top tokens to fetch (default: 10, max: 250)')
    return parser.parse_args()

def format_token_data(market_data: Dict, volume_30d: float) -> Dict:
    """Format token data according to the specified structure."""
    return {
        "symbol": market_data.get('symbol', ''),
        "coingecko_data": {
            "coingecko_name": market_data.get('name'),
            "coingecko_id": market_data.get('id'),
            "coingecko_image_url": market_data.get('image_url'),
            "coingecko_current_price": market_data.get('current_price'),
            "coingecko_market_cap_rank": market_data.get('market_cap_rank'),
            "coingecko_market_cap": market_data.get('market_cap'),
            "coingecko_fully_diluted_valuation": market_data.get('fully_diluted_valuation'),
            "coingecko_total_volume_24h": market_data.get('total_volume_24h'),
            "coingecko_mc_derived": market_data.get('market_cap'),  # Same as market_cap for now
            "coingecko_circulating": market_data.get('circulating_supply'),
            "coingecko_total_supply": market_data.get('total_supply'),
            "coingecko_max_supply": market_data.get('max_supply'),
            "coingecko_ath_price": market_data.get('ath_price'),
            "coingecko_ath_change_percentage": market_data.get('ath_change_percentage'),
            "coingecko_volume_30d": volume_30d
        }
    }

def run_detailed_test(num_tokens: int = 10) -> None:
    """
    Run a detailed test of CoinGecko API functionality.
    
    Args:
        num_tokens: Number of tokens to fetch data for (default: 10)
    """
    logger.info(f"Starting detailed test for {num_tokens} tokens")
    
    # Fetch market data for all tokens
    market_data = fetch_all_coingecko_market_data()
    if not market_data or isinstance(market_data, dict) and "error" in market_data:
        logger.error("Failed to fetch market data")
        return
    
    # Convert dictionary to list and sort by market cap
    tokens_list = []
    for coin_id, token_data in market_data.items():
        token_data['id'] = coin_id  # Add coin_id to the token data
        tokens_list.append(token_data)
    
    # Sort tokens by market cap and take top N
    sorted_tokens = sorted(
        tokens_list,
        key=lambda x: float(x.get("market_cap", 0) or 0),  # Handle None values
        reverse=True
    )[:num_tokens]
    
    # Get list of coin IDs for volume fetch
    coin_ids = [token.get("id") for token in sorted_tokens]
    logger.info(f"Fetching 30d volume data for top {len(coin_ids)} tokens")
    
    # Fetch 30d volume data for all tokens at once
    volume_data = fetch_all_coingecko_volumes(coin_ids)
    
    results = []
    total_tokens = len(sorted_tokens)
    
    for idx, token in enumerate(sorted_tokens, 1):
        coin_id = token.get("id")
        symbol = token.get("symbol", "").upper()
        
        # Get the 30d volume from the volume data
        total_volume = volume_data.get(coin_id, 0)
        
        results.append({
            "symbol": symbol,
            "name": token.get("name"),
            "market_cap": token.get("market_cap", 0),
            "total_volume_30d": total_volume
        })
        
        # Log progress
        progress = (idx / total_tokens) * 100
        logger.info(f"Progress: {progress:.1f}% ({idx}/{total_tokens})")
    
    # Format and display results
    if results:
        print("\nResults:")
        print(f"{'Symbol':<10} {'Name':<20} {'Market Cap':>15} {'30d Volume':>20}")
        print("-" * 65)
        
        for r in results:
            print(
                f"{r['symbol']:<10} "
                f"{r['name'][:20]:<20} "
                f"${r['market_cap']:>14,.0f} "
                f"${r['total_volume_30d']:>19,.0f}"
            )
    else:
        logger.warning("No results to display")

if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting detailed CoinGecko API tests for {args.num_tokens} tokens...")
    
    run_detailed_test(args.num_tokens)
    
    print("\nTests completed!") 