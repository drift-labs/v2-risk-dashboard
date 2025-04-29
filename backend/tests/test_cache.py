import logging
import time
from utils.cache_utils import ttl_cache
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a cached version of the CoinGecko API call
@ttl_cache(ttl_seconds=60)  # Set a short TTL for testing purposes
def get_bitcoin_price():
    """Get current Bitcoin price from CoinGecko API."""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd"
    }
    
    logger.info("Making API call to CoinGecko...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def test_cache():
    """Test the cache functionality."""
    logger.info("Starting cache test...")
    
    # First call - should be a cache miss
    logger.info("Making first API call...")
    result1 = get_bitcoin_price()
    logger.info(f"Bitcoin price (first call): ${result1['bitcoin']['usd']:,.2f}")
    
    # Second call - should be a cache hit
    logger.info("Making second API call (should be cached)...")
    result2 = get_bitcoin_price()
    logger.info(f"Bitcoin price (second call): ${result2['bitcoin']['usd']:,.2f}")
    
    # Third call - should be a cache hit
    logger.info("Making third API call (should be cached)...")
    result3 = get_bitcoin_price()
    logger.info(f"Bitcoin price (third call): ${result3['bitcoin']['usd']:,.2f}")
    
    # Wait a bit and make another call to show they're the same
    logger.info("Waiting 5 seconds...")
    time.sleep(5)
    
    # Fourth call - should still be a cache hit
    logger.info("Making fourth API call (should still be cached)...")
    result4 = get_bitcoin_price()
    logger.info(f"Bitcoin price (fourth call): ${result4['bitcoin']['usd']:,.2f}")
    
    # Verify all results are identical
    logger.info("\nVerifying results consistency:")
    logger.info(f"All results identical: {result1 == result2 == result3 == result4}")

if __name__ == "__main__":
    test_cache() 