# CCXT Testing Tool for Delist Recommender

This tool helps diagnose issues with the CCXT implementation in the delist recommender module, specifically related to fetching market volume data for cryptocurrency tokens.

## Setup

1. Install required dependencies:
   ```bash
   pip install -r test_ccxt_requirements.txt
   ```

2. For market cap data testing, set your CoinMarketCap API key:
   ```bash
   export CMC_API_KEY=your_coinmarketcap_api_key_here
   ```
   You can get a free API key from [CoinMarketCap](https://coinmarketcap.com/api/).

## Usage

Basic usage:
```bash
python test_ccxt_implementation.py SOL-PERP
```

This will test both exchange data retrieval and market cap data for SOL.

### Options

- `--verbose` or `-v`: Show more detailed output, including all potential market pairs
  ```bash
  python test_ccxt_implementation.py SOL-PERP --verbose
  ```

- `--ccxt-only`: Only test CCXT exchange data (skip CoinMarketCap)
  ```bash
  python test_ccxt_implementation.py SOL-PERP --ccxt-only
  ```

- `--cmc-only`: Only test CoinMarketCap data (skip exchange data)
  ```bash
  python test_ccxt_implementation.py SOL-PERP --cmc-only
  ```

## Interpreting Results

The script will show:

1. The normalized symbol used for matching (e.g., SOL for SOL-PERP)
2. For each exchange:
   - Available markets matching the symbol
   - Success/failure of data retrieval
   - Sample volume calculations
3. A summary table with volumes across exchanges
4. Market cap data from CoinMarketCap if available

## Troubleshooting Common Issues

- **No markets found**: The exchange may not list the token, or the symbol transformation may be incorrect.
- **Markets found but no data**: The API may be rate-limited or the market data endpoint may be unavailable.
- **Symbol not found in CoinMarketCap**: Check for alternative symbol representations or if the token is listed under a different ticker. 