#!/bin/bash
# Simple wrapper script for the CCXT testing tool

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.7 or higher."
    exit 1
fi

# Set up environment
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check for CMC API key
if [ -z "$CMC_API_KEY" ]; then
    echo "Warning: CMC_API_KEY is not set. Market cap data testing will fail."
    echo "You can set it by adding it to your .env file or by running: export CMC_API_KEY=your_key"
fi

# Install dependencies if needed
if ! pip list | grep -q "rich"; then
    echo "Installing required dependencies..."
    pip install -r ../requirements.txt
fi

# Default to SOL-PERP if no symbol provided
SYMBOL=${1:-"SOL-PERP"}
shift  # Remove the first argument

# Run the test script with all remaining arguments passed through
# Use Python's unbuffered mode (-u) instead of stdbuf
echo "Testing CCXT implementation for symbol: $SYMBOL"
python -u test_ccxt_implementation.py "$SYMBOL" "$@" 