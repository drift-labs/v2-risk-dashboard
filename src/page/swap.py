import streamlit as st
import requests
from typing import Dict, Any

def format_token_amount(amount: str, decimals: int = 9) -> float:
    """Convert token amount from base units to decimal representation."""
    return float(amount) / (10 ** decimals)

def get_jupiter_quote(input_mint: str, output_mint: str, amount: int) -> Dict[str, Any]:
    """Fetch quote from Jupiter API."""
    base_url = "https://quote-api.jup.ag/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount),  # Convert to string to avoid scientific notation
        "slippageBps": 50
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching quote: {str(e)}")
        return None

def show():
    st.title("Token Swap")
    
    # Token configurations
    tokens = {
        "SOL": {
            "mint": "So11111111111111111111111111111111111111112",
            "decimals": 9,
            "symbol": "SOL"
        },
        "USDC": {
            "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "decimals": 6,
            "symbol": "USDC"
                    },
        "mSOL": {
            "mint": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
            "decimals": 9,
            "symbol": "mSOL"
        },
        "wBTC": {
            "mint": "3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh",
            "decimals": 8,
            "symbol": "wBTC"
        },
        "wETH": {
            "mint": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",
            "decimals": 18,
            "symbol": "wETH"
        },
        "bSOL": {
            "mint": "bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1",
            "decimals": 9,
            "symbol": "bSOL"
        },
        "JTO": {
            "mint": "jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL",
            "decimals": 6,
            "symbol": "JTO"
        },
        "DRIFT": {
            "mint": "DriFtupJYLTosbwoN8koMbEYSx54aFAVLddWsbksjwg7",
            "decimals": 6,
            "symbol": "DRIFT"
        },
        "USDY": {
            "mint": "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo",
            "decimals": 6,
            "symbol": "USDY"
        },
        "USDS": {
            "mint": "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA",
            "decimals": 6,
            "symbol": "USDS"
        },
        "CLOUD": {
            "mint": "CLoUDKc4Ane7HeQcPpE3YHnznRxhMimJ4MyaUqyHFzAu",
            "decimals": 9,
            "symbol": "CLOUD"
        },
        "PYUSD": {
            "mint": "GyyHYVCrZGc2AQPuvNbcP1babmU3L42ptmxZthUfD9q",
            "decimals": 6,
            "symbol": "PYUSD"
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_token = st.selectbox(
            "From Token",
            options=list(tokens.keys()),
            key="input_token"
        )
        
        input_amount = st.number_input(
            f"Amount in {input_token}",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.8f"  # Increase precision to match jup.ag
        )
    
    with col2:
        output_token = st.selectbox(
            "To Token",
            options=[k for k in tokens.keys() if k != input_token],
            key="output_token"
        )
    
    if st.button("Get Quote"):
        if input_amount <= 0:
            st.error("Please enter an amount greater than 0")
            return
            
        # Convert input amount to base units (lamports/wei)
        input_decimals = tokens[input_token]["decimals"]
        amount_base_units = int(input_amount * (10 ** input_decimals))
        
        # Fetch quote from Jupiter
        quote = get_jupiter_quote(
            tokens[input_token]["mint"],
            tokens[output_token]["mint"],
            amount_base_units
        )
        
        if quote:
            st.subheader("Swap Quote")
            
            # Display quote details
            output_decimals = tokens[output_token]["decimals"]
            output_amount = format_token_amount(quote["outAmount"], output_decimals)
            min_received = format_token_amount(quote["otherAmountThreshold"], output_decimals)
            
            # Calculate USD value if available
            if input_token == "SOL" and output_token == "USDC":
                usd_value = output_amount
                st.write(f"You will receive: {output_amount:.6f} {output_token} (≈${usd_value:.2f})")
            elif input_token == "USDC" and output_token == "SOL":
                usd_value = input_amount
                st.write(f"You will receive: {output_amount:.6f} {output_token} (≈${usd_value:.2f})")
            else:
                st.write(f"You will receive: {output_amount:.6f} {output_token}")
            
            st.write(f"Price Impact: {float(quote['priceImpactPct']) * 100:.8f}%")
            st.write(f"Minimum Received (with slippage): {min_received:.6f} {output_token}")
            
            # Display route information
            st.subheader("Route Information")
            for step in quote["routePlan"]:
                st.write(f"Via {step['swapInfo']['label']} ({step['percent']}%)")
            
            # Display exchange rate
            if output_amount > 0:
                rate = input_amount / output_amount if output_token == "SOL" else output_amount / input_amount
                st.write(f"\nExchange Rate: 1 {input_token} ≈ {rate:.6f} {output_token}") 