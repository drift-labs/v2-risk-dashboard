import streamlit as st
import pandas as pd
import numpy as np
from lib.api import fetch_api_data

# --- Constants ---
DEFAULT_NUMBER_OF_TOKENS = 7  # Hardcoded number of tokens to analyze
DEBUG_MODE = True  # Enable debug logging

# --- Helper Functions ---

def format_large_number(num):
    """Formats large numbers into human-readable strings with $, M, B, K suffixes."""
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif abs(num) >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        elif abs(num) >= 1_000:
            return f"${num / 1_000:.2f}K"
        elif abs(num) < 1 and abs(num) > 0:
            return f"${num:.4f}" # Show more precision for small numbers
        else:
            return f"${num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(num):
    """Formats number as a percentage string."""
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        return f"{float(num):.3f}%"
    except (ValueError, TypeError):
        return "N/A"


def style_recommendation(rec):
    """Applies color styling based on recommendation."""
    if rec == 'List':
        color = 'green'
    elif rec == 'Increase Leverage':
        color = 'blue'
    elif rec == 'Decrease Leverage':
        color = 'orange'
    elif rec == 'Delist':
        color = 'red'
    elif rec == 'No Action':
        color = 'gray'
    else:
        color = 'black'
    return f'color: {color}; font-weight: bold;'

@st.cache_data(ttl=60) # Cache data for 1 minutes
def fetch_market_recommendations():
    """Fetches market recommendations from the backend API."""
    try:
        # Only log on actual API calls (not reruns)
        if DEBUG_MODE:
            print("Fetching market recommendations from API")
            endpoint = "market-recommender"
            path = "market-data"
            params = {"number_of_tokens": DEFAULT_NUMBER_OF_TOKENS}
            print(f"API call details: endpoint={endpoint}, path={path}, params={params}")
        
        response = fetch_api_data(
            "market-recommender", 
            "market-data", 
            params={"number_of_tokens": DEFAULT_NUMBER_OF_TOKENS},
            retry=True
        )
        
        if DEBUG_MODE:
            print("Successfully fetched data from API.")
        
        return response
    except Exception as e:
        error_message = f"An error occurred while fetching data: {e}"
        if DEBUG_MODE:
            print(f"ERROR: {error_message}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        # Return error information rather than displaying directly
        return {"error": error_message}

# --- Streamlit Page ---

def market_recommender_page():
    """
    Streamlit page for displaying market recommendations for Drift.
    Fetches pre-calculated data from a backend API.
    """
    # Initialize request flag if not present
    if "api_request_in_progress" not in st.session_state:
        st.session_state.api_request_in_progress = False
    
    st.title("📈 Market Recommender")
    st.markdown("This page provides recommendations for listing, delisting, or adjusting leverage for markets based on various metrics calculated by the backend.")
    st.markdown("THIS PAGE IS IN BETA AND SUBJECT TO CHANGE.")
    
    # Check if request is already in progress
    if st.session_state.api_request_in_progress:
        st.info("Data request in progress... Please wait.")
        with st.spinner("This may take a few minutes as the backend processes market data..."):
            # Add a small delay to prevent too frequent reruns
            import time
            time.sleep(2)
            # Try to get data from cache
            data = fetch_market_recommendations()
            # If we got data, clear the in-progress flag
            if data is not None and not isinstance(data, dict) or not data.get("error"):
                st.session_state.api_request_in_progress = False
            else:
                st.rerun()  # Rerun to check if data is ready
    else:
        # Try to get data from cache without starting a new request
        data = fetch_market_recommendations()
        
        # If no data in cache and no request in progress, start a new request
        if data is None and not st.session_state.api_request_in_progress:
            st.session_state.api_request_in_progress = True
            st.info("Requesting data from backend. This may take a few minutes...")
            # This will trigger the API call
            st.rerun()
    
    # Check for errors
    if isinstance(data, dict) and "error" in data:
        st.error(data["error"])
        if st.button("Retry"):
            st.session_state.api_request_in_progress = False
            st.cache_data.clear()
            st.rerun()
        return
    
    if data and isinstance(data, list):
        st.subheader(f"Analysis Results ({len(data)} tokens)")

        # Count recommendations
        list_count = sum(1 for item in data if item.get('recommendation') == 'List')
        increase_leverage_count = sum(1 for item in data if item.get('recommendation') == 'Increase Leverage')
        decrease_leverage_count = sum(1 for item in data if item.get('recommendation') == 'Decrease Leverage')
        delist_count = sum(1 for item in data if item.get('recommendation') == 'Delist')
        no_action_count = sum(1 for item in data if item.get('recommendation') == 'No Action')

        # Display Summary Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Recommend List", list_count, delta_color="off")
        with col2:
            st.metric("Increase Leverage", increase_leverage_count, delta_color="off")
        with col3:
            st.metric("No Action", no_action_count, delta_color="off")
        with col4:
            st.metric("Decrease Leverage", decrease_leverage_count, delta_color="off")
        with col5:
            st.metric("Recommend Delist", delist_count, delta_color="off")

        try:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)
            
            # Create checkbox for filtering to only show listed tokens
            show_only_listed = st.checkbox("Show only tokens listed on Drift", value=False)
            
            if show_only_listed:
                # Check if drift_data and drift_is_listed_perp are available
                if 'drift_data' in df.columns:
                    listed_tokens = []
                    for idx, row in df.iterrows():
                        drift_data = row.get('drift_data', {})
                        if isinstance(drift_data, dict) and drift_data.get('drift_is_listed_perp') == 'true':
                            listed_tokens.append(idx)
                    df = df.loc[listed_tokens]
                    st.write(f"Showing {len(df)} tokens currently listed on Drift")
                else:
                    st.warning("Unable to filter by listed status - missing data")
            
            # Extract key fields and create a display DataFrame
            display_df = pd.DataFrame()
            
            # First try to extract key fields directly
            try:
                display_df['Symbol'] = df['symbol']
            except (KeyError, TypeError):
                display_df['Symbol'] = [item.get('symbol', 'Unknown') for item in data]
            
            # Extract metrics or set defaults if not found
            try:
                display_df['Total Score'] = df['total_score']
            except (KeyError, TypeError):
                display_df['Total Score'] = [item.get('total_score', 0) for item in data]
                
            try:
                display_df['Recommendation'] = df['recommendation']
            except (KeyError, TypeError):
                display_df['Recommendation'] = [item.get('recommendation', 'Unknown') for item in data]
                
            # Extract metrics from nested structures if available
            display_df['Market Cap'] = 0.0  # Use float instead of int
            display_df['24h Volume'] = 0.0  # Use float instead of int
            display_df['Drift OI'] = 0.0  # Use float instead of int
            display_df['Max Leverage'] = 0.0  # Use float instead of int
            display_df['Drift Volume (30d)'] = 0.0  # Use float instead of int
            
            for idx, row in df.iterrows():
                # Extract from coingecko_data if available
                coingecko_data = row.get('coingecko_data', {})
                if isinstance(coingecko_data, dict):
                    display_df.at[idx, 'Market Cap'] = float(coingecko_data.get('coingecko_market_cap', 0))
                    display_df.at[idx, '24h Volume'] = float(coingecko_data.get('coingecko_total_volume_24h', 0))
                
                # Extract from drift_data if available
                drift_data = row.get('drift_data', {})
                if isinstance(drift_data, dict):
                    display_df.at[idx, 'Drift OI'] = float(drift_data.get('drift_open_interest', 0))
                    display_df.at[idx, 'Max Leverage'] = float(drift_data.get('drift_max_leverage', 0))
                    display_df.at[idx, 'Drift Volume (30d)'] = float(drift_data.get('drift_total_quote_volume_30d', 0))
            
            # Extract raw metrics if available for backup
            for idx, row in df.iterrows():
                raw_metrics = row.get('raw_metrics', {})
                if isinstance(raw_metrics, dict):
                    if display_df.at[idx, 'Market Cap'] == 0:
                        display_df.at[idx, 'Market Cap'] = float(raw_metrics.get('fdv', 0))
                    if display_df.at[idx, '24h Volume'] == 0:
                        display_df.at[idx, '24h Volume'] = float(raw_metrics.get('global_daily_volume', 0))
                    if display_df.at[idx, 'Drift OI'] == 0:
                        display_df.at[idx, 'Drift OI'] = float(raw_metrics.get('drift_open_interest', 0))
            
            # Sort by Total Score descending
            display_df = display_df.sort_values(by='Total Score', ascending=False)
            
            # Calculate component scores
            try:
                display_df['Drift Volume Score'] = df['drift_volume_score']
                display_df['OI Score'] = df['open_interest_score']
                display_df['Global Volume Score'] = df['global_volume_score']
                display_df['FDV Score'] = df['fdv_score']
            except (KeyError, TypeError):
                # Fallback to extracting from the item list
                display_df['Drift Volume Score'] = [item.get('drift_volume_score', 0) for item in data]
                display_df['OI Score'] = [item.get('open_interest_score', 0) for item in data]
                display_df['Global Volume Score'] = [item.get('global_volume_score', 0) for item in data]
                display_df['FDV Score'] = [item.get('fdv_score', 0) for item in data]
            
            # Set index to Symbol for display
            display_df = display_df.set_index('Symbol')
            
            # Define formatters for each column
            formatters = {
                'Market Cap': format_large_number,
                '24h Volume': format_large_number,
                'Drift OI': format_large_number,
                'Drift Volume (30d)': format_large_number,
                'Max Leverage': lambda x: f"{int(x)}x" if pd.notna(x) and x > 0 else "Not Listed",
                'Total Score': "{:.2f}",
                'Drift Volume Score': "{:.2f}",
                'OI Score': "{:.2f}",
                'Global Volume Score': "{:.2f}",
                'FDV Score': "{:.2f}"
            }
            
            # Apply formatting
            st.subheader("Market Recommendations")
            df_styled = display_df.style.format(formatters, na_rep="N/A")
            df_styled = df_styled.apply(lambda x: x.map(style_recommendation), subset=['Recommendation'])
            
            # Display the main table with score and recommendation
            st.dataframe(df_styled, use_container_width=True)
            
            # Expander for detailed view
            with st.expander("View Detailed Metrics"):
                # Show all numeric columns with formatting
                detail_cols = ['Market Cap', '24h Volume', 'Drift OI', 'Drift Volume (30d)', 
                               'Max Leverage', 'Total Score', 
                               'Drift Volume Score', 'OI Score', 'Global Volume Score', 'FDV Score']
                detail_df = display_df[detail_cols]
                detail_styled = detail_df.style.format(formatters, na_rep="N/A")
                st.dataframe(detail_styled, use_container_width=True)
                
            # Raw data expander for debugging
            with st.expander("Raw Data (for debugging)"):
                # Get the first item as a sample
                if data:
                    st.json(data[0])
        
        except Exception as e:
            st.error(f"Error processing data: {e}")
            st.write("Raw data received from API:")
            st.write(data[:5])  # Show first 5 items

    elif data:
        # Handle API error or unexpected format
        st.error("Received data in unexpected format. Here's what was returned:")
        st.write(data)
    else:
        # Handle connection error
        st.warning("Could not retrieve data from the backend. Please ensure it's running and accessible.")

    st.markdown("---")

    # Methodology Section
    st.markdown(
        """
        ### 📊 Methodology

        This page provides recommendations for listing new markets or adjusting leverage on existing markets based on several key metrics:

        **Scoring Components (Total 100 Points):**

        1. **Drift Volume Score (0-25 points)**
           * Based on 30-day trading volume on Drift
           * Higher volume indicates more trader interest and protocol revenue
           * Score tiers: $500M+ (25pts), $100M-$499M (20pts), $25M-$99M (15pts), $1M-$24M (10pts), $100K-$999K (5pts), <$100K (0pts)

        2. **Open Interest Score (0-25 points)**
           * Based on current open interest on Drift perpetual markets
           * Higher OI indicates more capital committed to positions
           * Score tiers: $5M+ (25pts), $1M-$4.9M (20pts), $250K-$999K (15pts), $50K-$249K (10pts), $5K-$49K (5pts), <$5K (0pts)

        3. **Global Volume Score (0-40 points)**
           * Based on 24-hour global trading volume across all exchanges
           * Higher global volume indicates broader market interest and liquidity
           * Score tiers: $500M+ (40pts), $250M-$499M (30pts), $100M-$249M (20pts), $25M-$99M (10pts), $5M-$24M (5pts), <$5M (0pts)

        4. **FDV Score (0-10 points)**
           * Based on Fully Diluted Valuation (or Market Cap if FDV unavailable)
           * Higher market cap indicates larger, more established projects
           * Score tiers: $10B+ (10pts), $1B-$9.9B (8pts), $500M-$999M (6pts), $100M-$499M (2pts), <$100M (0pts)

        **Recommendation Logic:**

        The total score (0-100) is compared against thresholds that vary based on the current maximum leverage:

        * For **unlisted markets**:
          * Score ≥ 45: **List**
          * Score < 45: **Do Nothing**

        * For **listed markets** with leverage = 2x:
          * Score ≤ 40: **Delist**
          * Score > 40: **No Action**

        * For **listed markets** with leverage = 4x:
          * Score ≥ 75: **Increase Leverage**
          * Score ≤ 50: **Decrease Leverage**
          * 50 < Score < 75: **No Action**

        * For **listed markets** with leverage = 5x:
          * Score ≥ 80: **Increase Leverage**
          * Score ≤ 60: **Decrease Leverage**
          * 60 < Score < 80: **No Action**

        * For **listed markets** with leverage = 10x:
          * Score ≥ 90: **Increase Leverage**
          * Score ≤ 70: **Decrease Leverage**
          * 70 < Score < 90: **No Action**

        * For **listed markets** with leverage = 20x:
          * Score ≤ 75: **Decrease Leverage**
          * Score > 75: **No Action**

        **Data Sources:**

        * **Drift Protocol**: Drift OI, trading volume, leverage data
        * **CoinGecko**: Market cap, global trading volume, and other token metrics
        
        **Important Notes:**

        * Recommendations are based solely on quantitative metrics and should be supplemented with qualitative analysis
        * Market conditions can change rapidly, so recommendations should be reviewed regularly
        * New or smaller markets may have limited data, which can affect scoring accuracy
        """
    )

    st.markdown("---")
    st.caption("Data is cached for 10 minutes. Click Refresh to fetch the latest data from the backend.")

    # Add a button to force refresh (clear cache and restart request)
    if st.button("Refresh Data"):
        st.session_state.api_request_in_progress = False
        st.cache_data.clear()
        st.rerun()

# Remove the direct execution
# if __name__ == "__main__":
#     market_recommender_page() 