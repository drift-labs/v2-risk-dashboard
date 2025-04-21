import streamlit as st
import pandas as pd
import numpy as np
from lib.api import fetch_api_data

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
    elif rec == 'Monitor':
        color = 'orange'
    else:
        color = 'grey'
    return f'color: {color}; font-weight: bold;'

@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_list_recommendations():
    """Fetches list recommendations from the backend API."""
    try:
        print("Fetching list recommendations from API") # Debug print
        response = fetch_api_data(
            "list-recommender", 
            "recommendations", 
            retry=True
        )
        print("Successfully fetched data from API.") # Debug print
        return response
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        return None

# --- Streamlit Page ---

def list_recommender_page():
    """
    Streamlit page for displaying listing recommendations for new Drift markets.
    Fetches pre-calculated data from a backend API.
    """
    st.title("📈 List Recommender")
    st.markdown("This page provides recommendations for listing new tokens or adjusting leverage for existing markets on Drift based on various metrics calculated by the backend.")
    st.markdown("THIS PAGE IS IN BETA AND SUBJECT TO CHANGE. USE AT YOUR OWN RISK.")
    
    # Fetch data
    data = fetch_list_recommendations()

    if data and data.get("status") == "success":
        recommendation_data = data.get("data")

        if recommendation_data:
            summary = recommendation_data.get("summary", {})
            results_dict = recommendation_data.get("results", {})
            score_boundaries = recommendation_data.get("score_boundaries", {})

            st.subheader("Analysis Results")

            # Display Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tokens Analyzed", summary.get("total_tokens", 0))
            with col2:
                st.metric("Recommend List", summary.get("list_tokens", 0), delta_color="off")
            with col3:
                st.metric("Recommend Increase Leverage", summary.get("increase_leverage_tokens", 0), delta_color="off")
            with col4:
                st.metric("Recommend Monitor", summary.get("monitor_tokens", 0), delta_color="off")

            # Display top candidates
            top_candidates = summary.get("top_candidates", [])
            if top_candidates:
                st.subheader("Top Listing Candidates")
                st.write(", ".join(top_candidates))

            if results_dict:
                # Convert results to DataFrame
                try:
                    df = pd.DataFrame(results_dict)
                    
                    # Select and order columns for display
                    display_cols = [
                        'Symbol',
                        'Score',
                        'Recommendation',
                        'MC $m',
                        'Spot Volume $m',
                        'Fut Volume $m',
                        'Volume $m',
                        'OI $m',
                        'Slip. $3k',
                        'Slip. $30k'
                    ]
                    # Filter df to only include columns that actually exist
                    display_cols_exist = [col for col in display_cols if col in df.columns]
                    df_display = df[display_cols_exist].copy() # Create a copy for formatting

                    # Define formatters, checking if columns exist
                    formatters = {}
                    if 'MC $m' in df_display.columns: formatters['MC $m'] = format_large_number
                    if 'Spot Volume $m' in df_display.columns: formatters['Spot Volume $m'] = format_large_number
                    if 'Fut Volume $m' in df_display.columns: formatters['Fut Volume $m'] = format_large_number
                    if 'Volume $m' in df_display.columns: formatters['Volume $m'] = format_large_number
                    if 'OI $m' in df_display.columns: formatters['OI $m'] = format_large_number
                    if 'Slip. $3k' in df_display.columns: formatters['Slip. $3k'] = format_percentage
                    if 'Slip. $30k' in df_display.columns: formatters['Slip. $30k'] = format_percentage
                    if 'Score' in df_display.columns: formatters['Score'] = "{:.2f}"

                    # Apply formatting and styling
                    st.subheader("Token Recommendations")
                    df_styled = df_display.style.format(formatters, na_rep="N/A")
                    if 'Recommendation' in df_display.columns:
                         df_styled = df_styled.apply(lambda x: x.map(style_recommendation), subset=['Recommendation'])

                    st.dataframe(df_styled, use_container_width=True)

                    # Expander for full data
                    with st.expander("Show Full Data Table"):
                        # Apply basic formatting to all relevant columns in the original df
                        full_formatters = {
                                'MC $m': format_large_number,
                                'Spot Volume $m': format_large_number,
                                'Spot Vol Geomean $m': format_large_number,
                                'Fut Volume $m': format_large_number,
                                'Fut Vol Geomean $m': format_large_number,
                                'Volume $m': format_large_number,
                                'OI $m': format_large_number,
                                'Slip. $3k': format_percentage,
                                'Slip. $30k': format_percentage
                        }
                        # Add score formatting for all score-related columns
                        score_cols = [col for col in df.columns if 'Score' in col or 'score' in col.lower()]
                        for col in score_cols:
                            full_formatters[col] = "{:.2f}"

                        # Filter out formatters for columns that don't exist or aren't numeric
                        valid_formatters = {}
                        for col, fmt in full_formatters.items():
                             if col in df.columns:
                                 # Attempt conversion to numeric, check if successful for at least one value
                                 is_numeric = pd.to_numeric(df[col], errors='coerce').notna().any()
                                 if is_numeric:
                                     valid_formatters[col] = fmt

                        st.dataframe(df.style.format(valid_formatters, na_rep="N/A"), use_container_width=True)

                    # Display Score Boundaries
                    with st.expander("Score Boundaries"):
                        st.write("Upper score bounds used for recommendations based on current max leverage:")
                        # Format boundaries for better readability
                        formatted_boundaries = {f"{k}x Leverage": v for k, v in score_boundaries.items()}
                        st.json(formatted_boundaries) # Display boundaries as JSON

                except Exception as e:
                    st.error(f"Error processing results data into DataFrame: {e}")
                    st.write("Raw results data received from API:")
                    st.json(results_dict) # Show raw if processing fails

            else:
                st.warning("No detailed results data found in the API response.")

        else:
            st.warning("No data payload found in the API response.")

    elif data:
        # Handle API error status
        st.error(f"API Error: {data.get('message', 'Unknown error')}")
        if data.get('data'):
            st.json(data.get('data')) # Show error details if available
    else:
        # Handle connection error (message already shown by fetch function)
        st.warning("Could not retrieve data from the backend. Please ensure it's running and accessible.")

    st.markdown("---")

    # Methodology Section
    with st.expander("📊 Methodology"):
        st.markdown(
            """
            This section outlines how the listing recommendations are generated, providing transparency into the data sources and calculations involved.

            **Goal:**

            The primary objective of this tool is to identify tokens that could potentially be listed as new perpetual markets on Drift. This is achieved by scoring tokens based on several key liquidity, activity, and market size metrics.

            **Data Sources:**

            The analysis aggregates data from multiple sources:

            1.  **Centralized Exchange (CEX) Data:**
                * **Spot & Futures Volume:** Fetches 30 days of daily OHLCV data from major CEXs (e.g., Binance, Bybit, OKX, Gate, Kucoin, MEXC, Coinbase, Kraken for spot; Bybit, Binance, Gate, MEXC, OKX, HTX, Bitmex for futures).
                * Calculates the *average daily USD volume* across these exchanges for both spot and futures markets.
                * Calculates the *geometric mean* of the top 3 exchange volumes for both spot and futures to reward tokens with liquidity distributed across multiple venues.
            2.  **CoinMarketCap (CMC) Data:**
                * **Market Capitalization (MC):** Fetches the current market cap (or Fully Diluted Valuation if market cap is unavailable) from the CoinMarketCap API.
            3.  **Thunderhead Data (Market Metrics):**
                * **Volume:** Estimated 30-day trading volume for the token
                * **Open Interest (OI):** Current open interest for the token
                * **Slippage:** Slippage metrics for $3k and $30k trades, giving insight into market depth

            **Scoring (Total 80 Points + Boost):**

            Tokens are scored out of a potential 80 points based on the following categories. Scoring uses exponential ranges, meaning higher values generally receive diminishing point returns:

            1.  **Market Cap Score (20 Points):**
                * Assesses the overall size and significance of the token.
                * Based on Market Cap (MC).
                * Range: 0 points for less than $1M MC, up to 20 points for >= $5B MC.
            2.  **Spot Volume Score (20 Points):**
                * Measures the liquidity of the token's spot markets on major CEXs.
                * Metrics Used:
                    * Sum of Average Daily Spot Volume (0-10 points): Range from < $10k/day to >= $1B/day.
                    * Geometric Mean of Top 3 Average Daily Spot Volumes (0-10 points): Range from < $10k/day to >= $1B/day.
            3.  **Futures Volume Score (20 Points):**
                * Measures the liquidity of the token's futures markets on major CEXs.
                * Metrics Used:
                    * Sum of Average Daily Futures Volume (0-10 points): Range from < $10k/day to >= $1B/day.
                    * Geometric Mean of Top 3 Average Daily Futures Volumes (0-10 points): Range from < $10k/day to >= $1B/day.
            4.  **Activity Score (20 Points):**
                * Measures how actively the token is traded.
                * Metrics Used:
                    * 30-Day Trading Volume (0-10 points): Range from < $1k to >= $1B.
                    * Open Interest (OI) (0-10 points): Range from < $1k to >= $1B.
            5.  **Liquidity Score (10 Points):**
                * Measures market depth and quality.
                * Metrics Used:
                    * Slippage for $3k trade (0-5 points): Range from 5% to 0%.
                    * Slippage for $30k trade (0-5 points): Range from 50% to 0%.

            **Score Boost (5 Points):**

            * Tokens listed in `STRICT_TOKENS` receive a +5 point boost to their final score.

            **Recommendation Logic:**

            The final recommendation ('List', 'Increase Leverage', 'Monitor') is determined by comparing the token's **Total Score** against dynamic upper bounds (`SCORE_UB`) that depend on the token's **Current Maximum Leverage** (which is 0 for unlisted tokens):

            * **Score Upper Bounds (`SCORE_UB`):**
                * Leverage 0x: Upper Bound = 62 points
                * Leverage 3x: Upper Bound = 75 points
                * Leverage 5x: Upper Bound = 85 points
                * Leverage 10x: Upper Bound = 101 points

            * **Decision Process:**
                1.  For unlisted tokens (Max Lev = 0), if `Total Score >= 62`, recommend **List**.
                2.  For existing markets, if `Total Score >= Applicable Upper Bound`, recommend **Increase Leverage**.
                3.  Otherwise, recommend **Monitor**.

            This approach ensures that tokens must meet a high quality threshold before being recommended for listing or leverage increases.
            """
        )

    st.markdown("---")
    st.caption("Data is cached for 10 minutes. Click Refresh to fetch the latest data from the backend.")

    # Add a button to force refresh
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Call the page function
if __name__ == "__main__":
    list_recommender_page() 