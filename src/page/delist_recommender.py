import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get backend URL from environment variable or use default
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_ENDPOINT = f"{BACKEND_URL}/api/delist-recommender/recommendations"

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
    if rec == 'Delist':
        color = 'red'
    elif rec == 'Decrease Leverage':
        color = 'orange'
    elif rec == 'Keep':
        color = 'green'
    else:
        color = 'grey'
    return f'color: {color}; font-weight: bold;'

@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_delist_recommendations():
    """Fetches delist recommendations from the backend API."""
    try:
        print(f"Fetching data from: {API_ENDPOINT}") # Debug print
        response = requests.get(API_ENDPOINT, timeout=120) # Increased timeout to 120s
        response.raise_for_status() # Raise an exception for bad status codes
        print("Successfully fetched data from API.") # Debug print
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Error: Timeout connecting to backend API at {API_ENDPOINT}. The request took longer than 120 seconds.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to backend API at {API_ENDPOINT}. Please ensure the backend is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching data: {e}")
        return None

# --- Streamlit Page ---

def delist_recommender_page():
    """
    Streamlit page for displaying delisting recommendations for Drift markets.
    Fetches pre-calculated data from a backend API.
    """
    st.title("📉 Delist Recommender")
    st.markdown("This page provides recommendations for delisting or adjusting leverage for Drift perpetual markets based on various metrics calculated by the backend.")

    # Fetch data
    data = fetch_delist_recommendations()

    if data and data.get("status") == "success":
        recommendation_data = data.get("data")

        if recommendation_data:
            slot = recommendation_data.get("slot", "N/A")
            summary = recommendation_data.get("summary", {})
            results_dict = recommendation_data.get("results", {})
            score_boundaries = recommendation_data.get("score_boundaries", {})

            st.subheader(f"Analysis Results (Slot: {slot})")

            # Display Summary Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Markets Analyzed", summary.get("total_markets", 0))
            with col2:
                st.metric("Recommend Keep", summary.get("keep_markets", 0), delta_color="off")
            with col3:
                st.metric("Recommend Decrease Leverage", summary.get("decrease_leverage_markets", 0), delta_color="off")
            with col4:
                st.metric("Recommend Delist", summary.get("delist_markets", 0), delta_color="off")

            if results_dict:
                # Convert results to DataFrame
                try:
                    # The 'index' key contains the symbols in the sample response
                    if 'index' in results_dict and 'Symbol' not in results_dict:
                         # If 'index' exists and 'Symbol' doesn't, use 'index' for symbols
                         df = pd.DataFrame(results_dict)
                         df = df.rename(columns={'index': 'Symbol'})
                    elif 'Symbol' in results_dict:
                         # If 'Symbol' key already exists
                         df = pd.DataFrame(results_dict)
                    else:
                         # Fallback if neither 'index' nor 'Symbol' is present
                         st.error("Could not find symbol information ('index' or 'Symbol' key) in API results.")
                         df = pd.DataFrame(results_dict) # Try creating DF anyway

                    # Use Market Index as index if Symbol isn't suitable or missing
                    if 'Market Index' in df.columns and 'Symbol' not in df.columns:
                         df = df.set_index('Market Index')
                    elif 'Symbol' in df.columns:
                         df = df.set_index('Symbol')

                    # Ensure correct sorting by Market Index if available
                    if 'Market Index' in df.columns:
                         df = df.sort_values(by='Market Index')

                    # Select and order columns for display
                    display_cols = [
                        'Market Index',
                        'Max Lev. on Drift',
                        'OI on Drift',
                        'Volume on Drift', # Estimated 30d Drift Volume
                        'Spot Volume', # Sum of Avg Daily CEX Spot Volume
                        'Fut Volume', # Sum of Avg Daily CEX Futures Volume
                        'MC', # Market Cap
                        'Score',
                        'Recommendation'
                    ]
                    # Filter df to only include columns that actually exist
                    display_cols_exist = [col for col in display_cols if col in df.columns]
                    df_display = df[display_cols_exist].copy() # Create a copy for formatting

                    # Define formatters, checking if columns exist
                    formatters = {}
                    if 'OI on Drift' in df_display.columns: formatters['OI on Drift'] = format_large_number
                    if 'Volume on Drift' in df_display.columns: formatters['Volume on Drift'] = format_large_number
                    if 'Spot Volume' in df_display.columns: formatters['Spot Volume'] = format_large_number
                    if 'Fut Volume' in df_display.columns: formatters['Fut Volume'] = format_large_number
                    if 'MC' in df_display.columns: formatters['MC'] = format_large_number
                    if 'Score' in df_display.columns: formatters['Score'] = "{:.2f}"
                    if 'Max Lev. on Drift' in df_display.columns:
                         formatters['Max Lev. on Drift'] = lambda x: f"{int(x)}x" if pd.notna(x) else "N/A"
                    if 'Market Index' in df_display.columns:
                          formatters['Market Index'] = lambda x: f"{int(x)}" if pd.notna(x) else "N/A"


                    # Apply formatting and styling
                    st.subheader("Market Recommendations")
                    df_styled = df_display.style.format(formatters, na_rep="N/A")
                    if 'Recommendation' in df_display.columns:
                         df_styled = df_styled.apply(lambda x: x.map(style_recommendation), subset=['Recommendation'])

                    st.dataframe(df_styled, use_container_width=True)

                    # Expander for full data
                    with st.expander("Show Full Data Table"):
                        # Apply basic formatting to all relevant columns in the original df
                        full_formatters = {
                                'OI on Drift': format_large_number,
                                'Volume on Drift': format_large_number,
                                'Spot Volume': format_large_number,
                                'Spot Vol Geomean': format_large_number,
                                'Fut Volume': format_large_number,
                                'Fut Vol Geomean': format_large_number,
                                'MC': format_large_number,
                                'Oracle Price': lambda x: f"${x:,.6f}" if pd.notna(x) else "N/A",
                                'Funding Rate % (1h)': format_percentage,
                                'Max Lev. on Drift': lambda x: f"{int(x)}x" if pd.notna(x) else "N/A",
                                'Market Index': lambda x: f"{int(x)}" if pd.notna(x) else "N/A",
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
                        st.write("Lower score bounds used for recommendations based on current max leverage:")
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
    st.markdown(
        """
        ### 📊 Methodology

        This section outlines how the delisting recommendations are generated, providing transparency into the data sources and calculations involved.

        **Goal:**

        The primary objective of this tool is to identify perpetual markets on Drift that may exhibit characteristics suggesting a need for review, potentially leading to leverage reduction or delisting. This is achieved by scoring markets based on several key liquidity, activity, and market size metrics.

        **Data Sources:**

        The analysis aggregates data from multiple sources:

        1.  **Drift Protocol Data:**
            * **Open Interest (OI):** Calculated by summing the absolute value of long and short positions held by users in each market, converted to USD using the current oracle price. This reflects the total value locked in open contracts on Drift.
            * **Trading Volume:** The total USD value of trades executed in the market over the past 30 days, fetched directly from the Drift API.
            * **Market Configuration:** Current maximum leverage allowed for each market.
            * **Oracle Price:** The current price feed used by Drift for the market.
        2.  **Centralized Exchange (CEX) Data:**
            * **Spot & Futures Volume:** Fetches 30 days of daily OHLCV data from major CEXs (e.g., Coinbase, OKX, Gate, Kucoin, MEXC, Kraken for spot; OKX, Gate, MEXC, HTX, BitMEX for futures) using the `ccxt` library.
            * Calculates the *average daily USD volume* across these exchanges for both spot and futures markets.
            * Calculates the *geometric mean* of the top 3 exchange volumes for both spot and futures to reward markets with liquidity distributed across multiple venues.
        3.  **CoinMarketCap (CMC) Data:**
            * **Market Capitalization (MC):** Fetches the current market cap (or Fully Diluted Valuation if market cap is unavailable) from the CoinMarketCap API.

        **Scoring (Total 80 Points + Boost):**

        Markets are scored out of a potential 80 points based on the following categories. Scoring uses exponential ranges, meaning higher values generally receive diminishing point returns. All input metrics (Volume, OI, Market Cap) are processed in their full dollar amounts.

        1.  **Market Cap Score (20 Points):**
            * Assesses the overall size and significance of the asset.
            * Based on CMC Market Cap (MC).
            * Range: 0 points for less than $1M MC, up to 20 points for >= $5B MC.
        2.  **Spot Volume Score (20 Points):**
            * Measures the liquidity of the asset's spot markets on major CEXs. Higher spot volume generally indicates better price stability and oracle reliability.
            * Metrics Used:
                * Sum of Average Daily Spot Volume (0-10 points): Range from < $10k/day to >= $1B/day.
                * Geometric Mean of Top 3 Average Daily Spot Volumes (0-10 points): Range from < $10k/day to >= $1B/day.
        3.  **Futures Volume Score (20 Points):**
            * Measures the liquidity of the asset's futures markets on major CEXs. Relevant for hedging and price discovery.
            * Metrics Used:
                * Sum of Average Daily Futures Volume (0-10 points): Range from < $10k/day to >= $1B/day.
                * Geometric Mean of Top 3 Average Daily Futures Volumes (0-10 points): Range from < $10k/day to >= $1B/day.
        4.  **Drift Activity Score (20 Points):**
            * Measures how actively the market is traded specifically on Drift.
            * Metrics Used:
                * 30-Day Drift Trading Volume (0-10 points): Range from < $1k to >= $500M.
                * Drift Open Interest (OI) (0-10 points): Range from < $1k to >= $500M.

        **Score Boost (Up to 10 Points):**

        * Markets listed in `DRIFT_SCORE_BOOST_SYMBOLS` (currently: `DRIFT-PERP`) receive a +10 point boost to their final score.

        **Recommendation Logic:**

        The final recommendation ('Keep', 'Decrease Leverage', 'Delist') is determined by comparing the market's **Total Score** against dynamic lower bounds (`SCORE_LB`) that depend on the market's **Current Maximum Leverage** on Drift:

        * **Score Lower Bounds (`SCORE_LB`):**
            * Leverage <= 5x: Lower Bound = 31 points
            * Leverage <= 10x: Lower Bound = 48 points
            * Leverage <= 20x: Lower Bound = 60 points
            * *(Note: Leverage 0x has a lower bound of 0)*

        * **Decision Process:**
            1.  Determine the applicable lower bound based on the market's current max leverage.
            2.  If `Total Score < Applicable Lower Bound`:
                * If `Current Max Leverage > 5x`: Recommend **Decrease Leverage**.
                * If `Current Max Leverage <= 5x`: Recommend **Delist**.
            3.  If `Total Score >= Applicable Lower Bound`: Recommend **Keep**.

        This approach aims to be more conservative with lower-leverage markets, requiring a higher relative score to maintain their status, while allowing higher-leverage markets more leeway before recommending a decrease.

        **Important Notes:**

        * This tool provides recommendations based on quantitative data. Qualitative factors (e.g., project roadmap, team, regulatory concerns) are not considered.
        * Data fetching from external APIs (CEXs, CMC) can occasionally fail or be incomplete, which might affect scores. The system attempts fallbacks where possible but defaults to zero values if data is unavailable.
        * Prediction market symbols (e.g., `TRUMP-WIN-2024-BET`) are explicitly excluded from the analysis.
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
    delist_recommender_page()