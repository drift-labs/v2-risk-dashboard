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
    st.caption("Data is cached for 10 minutes. Click Refresh to fetch the latest data from the backend.")

    # Add a button to force refresh
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Call the page function
if __name__ == "__main__":
    delist_recommender_page()