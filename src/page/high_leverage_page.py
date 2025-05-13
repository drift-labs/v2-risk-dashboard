import streamlit as st
import pandas as pd
import time
from lib.api import fetch_api_data
from utils import get_current_slot

# Retry delay can be adjusted or removed if backend processing isn't expected
RETRY_DELAY_SECONDS = 5 

def is_processing(result):
    """Checks if the API result indicates backend processing."""
    return isinstance(result, dict) and result.get("result") == "processing"

def has_error(result):
    """Checks if the API result indicates an error."""
    return result is None or (isinstance(result, dict) and result.get("result") == "error")

def high_leverage_page():
    st.title("High Leverage Usage on Drift")

    try:
        # --- 1. Fetch all data --- 
        result_stats = fetch_api_data(
            section="high-leverage", 
            path="stats",
            retry=False # Let the page handle retries via rerun
        )
        result_positions = fetch_api_data(
            section="high-leverage",
            path="positions/detailed",
            retry=False
        )

        # --- 2. Check for processing state (If backend uses caching/async) --- 
        # Simplified: If either endpoint is processing, wait and retry.
        # You might remove this if your high-leverage API is purely synchronous.
        if is_processing(result_stats) or is_processing(result_positions):
            st.info(f"Backend is processing high leverage data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds...")
            with st.spinner("Please wait..."):
                time.sleep(RETRY_DELAY_SECONDS)
            st.rerun() 
            return

        # --- 3. Check for errors --- 
        # Stats are essential for metrics
        if has_error(result_stats):
            error_msg = result_stats['message'] if isinstance(result_stats, dict) else "Could not connect or fetch stats data."
            st.error(f"Failed to fetch essential data (High Leverage Stats): {error_msg}")
            if st.button("Retry Fetch Stats"):
                st.rerun()
            return

        # Detailed positions are useful but not strictly essential for basic metrics
        if has_error(result_positions):
            st.warning("Could not fetch Detailed High Leverage Positions data. The position table will be empty.")
            # Proceed without detailed data

        # --- 4. Extract data and slot (if all checks passed) --- 
        # We know result_stats is valid here
        stats_data = result_stats # The stats endpoint returns the data directly
        slot = stats_data.get("slot", "N/A") # Assuming backend adds slot to stats response
        
        # Extract detailed positions data, defaulting to empty list if fetch failed
        positions_data = result_positions if isinstance(result_positions, list) else []
        # If the API wraps list in a dict like {"data": [...]}, adjust accordingly:
        # positions_data = result_positions.get("data", []) if isinstance(result_positions, dict) else []

        # --- 5. Display Slot Info --- 
        current_slot = get_current_slot()
        if slot != "N/A" and current_slot:
            try:
                slot_age = int(current_slot) - int(slot)
                st.info(f"Displaying data for slot {slot} (age: {slot_age} slots)")
            except (ValueError, TypeError):
                st.info(f"Displaying data for slot {slot}. Current slot: {current_slot}")
        else:
            st.info(f"Slot information unavailable. Current slot: {current_slot}")

        # --- 6. Prepare Positions DataFrame --- 
        df_positions = pd.DataFrame()
        if positions_data:
            try:
                df_positions = pd.DataFrame(positions_data)
                if not df_positions.empty:
                    df_positions.rename(columns={
                        'user_public_key': 'User Account',
                        'authority': 'Authority',
                        'market_index': 'Market Index',
                        'market_symbol': 'Market Symbol',
                        'base_asset_amount_ui': 'Base Asset Amount',
                        'position_value_usd': 'Notional Value (USD)',
                        'account_leverage': 'Account Leverage',
                        'position_leverage': 'Position Leverage'
                    }, inplace=True)
                    
                    # Ensure leverage columns are numeric for sorting
                    df_positions['Position Leverage'] = pd.to_numeric(df_positions['Position Leverage'], errors='coerce')
                    df_positions['Account Leverage'] = pd.to_numeric(df_positions['Account Leverage'], errors='coerce')

                    # Default sort by Position Leverage (descending)
                    df_positions = df_positions.sort_values(by='Position Leverage', ascending=False)
                    
                    # Select and order columns for display
                    display_df = df_positions[[
                        'Market Symbol',
                        'Base Asset Amount',
                        'Notional Value (USD)',
                        'Position Leverage',
                        'Account Leverage',
                        'User Account',
                        'Authority',
                        'Market Index' 
                    ]].copy() # Use .copy() to avoid SettingWithCopyWarning on the formatted DataFrame
                    
                    # Formatting for display
                    # Base Asset Amount and Notional Value formatting remains the same
                    display_df['Base Asset Amount'] = display_df['Base Asset Amount'].apply(lambda x: f"{x:,.4f}" if pd.notnull(x) else 'N/A')
                    display_df['Notional Value (USD)'] = display_df['Notional Value (USD)'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else 'N/A')
                    # Leverage columns formatted for display with 'x' - keep original df_positions for numerical sorting
                    display_df['Position Leverage'] = display_df['Position Leverage'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else 'N/A')
                    display_df['Account Leverage'] = display_df['Account Leverage'].apply(lambda x: f"{x:.2f}x" if pd.notnull(x) else 'N/A')

            except Exception as df_e:
                st.error(f"Error processing detailed position data into DataFrame: {df_e}")
                df_positions = pd.DataFrame() # Ensure it's empty on error
                display_df = pd.DataFrame() # Also ensure display_df is empty
        else:
            display_df = pd.DataFrame() # Ensure display_df is initialized if positions_data is empty

        # --- 7. Display Layout (Metrics and DataFrame) --- 
        st.subheader("High Leverage Usage Stats")
        cols = st.columns(4)
        cols[0].metric("Total High Leverage Spots", stats_data.get('total_spots', 'N/A'))
        cols[1].metric("Opted-In Users", stats_data.get('opted_in_spots', 'N/A'))
        cols[2].metric("Available Spots", stats_data.get('available_spots', 'N/A'))
        cols[3].metric("Bootable Spots (Opted-in, No Position)", stats_data.get('bootable_spots', 'N/A'))

        st.subheader("Detailed High Leverage Positions")
        if not display_df.empty:
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info("No detailed high leverage position data available, or an error occurred during processing.")

    except Exception as e:
        st.error(f"An error occurred while displaying the page: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    high_leverage_page()
