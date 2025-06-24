import streamlit as st
import pandas as pd
import time # Added import
from lib.api import fetch_api_data
from utils import get_current_slot

RETRY_DELAY_SECONDS = 5 # Define delay for auto-refresh

def is_processing(result):
    """Checks if the API result indicates backend processing."""
    # Check if result is a dictionary and has the specific processing structure
    return isinstance(result, dict) and result.get("result") == "processing"

def has_error(result):
    """Checks if the API result indicates an error."""
    # Check if result is None or a dictionary with an error message
    return result is None or (isinstance(result, dict) and result.get("result") == "error")

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_market_list():
    """Fetches the list of available markets for the dropdown."""
    return fetch_api_data(section="open-interest", path="markets", params={"bypass_cache": "true"})

def open_interest_page():
    st.title("Open Interest on Drift")

    market_list = get_market_list()
    if not isinstance(market_list, list) or not market_list:
        st.error("Could not fetch the list of markets. Defaulting to 'All'.")
        market_list = ["All"]

    selected_market = st.selectbox(
        "Select Market",
        options=market_list,
        index=0, # Default to "All"
        key='oi_market'
    )

    try:
        # --- 1. Fetch all data --- 
        api_params = {}
        if selected_market and selected_market != "All":
            api_params['market_name'] = selected_market

        result_authority = fetch_api_data(
            section="open-interest", 
            path="per-authority",
            params=api_params,
            retry=False # Let the page handle retries via rerun
        )
        result_account = fetch_api_data(
            section="open-interest",
            path="per-account",
            params=api_params,
            retry=False
        )
        result_detailed = fetch_api_data(
            section="open-interest",
            path="detailed-positions",
            params=api_params,
            retry=False
        )

        # --- 2. Check for processing state --- 
        if is_processing(result_authority) or is_processing(result_account) or is_processing(result_detailed):
            st.info(f"Backend is processing open interest data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds...")
            # Optionally add more details about which endpoint is processing if needed
            # e.g., if is_processing(result_authority): st.caption("Authority data processing...")
            with st.spinner("Please wait..."):
                time.sleep(RETRY_DELAY_SECONDS)
            st.rerun() # Rerun the script to fetch again
            return # Stop further execution in this run

        # --- 3. Check for errors --- 
        # Check critical data source first (e.g., authority for total OI metric)
        if has_error(result_authority):
            error_msg = result_authority['message'] if isinstance(result_authority, dict) else "Could not connect or fetch data."
            st.error(f"Failed to fetch essential data (OI by Authority): {error_msg}")
            if st.button("Retry Fetch"): 
                # Optionally clear streamlit cache if fetch_api_data uses it
                # st.cache_data.clear()
                st.rerun()
            return # Stop execution

        # Handle potential errors in other non-critical fetches (optional: display warnings instead of stopping)
        if has_error(result_account):
            st.warning("Could not fetch OI by Account data. This table will be empty.")
            # Proceed without account data
        
        if has_error(result_detailed):
            st.warning("Could not fetch Detailed Open Positions data. This table will be empty.")
            # Proceed without detailed data

        # --- 4. Extract data and slot (if all checks passed) --- 
        # We know result_authority is valid here
        data_authority = result_authority.get("data", [])
        slot = result_authority.get("slot", "N/A")
        
        # Extract data for others, defaulting to empty list if fetch failed (handled by warnings above)
        data_account = result_account.get("data", []) if isinstance(result_account, dict) else []
        data_detailed = result_detailed.get("data", []) if isinstance(result_detailed, dict) else []

        # --- 5. Display Slot Info --- 
        current_slot = get_current_slot()
        if slot != "N/A" and current_slot:
            try:
                slot_age = int(current_slot) - int(slot)
                st.info(f"Displaying data for slot {slot} (age: {slot_age} slots)")
            except ValueError:
                st.info(f"Displaying data for slot {slot}. Current slot: {current_slot}")
        else:
            st.info(f"Slot information unavailable. Current slot: {current_slot}")

        # --- 6. Process and Prepare DataFrames --- 
        df_authority = pd.DataFrame() 
        if data_authority:
            df_authority = pd.DataFrame(data_authority)
            if not df_authority.empty:
                df_authority.rename(columns={
                    'authority': 'User Authority',
                    'total_open_interest_usd': 'Total Open Interest (USD)'
                }, inplace=True)
                # Keep original numeric column for calculation, create a new one for display
                df_authority["Total Open Interest (USD) Display"] = df_authority["Total Open Interest (USD)"].apply(lambda x: f"${x:,.2f}")
        
        df_account = pd.DataFrame()
        if data_account:
            df_account = pd.DataFrame(data_account)
            if not df_account.empty:
                df_account.rename(columns={
                    'user_public_key': 'User Account',
                    'authority': 'Authority',
                    'total_open_interest_usd': 'Total Open Interest (USD)'
                }, inplace=True)
                # Create a new column for display
                df_account["Total Open Interest (USD) Display"] = df_account["Total Open Interest (USD)"].apply(lambda x: f"${x:,.2f}")

        df_detailed = pd.DataFrame()
        if data_detailed:
            df_detailed = pd.DataFrame(data_detailed)
            if not df_detailed.empty:
                df_detailed.rename(columns={
                    'market_index': 'Market Index',
                    'market_symbol': 'Market Symbol',
                    'base_asset_amount_ui': 'Base Asset Amount',
                    'position_value_usd': 'Notional Value (USD)',
                    'user_public_key': 'User Account',
                    'authority': 'Authority'
                }, inplace=True)
                # Create new columns for display
                df_detailed['Base Asset Amount Display'] = df_detailed['Base Asset Amount'].apply(lambda x: f"{x:,.4f}")
                df_detailed['Notional Value (USD) Display'] = df_detailed['Notional Value (USD)'].apply(lambda x: f"${x:,.2f}")

        # --- 7. Display Layout (Metrics and DataFrames) --- 
        total_oi_usd = 0.0
        total_long_oi_usd = 0.0
        total_short_oi_usd = 0.0
        if not df_authority.empty:
             try:
                # Calculate metric from the original numeric column
                total_oi_usd = df_authority['Total Open Interest (USD)'].sum()
                total_long_oi_usd = df_authority['long_oi_usd'].sum()
                total_short_oi_usd = df_authority['short_oi_usd'].sum()
             except Exception as calc_e:
                 st.warning(f"Could not calculate Total OI metric: {calc_e}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Authorities with Open Interest", len(df_authority) if not df_authority.empty else 0)
        with col2:
            st.metric("Total Open Interest (USD)", f"${total_oi_usd:,.2f}")
        with col3:
            st.metric("Total Long OI (USD)", f"${total_long_oi_usd:,.2f}")
        with col4:
            st.metric("Total Short OI (USD)", f"${total_short_oi_usd:,.2f}")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("OI by Authority")
            if not df_authority.empty:
                # Display only the relevant columns
                df_authority_display = df_authority[["User Authority", "Total Open Interest (USD) Display"]]
                df_authority_display.columns = ["User Authority", "Total Open Interest (USD)"]
                st.dataframe(df_authority_display, hide_index=True)
            else:
                st.info("Authority data not available or empty.")

        with col4:
            st.subheader("OI by Account")
            if not df_account.empty:
                # Display only the relevant columns
                df_account_display = df_account[['User Account', 'Authority', 'Total Open Interest (USD) Display']]
                df_account_display.columns = ['User Account', 'Authority', 'Total Open Interest (USD)']
                st.dataframe(df_account_display, hide_index=True)
            else:
                st.info("Account data not available or empty.")

        st.subheader("Detailed Open Positions")
        if not df_detailed.empty:
            # Display only the relevant columns
            df_detailed_display = df_detailed[[
                'Market Index', 'Market Symbol', 'Base Asset Amount Display', 'Notional Value (USD) Display',
                'User Account', 'Authority'
            ]]
            df_detailed_display.columns = [
                'Market Index', 'Market Symbol', 'Base Asset Amount', 'Notional Value (USD)',
                'User Account', 'Authority'
            ]
            st.dataframe(df_detailed_display, hide_index=True, use_container_width=True)
        else:
            st.info("Detailed position data not available or empty.")

    except Exception as e:
        st.error(f"An error occurred while displaying the page: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    open_interest_page()
