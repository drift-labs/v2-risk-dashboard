# This page will display the user retention data in a table format.
# It will receive the data from the user_retention endpoint from the backend
# It will also allow the user to filter the data by market and date range.
# It will also allow the user to download the data in a CSV format.
# It will also allow the user to export the data to a JSON file.

import streamlit as st
import pandas as pd
import json # For JSON export
import time # For potential retry delays
from lib.api import fetch_api_data # Assuming this is your helper for API calls

RETRY_DELAY_SECONDS = 5 # For retrying if API is processing

# Helper function to check if the API result indicates backend processing
def is_processing(result):
    return isinstance(result, dict) and result.get("result") == "processing"

# Helper function to check for errors in API result
def has_error(result):
    if result is None:
        return True
    if isinstance(result, dict):
        # Check for a specific error structure if your API has one
        # For example, if errors are always like {"detail": "error message"}
        if "detail" in result and isinstance(result["detail"], str):
            return True 
        # Or a more generic check if your API returns a specific error key or status
        if result.get("result") == "error" or result.get("status") == "error":
            return True
    return False

# Helper to get error message
def get_error_message(result):
    if isinstance(result, dict) and "detail" in result:
        return str(result["detail"])
    if isinstance(result, dict) and "message" in result:
        return str(result["message"])
    return "Could not connect or fetch data from the backend."

@st.cache_data(ttl=300) # Cache data for 5 minutes
def load_retention_data():
    """Fetches user retention summary data from the backend API."""
    # Adjust 'section' and 'path' according to your fetch_api_data implementation
    # and how the user_retention_api.router is mounted in your FastAPI app.
    # Assuming the new endpoint is mounted under a 'user-retention' section similar to 'high-leverage'
    data = fetch_api_data(section="user-retention", path="summary", retry=False)
    return data

def user_retention_page():
    st.title("User Retention Analysis for Hype Markets")

    st.markdown("""
    This page displays user retention data for specified "hype" markets. 
    It shows how many new traders (first-ever order in the market within 7 days of launch) 
    were retained by trading in *any other* market within 14 and 28 days.
    """)

    # Initial data load
    data = load_retention_data()

    # Handle processing state from API (if applicable)
    if is_processing(data):
        st.info(f"Backend is processing user retention data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds...")
        with st.spinner("Please wait..."):
            time.sleep(RETRY_DELAY_SECONDS)
        st.rerun()
        return

    # Handle errors from API
    if has_error(data):
        error_msg = get_error_message(data)
        st.error(f"Failed to fetch user retention data: {error_msg}")
        if st.button("Retry Data Load"):
            st.cache_data.clear() # Clear cache before retrying
            st.rerun()
        return
    
    # If data is a list (expected for summary items)
    if isinstance(data, list) and data:
        try:
            df = pd.DataFrame(data)

            # Ensure correct data types, especially for numeric columns that might be None
            df['new_traders'] = pd.to_numeric(df['new_traders'], errors='coerce').fillna(0).astype(int)
            df['retained_users_14d'] = pd.to_numeric(df['retained_users_14d'], errors='coerce').fillna(0).astype(int)
            df['retention_ratio_14d'] = pd.to_numeric(df['retention_ratio_14d'], errors='coerce').fillna(0.0).astype(float)
            df['retained_users_28d'] = pd.to_numeric(df['retained_users_28d'], errors='coerce').fillna(0).astype(int)
            df['retention_ratio_28d'] = pd.to_numeric(df['retention_ratio_28d'], errors='coerce').fillna(0.0).astype(float)

            # Ensure list columns are treated as objects and handle potential NaNs (e.g. if API returns null instead of [])
            list_columns = ['new_traders_list', 'retained_users_14d_list', 'retained_users_28d_list']
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
                else: # If a list column is missing from API response for some reason
                    df[col] = pd.Series([[] for _ in range(len(df))], dtype='object')

            # Rename columns for better display
            df.rename(columns={
                'market': 'Market',
                'new_traders': 'New Traders (Count)',
                'retained_users_14d': 'Retained at 14d (Count)',
                'retention_ratio_14d': 'Retention Rate 14d',
                'retained_users_28d': 'Retained at 28d (Count)',
                'retention_ratio_28d': 'Retention Rate 28d',
                'new_traders_list': 'New Traders (List)',
                'retained_users_14d_list': 'Retained 14d (List)',
                'retained_users_28d_list': 'Retained 28d (List)'
            }, inplace=True)
            
            # --- Filtering (Applied to the original comprehensive df) ---
            st.sidebar.header("Filters")
            all_markets = ["All"] + sorted(df["Market"].unique().tolist())
            selected_market = st.sidebar.selectbox("Filter by Market", all_markets)

            filtered_df = df.copy()
            if selected_market != "All":
                filtered_df = filtered_df[filtered_df["Market"] == selected_market]
            
            # (Optional: Date range filter if your data/API supports it. Not implemented based on current API)
            # selected_date_range = st.sidebar.date_input("Filter by Date Range (if applicable)", [])
            # if selected_date_range and len(selected_date_range) == 2:
            #     start_date, end_date = selected_date_range
            #     # Add logic to filter df by date if a date column exists
            #     pass 

            # --- Display Tables ---
            if filtered_df.empty:
                st.info("No data available for the selected filters.")
            else:
                # Table 1: Main Summary
                st.subheader("Overall Retention Summary")
                summary_cols = [
                    'Market',
                    'New Traders (Count)',
                    'Retained at 14d (Count)',
                    'Retention Rate 14d',
                    'Retained at 28d (Count)',
                    'Retention Rate 28d'
                ]
                main_summary_df = filtered_df[summary_cols]
                st.dataframe(
                    main_summary_df.style.format({
                        'Retention Rate 14d': '{:.2%}',
                        'Retention Rate 28d': '{:.2%}'
                    }), 
                    hide_index=True, 
                    use_container_width=True
                )

                # Table 2: New Traders List
                st.subheader("New Traders Lists")
                new_traders_list_df = filtered_df[['Market', 'New Traders (List)']]
                # Filter out rows where the list is empty to make the table cleaner
                new_traders_list_df = new_traders_list_df[new_traders_list_df['New Traders (List)'].apply(lambda x: len(x) > 0)]
                if not new_traders_list_df.empty:
                    st.dataframe(new_traders_list_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No new traders found for the selected market(s) or new trader lists are empty.")

                # Table 3: 14-Day Retained Users List
                st.subheader("14-Day Retained User Lists")
                retained_14d_list_df = filtered_df[['Market', 'Retained 14d (List)']]
                retained_14d_list_df = retained_14d_list_df[retained_14d_list_df['Retained 14d (List)'].apply(lambda x: len(x) > 0)]
                if not retained_14d_list_df.empty:
                    st.dataframe(retained_14d_list_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No users retained at 14 days for the selected market(s) or lists are empty.")

                # Table 4: 28-Day Retained Users List
                st.subheader("28-Day Retained User Lists")
                retained_28d_list_df = filtered_df[['Market', 'Retained 28d (List)']]
                retained_28d_list_df = retained_28d_list_df[retained_28d_list_df['Retained 28d (List)'].apply(lambda x: len(x) > 0)]
                if not retained_28d_list_df.empty:
                    st.dataframe(retained_28d_list_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No users retained at 28 days for the selected market(s) or lists are empty.")

                # --- Data Export (based on the main summary table) --- 
                st.subheader("Export Main Summary Data")
                col1, col2 = st.columns(2)

                # CSV Download
                csv_data = main_summary_df.to_csv(index=False).encode('utf-8')
                col1.download_button(
                    label="Download Summary as CSV",
                    data=csv_data,
                    file_name="user_retention_main_summary.csv",
                    mime="text/csv",
                )

                # JSON Export/Display
                json_data_str = main_summary_df.to_json(orient="records", indent=4)
                col2.download_button(
                    label="Download Summary as JSON",
                    data=json_data_str,
                    file_name="user_retention_main_summary.json",
                    mime="application/json",
                )
                with st.expander("View Main Summary JSON Data"):
                    st.json(json_data_str)
        
        except Exception as e:
            st.error(f"An error occurred while processing and displaying the data: {e}")
            import traceback
            st.text(traceback.format_exc())

    elif isinstance(data, list) and not data: # API returned empty list
        st.info("No user retention data found. The backend returned an empty list. This might mean no hype markets are configured or no new traders were found for any configured market.")
        if st.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()

    else: # Unexpected data format
        st.error("Received unexpected data format from the backend.")
        st.write("Data received:", data)
        if st.button("Attempt Reload"):
            st.cache_data.clear()
            st.rerun()

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    user_retention_page()