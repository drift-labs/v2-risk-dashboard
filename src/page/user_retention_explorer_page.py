import streamlit as st
import pandas as pd
import datetime
import time
from lib.api import fetch_api_data

def is_processing(result):
    """Checks if the API result indicates backend processing."""
    return isinstance(result, dict) and result.get("result") == "processing"

# Function to get the list of available markets from the new API endpoint
@st.cache_data(ttl=3600) # Cache for 1 hour
def get_market_list():
    """Fetches the list of available markets for the dropdown."""
    # The middleware expects `bypass_cache=true`, not `use_cache=False`.
    return fetch_api_data(section="user-retention-explorer", path="markets", params={"bypass_cache": "true"})

# Function to call the calculation endpoint
def calculate_retention(market, start_date):
    """Fetches retention data for a specific market and date."""
    params = {"market_name": market, "start_date": start_date.strftime("%Y-%m-%d")}
    # We expect this call to take time, so no retry logic here.
    # Errors will be caught and displayed.
    data = fetch_api_data(section="user-retention-explorer", path="calculate", params=params, retry=False)
    return data

def user_retention_explorer_page():
    st.title("User Retention Explorer")

    st.markdown("""
    This page allows you to perform a dynamic user retention analysis for a single market.
    1.  **Select a Market**: Choose the market you want to analyze.
    2.  **Select a Start Date**: This is the beginning of the 7-day window to identify "new traders".
    3.  **Click Execute**: The system will query the data warehouse to find new traders and calculate their retention at 14 and 28 days in *any other* market.
    
    **Note**: This query can take a few minutes to complete. Please be patient.
    """)

    # --- Step 1: Get market list for the dropdown ---
    market_list = get_market_list()
    
    if not isinstance(market_list, list) or not market_list:
        st.error("Could not fetch the list of markets from the backend. Please ensure the backend is running and `shared/markets.json` is populated.")
        st.write("Data received:", market_list)
        return

    # --- Step 2: User Inputs ---
    st.header("Analysis Parameters")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    selected_market = col1.selectbox(
        "Select Market",
        options=market_list,
        index=0,
        key='explorer_market'
    )
    
    selected_date = col2.date_input(
        "Select Start Date",
        value=datetime.date.today() - datetime.timedelta(days=30), # Default to 30 days ago
        help="The start of the 7-day window for identifying new traders.",
        key='explorer_date'
    )

    # Vertically align the button with the inputs
    col3.write("") # Spacer
    col3.write("") # Spacer
    execute_button = col3.button("Execute Analysis", type="primary")

    # --- Step 3: Execution Logic ---
    # When the button is clicked, set the state to start the analysis process.
    if execute_button:
        if not st.session_state.explorer_market or not st.session_state.explorer_date:
            st.warning("Please select a market and a start date.")
        else:
            st.session_state.analysis_running = True
            st.session_state.explorer_result = None # Clear previous results

    # If an analysis is in progress, call the backend and update the result.
    # This runs on the initial click and on subsequent reruns during polling.
    if st.session_state.get('analysis_running'):
        with st.spinner(f"Calculating retention for **{st.session_state.explorer_market}** from **{st.session_state.explorer_date}**... This may take a while."):
            try:
                st.session_state.explorer_result = calculate_retention(
                    st.session_state.explorer_market, 
                    st.session_state.explorer_date
                )
            except Exception as e:
                st.session_state.explorer_result = {"error": f"An unexpected error occurred during API call: {e}"}
                st.session_state.analysis_running = False # Stop on error

    # --- Step 4: Display results stored in session state ---
    if 'explorer_result' in st.session_state and st.session_state.explorer_result is not None:
        result = st.session_state.explorer_result

        st.divider()

        # Handle the case where the backend is still processing the data
        if is_processing(result):
            message = result.get("message", "Data generation in progress. Please wait.")
            st.info(message)
            with st.spinner("Auto-refreshing in 30 seconds to check for results..."):
                time.sleep(30)
            st.rerun()
            return # Stop further execution until the next rerun

        # Once processing is finished (or if it was immediate), stop the polling loop.
        st.session_state.analysis_running = False

        if "error" in result:
             st.error(result["error"])
        elif isinstance(result, dict) and result.get("detail"):
            error_msg = result.get("detail", "An unknown error occurred.")
            st.error(f"Failed to calculate retention data: {error_msg}")
        elif isinstance(result, dict) and "market" in result:
            st.header("Analysis Results")

            df_data = {
                'Metric': [
                    "Market", "Category", "Analysis Start Date", "New Traders (Count)", 
                    "Retained at 14d (Count)", "Retention Rate 14d",
                    "Retained at 28d (Count)", "Retention Rate 28d"
                ],
                'Value': [
                    result['market'],
                    ', '.join(result['category']),
                    result['start_date'],
                    result['new_traders'],
                    result['retained_users_14d'],
                    f"{result['retention_ratio_14d']:.2%}",
                    result['retained_users_28d'],
                    f"{result['retention_ratio_28d']:.2%}"
                ]
            }
            summary_df = pd.DataFrame(df_data)
            st.table(summary_df.set_index('Metric'))

            # Display lists of traders in expanders
            with st.expander("View New Traders List"):
                if result['new_traders_list']:
                    st.dataframe(pd.DataFrame(result['new_traders_list'], columns=["User Address"]), hide_index=True, use_container_width=True)
                else:
                    st.caption("No new traders found.")
            
            with st.expander("View 14-Day Retained Users List"):
                if result['retained_users_14d_list']:
                    st.dataframe(pd.DataFrame(result['retained_users_14d_list'], columns=["User Address"]), hide_index=True, use_container_width=True)
                else:
                    st.caption("No users retained at 14 days.")

            with st.expander("View 28-Day Retained Users List"):
                if result['retained_users_28d_list']:
                    st.dataframe(pd.DataFrame(result['retained_users_28d_list'], columns=["User Address"]), hide_index=True, use_container_width=True)
                else:
                    st.caption("No users retained at 28 days.")

        else:
            st.error("Received unexpected data format from the backend.")
            st.write("Data received:", result)

if __name__ == "__main__":
    user_retention_explorer_page() 