import datetime
import time

import pandas as pd
import streamlit as st

from lib.api import fetch_api_data


def is_processing(result):
    return isinstance(result, dict) and result.get("result") == "processing"


@st.cache_data(ttl=3600)
def get_market_list():
    return fetch_api_data(
        section="user-retention-explorer",
        path="markets",
        params={"bypass_cache": "true"},
    )


def calculate_retention(market, start_date):
    params = {
        "market_name": market,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "bypass_cache": "false",
    }
    data = fetch_api_data(
        section="user-retention-explorer", path="calculate", params=params, retry=False
    )
    return data


def user_retention_explorer_page():
    st.title("User Retention Explorer")

    st.markdown("""
    This page allows you to perform a dynamic user retention analysis for a single market.
    1.  **Select a Market**: Choose the market you want to analyze.
    2.  **Select a Start Date**: This is the beginning of the 7-day window to identify "new traders".
    3.  **Click Execute**: The system will query the data warehouse to find new traders and calculate their retention at 14 and 28 days in *any other* market.
    
    **Note on "New Traders"**: This analysis now filters for *newly trading authorities*. A user is only considered "new" if their first-ever trade on the platform occurs within the selected time window and is made with their primary account (`subaccountid=0`). This ensures we don't mistakenly include experienced users who are simply creating new sub-accounts.
    
    **Note**: This query can take a few minutes to complete. Please be patient.
    """)

    market_list = get_market_list()

    if not isinstance(market_list, list) or not market_list:
        st.error(
            "Could not fetch the list of markets from the backend. Please ensure the backend is running and `shared/markets.json` is populated."
        )
        st.write("Data received:", market_list)
        return

    st.header("Analysis Parameters")
    col1, col2, col3 = st.columns([2, 1, 1])

    selected_market = col1.selectbox(
        "Select Market", options=market_list, index=0, key="explorer_market"
    )

    selected_date = col2.date_input(
        "Select Start Date",
        value=datetime.date.today() - datetime.timedelta(days=30),
        help="The start of the 7-day window for identifying new traders.",
        key="explorer_date",
    )

    col3.write("")
    col3.write("")
    execute_button = col3.button("Execute Analysis", type="primary")

    if execute_button:
        if not st.session_state.explorer_market or not st.session_state.explorer_date:
            st.warning("Please select a market and a start date.")
        else:
            st.session_state.analysis_running = True
            st.session_state.explorer_result = None

    if st.session_state.get("analysis_running"):
        with st.spinner(
            f"Calculating retention for **{st.session_state.explorer_market}** from **{st.session_state.explorer_date}**... This may take a while."
        ):
            try:
                st.session_state.explorer_result = calculate_retention(
                    st.session_state.explorer_market, st.session_state.explorer_date
                )
            except Exception as e:
                st.session_state.explorer_result = {
                    "error": f"An unexpected error occurred during API call: {e}"
                }
                st.session_state.analysis_running = False

    if (
        "explorer_result" in st.session_state
        and st.session_state.explorer_result is not None
    ):
        result = st.session_state.explorer_result

        st.divider()

        if is_processing(result):
            message = result.get("message", "Data generation in progress. Please wait.")
            st.info(message)
            with st.spinner("Auto-refreshing in 30 seconds to check for results..."):
                time.sleep(30)
            st.rerun()
            return

        st.session_state.analysis_running = False

        if "error" in result:
            st.error(result["error"])
        elif isinstance(result, dict) and (
            result.get("detail") or result.get("result") == "error"
        ):
            error_msg = result.get("detail") or result.get(
                "message", "An unknown error occurred."
            )
            st.error(f"Failed to calculate retention data: {error_msg}")
        elif isinstance(result, dict) and "user_data" in result:
            st.header("Analysis Results")

            st.subheader("High-Level Volume Metrics")
            metric_cols = st.columns(4)
            metric_cols[0].metric(
                "ISMV7d",
                f"${result.get('total_initial_selected_market_volume_7d', 0):,.0f}",
                help="Initial Selected Market Volume (7d): Total volume in the selected market from new traders during their first 7 days.",
            )
            metric_cols[1].metric(
                "IOMV7d",
                f"${result.get('total_initial_other_market_volume_7d', 0):,.0f}",
                help="Initial Other Market Volume (7d): Total volume in all other markets from new traders during their first 7 days.",
            )
            metric_cols[2].metric(
                "TV14d",
                f"${result.get('total_volume_14d', 0):,.0f}",
                help="Total Volume (14d): Total volume from the cohort during the 14-day retention period (Days 8-21).",
            )
            metric_cols[3].metric(
                "TV28d",
                f"${result.get('total_volume_28d', 0):,.0f}",
                help="Total Volume (28d): Total volume from the cohort during the 28-day retention period (Days 8-35).",
            )

            st.subheader("Retention Summary")
            df_data = {
                "Metric": [
                    "Market",
                    "Category",
                    "Analysis Start Date",
                    "New Traders (Count)",
                    "Retained at 14d (Count)",
                    "Retention Rate 14d",
                    "Retained at 28d (Count)",
                    "Retention Rate 28d",
                ],
                "Value": [
                    result["market"],
                    ", ".join(result["category"]),
                    result["start_date"],
                    result["new_traders_count"],
                    result["retained_14d_count"],
                    f"{result['retention_ratio_14d']:.2%}",
                    result["retained_28d_count"],
                    f"{result['retention_ratio_28d']:.2%}",
                ],
            }
            summary_df = pd.DataFrame(df_data)
            st.table(summary_df.set_index("Metric"))

            st.subheader("User Activity Details")

            if not result["user_data"]:
                st.caption("No new traders found for the selected criteria.")
            else:
                user_df = pd.DataFrame(result["user_data"])

                user_df["url"] = (
                    "https://app.drift.trade/overview?userAccount="
                    + user_df["user_address"]
                )

                user_df["total_initial_volume"] = (
                    user_df["initial_selected_market_volume"]
                    + user_df["initial_other_market_volume"]
                )
                user_df["total_volume_14d"] = (
                    user_df["selected_market_volume_14d"]
                    + user_df["other_market_volume_14d"]
                )
                user_df["total_volume_28d"] = (
                    user_df["selected_market_volume_28d"]
                    + user_df["other_market_volume_28d"]
                )
                user_df["aggregate_volume"] = (
                    user_df["total_initial_volume"] + user_df["total_volume_28d"]
                )

                user_df = user_df.rename(
                    columns={
                        "user_address": "User Address",
                        "initial_selected_market_volume": "Initial Selected Market Volume (7d)",
                        "initial_other_market_volume": "Initial Other Market Volume (7d)",
                        "total_initial_volume": "Total Initial Volume (7d)",
                        "selected_market_volume_14d": "Selected Market Volume (14d)",
                        "other_market_volume_14d": "Other Market Volume (14d)",
                        "selected_market_volume_28d": "Selected Market Volume (28d)",
                        "other_market_volume_28d": "Other Market Volume (28d)",
                        "total_volume_14d": "Total Volume (14d)",
                        "total_volume_28d": "Total Volume (28d)",
                        "aggregate_volume": "Aggregate Volume",
                    }
                )

                column_config = {
                    "url": st.column_config.LinkColumn(
                        "User Address",
                        help="The user's wallet address. Click to view on Drift.",
                        display_text=".*userAccount=(.*)",
                    ),
                    "Initial Selected Market Volume (7d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in the selected market during the initial 7-day period (Days 1-7).",
                    ),
                    "Initial Other Market Volume (7d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in all other markets during the initial 7-day period (Days 1-7).",
                    ),
                    "Total Initial Volume (7d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Total volume across all markets during the initial 7-day period (Days 1-7).",
                    ),
                    "Selected Market Volume (14d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in the selected market during the 14-day retention period (Days 8-21).",
                    ),
                    "Other Market Volume (14d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in all other markets during the 14-day retention period (Days 8-21).",
                    ),
                    "Total Volume (14d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Total volume across all markets during the 14-day retention period (Days 8-21).",
                    ),
                    "Selected Market Volume (28d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in the selected market during the 28-day retention period (Days 8-35).",
                    ),
                    "Other Market Volume (28d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Volume in all other markets during the 28-day retention period (Days 8-35).",
                    ),
                    "Total Volume (28d)": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Total volume across all markets during the 28-day retention period (Days 8-35).",
                    ),
                    "Aggregate Volume": st.column_config.NumberColumn(
                        format="$%.2f",
                        help="Total volume across all markets and all time periods (initial 7d + 28d retention).",
                    ),
                }

                st.dataframe(
                    user_df[
                        [
                            "url",
                            "Initial Selected Market Volume (7d)",
                            "Initial Other Market Volume (7d)",
                            "Total Initial Volume (7d)",
                            "Selected Market Volume (14d)",
                            "Other Market Volume (14d)",
                            "Total Volume (14d)",
                            "Selected Market Volume (28d)",
                            "Other Market Volume (28d)",
                            "Total Volume (28d)",
                            "Aggregate Volume",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config=column_config,
                )

        else:
            st.error("Received unexpected data format from the backend.")
            st.write("Data received:", result)


if __name__ == "__main__":
    user_retention_explorer_page()
