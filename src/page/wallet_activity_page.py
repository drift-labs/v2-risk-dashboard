import datetime
import time

import pandas as pd
import streamlit as st

from lib.api import fetch_api_data


def is_processing(result):
    return isinstance(result, dict) and result.get("result") == "processing"


def calculate_wallet_activity(since_date):
    params = {
        "since_date": since_date.strftime("%Y-%m-%d"),
        "bypass_cache": "false",
    }
    data = fetch_api_data(
        section="wallet-activity", path="calculate", params=params, retry=False
    )
    return data


def wallet_activity_page():
    st.title("User Activity Dashboard")

    st.markdown("""
    This dashboard analyzes user activity since a given date, categorizing users into three groups:
    
    1. **New Users**: Users that placed an order for the first time after the selected date
    2. **Reactivated Users**: Users that had **no trading activity for more than 15 days prior to the selected date**, but traded again after
    3. **Active Users**: Users that **did trade within 15 days before the selected date**, and continued trading after
    
    The analysis provides both wallet counts and cumulative trading volume for each category.
    
    **Note**: This query can take a few minutes to complete. Please be patient.
    """)

    st.header("Analysis Parameters")
    col1, col2 = st.columns([2, 1])

    _selected_date = col1.date_input(
        "Select Analysis Date",
        value=datetime.date.today() - datetime.timedelta(days=1),
        help="The date from which to analyze wallet activity. Wallets will be categorized based on their trading activity relative to this date.",
        key="wallet_activity_date",
    )

    col2.write("")
    col2.write("")
    execute_button = col2.button("Execute Analysis", type="primary")

    if execute_button:
        if not st.session_state.wallet_activity_date:
            st.warning("Please select an analysis date.")
        else:
            st.session_state.wallet_analysis_running = True
            st.session_state.wallet_activity_result = None

    if st.session_state.get("wallet_analysis_running"):
        with st.spinner(
            f"Analyzing wallet activity since **{st.session_state.wallet_activity_date}**... This may take a while."
        ):
            try:
                st.session_state.wallet_activity_result = calculate_wallet_activity(
                    st.session_state.wallet_activity_date
                )
            except Exception as e:
                st.session_state.wallet_activity_result = {
                    "error": f"An unexpected error occurred during API call: {e}"
                }
                st.session_state.wallet_analysis_running = False

    if (
        "wallet_activity_result" in st.session_state
        and st.session_state.wallet_activity_result is not None
    ):
        result = st.session_state.wallet_activity_result

        st.divider()

        if is_processing(result):
            message = result.get("message", "Data generation in progress. Please wait.")
            st.info(message)
            with st.spinner("Auto-refreshing in 30 seconds to check for results..."):
                time.sleep(30)
            st.rerun()
            return

        st.session_state.wallet_analysis_running = False

        if "error" in result:
            st.error(result["error"])
        elif isinstance(result, dict) and "volume_df" in result:
            st.header("Analysis Results")
            volume_df = pd.DataFrame(result["volume_df"])
            first_trades_df = pd.DataFrame(result["first_trades_df"])
            last_trades_df = pd.DataFrame(result["last_trades_df"])
            first_trades_df = columns_to_datetime(first_trades_df, ["first_trade_ts"])
            last_trades_df = columns_to_datetime(last_trades_df, ["last_trade_ts"])
            merged_df = first_trades_df.merge(last_trades_df, on="user", how="left")
            merged_df = merged_df.merge(volume_df, on="user", how="left")

            def categorize_wallet(row):
                if row["first_trade_ts"] >= pd.to_datetime(
                    st.session_state.wallet_activity_date
                ):
                    return "new"
                if row["last_trade_ts"] < pd.to_datetime(
                    st.session_state.wallet_activity_date
                ) - datetime.timedelta(days=15):
                    return "reactivated"
                return "active"

            merged_df["category"] = merged_df.apply(categorize_wallet, axis=1)

            st.metric(
                "New User Accounts",
                len(merged_df[merged_df["category"] == "new"]),
            )
            st.metric(
                "Reactivated User Accounts",
                len(merged_df[merged_df["category"] == "reactivated"]),
            )
            st.metric(
                "Active User Accounts",
                len(merged_df[merged_df["category"] == "active"]),
            )

            st.divider()
            st.dataframe(merged_df)
        else:
            st.error("Received unexpected data format from the backend.")
            st.write("Data received:", result)


def columns_to_front(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns + [col for col in df.columns if col not in columns]]


def columns_to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = pd.to_datetime(df[col], unit="s")
    return df


if __name__ == "__main__":
    wallet_activity_page()
