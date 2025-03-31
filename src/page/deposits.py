import pandas as pd
import streamlit as st
import logging
from typing import Optional
from driftpy.constants.spot_markets import mainnet_spot_market_configs

from lib.api import fetch_api_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_authority(authority: str) -> str:
    """Format authority to show first and last 4 chars"""
    return f"{authority[:4]}...{authority[-4:]}"

def get_market_symbol(market_index: int) -> str:
    """
    Get market symbol for a given market index with validation.
    
    Args:
        market_index: The market index to look up
        
    Returns:
        str: Formatted string with market index and symbol
    """
    try:
        if not isinstance(market_index, (int, float)):
            raise ValueError(f"Invalid market index type: {type(market_index)}")
            
        market_index = int(market_index)
        valid_indices = [x.market_index for x in mainnet_spot_market_configs]
        
        if market_index not in valid_indices:
            logger.error(f"Invalid market index: {market_index}. Valid indices: {valid_indices}")
            return f"{market_index} (Unknown)"
            
        return f"{market_index} ({mainnet_spot_market_configs[market_index].symbol})"
    except Exception as e:
        logger.error(f"Error processing market index {market_index}: {str(e)}")
        return f"{market_index} (Error)"

def deposits_page():
    try:
        params = st.query_params
        market_index = int(params.get("market_index", 0))

        radio_option = st.radio(
            "Aggregate by",
            ["All", "By Market"],
            index=0,
        )
        col1, col2 = st.columns([2, 2])

        if radio_option == "All":
            market_index = 0
        else:
            valid_indices = [x.market_index for x in mainnet_spot_market_configs]
            if market_index not in valid_indices:
                logger.warning(f"Invalid market index in params: {market_index}. Defaulting to first available market.")
                market_index = valid_indices[0] if valid_indices else 0

            with col2:
                market_index = st.selectbox(
                    "Market index",
                    valid_indices,
                    index=valid_indices.index(market_index) if market_index in valid_indices else 0,
                    format_func=lambda x: get_market_symbol(x),
                )
            st.query_params.update({"market_index": str(market_index)})

        # Log API request parameters
        logger.info(f"Fetching deposits for market_index: {market_index if radio_option == 'By Market' else 'All'}")
        
        if radio_option == "All":
            result = fetch_api_data(
                "deposits",
                "deposits",
                params={"market_index": None},
                retry=True,
            )
        else:
            result = fetch_api_data(
                "deposits",
                "deposits",
                params={"market_index": market_index},
                retry=True,
            )

        if result is None:
            logger.error("API returned no deposits data")
            st.error("No deposits found")
            return

        df = pd.DataFrame(result["deposits"])
        if df.empty:
            logger.warning("Empty deposits dataframe created")
            st.warning("No deposits data available")
            return

        total_number_of_deposited = sum([x["balance"] for x in result["deposits"]])

        exclude_vaults = st.checkbox("Exclude Vaults", value=True)

        if exclude_vaults:
            original_len = len(df)
            df = df[~df["authority"].isin(result["vaults"])]
            logger.info(f"Excluded {original_len - len(df)} vault entries")

        with col1:
            min_balance = st.number_input(
                "Minimum Balance",
                min_value=0.0,
                max_value=float(df["balance"].max()),
                value=0.0,
                step=0.1,
            )

        tabs = st.tabs(["By Position", "By Authority"])

        with tabs[0]:
            filtered_df = df[df["balance"] >= min_balance]
            st.write(f"Total deposits value: **${filtered_df['value'].sum():,.2f}**")
            st.write(f"Number of depositor user accounts: **{len(filtered_df):,}**")

            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download All Deposits CSV",
                csv,
                "all_deposits.csv",
                "text/csv",
                key="download-all-deposits",
            )
            
            # Safely map market indices to symbols
            filtered_df["market_index"] = filtered_df["market_index"].map(get_market_symbol)

            st.dataframe(
                filtered_df.sort_values("value", ascending=False),
                column_config={
                    "authority": st.column_config.TextColumn(
                        "Authority",
                        help="Account authority",
                    ),
                    "user_account": st.column_config.TextColumn(
                        "User Account",
                        help="User account address",
                    ),
                    "value": st.column_config.NumberColumn(
                        "Value (USD)",
                        step=0.01,
                    ),
                    "balance": st.column_config.NumberColumn(
                        "Balance (USD)",
                        step=0.01,
                    ),
                },
                hide_index=True,
            )

        with tabs[1]:
            grouped_df = (
                df.groupby("authority")
                .agg({"value": "sum", "balance": "sum", "user_account": "count"})
                .reset_index()
            )
            grouped_df = grouped_df[grouped_df["value"] >= min_balance]
            st.write(f"Total deposits value: **${grouped_df['value'].sum():,.2f}**")
            st.write(f"Total number of authorities with deposits: **{len(grouped_df):,}**")
            grouped_df = grouped_df.rename(columns={"user_account": "num_accounts"})
            grouped_df = grouped_df.sort_values("value", ascending=False)
            grouped_df.drop(columns=["balance"], inplace=True)

            csv_grouped = grouped_df.to_csv(index=False)
            st.download_button(
                "Download Authority Summary CSV",
                csv_grouped,
                "deposits_by_authority.csv",
                "text/csv",
                key="download-grouped-deposits",
            )

            st.dataframe(
                grouped_df,
                column_config={
                    "authority": st.column_config.TextColumn(
                        "Authority",
                        help="Account authority",
                    ),
                    "value": st.column_config.NumberColumn(
                        "Total Value (USD)",
                        step=0.01,
                    ),
                    "num_accounts": st.column_config.NumberColumn(
                        "Number of Accounts",
                        step=1,
                    ),
                },
                hide_index=True,
            )
    except Exception as e:
        logger.exception("Unexpected error in deposits_page")
        st.error(f"An unexpected error occurred: {str(e)}")
