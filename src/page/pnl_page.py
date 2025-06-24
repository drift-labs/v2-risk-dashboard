import pandas as pd
import streamlit as st
import time

from lib.api import fetch_api_data
from src.utils import get_current_slot

RETRY_DELAY_SECONDS = 5


def is_processing(result):
    """Checks if the API result indicates backend processing."""
    return isinstance(result, dict) and result.get("result") == "processing"


def has_error(result):
    """Checks if the API result indicates an error."""
    return result is None or (
        isinstance(result, dict) and result.get("result") == "error"
    ) or ("pnl" not in result if isinstance(result, dict) else False)


def pnl_page():
    st.title("Top PnL by User (All Time)")
    response = fetch_api_data("pnl", "top_pnl", retry=False)

    if is_processing(response):
        st.info(
            f"Backend is processing PnL data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
        )
        with st.spinner("Please wait..."):
            time.sleep(RETRY_DELAY_SECONDS)
        st.rerun()
        return

    if has_error(response):
        error_msg = (
            response["message"]
            if isinstance(response, dict) and "message" in response
            else "Could not connect or fetch PnL data."
        )
        st.error(f"Failed to fetch PnL data: {error_msg}")
        if st.button("Retry"):
            st.rerun()
        return

    pnl_data = response["pnl"]
    if not pnl_data:
        st.info("No PnL data available.")
        return

    slot = response.get("slot", 0)
    current_slot = get_current_slot()
    if slot > 0 and current_slot > 0:
        slot_age = current_slot - slot
        st.info(f"Data from slot {slot}, which is {slot_age} slots old.")

    df = pd.DataFrame(pnl_data)
    for col in ["realized_pnl", "unrealized_pnl", "total_pnl"]:
        df[col] = df[col].map("${:,.2f}".format)

    csv = df.to_csv(index=False)
    st.download_button(
        "Download PnL Data CSV", csv, "top_pnl.csv", "text/csv", key="download-pnl"
    )
    st.dataframe(
        df,
        height=650,
        column_config={
            "authority": st.column_config.TextColumn(
                "Authority",
                help="Authority address",
            ),
            "user_key": st.column_config.TextColumn(
                "User Account",
                help="User account address",
            ),
            "realized_pnl": st.column_config.NumberColumn("All Time Realized PnL"),
            "unrealized_pnl": st.column_config.NumberColumn("Unrealized PnL"),
            "total_pnl": st.column_config.NumberColumn("Total PnL"),
        },
        hide_index=True,
    )
