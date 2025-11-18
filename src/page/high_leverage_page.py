import time

import pandas as pd
import streamlit as st

from lib.api import fetch_api_data
from utils import get_current_slot

RETRY_DELAY_SECONDS = 5


def is_processing(result):
    return isinstance(result, dict) and result.get("result") == "processing"


def has_error(result):
    return result is None or (
        isinstance(result, dict) and result.get("result") == "error"
    )


def high_leverage_page():
    st.title("High Leverage Usage on Drift")

    try:
        config_data = fetch_api_data(
            section="high-leverage",
            path="config",
            retry=False,
        )
        result_positions = fetch_api_data(
            section="high-leverage", path="positions/detailed", retry=False
        )

        if is_processing(config_data) or is_processing(result_positions):
            st.info(
                f"Backend is processing high leverage data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
            )
            with st.spinner("Please wait..."):
                time.sleep(RETRY_DELAY_SECONDS)
            st.rerun()

        if has_error(config_data):
            error_msg = (
                config_data["message"]
                if isinstance(config_data, dict)
                else "Could not connect or fetch stats data."
            )
            st.error(
                f"Failed to fetch essential data (High Leverage Stats): {error_msg}"
            )
            if st.button("Retry Fetch Stats"):
                st.rerun()
            return

        if has_error(result_positions):
            st.warning(
                "Could not fetch Detailed High Leverage Positions data. The position table will be empty."
            )

        slot = config_data.get("slot", "N/A")
        positions_data = result_positions if isinstance(result_positions, list) else []
        st.subheader("High Leverage Usage Stats")
        st.write(config_data)
        cols = st.columns(4)
        cols[0].metric(
            "Config: Total High Leverage Spots", config_data.get("total_spots", "N/A")
        )
        cols[1].metric(
            "Config: Opted-In Users", config_data.get("opted_in_spots", "N/A")
        )
        unique_users = len(
            set([position["user_public_key"] for position in result_positions])
        )
        cols[2].metric("Usermap: Unique users", unique_users)
        cols[3].metric("Usermap: Total HL positions", len(result_positions))
        unique_high_leverage_users = len(
            set(
                [
                    position["user_public_key"]
                    for position in result_positions
                    if position["leverage_category"] == "high_leverage"
                ]
            )
        )
        unique_high_leverage_maintenance_users = len(
            set(
                [
                    position["user_public_key"]
                    for position in result_positions
                    if position["leverage_category"] == "high_leverage_maintenance"
                ]
            )
        )
        row2_cols = st.columns(4)
        row2_cols[0].metric("Usermap: Unique HL users", unique_high_leverage_users)
        row2_cols[1].metric(
            "Usermap: Unique HL maintenance users",
            unique_high_leverage_maintenance_users,
        )

        df_positions = pd.DataFrame()
        if positions_data:
            try:
                df_positions = pd.DataFrame(positions_data)
                if not df_positions.empty:
                    df_positions.rename(
                        columns={
                            "user_public_key": "User Account",
                            "authority": "Authority",
                            "market_index": "Market Index",
                            "market_symbol": "Market Symbol",
                            "base_asset_amount_ui": "Base Asset Amount",
                            "position_value_usd": "Notional Value (USD)",
                            "account_leverage": "Account Leverage",
                            "position_leverage": "Position Leverage",
                            "leverage_category": "Leverage Category",
                        },
                        inplace=True,
                    )

                    df_positions["Base Asset Amount"] = pd.to_numeric(
                        df_positions["Base Asset Amount"], errors="coerce"
                    )
                    df_positions["Notional Value (USD)"] = pd.to_numeric(
                        df_positions["Notional Value (USD)"], errors="coerce"
                    )
                    df_positions["Position Leverage"] = pd.to_numeric(
                        df_positions["Position Leverage"], errors="coerce"
                    )
                    df_positions["Account Leverage"] = pd.to_numeric(
                        df_positions["Account Leverage"], errors="coerce"
                    )

                    df_positions = df_positions.sort_values(
                        by="Account Leverage", ascending=False
                    )

                    display_df = df_positions[
                        [
                            "Market Symbol",
                            "Base Asset Amount",
                            "Notional Value (USD)",
                            "Position Leverage",
                            "Account Leverage",
                            "User Account",
                            "Authority",
                            "Market Index",
                            "Leverage Category",
                        ]
                    ].copy()

            except Exception as df_e:
                st.error(
                    f"Error processing detailed position data into DataFrame: {df_e}"
                )
                df_positions = pd.DataFrame()
                display_df = pd.DataFrame()
        else:
            display_df = pd.DataFrame()

        current_slot = get_current_slot()
        if slot != "N/A" and current_slot:
            try:
                slot_age = int(current_slot) - int(slot)
                st.info(f"Below data for slot {slot} (age: {slot_age} slots)")
            except (ValueError, TypeError):
                st.info(f"Below data for slot {slot}. Current slot: {current_slot}")
        else:
            st.info(f"Slot information unavailable. Current slot: {current_slot}")

        st.subheader("Detailed High Leverage Positions")
        if not display_df.empty:
            styled_df = display_df.style.format(
                {
                    "Base Asset Amount": "{:,.4f}",
                    "Notional Value (USD)": "${:,.2f}",
                    "Position Leverage": "{:.2f}x",
                    "Account Leverage": "{:.2f}x",
                }
            )
            st.dataframe(
                styled_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Market Symbol": st.column_config.TextColumn(label="Market Symbol"),
                    "User Account": st.column_config.TextColumn(label="User Account"),
                    "Authority": st.column_config.TextColumn(label="Authority"),
                    "Market Index": st.column_config.TextColumn(label="Market Index"),
                },
            )
        else:
            st.info(
                "No detailed high leverage position data available, or an error occurred during processing."
            )

        st.subheader("Bootable User Details (Inactive High Leverage Users)")
        result_bootable_users = fetch_api_data(
            section="high-leverage", path="bootable-users", retry=False
        )

        if is_processing(result_bootable_users):
            st.info(
                f"Backend is processing bootable user data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
            )
            with st.spinner("Loading bootable users..."):
                time.sleep(RETRY_DELAY_SECONDS)
            st.rerun()
            return

        if has_error(result_bootable_users):
            st.warning(
                "Could not fetch Bootable User Details. This table will be empty."
            )
            bootable_users_data = []
        else:
            bootable_users_data = (
                result_bootable_users if isinstance(result_bootable_users, list) else []
            )

        df_bootable = pd.DataFrame()
        if bootable_users_data:
            try:
                df_bootable = pd.DataFrame(bootable_users_data)
                if not df_bootable.empty:
                    df_bootable.rename(
                        columns={
                            "user_public_key": "User Account",
                            "authority": "Authority",
                            "account_leverage": "Account Leverage",
                            "activity_staleness_slots": "Activity Staleness (Slots)",
                            "last_active_slot": "Last Active Slot",
                            "initial_margin_requirement_usd": "Initial Margin Req. (USD)",
                            "total_collateral_usd": "Total Collateral (USD)",
                            "health_percent": "Health (%)",
                        },
                        inplace=True,
                    )

                    df_bootable["Account Leverage"] = pd.to_numeric(
                        df_bootable["Account Leverage"], errors="coerce"
                    )
                    df_bootable["Activity Staleness (Slots)"] = pd.to_numeric(
                        df_bootable["Activity Staleness (Slots)"], errors="coerce"
                    )
                    df_bootable["Initial Margin Req. (USD)"] = pd.to_numeric(
                        df_bootable["Initial Margin Req. (USD)"], errors="coerce"
                    )
                    df_bootable["Total Collateral (USD)"] = pd.to_numeric(
                        df_bootable["Total Collateral (USD)"], errors="coerce"
                    )
                    df_bootable["Health (%)"] = pd.to_numeric(
                        df_bootable["Health (%)"], errors="coerce"
                    )

                    df_bootable = df_bootable.sort_values(
                        by="Activity Staleness (Slots)", ascending=False
                    )

                    display_bootable_df = df_bootable[
                        [
                            "User Account",
                            "Authority",
                            "Account Leverage",
                            "Activity Staleness (Slots)",
                            "Last Active Slot",
                            "Initial Margin Req. (USD)",
                            "Total Collateral (USD)",
                            "Health (%)",
                        ]
                    ].copy()

                    styled_bootable_df = display_bootable_df.style.format(
                        {
                            "Account Leverage": "{:.2f}x",
                            "Activity Staleness (Slots)": "{:,.0f}",
                            "Initial Margin Req. (USD)": "${:,.2f}",
                            "Total Collateral (USD)": "${:,.2f}",
                            "Health (%)": "{:.0f}%",
                        }
                    )
                    st.dataframe(
                        styled_bootable_df, hide_index=True, use_container_width=True
                    )
                else:
                    st.info("No bootable users found meeting the criteria.")
            except Exception as df_boot_e:
                st.error(
                    f"Error processing bootable user data into DataFrame: {df_boot_e}"
                )
        else:
            st.info("No bootable user data to display.")

    except Exception as e:
        st.error(f"An error occurred while displaying the page: {e}")
        import traceback

        st.text(traceback.format_exc())


if __name__ == "__main__":
    high_leverage_page()
