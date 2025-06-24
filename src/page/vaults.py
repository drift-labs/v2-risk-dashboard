import pandas as pd
import streamlit as st

from lib.api import fetch_api_data


def vaults_page():
    st.title("Drift Vaults")
    st.write(
        "View statistics and analytics for Drift Protocol vaults. More information and performance metrics of specific vaults can be found on the [Drift app](https://app.drift.trade/vaults/strategy-vaults). "
        "This page may be out of date up to 30 minutes."
    )

    with st.spinner("Loading vault data..."):
        response = fetch_api_data("vaults", "data", retry=True, max_wait_time=60)
        if not response or "data" not in response:
            st.error("Failed to load vault data")
            return

        data = response["data"]
        analytics = data["analytics"]
        all_depositors = data["depositors"]

    unique_depositors = set()
    for depositors in all_depositors.values():
        for depositor in depositors:
            unique_depositors.add(depositor["pubkey"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vaults", analytics["total_vaults"])
    with col2:
        st.metric("Total Value Locked", f"${analytics['total_deposits']:,.2f}")
    with col3:
        st.metric("Total Unique Depositor accounts", f"{len(unique_depositors):,}")

    st.subheader("All Vaults")
    st.markdown(
        "Note that this list includes many test vaults or otherwise inactive vaults. For the full list of active vaults, please refer to the [page on Drift](https://app.drift.trade/vaults/strategy-vaults)"
    )

    all_vaults_df = pd.DataFrame(analytics["vaults"])
    all_vaults_df = all_vaults_df.sort_values(by="true_net_deposits", ascending=False)
    all_vaults_df = all_vaults_df[
        ["name", "true_net_deposits", "depositor_count", "pubkey"]
    ]

    st.dataframe(
        all_vaults_df,
        column_config={
            "name": st.column_config.TextColumn("Vault Name"),
            "true_net_deposits": st.column_config.NumberColumn(
                "Deposits (USD)", step=0.01
            ),
            "depositor_count": st.column_config.NumberColumn("Depositors"),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Vault Depositor Details")

    all_vaults_df_indexed = all_vaults_df.set_index("pubkey")

    def format_vault_option(pubkey: str) -> str:
        try:
            vault_info = all_vaults_df_indexed.loc[pubkey]
            name = vault_info["name"]
            net_deposits = vault_info["true_net_deposits"]
            return f"{name} (${net_deposits:,.2f} USD) - {pubkey[:4]}...{pubkey[-4:]}"
        except KeyError:
            return f"Unknown Vault - {pubkey[:4]}...{pubkey[-4:]}"

    selected_vault = st.selectbox(
        "Select Vault",
        all_vaults_df["pubkey"].tolist(),
        format_func=format_vault_option,
    )

    if selected_vault:
        depositors = [
            dep for dep in all_depositors.get(selected_vault, []) if dep["shares"] > 0
        ]

        if depositors:
            st.write(f"Depositors for vault {selected_vault}")

            depositors_df = pd.DataFrame(depositors)
            st.dataframe(
                depositors_df,
                column_config={
                    "pubkey": st.column_config.TextColumn("Address"),
                    "shares": st.column_config.NumberColumn("Shares", step=0.01),
                    "share_percentage": st.column_config.NumberColumn(
                        "Percentage",
                        format="%.4f%%",
                    ),
                },
                hide_index=True,
            )
        else:
            st.write("No depositors found for this vault")
