import asyncio
import io
import time
import os
import aiohttp
import msgpack
import zipfile

import pandas as pd  # type: ignore
import streamlit as st

from typing import Any
import datetime as dt
from asyncio import AbstractEventLoop

from solana.rpc.async_api import AsyncClient

from anchorpy import Wallet

from driftpy.drift_client import DriftClient
from driftpy.account_subscription_config import AccountSubscriptionConfig

from health_utils import *
from utils import load_newest_files, load_vat, clear_local_pickles
from sections.asset_liab_matrix import asset_liab_matrix_page
from sections.ob import ob_cmp_page
from sections.scenario import plot_page
from sections.liquidation_curves import plot_liquidation_curve
from sections.margin_model import margin_model


SERVER_URL = "http://54.74.185.225:8080"


async def fetch_context(session: aiohttp.ClientSession, req: str) -> dict[str, Any]:
    async with session.get(req) as response:
        return msgpack.unpackb(await response.read(), strict_map_key=False)


async def fetch_pickles(session: aiohttp.ClientSession, req: str) -> dict[str, Any]:
    async with session.get(req) as response:
        content = await response.read()
        with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
            zip_ref.extractall(os.getcwd() + "/pickles")


async def setup_context(dc: DriftClient, loop: AbstractEventLoop, env):
    start_dashboard_ready = time.time()
    async with aiohttp.ClientSession() as session:
        print("fetching context")
        start = time.time()

        tasks = [
            fetch_pickles(session, f"{SERVER_URL}/pickles"),
            fetch_context(session, f"{SERVER_URL}/{env}_context"),
        ]
        _, context_data = await asyncio.gather(*tasks)
        print("context fetched in ", time.time() - start)

        filepath = os.getcwd() + "/pickles"
        newest_snapshot = load_newest_files(filepath)
        start_load_vat = time.time()
        vat = await load_vat(dc, newest_snapshot, loop, env)
        clear_local_pickles(filepath)
        st.session_state["vat"] = vat
        print(f"loaded vat in {time.time() - start_load_vat}")

        levs = [
            context_data["levs_none"],
            context_data["levs_init"],
            context_data["levs_maint"],
        ]
        user_keys = context_data["user_keys"]
        margin = [pd.DataFrame(context_data["res"]), pd.DataFrame(context_data["df"])]

        st.session_state["margin"] = tuple(margin)
        st.session_state["asset_liab_data"] = tuple(levs), user_keys
    st.session_state["context"] = True
    print(f"dashboard ready in: {time.time() - start_dashboard_ready}")


def main():
    st.set_page_config(layout="wide")

    query_index = 0

    env = st.sidebar.radio("Environment:", ["prod", "dev"])

    def query_string_callback():
        st.query_params["tab"] = st.session_state.query_key

    query_tab = st.query_params.get("tab", ["Welcome"])[0]
    tab_options = (
        "Welcome",
        "Health",
        "Price-Shock",
        "Asset-Liab-Matrix",
        "Orderbook",
        "Liquidations",
        "Margin-Model",
    )
    for idx, x in enumerate(tab_options):
        if x.lower() == query_tab.lower():
            query_index = idx

    tab = st.sidebar.radio(
        "Select Tab:",
        tab_options,
        query_index,
        on_change=query_string_callback,
        key="query_key",
    )

    if tab is None:
        tab = "Welcome"

    drift_client = DriftClient(
        AsyncClient("https://api.mainnet-beta.solana.com/"),
        Wallet.dummy(),
        account_subscription=AccountSubscriptionConfig("cached"),
    )

    def func():
        if (
            tab.lower()
            in [
                "welcome",
                "health",
                "price-shock",
                "asset-liab-matrix",
                "liquidations",
                "margin-model",
            ]
            and "vat" not in st.session_state
        ):
            md = st.empty()
            md.markdown("`Loading dashboard, do not leave this page`")
            if st.session_state["context"] == False:
                loop.run_until_complete(setup_context(drift_client, loop, env))
            md.markdown("`Dashboard ready!`")

    st.sidebar.button("Start Dashboard", on_click=func)

    loop: AbstractEventLoop = asyncio.new_event_loop()
    st.session_state["context"] = False

    if tab.lower() == "welcome":
        st.header("Welcome to the Drift v2 Risk Analytics Dashboard!")
        st.metric(
            "protocol has been live for:",
            str(
                int(
                    (dt.datetime.now() - pd.to_datetime("2022-11-05")).total_seconds()
                    / (60 * 60 * 24)
                )
            )
            + " days",
        )
        st.write(
            "Click `Start Dashboard` to load the dashboard on the selected `Environment`"
        )

    if tab.lower() in [
        "health",
        "price-shock",
        "asset-liab-matrix",
        "liquidations",
        "margin model",
    ]:
        vat = st.session_state["vat"]

    if tab.lower() == "health":
        health_distribution = get_account_health_distribution(vat)

        with st.container():
            st.plotly_chart(health_distribution, use_container_width=True)

        perp_col, spot_col = st.columns([1, 1])

        with perp_col:
            largest_perp_positions = get_largest_perp_positions(vat)
            st.markdown("### **Largest perp positions:**")
            st.table(largest_perp_positions)
            most_levered_positions = get_most_levered_perp_positions_above_1m(vat)
            st.markdown("### **Most levered perp positions > $1m:**")
            st.table(most_levered_positions)

        with spot_col:
            largest_spot_borrows = get_largest_spot_borrows(vat)
            st.markdown("### **Largest spot borrows:**")
            st.table(largest_spot_borrows)
            most_levered_borrows = get_most_levered_spot_borrows_above_1m(vat)
            st.markdown("### **Most levered spot borrows > $750k:**")
            st.table(most_levered_borrows)

    elif tab.lower() == "price-shock":
        plot_page(loop, vat, drift_client)
    elif tab.lower() == "asset-liab-matrix":
        asset_liab_matrix_page(loop, vat, drift_client)
    elif tab.lower() == "orderbook":
        ob_cmp_page()
    elif tab.lower() == "liquidations":
        plot_liquidation_curve(vat)
    elif tab.lower() == "margin-model":
        margin_model(loop, drift_client)


main()
