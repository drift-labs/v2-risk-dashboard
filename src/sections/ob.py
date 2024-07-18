import asyncio
import heapq
import time
import os

from asyncio import AbstractEventLoop
import plotly.express as px  # type: ignore
import pandas as pd  # type: ignore

from typing import Any
import streamlit as st
from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat

from driftpy.constants.spot_markets import (
    mainnet_spot_market_configs,
    devnet_spot_market_configs,
)
from driftpy.constants.perp_markets import (
    mainnet_perp_market_configs,
    devnet_perp_market_configs,
)

from scenario import get_usermap_df
import requests


def fetch_ob_data(coin, size):
    # Define the POST request details
    post_url = "https://api.hyperliquid.xyz/info"
    payload = {"type": "metaAndAssetCtxs"}
    payload2 = {"type": "l2Book", "coin": coin}

    post_headers = {"Content-Type": "application/json"}

    results = {}

    for nom, pay in [("hl_cxt", payload), ("hl_book", payload2)]:
        # Send the POST request
        post_response = requests.post(post_url, json=pay, headers=post_headers)
        # Print the POST request response
        print("POST request to https://api.hyperliquid.xyz/info")
        print("Status Code:", post_response.status_code)
        if post_response.status_code == 200:
            print("Response JSON:", post_response.json(), "\n")
            results[nom] = post_response.json()
        else:
            print("Error:", post_response.text, "\n")

    # Define the GET request URL
    get_url = "https://dlob.drift.trade/l2"
    get_params = {
        "marketName": coin + "-PERP",
        "depth": 5,
        "includeOracle": "true",
        "includeVamm": "true",
    }

    # Send the GET request
    get_response = requests.get(get_url, params=get_params)

    # Print the GET request response
    print("GET request to https://dlob.drift.trade/l2")
    print("Status Code:", get_response.status_code)
    if get_response.status_code == 200:
        print("Response JSON:", get_response.json())
    else:
        print("Error:", get_response.text)

    results["dr_book"] = get_response.json()

    def calculate_average_fill_price_dr(order_book, volume):
        # Adjust volume to match size precision (1e9)
        volume = volume

        bids = order_book["bids"]
        asks = order_book["asks"]

        print(f'{float(bids[0]["price"])/1e6}/{float(asks[0]["price"])/1e6}')

        def average_price(levels, volume, is_buy):
            total_volume = 0
            total_cost = 0.0

            for level in levels:
                # Price is in 1e6 precision, size is in 1e9 precision
                price = float(level["price"]) / 1e6
                size = float(level["size"]) / 1e9

                if total_volume + size >= volume:
                    # Only take the remaining required volume at this level
                    remaining_volume = volume - total_volume
                    total_cost += remaining_volume * price
                    total_volume += remaining_volume
                    break
                else:
                    # Take the whole size at this level
                    total_cost += size * price
                    total_volume += size

            if total_volume < volume:
                raise ValueError(
                    "Insufficient volume in the order book to fill the order"
                )

            return total_cost / volume

        try:
            buy_price = average_price(asks, volume, is_buy=True)
            sell_price = average_price(bids, volume, is_buy=False)
        except ValueError as e:
            return str(e)

        return {"average_buy_price": buy_price, "average_sell_price": sell_price}

    def calculate_average_fill_price_hl(order_book, volume):
        buy_levels = order_book["levels"][1]  # Bids (lower prices first)
        sell_levels = order_book["levels"][0]  # Asks (higher prices first)

        def average_price(levels, volume):
            total_volume = 0
            total_cost = 0.0

            for level in levels:
                px = float(level["px"])
                sz = float(level["sz"])

                if total_volume + sz >= volume:
                    # Only take the remaining required volume at this level
                    remaining_volume = volume - total_volume
                    total_cost += remaining_volume * px
                    total_volume += remaining_volume
                    break
                else:
                    # Take the whole size at this level
                    total_cost += sz * px
                    total_volume += sz

            if total_volume < volume:
                raise ValueError(
                    "Insufficient volume in the order book to fill the order"
                )

            return total_cost / volume

        try:
            buy_price = average_price(buy_levels, volume)
            sell_price = average_price(sell_levels, volume)
        except ValueError as e:
            return str(e)

        return {"average_buy_price": buy_price, "average_sell_price": sell_price}

    r = calculate_average_fill_price_hl(results["hl_book"], size)
    d = calculate_average_fill_price_dr(results["dr_book"], size)
    return (r, d, results["dr_book"]["oracle"] / 1e6, results["hl_cxt"])


def ob_cmp_page():
    if st.button("Refresh"):
        st.cache_data.clear()
    s1, s2 = st.columns(2)

    coin = s1.selectbox("coin:", ["SOL", "BTC", "ETH"])
    size = s2.number_input("size:", min_value=0.1, value=1.0, help="in base units")
    hl, dr, dr_oracle, hl_ctx = fetch_ob_data(coin, size)

    uni_id = [i for (i, x) in enumerate(hl_ctx[0]["universe"]) if coin == x["name"]]
    # hl_oracle = hl_ctx

    o1, o2 = st.columns(2)
    o1.header("hyperliquid")
    o1.write(float(hl_ctx[1][uni_id[0]]["oraclePx"]))
    o1.write(hl)

    o2.header("drift")
    o2.write(dr_oracle)
    o2.write(dr)
