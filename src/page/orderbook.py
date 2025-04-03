from typing import Dict, Optional, Tuple

import requests
import streamlit as st

HL_BASE_URL = "https://api.hyperliquid.xyz/info"
DRIFT_BASE_URL = "https://dlob.drift.trade/l2"


def fetch_hyperliquid_data(coin: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    meta = requests.post(
        HL_BASE_URL,
        json={"type": "metaAndAssetCtxs"},
        headers={"Content-Type": "application/json"},
    ).json()
    book = requests.post(
        HL_BASE_URL,
        json={"type": "l2Book", "coin": coin},
        headers={"Content-Type": "application/json"},
    ).json()
    return meta, book


def fetch_drift_data(coin: str) -> Optional[Dict]:
    params = {
        "marketName": f"{coin}-PERP",
        "depth": 500,
        "includeOracle": "true",
        "includeVamm": "true",
    }
    return requests.get(DRIFT_BASE_URL, params=params).json()


def calculate_avg_fill_price(
    levels: list, volume: float, is_drift: bool = False
) -> float:
    total_volume = 0
    total_cost = 0.0

    for level in levels:
        if is_drift:
            price = float(level["price"]) / 1e6
            size = float(level["size"]) / 1e9
        else:
            price = float(level["px"])
            size = float(level["sz"])

        if total_volume + size >= volume:
            remaining_volume = volume - total_volume
            total_cost += remaining_volume * price
            total_volume += remaining_volume
            break
        else:
            total_cost += size * price
            total_volume += size

    if total_volume < volume:
        raise ValueError("Insufficient volume in the order book")

    return total_cost / volume


def get_orderbook_data(coin: str, size: float) -> Tuple[Dict, Dict, float, Dict]:
    hl_meta, hl_book = fetch_hyperliquid_data(coin)
    drift_book = fetch_drift_data(coin)

    try:
        hl_buy = calculate_avg_fill_price(hl_book["levels"][1], size)
        hl_sell = calculate_avg_fill_price(hl_book["levels"][0], size)
        hl_prices = {"average_buy_price": hl_buy, "average_sell_price": hl_sell}
    except (ValueError, KeyError) as e:
        hl_prices = str(e)

    try:
        drift_buy = calculate_avg_fill_price(drift_book["asks"], size, is_drift=True)
        drift_sell = calculate_avg_fill_price(drift_book["bids"], size, is_drift=True)
        drift_prices = {
            "average_buy_price": drift_buy,
            "average_sell_price": drift_sell,
        }
    except (ValueError, KeyError) as e:
        drift_prices = str(e)

    return hl_prices, drift_prices, drift_book["oracle"] / 1e6, hl_meta


def orderbook_page():
    st.title("Orderbook comparison")
    col1, col2 = st.columns(2)
    coin = col1.selectbox("Coin:", ["SOL", "BTC", "ETH"])
    size = col2.number_input("Size:", min_value=0.1, value=1.0, help="in base units")

    hl_prices, drift_prices, drift_oracle, hl_meta = get_orderbook_data(coin, size)

    coin_idx = next(
        i for i, x in enumerate(hl_meta[0]["universe"]) if coin == x["name"]
    )
    col1, col2 = st.columns(2)

    with col1:
        st.header("Hyperliquid")
        st.write(float(hl_meta[1][coin_idx]["oraclePx"]))
        st.write(hl_prices)

    with col2:
        st.header("Drift")
        st.write(drift_oracle)
        st.write(drift_prices)

    if st.button("Refresh"):
        st.cache_data.clear()
