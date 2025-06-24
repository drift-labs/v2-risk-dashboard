from collections import defaultdict
from typing import List, Tuple, TypedDict
import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs

from lib.api import fetch_api_data
from src.utils import get_current_slot

RETRY_DELAY_SECONDS = 5


class PriceData(TypedDict):
    notional: float
    positions: List[Tuple[str, float]]


def is_processing(result):
    """Checks if the API result indicates backend processing."""
    return isinstance(result, dict) and result.get("result") == "processing"


def has_error(result):
    """Checks if the API result indicates an error or is missing essential data."""
    return result is None or (
        isinstance(result, dict) and "liquidations_long" not in result
    )


def plot_liquidation_curves(liquidation_data):
    liquidations_long = liquidation_data["liquidations_long"]
    liquidations_short = liquidation_data["liquidations_short"]
    market_price_ui = liquidation_data["market_price_ui"]

    def filter_outliers(
        liquidations, upper_bound_multiplier=2.0, lower_bound_multiplier=0.5
    ):
        """Filter out liquidations based on a range multiplier of the market price."""
        return [
            (price, notional, pubkey)
            for price, notional, pubkey in liquidations
            if lower_bound_multiplier * market_price_ui
            <= price
            <= upper_bound_multiplier * market_price_ui
        ]

    def aggregate_liquidations(liquidations):
        """Aggregate liquidations to calculate cumulative notional amounts and track pubkeys with their sizes."""
        price_to_data: defaultdict[float, PriceData] = defaultdict(
            lambda: {"notional": 0.0, "positions": []}
        )
        for price, notional, pubkey in liquidations:
            price_to_data[price]["notional"] += notional
            price_to_data[price]["positions"].append((pubkey, notional))
        return price_to_data

    def prepare_data_for_plot(aggregated_data, reverse=False):
        sorted_prices = sorted(aggregated_data.keys(), reverse=reverse)
        cumulative_notional = np.cumsum(
            [aggregated_data[price]["notional"] for price in sorted_prices]
        )
        cumulative_positions = []
        current_positions = []
        for price in sorted_prices:
            current_positions.extend(aggregated_data[price]["positions"])
            cumulative_positions.append(list(current_positions))
        return sorted_prices, cumulative_notional, cumulative_positions

    liquidations_long = filter_outliers(liquidations_long, 2, 0.2)
    liquidations_short = filter_outliers(liquidations_short, 5, 0.5)

    aggregated_long = aggregate_liquidations(liquidations_long)
    aggregated_short = aggregate_liquidations(liquidations_short)

    long_prices, long_cum_notional, long_pubkeys = prepare_data_for_plot(
        aggregated_long, reverse=True
    )
    short_prices, short_cum_notional, short_pubkeys = prepare_data_for_plot(
        aggregated_short
    )

    if not long_prices or not short_prices:
        return None

    long_fig = go.Figure()
    short_fig = go.Figure()

    long_fig.add_trace(
        go.Scatter(
            x=long_prices,
            y=long_cum_notional,
            mode="lines",
            name="Long Positions",
            line=dict(color="purple", width=2),
            hovertemplate="Price: %{x}<br>Cumulative Notional: %{y}<br>Accounts: %{text}<extra></extra>",
            text=[f"{len(pubkeys)} accounts" for pubkeys in long_pubkeys],
        )
    )
    short_fig.add_trace(
        go.Scatter(
            x=short_prices,
            y=short_cum_notional,
            mode="lines",
            name="Short Positions",
            line=dict(color="turquoise", width=2),
            hovertemplate="Price: %{x}<br>Cumulative Notional: %{y}<br>Accounts: %{text}<extra></extra>",
            text=[f"{len(pubkeys)} accounts" for pubkeys in short_pubkeys],
        )
    )

    long_fig.update_layout(
        title="Long Liquidation Curve",
        xaxis_title="Asset Price",
        yaxis_title="Liquidations (Notional)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )
    short_fig.update_layout(
        title="Short Liquidation Curve",
        xaxis_title="Asset Price",
        yaxis_title="Liquidations (Notional)",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )

    return long_fig, short_fig, long_pubkeys, short_pubkeys, long_prices, short_prices


def liquidation_curves_page():
    options = [market.market_index for market in mainnet_perp_market_configs]
    labels = [market.symbol for market in mainnet_perp_market_configs]
    st.write("Liquidation Curves")

    params = st.query_params
    market_index = int(params.get("market_index", "0"))

    market_index = st.selectbox(
        "Market",
        options,
        format_func=lambda x: labels[x],
        index=options.index(market_index),
    )

    st.query_params.update({"market_index": str(market_index)})  # type: ignore

    liquidation_data = None
    try:
        liquidation_data = fetch_api_data(
            "liquidation-curves",
            "liquidation-curves",
            params=dict(st.query_params),
            retry=False,
        )
    except Exception as e:
        st.write(e)
        st.stop()

    if is_processing(liquidation_data):
        st.info(
            f"Backend is processing liquidation data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
        )
        with st.spinner("Please wait..."):
            time.sleep(RETRY_DELAY_SECONDS)
        st.rerun()
        return

    if has_error(liquidation_data):
        st.write("Fetching data for the first time...")
        st.image("images/pacman.gif")
        st.write("Check again in one minute!")
        st.stop()

    # Display slot age information
    slot = liquidation_data.get("slot", 0)
    current_slot = get_current_slot()
    slot_age = current_slot - slot
    st.info(f"Data from slot {slot}, which is {slot_age} slots old.")

    (
        long_fig,
        short_fig,
        long_pubkeys,
        short_pubkeys,
        long_prices,
        short_prices,
    ) = plot_liquidation_curves(liquidation_data) or (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if long_prices is None or short_prices is None:
        st.write("No liquidation data available")
        st.stop()
    else:
        long_col, short_col = st.columns([1, 1])

        with long_col:
            st.plotly_chart(long_fig, use_container_width=True)

            with st.expander("Long Position Accounts"):
                if long_pubkeys and len(long_pubkeys[-1]) > 0:
                    st.write(f"Total Accounts: {len(long_pubkeys[-1])}")
                    long_data = []
                    for i, positions in enumerate(long_pubkeys):
                        if i > 0:
                            new_positions = set(positions) - set(long_pubkeys[i - 1])
                            if new_positions:
                                for pubkey, size in new_positions:
                                    long_data.append(
                                        {
                                            "Price": f"{long_prices[i]:.2f}",
                                            "Size": f"{size:,.2f}",
                                            "Account": pubkey,
                                            "Link": f"https://app.drift.trade/overview?userAccount={pubkey}",
                                        }
                                    )
                    if long_data:
                        st.dataframe(
                            long_data,
                            column_config={
                                "Price": st.column_config.TextColumn("Price"),
                                "Size": st.column_config.TextColumn("Size"),
                                "Account": st.column_config.TextColumn(
                                    "Account", width="large"
                                ),
                                "Link": st.column_config.LinkColumn(
                                    "Link", display_text="View"
                                ),
                            },
                            hide_index=True,
                        )
                else:
                    st.write("No long positions found")

        with short_col:
            st.plotly_chart(short_fig, use_container_width=True)

            with st.expander("Short Position Accounts"):
                if short_pubkeys and len(short_pubkeys[-1]) > 0:
                    st.write(f"Total Accounts: {len(short_pubkeys[-1])}")
                    short_data = []
                    for i, positions in enumerate(short_pubkeys):
                        if i > 0:
                            new_positions = set(positions) - set(
                                short_pubkeys[i - 1]
                            )
                            if new_positions:
                                for pubkey, size in new_positions:
                                    short_data.append(
                                        {
                                            "Price": f"{short_prices[i]:.2f}",
                                            "Size": f"{size:,.2f}",
                                            "Account": pubkey,
                                            "Link": f"https://app.drift.trade/overview?userAccount={pubkey}",
                                        }
                                    )
                    if short_data:
                        st.dataframe(
                            short_data,
                            column_config={
                                "Price": st.column_config.TextColumn("Price"),
                                "Size": st.column_config.TextColumn("Size"),
                                "Account": st.column_config.TextColumn(
                                    "Account", width="large"
                                ),
                                "Link": st.column_config.LinkColumn(
                                    "Link", display_text="View"
                                ),
                            },
                            hide_index=True,
                        )
                else:
                    st.write("No short positions found")
