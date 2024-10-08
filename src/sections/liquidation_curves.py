from collections import defaultdict

from lib.api import api
import numpy as np
import plotly.graph_objects as go
import streamlit as st


options = [0, 1, 2]
labels = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]


def plot_liquidation_curves(liquidation_data):
    liquidations_long = liquidation_data["liquidations_long"]
    liquidations_short = liquidation_data["liquidations_short"]
    market_price_ui = liquidation_data["market_price_ui"]

    def filter_outliers(
        liquidations, upper_bound_multiplier=2.0, lower_bound_multiplier=0.5
    ):
        """Filter out liquidations based on a range multiplier of the market price."""
        return [
            (price, notional)
            for price, notional in liquidations
            if lower_bound_multiplier * market_price_ui
            <= price
            <= upper_bound_multiplier * market_price_ui
        ]

    def aggregate_liquidations(liquidations):
        """Aggregate liquidations to calculate cumulative notional amounts."""
        price_to_notional = defaultdict(float)
        for price, notional in liquidations:
            price_to_notional[price] += notional
        return price_to_notional

    def prepare_data_for_plot(aggregated_data, reverse=False):
        """Prepare and sort data for plotting, optionally reversing the cumulative sum for descending plots."""
        sorted_prices = sorted(aggregated_data.keys(), reverse=reverse)
        cumulative_notional = np.cumsum(
            [aggregated_data[price] for price in sorted_prices]
        )
        return sorted_prices, cumulative_notional

    # Filter outliers based on defined criteria
    liquidations_long = filter_outliers(
        liquidations_long, 2, 0.2
    )  # Example multipliers for long positions
    liquidations_short = filter_outliers(
        liquidations_short, 5, 0.5
    )  # Example multipliers for short positions

    # Aggregate and prepare data
    aggregated_long = aggregate_liquidations(liquidations_long)
    aggregated_short = aggregate_liquidations(liquidations_short)

    long_prices, long_cum_notional = prepare_data_for_plot(
        aggregated_long, reverse=True
    )
    short_prices, short_cum_notional = prepare_data_for_plot(aggregated_short)

    if not long_prices or not short_prices:
        return None

    # Create Plotly figures
    long_fig = go.Figure()
    short_fig = go.Figure()

    # Add traces for long and short positions
    long_fig.add_trace(
        go.Scatter(
            x=long_prices,
            y=long_cum_notional,
            mode="lines",
            name="Long Positions",
            line=dict(color="purple", width=2),
        )
    )
    short_fig.add_trace(
        go.Scatter(
            x=short_prices,
            y=short_cum_notional,
            mode="lines",
            name="Short Positions",
            line=dict(color="turquoise", width=2),
        )
    )

    # Update layout with axis titles and grid settings
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

    return long_fig, short_fig


def plot_liquidation_curve():  # (vat: Vat):
    st.write("Liquidation Curves")

    market_index = st.selectbox(
        "Market",
        options,
        format_func=lambda x: labels[x],
    )

    if market_index is None:
        market_index = 0

    liquidation_data = api(
        "liquidation", "liquidation-curve", str(market_index), as_json=True
    )
    (long_fig, short_fig) = plot_liquidation_curves(liquidation_data)

    long_col, short_col = st.columns([1, 1])

    with long_col:
        st.plotly_chart(long_fig, use_container_width=True)

    with short_col:
        st.plotly_chart(short_fig, use_container_width=True)
