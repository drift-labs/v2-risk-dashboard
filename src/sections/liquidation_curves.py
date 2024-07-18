from collections import defaultdict
from driftpy.pickle.vat import Vat
from driftpy.constants.numeric_constants import (
    BASE_PRECISION,
    PRICE_PRECISION,
    QUOTE_PRECISION,
)
from driftpy.drift_user import DriftUser
from driftpy.math.margin import MarginCategory
import numpy as np
import plotly.graph_objects as go  # type: ignore
import streamlit as st
from solders.pubkey import Pubkey  # type: ignore
import pandas as pd
from driftpy.constants.perp_markets import mainnet_perp_market_configs

# options = [0, 1, 2]
# labels = ["SOL-PERP", "BTC-PERP", "ETH-PERP"]

options = {market.symbol: market.market_index for market in mainnet_perp_market_configs}


def get_liquidation_curve(vat: Vat, market_index: int, use_liq_buffer=False):
    liquidations_long: list[tuple[float, float, Pubkey]] = []
    liquidations_short: list[tuple[float, float, Pubkey]] = []
    market_price = vat.perp_oracles.get(market_index)
    market_price_ui = market_price.price / PRICE_PRECISION  # type: ignore
    for user in vat.users.user_map.values():
        user: DriftUser = user
        perp_position = user.get_perp_position(market_index)
        if perp_position is not None:
            liquidation_price = user.get_perp_liq_price(market_index)

            if liquidation_price is not None:
                liquidation_price_ui = liquidation_price / PRICE_PRECISION

                if use_liq_buffer:
                    perp_market = user.drift_client.get_perp_market_account(
                        market_index
                    )
                    oracle_str = str(perp_market.amm.oracle)
                    prev_price = user.drift_client.account_subscriber.cache[
                        "oracle_price_data"
                    ][oracle_str].price
                    user.drift_client.account_subscriber.cache["oracle_price_data"][
                        oracle_str
                    ].price = liquidation_price

                    position_notional = (
                        user.get_margin_requirement(MarginCategory.MAINTENANCE, 100)
                        - user.get_total_collateral()
                    ) / QUOTE_PRECISION
                    user.drift_client.account_subscriber.cache["oracle_price_data"][
                        oracle_str
                    ].price = prev_price

                else:
                    position_size = (
                        abs(perp_position.base_asset_amount) / BASE_PRECISION
                    )
                    position_notional = position_size * market_price_ui

                is_zero = round(position_notional) == 0
                is_short = perp_position.base_asset_amount < 0
                is_long = perp_position.base_asset_amount > 0
                if is_zero:
                    continue
                if is_short and liquidation_price_ui > market_price_ui:
                    liquidations_short.append(
                        (liquidation_price_ui, position_notional, user.user_public_key)
                    )
                elif is_long and liquidation_price_ui < market_price_ui:
                    liquidations_long.append(
                        (liquidation_price_ui, position_notional, user.user_public_key)
                    )
                else:
                    pass
                    # print(f"liquidation price for user {user.user_public_key} is {liquidation_price_ui} and market price is {market_price_ui} - is_short: {is_short} - size {position_size} - notional {position_notional}")

    liquidations_long.sort(key=lambda x: x[0])
    liquidations_short.sort(key=lambda x: x[0])

    # for (price, size) in liquidations_long:
    #     print(f"Long liquidation for {size} @ {price}")

    # for (price, size) in liquidations_short:
    #     print(f"Short liquidation for {size} @ {price}")

    return plot_liquidation_curves(
        liquidations_long, liquidations_short, market_price_ui
    ), (liquidations_long, liquidations_short)


def plot_liquidation_curves(liquidations_long, liquidations_short, market_price_ui):
    def filter_outliers(
        liquidations, upper_bound_multiplier=2.0, lower_bound_multiplier=0.5
    ):
        """Filter out liquidations based on a range multiplier of the market price."""
        return [
            (price, notional)
            for price, notional, _ in liquidations
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
        # if reverse:
        #     cumulative_notional = cumulative_notional[::-1]  # Reverse cumulative sum for descending plots
        return sorted_prices, cumulative_notional

    # Filter outliers based on defined criteria
    liquidations_long = filter_outliers(
        liquidations_long, 2, 0.5
    )  # Example multipliers for long positions
    liquidations_short = filter_outliers(
        liquidations_short, 3, 0.5
    )  # Example multipliers for short positions

    # Aggregate and prepare data
    aggregated_long = aggregate_liquidations(liquidations_long)
    aggregated_short = aggregate_liquidations(liquidations_short)

    long_prices, long_cum_notional = prepare_data_for_plot(
        aggregated_long, reverse=True
    )
    short_prices, short_cum_notional = prepare_data_for_plot(aggregated_short)

    st.write(
        "long/short cum notional:", sum(long_cum_notional), sum(short_cum_notional)
    )

    if not long_prices or not short_prices:
        st.warning("No data available for plotting.")
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


def plot_liquidation_curve(vat: Vat):
    st.write("Liquidation Curves")

    default_index = list(options.keys()).index("SOL-PERP")
    market = st.selectbox("Market", options=list(options.keys()), index=default_index)

    market_index = options[market]  # type: ignore

    if market_index is None:
        market_index = 0

    (long_fig, short_fig), (
        liquidations_long,
        liquidations_short,
    ) = get_liquidation_curve(vat, int(market_index), True)
    (long_fig2, short_fig2), (
        liquidations_long2,
        liquidations_short2,
    ) = get_liquidation_curve(vat, int(market_index), False)

    long_col, short_col = st.columns([1, 1])

    use_liq_buffer = st.radio(
        "use liq buffer in details:", [True, False], horizontal=True
    )

    with long_col:
        st.header("liq notional")
        st.plotly_chart(long_fig, use_container_width=True)

        st.header("position notional")
        st.plotly_chart(long_fig2, use_container_width=True)
        with st.expander("user details"):
            st.dataframe(
                pd.DataFrame(
                    liquidations_long2 if use_liq_buffer else liquidations_long,
                    columns=["liq_price", "notional", "user_pubkey"],
                ).sort_values("notional", ascending=False)
            )

    with short_col:
        st.header("liq notional")
        st.plotly_chart(short_fig, use_container_width=True)

        st.header("position notional")
        st.plotly_chart(short_fig2, use_container_width=True)
        with st.expander("user details"):
            st.dataframe(
                pd.DataFrame(
                    liquidations_short2 if use_liq_buffer else liquidations_short,
                    columns=["liq_price", "notional", "user_pubkey"],
                ).sort_values("notional", ascending=False)
            )
