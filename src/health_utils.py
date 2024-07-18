import asyncio
import heapq
import time
import os

from asyncio import AbstractEventLoop
import plotly.express as px  # type: ignore
import pandas as pd  # type: ignore

from typing import Any

from solana.rpc.async_api import AsyncClient

from anchorpy import Wallet

import streamlit as st
from driftpy.drift_user import DriftUser
from driftpy.drift_client import DriftClient
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.constants.numeric_constants import (
    BASE_PRECISION,
    SPOT_BALANCE_PRECISION,
    PRICE_PRECISION,
)
from driftpy.types import is_variant
from driftpy.pickle.vat import Vat
from driftpy.constants.spot_markets import (
    mainnet_spot_market_configs,
    devnet_spot_market_configs,
)
from driftpy.constants.perp_markets import (
    mainnet_perp_market_configs,
    devnet_perp_market_configs,
)

from utils import load_newest_files, load_vat, to_financial


def get_largest_perp_positions(vat: Vat):
    top_positions: list[Any] = []

    for user in vat.users.values():
        for position in user.get_user_account().perp_positions:
            if position.base_asset_amount > 0:
                market_price = vat.perp_oracles.get(position.market_index)
                if market_price is not None:
                    market_price_ui = market_price.price / PRICE_PRECISION
                    base_asset_value = (
                        abs(position.base_asset_amount) / BASE_PRECISION
                    ) * market_price_ui
                    heap_item = (
                        to_financial(base_asset_value),
                        user.user_public_key,
                        position.market_index,
                        position.base_asset_amount / BASE_PRECISION,
                    )

                    if len(top_positions) < 10:
                        heapq.heappush(top_positions, heap_item)
                    else:
                        heapq.heappushpop(top_positions, heap_item)

    positions = sorted(
        (value, pubkey, market_idx, amt)
        for value, pubkey, market_idx, amt in top_positions
    )

    positions.reverse()

    data = {
        "Market Index": [pos[2] for pos in positions],
        "Value": [f"${pos[0]:,.2f}" for pos in positions],
        "Base Asset Amount": [f"{pos[3]:,.2f}" for pos in positions],
        "Public Key": [pos[1] for pos in positions],
    }

    return data


def get_largest_spot_borrows(vat: Vat):
    top_borrows: list[Any] = []

    for user in vat.users.values():
        for position in user.get_user_account().spot_positions:
            if position.scaled_balance > 0 and is_variant(
                position.balance_type, "Borrow"
            ):
                market_price = vat.spot_oracles.get(position.market_index)
                if market_price is not None:
                    market_price_ui = market_price.price / PRICE_PRECISION
                    borrow_value = (
                        position.scaled_balance / SPOT_BALANCE_PRECISION
                    ) * market_price_ui
                    heap_item = (
                        to_financial(borrow_value),
                        user.user_public_key,
                        position.market_index,
                        position.scaled_balance / SPOT_BALANCE_PRECISION,
                    )

                    if len(top_borrows) < 10:
                        heapq.heappush(top_borrows, heap_item)
                    else:
                        heapq.heappushpop(top_borrows, heap_item)

    borrows = sorted(
        (value, pubkey, market_idx, amt)
        for value, pubkey, market_idx, amt in top_borrows
    )

    borrows.reverse()

    data = {
        "Market Index": [pos[2] for pos in borrows],
        "Value": [f"${pos[0]:,.2f}" for pos in borrows],
        "Scaled Balance": [f"{pos[3]:,.2f}" for pos in borrows],
        "Public Key": [pos[1] for pos in borrows],
    }

    return data


def get_account_health_distribution(vat: Vat):
    health_notional_distributions = {
        "0-10%": 0,
        "10-20%": 0,
        "20-30%": 0,
        "30-40%": 0,
        "40-50%": 0,
        "50-60%": 0,
        "60-70%": 0,
        "70-80%": 0,
        "80-90%": 0,
        "90-100%": 0,
    }
    health_counts = {
        "0-10%": 0,
        "10-20%": 0,
        "20-30%": 0,
        "30-40%": 0,
        "40-50%": 0,
        "50-60%": 0,
        "60-70%": 0,
        "70-80%": 0,
        "80-90%": 0,
        "90-100%": 0,
    }

    for user in vat.users.values():
        total_collateral = user.get_total_collateral() / PRICE_PRECISION
        current_health = user.get_health()
        match current_health:
            case _ if current_health < 10:
                health_notional_distributions["0-10%"] += total_collateral
                health_counts["0-10%"] += 1
            case _ if current_health < 20:
                health_notional_distributions["10-20%"] += total_collateral
                health_counts["10-20%"] += 1
            case _ if current_health < 30:
                health_notional_distributions["20-30%"] += total_collateral
                health_counts["20-30%"] += 1
            case _ if current_health < 40:
                health_notional_distributions["30-40%"] += total_collateral
                health_counts["30-40%"] += 1
            case _ if current_health < 50:
                health_notional_distributions["40-50%"] += total_collateral
                health_counts["40-50%"] += 1
            case _ if current_health < 60:
                health_notional_distributions["50-60%"] += total_collateral
                health_counts["50-60%"] += 1
            case _ if current_health < 70:
                health_notional_distributions["60-70%"] += total_collateral
                health_counts["60-70%"] += 1
            case _ if current_health < 80:
                health_notional_distributions["70-80%"] += total_collateral
                health_counts["70-80%"] += 1
            case _ if current_health < 90:
                health_notional_distributions["80-90%"] += total_collateral
                health_counts["80-90%"] += 1
            case _:
                health_notional_distributions["90-100%"] += total_collateral
                health_counts["90-100%"] += 1
    df = pd.DataFrame(
        {
            "Health Range": list(health_counts.keys()),
            "Counts": list(health_counts.values()),
            "Notional Values": list(health_notional_distributions.values()),
        }
    )

    fig = px.bar(
        df,
        x="Health Range",
        y="Counts",
        title="Health Distribution",
        hover_data={"Notional Values": ":,"},  # Custom format for notional values
        labels={"Counts": "Num Users", "Notional Values": "Notional Value ($)"},
    )

    fig.update_traces(
        hovertemplate="<b>Health Range: %{x}</b><br>Count: %{y}<br>Notional Value: $%{customdata[0]:,.0f}<extra></extra>"
    )

    return fig


def get_most_levered_perp_positions_above_1m(vat: Vat):
    top_positions: list[Any] = []

    for user in vat.users.values():
        total_collateral = user.get_total_collateral() / PRICE_PRECISION
        if total_collateral > 0:
            for position in user.get_user_account().perp_positions:
                if position.base_asset_amount > 0:
                    market_price = vat.perp_oracles.get(position.market_index)
                    if market_price is not None:
                        market_price_ui = market_price.price / PRICE_PRECISION
                        base_asset_value = (
                            abs(position.base_asset_amount) / BASE_PRECISION
                        ) * market_price_ui
                        leverage = base_asset_value / total_collateral
                        if base_asset_value > 1_000_000:
                            heap_item = (
                                to_financial(base_asset_value),
                                user.user_public_key,
                                position.market_index,
                                position.base_asset_amount / BASE_PRECISION,
                                leverage,
                            )

                            if len(top_positions) < 10:
                                heapq.heappush(top_positions, heap_item)
                            else:
                                heapq.heappushpop(top_positions, heap_item)

    positions = sorted(
        top_positions,  # We can sort directly the heap result
        key=lambda x: x[
            4
        ],  # Sort by leverage, which is the fifth element in your tuple
    )

    positions.reverse()

    data = {
        "Market Index": [pos[2] for pos in positions],
        "Value": [f"${pos[0]:,.2f}" for pos in positions],
        "Base Asset Amount": [f"{pos[3]:,.2f}" for pos in positions],
        "Leverage": [f"{pos[4]:,.2f}" for pos in positions],
        "Public Key": [pos[1] for pos in positions],
    }

    return data


def get_most_levered_spot_borrows_above_1m(vat: Vat):
    top_borrows: list[Any] = []

    for user in vat.users.values():
        total_collateral = user.get_total_collateral() / PRICE_PRECISION
        if total_collateral > 0:
            for position in user.get_user_account().spot_positions:
                if (
                    is_variant(position.balance_type, "Borrow")
                    and position.scaled_balance > 0
                ):
                    market_price = vat.spot_oracles.get(position.market_index)
                    if market_price is not None:
                        market_price_ui = market_price.price / PRICE_PRECISION
                        borrow_value = (
                            position.scaled_balance / SPOT_BALANCE_PRECISION
                        ) * market_price_ui
                        leverage = borrow_value / total_collateral
                        if borrow_value > 750_000:
                            heap_item = (
                                to_financial(borrow_value),
                                user.user_public_key,
                                position.market_index,
                                position.scaled_balance / SPOT_BALANCE_PRECISION,
                                leverage,
                            )

                            if len(top_borrows) < 10:
                                heapq.heappush(top_borrows, heap_item)
                            else:
                                heapq.heappushpop(top_borrows, heap_item)

    borrows = sorted(
        top_borrows,
        key=lambda x: x[4],
    )

    borrows.reverse()

    data = {
        "Market Index": [pos[2] for pos in borrows],
        "Value": [f"${pos[0]:,.2f}" for pos in borrows],
        "Scaled Balance": [f"{pos[3]:,.2f}" for pos in borrows],
        "Leverage": [f"{pos[4]:,.2f}" for pos in borrows],
        "Public Key": [pos[1] for pos in borrows],
    }

    return data
