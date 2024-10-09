import asyncio
from asyncio import AbstractEventLoop
import os
import time
from typing import Any, TypedDict

from anchorpy import Wallet
from driftpy.account_subscription_config import AccountSubscriptionConfig
from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat
from lib.api import api
import pandas as pd
import plotly.graph_objects as go
from solana.rpc.async_api import AsyncClient
import streamlit as st


class UserLeveragesResponse(TypedDict):
    leverages_none: list[Any]
    leverages_up: list[Any]
    leverages_down: list[Any]
    user_keys: list[str]
    distorted_oracles: list[str]


def create_dataframes(leverages):
    return [pd.DataFrame(lev) for lev in leverages]


def calculate_spot_bankruptcies(df):
    spot_bankrupt = df[
        (df["spot_asset"] < df["spot_liability"]) & (df["net_usd_value"] < 0)
    ]
    return (spot_bankrupt["spot_liability"] - spot_bankrupt["spot_asset"]).sum()


def calculate_total_bankruptcies(df):
    return -df[df["net_usd_value"] < 0]["net_usd_value"].sum()


def generate_oracle_moves(num_scenarios, oracle_distort):
    return (
        [-oracle_distort * (i + 1) * 100 for i in range(num_scenarios)]
        + [0]
        + [oracle_distort * (i + 1) * 100 for i in range(num_scenarios)]
    )


def price_shock_plot(user_leverages, oracle_distort: float):
    levs = user_leverages
    dfs = (
        create_dataframes(levs["leverages_down"])
        + [pd.DataFrame(levs["leverages_none"])]
        + create_dataframes(levs["leverages_up"])
    )

    spot_bankruptcies = [calculate_spot_bankruptcies(df) for df in dfs]
    total_bankruptcies = [calculate_total_bankruptcies(df) for df in dfs]

    num_scenarios = len(levs["leverages_down"])
    oracle_moves = generate_oracle_moves(num_scenarios, oracle_distort)

    df_plot = pd.DataFrame(
        {
            "Oracle Move (%)": oracle_moves,
            "Total Bankruptcy ($)": total_bankruptcies,
            "Spot Bankruptcy ($)": spot_bankruptcies,
        }
    ).sort_values("Oracle Move (%)")

    df_plot["Perp Bankruptcy ($)"] = (
        df_plot["Total Bankruptcy ($)"] - df_plot["Spot Bankruptcy ($)"]
    )

    fig = go.Figure()
    for column in [
        "Total Bankruptcy ($)",
        "Spot Bankruptcy ($)",
        "Perp Bankruptcy ($)",
    ]:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Oracle Move (%)"],
                y=df_plot[column],
                mode="lines+markers",
                name=column,
            )
        )

    fig.update_layout(
        title="Bankruptcies in Crypto Price Scenarios",
        xaxis_title="Oracle Move (%)",
        yaxis_title="Bankruptcy ($)",
        legend_title="Bankruptcy Type",
    )

    return fig


def price_shock_page():
    cov_col, distort_col = st.columns(2)
    cov = cov_col.selectbox(
        "covariance:",
        [
            "ignore stables",
            "sol + lst only",
            "meme",
        ],
        index=0,
    )

    oracle_distort = distort_col.selectbox(
        "oracle distortion:",
        [0.05, 0.1, 0.2, 0.5, 1],
        index=0,
        help="step size of oracle distortions",
    )

    result = api(
        "price-shock",
        "usermap",
        params={
            "asset_group": cov,
            "oracle_distortion": oracle_distort,
            "n_scenarios": 5,
        },
        as_json=True,
    )
    st.write(result)

    fig = price_shock_plot(result, oracle_distort)
    st.plotly_chart(fig)

    oracle_down_max = pd.DataFrame(
        result["leverages_down"][-1][-1], index=result["user_keys"]
    )
    with st.expander(
        str("oracle down max bankrupt count=")
        + str(len(oracle_down_max[oracle_down_max.net_usd_value < 0]))
    ):
        st.dataframe(oracle_down_max)

    oracle_up_max = pd.DataFrame(
        result["leverages_up"][-1][-1], index=result["user_keys"]
    )
    with st.expander(
        str("oracle up max bankrupt count=")
        + str(len(oracle_up_max[oracle_up_max.net_usd_value < 0]))
    ):
        st.dataframe(oracle_up_max)

    with st.expander("distorted oracle keys"):
        st.write(result["distorted_oracles"])
