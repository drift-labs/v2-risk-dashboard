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


from scenario import get_usermap_df


def price_shock_plot(price_scenario_users: list[Any], oracle_distort: float):
    levs = price_scenario_users
    dfs = (
        [pd.DataFrame(levs[2][i]) for i in range(len(levs[2]))]
        + [pd.DataFrame(levs[0])]
        + [pd.DataFrame(levs[1][i]) for i in range(len(levs[1]))]
    )

    spot_bankrs = []
    for df in dfs:
        spot_b_t1 = -(
            df[(df["spot_asset"] < df["spot_liability"]) & (df["net_usd_value"] < 0)]
        )
        spot_bankrs.append(
            -(spot_b_t1["spot_liability"] - spot_b_t1["spot_asset"]).sum()
        )

    xdf = [
        [-df[df["net_usd_value"] < 0]["net_usd_value"].sum() for df in dfs],
        spot_bankrs,
    ]
    toplt_fig = pd.DataFrame(
        xdf,
        index=["bankruptcy", "spot bankrupt"],
        columns=[oracle_distort * (i + 1) * -100 for i in range(len(levs[2]))]
        + [0]
        + [oracle_distort * (i + 1) * 100 for i in range(len(levs[1]))],
    ).T
    toplt_fig["perp bankrupt"] = toplt_fig["bankruptcy"] - toplt_fig["spot bankrupt"]
    toplt_fig = toplt_fig.sort_index()
    toplt_fig = toplt_fig.plot()
    # Customize the layout if needed
    toplt_fig.update_layout(
        title="Bankruptcies in crypto price scenarios",
        xaxis_title="Oracle Move (%)",
        yaxis_title="Bankruptcy ($)",
    )
    st.plotly_chart(toplt_fig)


def plot_page(loop: AbstractEventLoop, vat: Vat, drift_client: DriftClient):
    if st.session_state["context"] == False:
        st.write("Please load dashboard before viewing this page")
    else:
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

        user_keys = list(vat.users.user_map.keys())
        st.write(len(user_keys), "drift users")
        start_time = time.time()

        price_scenario_users, user_keys, distorted_oracles = loop.run_until_complete(
            get_usermap_df(
                drift_client, vat.users, "oracles", oracle_distort, None, cov
            )
        )
        # levs[0]
        end_time = time.time()
        time_to_run = end_time - start_time
        st.write(
            time_to_run,
            "seconds to run",
            1 + len(price_scenario_users[1]) + len(price_scenario_users[2]),
            "price-shock scenarios",
        )

        price_shock_plot(price_scenario_users, oracle_distort)

        oracle_down_max = pd.DataFrame(price_scenario_users[-1][-1], index=user_keys)
        with st.expander(
            str("oracle down max bankrupt count=")
            + str(len(oracle_down_max[oracle_down_max.net_usd_value < 0]))
        ):
            st.dataframe(oracle_down_max)

        oracle_up_max = pd.DataFrame(price_scenario_users[1][-1], index=user_keys)
        with st.expander(
            str("oracle up max bankrupt count=")
            + str(len(oracle_up_max[oracle_up_max.net_usd_value < 0]))
        ):
            st.dataframe(oracle_up_max)

        with st.expander("distorted oracle keys"):
            st.write(distorted_oracles)
