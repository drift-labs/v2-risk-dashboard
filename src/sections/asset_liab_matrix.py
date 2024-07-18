from asyncio import AbstractEventLoop
import pandas as pd  # type: ignore

import streamlit as st
from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat

from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.constants.perp_markets import mainnet_perp_market_configs

from scenario import NUMBER_OF_SPOT  # type: ignore
from cache import get_cached_asset_liab_dfs  # type: ignore

options = [0, 1, 2, 3]
labels = [
    "none",
    "liq within 50% of oracle",
    "maint. health < specified %",
    "init. health < specified %",
]


def get_matrix(
    loop: AbstractEventLoop,
    vat: Vat,
    drift_client: DriftClient,
    env="mainnet",
    mode=0,
    perp_market_inspect=0,
    health=50,
):
    if "asset_liab_data" not in st.session_state:
        st.session_state["asset_liab_data"] = get_cached_asset_liab_dfs(
            drift_client, vat, loop
        )
        (levs_none, levs_init, levs_maint) = st.session_state["asset_liab_data"][0]
        user_keys = st.session_state["asset_liab_data"][1]
    else:
        (levs_none, levs_init, levs_maint) = st.session_state["asset_liab_data"][0]
        user_keys = st.session_state["asset_liab_data"][1]

    if mode == 2:
        levs_maint = [x for x in levs_maint if int(x["health"]) <= int(health)]
    if mode == 3:
        levs_init = [x for x in levs_init if int(x["health"]) <= int(health)]

    df: pd.DataFrame
    match mode:
        case 0:  # nothing
            df = pd.DataFrame(levs_none, index=user_keys)
        case 1:  # liq within 50% of oracle
            df = pd.DataFrame(levs_none, index=user_keys)
        case 2:  # maint. health < 10%
            user_keys = [x["user_key"] for x in levs_init]
            df = pd.DataFrame(levs_init, index=user_keys)
        case 3:  # init. health < 10%
            user_keys = [x["user_key"] for x in levs_maint]
            df = pd.DataFrame(levs_maint, index=user_keys)

    def get_rattt(row):
        calculations = [
            (
                "all_assets",
                lambda v: v if v > 0 else 0,
            ),
            (
                "all",
                lambda v: (
                    v
                    / row["spot_asset"]
                    * (row["perp_liability"] + row["spot_liability"])
                    if v > 0
                    else 0
                ),
            ),
            ("leverage", lambda v: row["leverage"]),
            (
                "all_perp",
                lambda v: v / row["spot_asset"] * row["perp_liability"] if v > 0 else 0,
            ),
            (
                "all_spot",
                lambda v: v / row["spot_asset"] * row["spot_liability"] if v > 0 else 0,
            ),
            (
                f"perp_{perp_market_inspect}_long",
                lambda v: (
                    v / row["spot_asset"] * row["net_p"][perp_market_inspect]
                    if v > 0 and row["net_p"][0] > 0
                    else 0
                ),
            ),
            (
                f"perp_{perp_market_inspect}_short",
                lambda v: (
                    v / row["spot_asset"] * row["net_p"][perp_market_inspect]
                    if v > 0 and row["net_p"][perp_market_inspect] < 0
                    else 0
                ),
            ),
        ]

        series_list = []
        for suffix, calc_func in calculations:
            series = pd.Series([calc_func(val) for key, val in row["net_v"].items()])
            series.index = [f"spot_{x}_{suffix}" for x in series.index]
            series_list.append(series)

        return pd.concat(series_list)

    df = pd.concat([df, df.apply(get_rattt, axis=1)], axis=1)

    def calculate_effective_leverage(group):
        assets = group["all_assets"]
        liabilities = group["all_liabilities"]
        return liabilities / assets if assets != 0 else 0

    def format_with_checkmark(value, condition, mode, financial=False):
        if financial:
            formatted_value = f"${value:,.2f}"
        else:
            formatted_value = f"{value:.2f}"

        if condition and mode > 0:
            return f"{formatted_value} ✅"
        return formatted_value

    res = pd.DataFrame(
        {
            ("spot" + str(i)): (
                df[f"spot_{i}_all_assets"].sum(),
                format_with_checkmark(
                    df[f"spot_{i}_all"].sum(),
                    0 < df[f"spot_{i}_all"].sum() < 1_000_000,
                    mode,
                    financial=True,
                ),
                format_with_checkmark(
                    calculate_effective_leverage(
                        {
                            "all_assets": df[f"spot_{i}_all_assets"].sum(),
                            "all_liabilities": df[f"spot_{i}_all"].sum(),
                        }
                    ),
                    0
                    < calculate_effective_leverage(
                        {
                            "all_assets": df[f"spot_{i}_all_assets"].sum(),
                            "all_liabilities": df[f"spot_{i}_all"].sum(),
                        }
                    )
                    < 2,
                    mode,
                ),
                df[f"spot_{i}_all_spot"].sum(),
                df[f"spot_{i}_all_perp"].sum(),
                df[f"spot_{i}_perp_{perp_market_inspect}_long"].sum(),
                df[f"spot_{i}_perp_{perp_market_inspect}_short"].sum(),
            )
            for i in range(NUMBER_OF_SPOT)
        },
        index=[
            "all_assets",
            "all_liabilities",
            "effective_leverage",
            "all_spot",
            "all_perp",
            f"perp_{perp_market_inspect}_long",
            f"perp_{perp_market_inspect}_short",
        ],
    ).T

    res["all_liabilities"] = res["all_liabilities"].astype(str)
    res["effective_leverage"] = res["effective_leverage"].astype(str)

    if env == "mainnet":  # mainnet_spot_market_configs
        res.index = [x.symbol for x in mainnet_spot_market_configs]
        res.index.name = "spot assets"  # type: ignore

    return res, df


def asset_liab_matrix_page(
    loop: AbstractEventLoop, vat: Vat, drift_client: DriftClient, env="mainnet"
):
    mode = st.selectbox("Mode", options, format_func=lambda x: labels[x])

    if mode is None:
        mode = 0

    perp_market_inspect = st.selectbox(
        "Market index", [x.market_index for x in mainnet_perp_market_configs]
    )

    if perp_market_inspect is None:
        perp_market_inspect = 0

    slider_disabled = mode not in [2, 3]

    health = st.slider(
        "Upper health cutoff (only for init. health and maint. health modes)",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        disabled=slider_disabled,
    )

    if mode not in [1, 2]:
        if "margin" not in st.session_state:
            st.session_state["margin"] = get_matrix(loop, vat, drift_client)
            res = st.session_state["margin"][0]
            df = st.session_state["margin"][1]
        else:
            res = st.session_state["margin"][0]
            df = st.session_state["margin"][1]
    else:
        import time

        start = time.time()
        res, df = get_matrix(
            loop, vat, drift_client, env, mode, perp_market_inspect, health
        )
        st.write(f"matrix in {time.time() - start:.2f} seconds")

    st.write(f"{df.shape[0]} users for scenario")

    st.write(res)

    tabs = st.tabs(["FULL"] + [x.symbol for x in mainnet_spot_market_configs])

    tabs[0].dataframe(df)

    for idx, tab in enumerate(tabs[1:]):
        important_cols = [x for x in df.columns if "spot_" + str(idx) in x]
        toshow = df[["spot_asset", "net_usd_value"] + important_cols]
        toshow = toshow[toshow[important_cols].abs().sum(axis=1) != 0].sort_values(
            by="spot_" + str(idx) + "_all", ascending=False
        )
        tab.write(f"{ len(toshow)} users with this asset to cover liabilities")
        tab.dataframe(toshow)
