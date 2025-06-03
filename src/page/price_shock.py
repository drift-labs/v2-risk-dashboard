import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.api import fetch_cached_data
from shared.types import PriceShockAssetGroup
from utils import get_current_slot


def price_shock_plot(df_plot):
    fig = go.Figure()
    for column in [
        "Total Bankruptcy ($)",
        "Spot Bankruptcy ($)",
        "Perpetual Bankruptcy ($)",
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
        title="Bankruptcies in Cryptocurrency Price Scenarios",
        xaxis_title="Oracle Move (%)",
        yaxis_title="Bankruptcy ($)",
        legend_title="Bankruptcy Type",
        template="plotly_dark",
    )

    return fig


def price_shock_cached_page():
    params = st.query_params
    asset_group = params.get("asset_group", PriceShockAssetGroup.IGNORE_STABLES.value)
    n_scenarios_param = params.get("n_scenarios", 5)
    pool_id_param = params.get("pool_id", "all")

    asset_groups = [
        PriceShockAssetGroup.IGNORE_STABLES.value,
        PriceShockAssetGroup.JLP_ONLY.value,
    ]

    asset_group = st.selectbox(
        "Asset Group", asset_groups, index=asset_groups.index(asset_group)
    )
    st.query_params.update({"asset_group": asset_group})

    scenario_options = [5, 10]
    radio = st.radio(
        "Scenarios",
        scenario_options,
        index=scenario_options.index(int(n_scenarios_param)),
        key="n_scenarios",
    )
    n_scenarios = radio

    st.query_params.update({"n_scenarios": n_scenarios})
    
    # Pool ID selection
    pool_options = ["all", "0", "1", "3"]
    pool_display_names = ["All Pools", "Main Pool (0)", "Isolated Pool 1", "Isolated Pool 3"]
    
    try:
        pool_index = pool_options.index(str(pool_id_param))
    except ValueError:
        pool_index = 0  # Default to "all"
    
    pool_selection = st.selectbox(
        "Pool Filter", 
        pool_display_names,
        index=pool_index,
        help="Filter positions by pool ID. Main Pool (0) has more lenient parameters, Isolated Pools (>0) have stricter risk parameters."
    )
    
    # Map display name back to value
    selected_pool_id = pool_options[pool_display_names.index(pool_selection)]
    st.query_params.update({"pool_id": selected_pool_id})
    
    if n_scenarios == 5:
        oracle_distort = 0.05
    else:
        oracle_distort = 0.1
        
    # Prepare API parameters
    api_params = {
        "asset_group": asset_group,
        "oracle_distortion": oracle_distort,
        "n_scenarios": n_scenarios,
    }
    
    # Add pool_id to params if not "all"
    if selected_pool_id != "all":
        api_params["pool_id"] = int(selected_pool_id)
    
    # Create cache key that includes pool_id
    cache_key_parts = [
        "price-shock/usermap",
        asset_group,
        str(oracle_distort),
        str(n_scenarios),
        selected_pool_id
    ]
    cache_key = "_".join(cache_key_parts)
    
    try:
        result = fetch_cached_data(
            "price-shock/usermap",
            _params=api_params,
            key=cache_key,
        )
    except Exception as e:
        print("HIT AN EXCEPTION...", e)
        st.error("Failed to fetch data")
        return

    if "result" in result and result["result"] == "miss":
        st.write("Fetching data for the first time...")
        st.image(
            "https://i.gifer.com/origin/8a/8a47f769c400b0b7d81a8f6f8e09a44a_w200.gif"
        )
        st.write("Check again in one minute!")
        st.stop()

    current_slot = get_current_slot()
    pool_display = f" (Pool {selected_pool_id})" if selected_pool_id != "all" else " (All Pools)"
    st.info(
        f"This data is for slot {result['slot']}{pool_display}, which is now {int(current_slot) - int(result['slot'])} slots old"
    )
    df_plot = pd.DataFrame(json.loads(result["result"]))

    fig = price_shock_plot(df_plot)
    st.plotly_chart(fig)

    col1, col2 = st.columns(2)
    with col1:
        df_liquidations = df_plot.drop(
            columns=["Spot Bankruptcy ($)", "Total Bankruptcy ($)"]
        )
        df_liquidations.rename(
            columns={
                "Perpetual Bankruptcy ($)": "Liquidations ($)",
                "Oracle Move (%)": "Oracle Move (%)",
            },
            inplace=True,
        )
        st.dataframe(df_liquidations)

    oracle_down_max = pd.DataFrame(json.loads(result["oracle_down_max"]))
    oracle_up_max = pd.DataFrame(json.loads(result["oracle_up_max"]))

    with col2:
        df_bad_debts = df_plot.drop(
            columns=["Perpetual Bankruptcy ($)", "Total Bankruptcy ($)"]
        )
        df_bad_debts.rename(
            columns={
                "Spot Bankruptcy ($)": "Bad Debts ($)",
                "Oracle Move (%)": "Oracle Move (%)",
            },
            inplace=True,
        )
        st.dataframe(df_bad_debts)

    with st.expander(
        str("oracle down max bankrupt count=")
        + str(len(oracle_down_max[oracle_down_max.net_usd_value < 0]))
    ):
        st.dataframe(oracle_down_max)

    with st.expander(
        str("oracle up max bankrupt count=")
        + str(len(oracle_up_max[oracle_up_max.net_usd_value < 0]))
    ):
        st.dataframe(oracle_up_max)

    with st.expander("distorted oracle keys"):
        st.write(result["distorted_oracles"])
