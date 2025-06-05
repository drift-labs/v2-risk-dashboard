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

# Helper to initialize session state for a specific widget
def initialize_widget_state(param_name, session_state_key, options, default_value, type_converter=None):
    query_val_str = st.query_params.get(param_name)
    if query_val_str is not None:
        try:
            converted_val = type_converter(query_val_str) if type_converter else query_val_str
            if converted_val in options:
                st.session_state[session_state_key] = converted_val
                return 
            else: # Invalid value in query param
                st.session_state[session_state_key] = default_value
        except (ValueError, TypeError): # Conversion failed
            st.session_state[session_state_key] = default_value
    elif session_state_key not in st.session_state:
        st.session_state[session_state_key] = default_value
    # If query_val_str is None but session_state_key IS in st.session_state,
    # it means the user interacted, then maybe removed query param. Let existing session_state value persist.

# Define callbacks (these need to be defined before use in widgets)
def on_asset_group_change():
    st.query_params["asset_group"] = st.session_state.selected_asset_group

def on_n_scenarios_change():
    st.query_params["n_scenarios"] = str(st.session_state.n_scenarios) # query params are strings

def on_pool_id_change():
    st.query_params["pool_id"] = st.session_state.selected_pool_id

def price_shock_cached_page():
    # Define options lists
    asset_group_options = [
        PriceShockAssetGroup.IGNORE_STABLES.value,
        PriceShockAssetGroup.JLP_ONLY.value,
    ]
    scenario_options = [5, 10]
    pool_value_options = ["all", "0", "1", "3"]
    pool_display_names_map = {
        "all": "All Pools",
        "0": "Main Pool (0)",
        "1": "Isolated Pool 1",
        "3": "Isolated Pool 3"
    }

    # Initialize widget states from query_params or defaults
    initialize_widget_state("asset_group", "selected_asset_group", asset_group_options, PriceShockAssetGroup.IGNORE_STABLES.value)
    initialize_widget_state("n_scenarios", "n_scenarios", scenario_options, 5, type_converter=int)
    initialize_widget_state("pool_id", "selected_pool_id", pool_value_options, "all")

    # Asset Group Selection
    asset_group = st.selectbox(
        "Asset Group", 
        options=asset_group_options, 
        index=asset_group_options.index(st.session_state.selected_asset_group),
        key="selected_asset_group",
        on_change=on_asset_group_change
    )

    # Scenarios Selection
    n_scenarios = st.radio(
        "Scenarios",
        options=scenario_options,
        index=scenario_options.index(st.session_state.n_scenarios),
        key="n_scenarios",
        on_change=on_n_scenarios_change
    )
    
    # Pool ID selection
    def format_pool_id(pool_id_value):
        return pool_display_names_map.get(pool_id_value, str(pool_id_value))

    selected_pool_id = st.selectbox(
        "Pool Filter", 
        options=pool_value_options,
        index=pool_value_options.index(st.session_state.selected_pool_id),
        format_func=format_pool_id,
        key="selected_pool_id",
        on_change=on_pool_id_change,
        help="Filter positions by pool ID. Main Pool (0) has more lenient parameters, Isolated Pools (>0) have stricter risk parameters."
    )
    
    # Determine oracle_distort based on n_scenarios from session_state
    if st.session_state.n_scenarios == 5:
        oracle_distort = 0.05
    else:
        oracle_distort = 0.1
        
    # Prepare API parameters using values from st.session_state
    api_params = {
        "asset_group": st.session_state.selected_asset_group,
        "oracle_distortion": oracle_distort,
        "n_scenarios": st.session_state.n_scenarios,
    }
    
    if st.session_state.selected_pool_id != "all":
        api_params["pool_id"] = int(st.session_state.selected_pool_id)
    
    cache_key_parts = [
        "price-shock/usermap",
        st.session_state.selected_asset_group,
        str(oracle_distort),
        str(st.session_state.n_scenarios),
        st.session_state.selected_pool_id
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
    # Use selected_pool_id from session_state for display
    pool_display_text = f" (Pool {st.session_state.selected_pool_id})" if st.session_state.selected_pool_id != "all" else " (All Pools)"
    st.info(
        f"This data is for slot {result['slot']}{pool_display_text}, which is now {int(current_slot) - int(result['slot'])} slots old"
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
