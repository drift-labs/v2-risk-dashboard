import pandas as pd
import streamlit as st
from driftpy.constants.spot_markets import mainnet_spot_market_configs

from lib.api import fetch_api_data
from lib.page import needs_backend


def get_maintenance_asset_weight(market_index: int) -> float:
    """
    Get the maintenance asset weight for a spot market.
    """
    maintenance_asset_weights = fetch_api_data(
        "asset-liability",
        "maintenance-asset-weights",
    )
    return maintenance_asset_weights["maintenance_asset_weights"][str(market_index)]


@needs_backend
def liquidation_simulation_page():
    st.title("🔬 Maintenance Asset Weight Simulation")
    st.write(
        "Simulate lowering maintenance asset weight for a spot market to see how many users would be liquidated."
    )

    col1, col2 = st.columns(2)
    with col1:
        spot_market_index = st.selectbox(
            "Spot Market",
            [x.market_index for x in mainnet_spot_market_configs],
            format_func=lambda x: f"{x} ({mainnet_spot_market_configs[int(x)].symbol})",
            key="liquidation_sim_market",
        )
        current_maint_weight = get_maintenance_asset_weight(spot_market_index)
        st.info(f"Current maintenance asset weight: **{current_maint_weight:.4f}**")

    with col2:
        new_maint_weight = st.number_input(
            "New Maintenance Asset Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_maint_weight,
            step=0.01,
            key="liquidation_sim_weight",
        )

    if st.button("Run Simulation", key="run_liquidation_sim"):
        with st.spinner("Running simulation..."):
            try:
                sim_result = fetch_api_data(
                    "asset-liability",
                    "liquidation-simulation",
                    params={
                        "spot_market_index": spot_market_index,
                        "new_maintenance_asset_weight": new_maint_weight,
                        "bypass_cache": "true",  # Always run fresh simulation
                    },
                    retry=True,  # Retry if processing
                    max_wait_time=120,
                )

                if sim_result is None:
                    st.error("Simulation timed out. Please try again.")
                elif "error" in sim_result:
                    st.error(f"Error: {sim_result['error']}")
                elif (
                    isinstance(sim_result, dict)
                    and sim_result.get("result") == "processing"
                ):
                    st.warning(
                        "Simulation is still processing. Please try again in a moment."
                    )
                else:
                    st.success("Simulation completed!")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Users",
                            f"{sim_result['total_users']:,}",
                        )
                    with col2:
                        newly_liquidated = sim_result.get("newly_liquidated_count", 0)
                        st.metric(
                            "Newly Liquidated",
                            f"{newly_liquidated:,}",
                            delta=f"+{newly_liquidated}",
                            help="Users who would be liquidated due to the change",
                        )

                    if newly_liquidated > 0:
                        st.warning(
                            f"⚠️ **{newly_liquidated} users would be newly liquidated** if maintenance asset weight is lowered to {new_maint_weight:.4f}"
                        )
                        with st.expander(
                            f"View {newly_liquidated} Newly Liquidated Users"
                        ):
                            newly_df = pd.DataFrame(
                                {
                                    "User Public Key": sim_result.get(
                                        "newly_liquidated_user_keys", []
                                    ),
                                    "Link": [
                                        f"https://app.drift.trade/overview?userAccount={key}"
                                        for key in sim_result.get(
                                            "newly_liquidated_user_keys", []
                                        )
                                    ],
                                }
                            )
                            st.dataframe(
                                newly_df,
                                hide_index=True,
                                column_config={
                                    "Link": st.column_config.LinkColumn(
                                        "Link", display_text="View"
                                    ),
                                },
                            )
                    else:
                        st.success(
                            f"✅ **No new liquidations** would occur if maintenance asset weight is lowered to {new_maint_weight:.4f}"
                        )
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                st.exception(e)
