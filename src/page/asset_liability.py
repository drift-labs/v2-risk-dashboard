import pandas as pd
import streamlit as st
import time
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

from lib.api import api2
from utils import get_current_slot


options = [0, 1, 2, 3]
labels = [
    "none",
    "liq within 50% of oracle",
    "maint. health < 10%",
    "init. health < 10%",
]


def calculate_effective_leverage(assets: float, liabilities: float) -> float:
    return liabilities / assets if assets != 0 else 0


def format_metric(
    value: float, should_highlight: bool, mode: int, financial: bool = False
) -> str:
    formatted = f"{value:,.2f}" if financial else f"{value:.2f}"
    return f"{formatted} ✅" if should_highlight and mode > 0 else formatted


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.1f}s"


def generate_summary_data(
    df: pd.DataFrame, mode: int, perp_market_index: int
) -> pd.DataFrame:
    summary_data = {}
    for market in mainnet_spot_market_configs:
        i = market.market_index
        prefix = f"spot_{i}"
        try:
            assets = df[f"{prefix}_all_assets"].sum()
            liabilities = df[f"{prefix}_all"].sum()

            summary_data[f"spot{i}"] = {
                "all_assets": assets,
                "all_liabilities": format_metric(
                    liabilities, 0 < liabilities < 1_000_000, mode, financial=True
                ),
                "effective_leverage": format_metric(
                    calculate_effective_leverage(assets, liabilities),
                    0 < calculate_effective_leverage(assets, liabilities) < 2,
                    mode,
                ),
                "all_spot": df[f"{prefix}_all_spot"].sum(),
                "all_perp": df[f"{prefix}_all_perp"].sum(),
                f"perp_{perp_market_index}_long": df[
                    f"{prefix}_perp_{perp_market_index}_long"
                ].sum(),
                f"perp_{perp_market_index}_short": df[
                    f"{prefix}_perp_{perp_market_index}_short"
                ].sum(),
            }
        except KeyError as e:
            print(f"Warning: Missing data for market {i} ({market.symbol}): {e}")
            continue
    return pd.DataFrame(summary_data).T


def asset_liab_matrix_cached_page():
    if "min_leverage" not in st.session_state:
        st.session_state.min_leverage = 0.0
    if "only_high_leverage_mode_users" not in st.session_state:
        st.session_state.only_high_leverage_mode_users = False
    if "load_times" not in st.session_state:
        st.session_state.load_times = {}

    params = st.query_params
    mode = int(params.get("mode", 0))
    perp_market_index = int(params.get("perp_market_index", 0))

    mode = st.selectbox(
        "Options", options, format_func=lambda x: labels[x], index=options.index(mode)
    )
    st.query_params.update({"mode": str(mode)})

    perp_market_index = st.selectbox(
        "Market index",
        [x.market_index for x in mainnet_perp_market_configs],
        index=[x.market_index for x in mainnet_perp_market_configs].index(
            perp_market_index
        ),
        format_func=lambda x: f"{x} ({mainnet_perp_market_configs[int(x)].symbol})",
    )
    st.query_params.update({"perp_market_index": str(perp_market_index)})

    # Create containers for timing information
    timing_container = st.container()
    with timing_container:
        #st.write("### Loading Times")
        timing_table = st.empty()
    
    # Create a container for the loading spinner and status
    loading_container = st.container()
    
    try:
        start_time = time.time()
        load_times = {}
        
        # Create a placeholder for the timer
        timer_placeholder = st.empty()
        status_container = st.empty()
        
        while True:
            # Update timer
            elapsed = time.time() - start_time
            timer_placeholder.markdown(f"### Loading asset-liability matrix... ({format_time(elapsed)})")
            
            try:
                # Fetching data from backend
                step_start = time.time()
                status_container.info("⏳ Fetching data from backend...")
                
                result = api2(
                    "asset-liability/matrix",
                    _params={"mode": mode, "perp_market_index": perp_market_index},
                    key=f"asset-liability/matrix_{mode}_{perp_market_index}_{int(time.time())}",
                )
                
                step_duration = time.time() - step_start
                load_times["Fetch Data"] = step_duration
                break  # Exit the loop once data is fetched
                
            except Exception as e:
                if "timeout" not in str(e).lower():  # If it's not a timeout error, raise it
                    raise e
                time.sleep(0.1)  # Small delay before retrying
                continue
        
        # Processing market data
        step_start = time.time()
        status_container.info("⚡ Processing market data...")
        
        if not isinstance(result, dict) or "df" not in result:
            st.error("Invalid response format from API")
            return

        df = pd.DataFrame(result["df"])
        if df.empty:
            st.warning("No data available for the selected parameters")
            return
        
        step_duration = time.time() - step_start
        load_times["Process Data"] = step_duration

        # Calculating metrics
        step_start = time.time()
        status_container.info("📊 Calculating market metrics...")
        
        if st.session_state.only_high_leverage_mode_users:
            if "is_high_leverage" not in df.columns:
                st.error("High leverage mode data is not available")
                st.session_state.only_high_leverage_mode_users = False
            else:
                df = df[df["is_high_leverage"]]

        filtered_df = df[df["leverage"] >= st.session_state.min_leverage].sort_values(
            "leverage", ascending=False
        )
        
        step_duration = time.time() - step_start
        load_times["Calculate Metrics"] = step_duration

        # Generating summary
        step_start = time.time()
        status_container.info("🔄 Generating summary data...")
        
        summary_df = generate_summary_data(filtered_df, mode, perp_market_index)
        
        step_duration = time.time() - step_start
        load_times["Generate Summary"] = step_duration
        
        # Calculate total time
        total_time = time.time() - start_time
        load_times["Total Time"] = total_time
        
        # Update final timer display
        timer_placeholder.markdown(f"### Loading Times")
        
        # Store the load times in session state
        st.session_state.load_times = load_times
        
        # Clear the status container after loading is complete
        status_container.empty()

        # Display timing information
        timing_df = pd.DataFrame(
            {
                "Duration": [format_time(t) for t in load_times.values()],
                "Percentage": [f"{(t/load_times['Total Time'])*100:.1f}%" for t in load_times.values()]
            },
            index=load_times.keys()
        )
        st.table(timing_df)

    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return

    # Get slot information if available
    slot = result.get("slot")
    if slot is not None:
        try:
            current_slot = get_current_slot()
            st.info(
                f"This data is for slot {slot}, which is now {int(current_slot) - int(slot)} slots old"
            )
        except Exception as e:
            print(f"Error getting current slot: {e}")
            # Continue without showing slot information

    st.write(f"{df.shape[0]} users")
    st.checkbox(
        "Only show high leverage mode users", key="only_high_leverage_mode_users"
    )
    st.slider(
        "Filter by minimum leverage",
        0.0,
        110.0,
        0.0,
        key="min_leverage",
    )
    st.write(summary_df)

    tabs = st.tabs(["FULL"] + [x.symbol for x in mainnet_spot_market_configs])

    with tabs[0]:
        if st.session_state.only_high_leverage_mode_users:
            st.write(
                f"There are **{len(filtered_df)}** users with high leverage mode and {st.session_state.min_leverage}x leverage or more"
            )
        else:
            st.write(
                f"There are **{len(filtered_df)}** users with this **{st.session_state.min_leverage}x** leverage or more"
            )
        st.write(f"Total USD value: **{filtered_df['net_usd_value'].sum():,.2f}**")
        st.write(f"Total collateral: **{filtered_df['spot_asset'].sum():,.2f}**")
        st.write(f"Total liabilities: **{filtered_df['spot_liability'].sum():,.2f}**")
        st.dataframe(filtered_df, hide_index=True)

    for idx, tab in enumerate(tabs[1:], 1):
        market = mainnet_spot_market_configs[idx - 1]
        market_index = market.market_index
        prefix = f"spot_{market_index}"
        
        # Check if the required columns exist
        required_cols = [f"{prefix}_all", f"{prefix}_all_assets", f"{prefix}_all_perp", f"{prefix}_all_spot"]
        if not all(col in filtered_df.columns for col in required_cols):
            tab.warning(f"No data available for {market.symbol} (market index {market_index})")
            continue

        important_cols = [x for x in filtered_df.columns if prefix in x]
        
        toshow = filtered_df[
            ["user_key", "spot_asset", "net_usd_value"] + important_cols
        ]
        
        # Filter rows where any of the market-specific columns have non-zero values
        non_zero_mask = toshow[important_cols].abs().sum(axis=1) != 0
        toshow = toshow[non_zero_mask]
        
        if len(toshow) > 0:
            # Sort by the 'all' column if it exists, otherwise don't sort
            try:
                toshow = toshow.sort_values(by=f"{prefix}_all", ascending=False)
            except KeyError:
                pass  # Skip sorting if column doesn't exist
            
            tab.write(
                f"{len(toshow)} users with this asset to cover liabilities (with {st.session_state.min_leverage}x leverage or more)"
            )
            tab.dataframe(toshow, hide_index=True)
        else:
            tab.info("No users found with non-zero positions in this market")
