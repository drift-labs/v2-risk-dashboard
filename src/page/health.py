import pandas as pd
import plotly.express as px
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

from lib.api import fetch_api_data


def health_page():
    """
    Health page for viewing account health distribution and various largest positions.
    Added a Debug Options toggle in the sidebar that, when enabled, displays additional
    developer buttons: "Show Current Metadata", "List Available Pickles",
    and "Force Refresh Pickle".
    """
    # Debug mode toggle in sidebar
    with st.sidebar:
        st.write("---")
        st.write("Debug Options")
        debug_mode = st.toggle(
            "Enable Debug Mode",
            value=False,
            help="Show debugging features for the health page",
        )

    if debug_mode:
        st.subheader("Developer Tools")
        st.write("Use these buttons to inspect backend metadata and manage snapshots.")
        col1, col2, col3, col4 = st.columns(4)

        # Show Current Metadata
        if col1.button("Show Current Metadata"):
            metadata_info = fetch_api_data("metadata", "", retry=True)
            st.json(metadata_info)

        # List Available Pickles
        if col2.button("List Available Pickles"):
            pickles = fetch_api_data("metadata", "list_pickles", retry=True)
            st.write(pickles)

        # Force Refresh Pickle
        if col3.button("Force Refresh Pickle"):
            refresh_resp = fetch_api_data("metadata", "force_refresh", retry=True)
            st.json(refresh_resp)
            
        # Bypass Cache for Perp Positions
        if col4.button("Refresh Perp Positions"):
            st.info("Bypassing cache and fetching fresh data for largest perp positions...")
            # Get the current number input value
            num_positions = st.session_state.get("num_perp_positions", 10)
            fresh_perp_positions = fetch_api_data(
                "health",
                "largest_perp_positions",
                params={"number_of_positions": num_positions, "bypass_cache": "true"},
                retry=True,
            )
            st.dataframe(pd.DataFrame(fresh_perp_positions), hide_index=True)
            st.success("Successfully fetched fresh data (bypassed cache)")

    # Main Health content
    st.markdown("# Health")

    st.markdown(
        """
        Account health is a measure of the health of a user's account.
        It is calculated as the ratio of the user's collateral to the user's debt.
        For more information about how account health is calculated, see
        [account health](https://docs.drift.trade/trading/account-health) in the docs.
        """
    )
    health_distribution = fetch_api_data(
        "health",
        "health_distribution",
        retry=True,
    )

    fig = px.bar(
        pd.DataFrame(health_distribution),
        x="Health Range",
        y="Counts",
        title="Health Distribution",
        hover_data={"Notional Values": ":,"},  # Custom format for notional values
        labels={"Counts": "Num Users", "Notional Values": "Notional Value ($)"},
    )

    fig.update_traces(
        hovertemplate="<b>Health Range: %{x}</b><br>Count: %{y}<br>Notional Value: $%{customdata[0]:,.0f}<extra></extra>"
    )

    with st.container():
        st.plotly_chart(fig, use_container_width=True)

    perp_col, spot_col = st.columns([1, 1])

    with perp_col:
        st.markdown("### **Largest perp positions:**")
        
        # Create a container for the filter controls
        perp_filter_col1, perp_filter_col2 = st.columns(2)
        
        # Add number input to control how many positions to show
        num_positions = perp_filter_col1.number_input(
            "Number of values to return for perp positions",
            min_value=10,
            max_value=100,
            value=10,
            step=10,
            key="num_perp_positions"
        )
        
        # Add market index selector
        perp_market_options = [{"label": f"{idx} ({cfg.symbol})", "value": idx} 
                              for idx, cfg in enumerate(mainnet_perp_market_configs) if cfg is not None]
        perp_market_options.insert(0, {"label": "All Markets", "value": None})
        
        selected_perp_market = perp_filter_col2.selectbox(
            "Filter by market",
            options=[opt["value"] for opt in perp_market_options],
            format_func=lambda x: next((opt["label"] for opt in perp_market_options if opt["value"] == x), "All Markets"),
            key="perp_market_filter"
        )
        
        # Add bypass cache toggle when in debug mode
        bypass_cache = False
        if debug_mode:
            bypass_cache = st.toggle(
                "Bypass cache for perp positions",
                value=False,
                help="When enabled, the API will bypass the cache and fetch fresh data directly",
                key="bypass_perp_cache"
            )
        
        largest_perp_positions = fetch_api_data(
            "health",
            "largest_perp_positions",
            params={
                "number_of_positions": num_positions,
                "market_index": selected_perp_market,
                "bypass_cache": "true" if bypass_cache else "false"
            },
            retry=True,
        )
        
        # Convert to DataFrame and add pagination
        df = pd.DataFrame(largest_perp_positions)
        
        # Find market index column regardless of capitalization
        market_index_col = next((col for col in df.columns if col.lower() == 'market index' or col.lower() == 'market_index'), None)
        
        # Add market symbol to market_index column if it exists
        if market_index_col:
            df[market_index_col] = df[market_index_col].map(
                lambda x: f"{x} ({mainnet_perp_market_configs[int(x)].symbol})" if pd.notna(x) else x
            )
            
        total_rows = len(df)
        page_size = 10
        total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
        
        if total_pages > 1:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key="perp_positions_page"
            )
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            st.write(f"Showing positions {start_idx + 1}-{end_idx} of {total_rows}")
            st.dataframe(df.iloc[start_idx:end_idx], hide_index=True)
        else:
            st.dataframe(df, hide_index=True)

        most_levered_positions = fetch_api_data(
            "health",
            "most_levered_perp_positions_above_1m",
            params={
                "number_of_positions": num_positions,
                "market_index": selected_perp_market
            },
            retry=True,
        )
        st.markdown("### **Most levered perp positions > $1m:**")
        
        # Format market index in most_levered_positions too
        most_levered_df = pd.DataFrame(most_levered_positions)
        
        # Find market index column regardless of capitalization
        ml_market_index_col = next((col for col in most_levered_df.columns if col.lower() == 'market index' or col.lower() == 'market_index'), None)
        
        # Add market symbol to market_index column if it exists
        if ml_market_index_col:
            most_levered_df[ml_market_index_col] = most_levered_df[ml_market_index_col].map(
                lambda x: f"{x} ({mainnet_perp_market_configs[int(x)].symbol})" if pd.notna(x) else x
            )
        
        # Add pagination
        total_rows = len(most_levered_df)
        page_size = 10
        total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
        
        if total_pages > 1:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key="levered_perp_positions_page"
            )
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            st.write(f"Showing positions {start_idx + 1}-{end_idx} of {total_rows}")
            st.dataframe(most_levered_df.iloc[start_idx:end_idx], hide_index=True)
        else:
            st.dataframe(most_levered_df, hide_index=True)

    with spot_col:
        st.markdown("### **Largest spot borrows:**")
        # Create a container for the spot filter controls
        spot_filter_col1, spot_filter_col2 = st.columns(2)
        
        # Add number input to control how many positions to show
        spot_num_positions = spot_filter_col1.number_input(
            "Number of values to return for spot borrows",
            min_value=10,
            max_value=100,
            value=10,
            step=10,
            key="num_spot_positions"
        )
        
        # Add market index selector for spot markets
        spot_market_options = [{"label": f"{idx} ({cfg.symbol})", "value": idx} 
                              for idx, cfg in enumerate(mainnet_spot_market_configs) if cfg is not None]
        spot_market_options.insert(0, {"label": "All Markets", "value": None})
        
        selected_spot_market = spot_filter_col2.selectbox(
            "Filter by market",
            options=[opt["value"] for opt in spot_market_options],
            format_func=lambda x: next((opt["label"] for opt in spot_market_options if opt["value"] == x), "All Markets"),
            key="spot_market_filter"
        )
        
        # Add bypass cache toggle when in debug mode
        spot_bypass_cache = False
        if debug_mode:
            spot_bypass_cache = st.toggle(
                "Bypass cache for spot borrows",
                value=False,
                help="When enabled, the API will bypass the cache and fetch fresh data directly",
                key="bypass_spot_cache"
            )
            
        largest_spot_borrows = fetch_api_data(
            "health",
            "largest_spot_borrows",
            params={
                "number_of_positions": spot_num_positions,
                "market_index": selected_spot_market,
                "bypass_cache": "true" if spot_bypass_cache else "false"
            },
            retry=True,
        )
        
        # Convert to dataframe and add market symbols
        spot_df = pd.DataFrame(largest_spot_borrows)
        
        # Find market index column regardless of capitalization
        spot_market_index_col = next((col for col in spot_df.columns if col.lower() == 'market index' or col.lower() == 'market_index'), None)
        
        # Add market symbol to market_index column if it exists
        if spot_market_index_col:
            spot_df[spot_market_index_col] = spot_df[spot_market_index_col].map(
                lambda x: f"{x} ({mainnet_spot_market_configs[int(x)].symbol})" if pd.notna(x) else x
            )
            
        # Add pagination
        total_rows = len(spot_df)
        page_size = 10
        total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
        
        if total_pages > 1:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key="spot_borrows_page"
            )
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            st.write(f"Showing borrows {start_idx + 1}-{end_idx} of {total_rows}")
            st.dataframe(spot_df.iloc[start_idx:end_idx], hide_index=True)
        else:
            st.dataframe(spot_df, hide_index=True)

        most_levered_borrows = fetch_api_data(
            "health",
            "most_levered_spot_borrows_above_1m",
            params={
                "number_of_positions": spot_num_positions,
                "market_index": selected_spot_market
            },
            retry=True,
        )
        st.markdown("### **Most levered spot borrows > $750k:**")
        
        # Convert to dataframe and add market symbols
        levered_spot_df = pd.DataFrame(most_levered_borrows)
        
        # Find market index column regardless of capitalization
        levered_spot_market_index_col = next((col for col in levered_spot_df.columns if col.lower() == 'market index' or col.lower() == 'market_index'), None)
        
        # Add market symbol to market_index column if it exists
        if levered_spot_market_index_col:
            levered_spot_df[levered_spot_market_index_col] = levered_spot_df[levered_spot_market_index_col].map(
                lambda x: f"{x} ({mainnet_spot_market_configs[int(x)].symbol})" if pd.notna(x) else x
            )
            
        # Check if there are any error messages in the data
        has_errors = 'Error' in levered_spot_df.columns and levered_spot_df['Error'].any()
        if has_errors:
            error_records = levered_spot_df[levered_spot_df['Error'].notna() & (levered_spot_df['Error'] != '')]
            if not error_records.empty:
                st.warning(f"Found {len(error_records)} positions with errors. Please check with the team.")
                with st.expander("View Error Details"):
                    for idx, row in error_records.iterrows():
                        market_index = row[levered_spot_market_index_col] if levered_spot_market_index_col in row else row.get('Market Index', 'Unknown')
                        st.markdown(f"**Market {market_index}:** {row['Error']}")
        
        # Add pagination
        total_rows = len(levered_spot_df)
        page_size = 10
        total_pages = (total_rows + page_size - 1) // page_size  # Ceiling division
        
        if total_pages > 1:
            page_number = st.number_input(
                "Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                key="levered_spot_borrows_page"
            )
            start_idx = (page_number - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)
            
            st.write(f"Showing borrows {start_idx + 1}-{end_idx} of {total_rows}")
            st.dataframe(levered_spot_df.iloc[start_idx:end_idx], hide_index=True)
        else:
            st.dataframe(levered_spot_df, hide_index=True)