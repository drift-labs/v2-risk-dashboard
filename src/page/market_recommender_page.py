import time

import pandas as pd
import streamlit as st

from lib.api import fetch_api_data

DEFAULT_NUMBER_OF_TOKENS = 50
DEBUG_MODE = True
RETRY_DELAY_SECONDS = 5


def format_large_number(num):
    """Formats large numbers into human-readable strings with $, M, B, K suffixes."""
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        num = float(num)
        if abs(num) >= 1_000_000_000:
            return f"${num / 1_000_000_000:.2f}B"
        elif abs(num) >= 1_000_000:
            return f"${num / 1_000_000:.2f}M"
        elif abs(num) >= 1_000:
            return f"${num / 1_000:.2f}K"
        elif abs(num) < 1 and abs(num) > 0:
            return f"${num:.4f}"
        else:
            return f"${num:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def format_percentage(num):
    """Formats number as a percentage string."""
    if pd.isna(num) or num is None:
        return "N/A"
    try:
        return f"{float(num):.3f}%"
    except (ValueError, TypeError):
        return "N/A"


def style_recommendation(rec):
    """Applies color styling based on recommendation."""
    if rec == "List":
        color = "green"
    elif rec == "Increase Leverage":
        color = "deepskyblue"
    elif rec == "Decrease Leverage":
        color = "orange"
    elif rec == "Delist":
        color = "red"
    elif rec in ["Maintain Leverage", "Keep Unlisted"]:
        color = "gray"
    else:
        color = "white"
    return f"color: {color}; font-weight: bold;"


@st.cache_data(ttl=60)
def fetch_market_recommendations_cached():
    """Fetches market recommendations from the backend API. Handles processing state."""
    try:
        if DEBUG_MODE:
            print(
                f"[{time.strftime('%X')}] Attempting to fetch/get market recommendations from cache/API"
            )
            endpoint = "market-recommender"
            path = "market-data"
            params = {"number_of_tokens": DEFAULT_NUMBER_OF_TOKENS}
            print(
                f"API call details: endpoint={endpoint}, path={path}, params={params}"
            )

        response = fetch_api_data(
            "market-recommender",
            "market-data",
            params={"number_of_tokens": DEFAULT_NUMBER_OF_TOKENS},
        )

        if DEBUG_MODE:
            if isinstance(response, list):
                print(
                    f"[{time.strftime('%X')}] Successfully fetched data ({len(response)} items)."
                )
            elif isinstance(response, dict) and response.get("result") == "processing":
                print(f"[{time.strftime('%X')}] Received 'processing' status from API.")
            elif isinstance(response, dict) and "error" in response:
                print(
                    f"[{time.strftime('%X')}] Received error from API: {response['error']}"
                )
            elif response is None:
                print(
                    f"[{time.strftime('%X')}] Received None from API (initial miss?)."
                )
            else:
                print(
                    f"[{time.strftime('%X')}] Received unexpected response type from API: {type(response)}"
                )

        return response
    except Exception as e:
        error_message = f"An error occurred in fetch_market_recommendations_cached: {e}"
        if DEBUG_MODE:
            print(f"ERROR: {error_message}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}")
        return {"error": error_message}


def market_recommender_page():
    """
    Streamlit page for displaying market recommendations for Drift.
    Fetches pre-calculated data from a backend API, handling processing states.
    """
    st.title("📈 Market Recommender")
    st.markdown(
        "This page provides recommendations for listing, delisting, or adjusting leverage for markets based on various metrics calculated by the backend."
    )
    st.markdown("THIS PAGE IS IN BETA AND SUBJECT TO CHANGE.")

    data = fetch_market_recommendations_cached()

    if isinstance(data, dict) and data.get("result") == "processing":
        st.info(
            f"Backend is processing the market data. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
        )
        with st.spinner("Please wait..."):
            time.sleep(RETRY_DELAY_SECONDS)
        st.rerun()

    elif isinstance(data, dict) and "error" in data:
        st.error(f"Failed to fetch market recommendations: {data['error']}")
        if st.button("Retry Fetch"):
            st.cache_data.clear()
            st.rerun()
        return

    elif data is None:
        st.info(
            f"Requesting data from backend. Auto-refreshing in {RETRY_DELAY_SECONDS} seconds..."
        )
        with st.spinner("Please wait..."):
            time.sleep(RETRY_DELAY_SECONDS)
        st.rerun()

    elif isinstance(data, list):
        st.success("Market data loaded successfully.")
        st.subheader(f"Analysis Results ({len(data)} tokens)")

        list_count = sum(1 for item in data if item.get("recommendation") == "List")
        increase_leverage_count = sum(
            1 for item in data if item.get("recommendation") == "Increase Leverage"
        )
        decrease_leverage_count = sum(
            1 for item in data if item.get("recommendation") == "Decrease Leverage"
        )
        delist_count = sum(1 for item in data if item.get("recommendation") == "Delist")
        maintain_leverage_count = sum(
            1 for item in data if item.get("recommendation") == "Maintain Leverage"
        )
        keep_unlisted_count = sum(
            1 for item in data if item.get("recommendation") == "Keep Unlisted"
        )

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Recommend List", list_count, delta_color="off")
        with col2:
            st.metric("Increase Leverage", increase_leverage_count, delta_color="off")
        with col3:
            st.metric("Maintain Leverage", maintain_leverage_count, delta_color="off")
        with col4:
            st.metric("Decrease Leverage", decrease_leverage_count, delta_color="off")
        with col5:
            st.metric("Recommend Delist", delist_count, delta_color="off")
        with col6:
            st.metric("Keep Unlisted", keep_unlisted_count, delta_color="off")

        try:
            df = pd.DataFrame(data)
            if "recommendation" not in df.columns:
                df["recommendation"] = [
                    item.get("recommendation", "Unknown") for item in data
                ]

            st.subheader("Filter Options")
            col1, col2 = st.columns(2)

            with col1:
                show_only_listed = st.checkbox(
                    "Show only tokens listed on Drift", value=False
                )
                show_only_unlisted = st.checkbox(
                    "Show only tokens not listed on Drift", value=False
                )

                if show_only_listed and show_only_unlisted:
                    st.error(
                        "Cannot show both listed and unlisted tokens simultaneously. Please uncheck one option."
                    )
                    show_only_unlisted = False

            with col2:
                unique_recommendations = sorted(df["recommendation"].unique().tolist())
                selected_recommendations = st.multiselect(
                    "Filter by Recommendation",
                    options=unique_recommendations,
                    help="Select one or more recommendations to filter the table",
                )

            display_df = pd.DataFrame()

            if "symbol" in df.columns:
                display_df["Symbol"] = df["symbol"].astype(str)
            else:
                display_df["Symbol"] = [
                    str(item.get("symbol", "Unknown")) for item in data
                ]

            metric_cols_init = {
                "Total Score": 0.0,
                "Recommendation": "Unknown",
                "Market Cap": 0.0,
                "30d Volume": 0.0,
                "24h Volume": 0.0,
                "Drift OI": 0.0,
                "Max Leverage": 0.0,
                "Drift Volume (30d)": 0.0,
                "Drift Volume Score": 0.0,
                "OI Score": 0.0,
                "Global Volume Score": 0.0,
                "FDV Score": 0.0,
            }
            for col, default_val in metric_cols_init.items():
                display_df[col] = default_val

            for idx, row in df.iterrows():
                display_df.at[idx, "Total Score"] = float(row.get("total_score", 0))
                display_df.at[idx, "Recommendation"] = str(
                    row.get("recommendation", "Unknown")
                )
                display_df.at[idx, "Drift Volume Score"] = float(
                    row.get("drift_volume_score", 0)
                )
                display_df.at[idx, "OI Score"] = float(
                    row.get("open_interest_score", 0)
                )
                display_df.at[idx, "Global Volume Score"] = float(
                    row.get("global_volume_score", 0)
                )
                display_df.at[idx, "FDV Score"] = float(row.get("fdv_score", 0))

                coingecko_data = row.get("coingecko_data", {})
                if isinstance(coingecko_data, dict):
                    fdv = coingecko_data.get("coingecko_fully_diluted_valuation")
                    mc = coingecko_data.get("coingecko_market_cap", 0)
                    display_df.at[idx, "Market Cap"] = float(
                        fdv if fdv is not None and fdv > 0 else mc
                    )
                    display_df.at[idx, "30d Volume"] = float(
                        coingecko_data.get("coingecko_30d_volume", 0)
                    )
                    display_df.at[idx, "24h Volume"] = float(
                        coingecko_data.get("coingecko_total_volume_24h", 0)
                    )

                drift_data = row.get("drift_data", {})
                if isinstance(drift_data, dict):
                    display_df.at[idx, "Drift OI"] = float(
                        drift_data.get("drift_open_interest", 0)
                    )
                    display_df.at[idx, "Max Leverage"] = float(
                        drift_data.get("drift_max_leverage", 0)
                    )
                    display_df.at[idx, "Drift Volume (30d)"] = float(
                        drift_data.get("drift_total_quote_volume_30d", 0)
                    )

            indices_to_keep = list(df.index)

            if show_only_listed or show_only_unlisted:
                listing_indices = []
                for idx, row in df.iterrows():
                    drift_data = row.get("drift_data", {})
                    is_listed = (
                        isinstance(drift_data, dict)
                        and drift_data.get("drift_is_listed_perp") == "true"
                    )
                    if (show_only_listed and is_listed) or (
                        show_only_unlisted and not is_listed
                    ):
                        listing_indices.append(idx)
                indices_to_keep = [
                    idx for idx in indices_to_keep if idx in listing_indices
                ]

            if selected_recommendations:
                recommendation_indices = df[
                    df["recommendation"].isin(selected_recommendations)
                ].index.tolist()
                indices_to_keep = [
                    idx for idx in indices_to_keep if idx in recommendation_indices
                ]

            if indices_to_keep:
                display_df = display_df.loc[indices_to_keep].copy()
            else:
                display_df = pd.DataFrame(columns=display_df.columns)

            total_tokens = len(df)
            filtered_tokens = len(display_df)
            st.write(
                f"Showing {filtered_tokens} out of {total_tokens} tokens based on selected filters"
            )

            display_df = display_df.sort_values(by="Total Score", ascending=False)
            display_df = display_df.set_index("Symbol")

            formatters = {
                "Market Cap": format_large_number,
                "30d Volume": format_large_number,
                "24h Volume": format_large_number,
                "Drift OI": format_large_number,
                "Drift Volume (30d)": format_large_number,
                "Max Leverage": lambda x: f"{int(x)}x"
                if pd.notna(x) and x > 0
                else "N/A",
                "Total Score": "{:.2f}",
                "Drift Volume Score": "{:.2f}",
                "OI Score": "{:.2f}",
                "Global Volume Score": "{:.2f}",
                "FDV Score": "{:.2f}",
            }

            st.subheader("Market Recommendations Summary")
            summary_cols = [
                "Recommendation",
                "Total Score",
                "Max Leverage",
                "Drift Volume (30d)",
                "Drift OI",
                "30d Volume",
                "Market Cap",
            ]
            summary_cols_exist = [
                col for col in summary_cols if col in display_df.columns
            ]
            main_display_df = display_df[summary_cols_exist]
            df_styled = main_display_df.style.format(formatters, na_rep="N/A")
            if "Recommendation" in df_styled.columns:
                df_styled = df_styled.apply(
                    lambda x: x.map(style_recommendation),
                    subset=["Recommendation"],
                    axis=0,
                )
            st.dataframe(df_styled, use_container_width=True)

            with st.expander("View Detailed Metrics & Scores"):
                detail_cols = [
                    "Recommendation",
                    "Total Score",
                    "Drift Volume Score",
                    "OI Score",
                    "Global Volume Score",
                    "FDV Score",
                    "Max Leverage",
                    "Drift Volume (30d)",
                    "Drift OI",
                    "30d Volume",
                    "24h Volume",
                    "Market Cap",
                ]
                detail_cols_exist = [
                    col for col in detail_cols if col in display_df.columns
                ]
                detail_df = display_df[detail_cols_exist]
                detail_styled = detail_df.style.format(formatters, na_rep="N/A")
                if "Recommendation" in detail_styled.columns:
                    detail_styled = detail_styled.apply(
                        lambda x: x.map(style_recommendation),
                        subset=["Recommendation"],
                        axis=0,
                    )
                st.dataframe(detail_styled, use_container_width=True)

            with st.expander("Raw Data (for debugging)"):
                if data:
                    st.json(data[0])

        except Exception as e:
            st.error(f"Error processing or displaying data: {e}")
            import traceback

            st.error(f"Traceback: {traceback.format_exc()}")
            st.write("Original data received from API (first 5 items):")
            st.write(data[:5])

    else:
        st.error(
            "Received data in an unexpected format from the backend or cache. Here's what was returned:"
        )
        st.write(data)
        if st.button("Clear Cache and Retry"):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    st.markdown(
        """
        ### 📊 Methodology

        This page provides recommendations for listing new markets or adjusting leverage on existing markets based on several key metrics:

        **Scoring Components (Total 100 Points):**

        1.  **Drift Volume Score (0-25 points)**
            * Based on 30-day trading volume on Drift (Perp & Spot combined)
            * Higher volume indicates more trader interest and potential protocol revenue
            * Score tiers: $500M+ (25pts), $100M-$499M (20pts), $25M-$99M (15pts), $1M-$24M (10pts), $100K-$999K (5pts), <$100K (0pts)

        2.  **Open Interest Score (0-25 points)**
            * Based on current open interest on Drift perpetual markets
            * Higher OI indicates more capital committed to positions
            * Score tiers: $5M+ (25pts), $1M-$4.9M (20pts), $250K-$999K (15pts), $50K-$249K (10pts), $5K-$49K (5pts), <$5K (0pts)

        3.  **Global Volume Score (0-40 points)**
            * Based on 30-day global trading volume across all exchanges (Source: CoinGecko)
            * Higher global volume indicates broader market interest and liquidity
            * Score tiers: $15B+ (40pts), $7.5B-$14.9B (30pts), $3B-$7.49B (20pts), $750M-$2.9B (10pts), $150M-$749M (5pts), <$150M (0pts) (*Note: Tiers adjusted based on backend calculation*)

        4.  **FDV Score (0-10 points)**
            * Based on Fully Diluted Valuation (or Market Cap if FDV unavailable)
            * Higher market cap/FDV generally indicates larger, more established projects
            * Score tiers: $10B+ (10pts), $1B-$9.9B (8pts), $500M-$999M (6pts), $100M-$499M (2pts), <$100M (0pts)

        **Recommendation Logic:**

        The total score (0-100) is compared against thresholds that vary based on the current maximum leverage offered for the **perpetual market**:

        * For **unlisted markets** (Max Leverage = N/A or 0):
            * Score ≥ 45: **List** (Consider listing with initial leverage, e.g., 2x)
            * Score < 45: **Keep Unlisted** (Market doesn't meet minimum criteria)

        * For **listed markets** with Max Leverage = 2x:
            * Score ≤ 40: **Delist** (Market performance is below minimum threshold for listing)
            * Score > 40: **Maintain Leverage** (Score doesn't justify delisting or increase yet)

        * For **listed markets** with Max Leverage = 4x:
            * Score ≥ 75: **Increase Leverage** (Consider increasing to 5x)
            * Score ≤ 50: **Decrease Leverage** (Consider decreasing to 2x)
            * 50 < Score < 75: **Maintain Leverage**

        * For **listed markets** with Max Leverage = 5x:
            * Score ≥ 80: **Increase Leverage** (Consider increasing to 10x)
            * Score ≤ 60: **Decrease Leverage** (Consider decreasing to 4x)
            * 60 < Score < 80: **Maintain Leverage**

        * For **listed markets** with Max Leverage = 10x:
            * Score ≥ 90: **Increase Leverage** (Consider increasing to 20x)
            * Score ≤ 70: **Decrease Leverage** (Consider decreasing to 5x)
            * 70 < Score < 90: **Maintain Leverage**

        * For **listed markets** with Max Leverage = 20x:
            * Score ≤ 75: **Decrease Leverage** (Consider decreasing to 10x)
            * Score > 75: **Maintain Leverage** (Already at max leverage or score justifies it)

        **Data Sources:**

        * **Drift Protocol (via `driftpy` & API)**: Drift OI, trading volume (30d), funding rates, oracle prices, max leverage
        * **CoinGecko**: Market cap, FDV, global trading volume (30d & 24h), token info

        **Important Notes:**

        * Recommendations are based solely on quantitative metrics and should be supplemented with qualitative analysis (e.g., project fundamentals, security audits, community interest).
        * Market conditions can change rapidly; recommendations should be reviewed regularly.
        * The scoring model and thresholds are subject to refinement.
        """
    )

    st.markdown("---")
    st.caption(
        "Data is cached for 60 seconds. Refreshing the page will check for new data from the backend or cache."
    )

    if st.button("Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()
