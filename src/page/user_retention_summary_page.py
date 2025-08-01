import pandas as pd
import streamlit as st

from lib.api import fetch_api_data


def has_error(result):
    if result is None:
        return True
    if isinstance(result, dict):
        if "detail" in result and isinstance(result["detail"], str):
            return True
        if result.get("result") == "error" or result.get("status") == "error":
            return True
    return False


def get_error_message(result):
    if isinstance(result, dict) and "detail" in result:
        return str(result["detail"])
    if isinstance(result, dict) and "message" in result:
        return str(result["message"])
    return "Could not connect or fetch data from the backend. The static summary file may be missing or invalid."


@st.cache_data(ttl=300)
def load_retention_data():
    data = fetch_api_data(
        section="user-retention-summary",
        path="summary",
        params={"bypass_cache": "true"},
        retry=False,
    )
    return data


def user_retention_summary_page():
    st.title("User Retention Analysis for All Markets")

    st.markdown("""
    This page displays a pre-computed summary of user retention data for all markets on Drift.
    For each market, it identifies "new traders" (those whose first-ever order occurred within 7 days of the market's launch).
    It then measures how many of those new traders were retained by trading in *any other* market within 14 and 28 days.
    
    This data is loaded from a static file and is not calculated in real-time. For dynamic analysis, please use the **User Retention Explorer**.
    """)

    data = load_retention_data()

    if has_error(data):
        error_msg = get_error_message(data)
        st.error(f"Failed to fetch user retention data: {error_msg}")
        if st.button("Retry Data Load"):
            st.cache_data.clear()
            st.rerun()
        return

    if isinstance(data, list) and data:
        try:
            df = pd.DataFrame(data)

            df["new_traders"] = (
                pd.to_numeric(df["new_traders"], errors="coerce").fillna(0).astype(int)
            )
            df["retained_users_14d"] = (
                pd.to_numeric(df["retained_users_14d"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            df["retention_ratio_14d"] = (
                pd.to_numeric(df["retention_ratio_14d"], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
            df["retained_users_28d"] = (
                pd.to_numeric(df["retained_users_28d"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            df["retention_ratio_28d"] = (
                pd.to_numeric(df["retention_ratio_28d"], errors="coerce")
                .fillna(0.0)
                .astype(float)
            )

            list_columns = [
                "new_traders_list",
                "retained_users_14d_list",
                "retained_users_28d_list",
                "category",
            ]
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
                else:
                    df[col] = pd.Series([[] for _ in range(len(df))], dtype="object")

            df.rename(
                columns={
                    "market": "Market",
                    "category": "Category",
                    "new_traders": "New Traders (Count)",
                    "retained_users_14d": "Retained at 14d (Count)",
                    "retention_ratio_14d": "Retention Rate 14d",
                    "retained_users_28d": "Retained at 28d (Count)",
                    "retention_ratio_28d": "Retention Rate 28d",
                    "new_traders_list": "New Traders (List)",
                    "retained_users_14d_list": "Retained 14d (List)",
                    "retained_users_28d_list": "Retained 28d (List)",
                },
                inplace=True,
            )

            if (
                "all_categories" not in st.session_state
                or not st.session_state.all_categories
            ):
                all_cats = set([cat for sublist in df["Category"] for cat in sublist])
                st.session_state.all_categories = sorted(list(all_cats))
            if "selected_categories" not in st.session_state:
                st.session_state.selected_categories = []

            st.sidebar.header("Filters")

            search_market = st.sidebar.text_input("Search by Market Name", "")

            st.session_state.selected_categories = st.sidebar.multiselect(
                "Filter by Category",
                options=st.session_state.all_categories,
                default=st.session_state.selected_categories,
            )

            filtered_df = df.copy()
            if search_market:
                filtered_df = filtered_df[
                    filtered_df["Market"].str.contains(
                        search_market, case=False, na=False
                    )
                ]

            if st.session_state.selected_categories:
                filtered_df = filtered_df[
                    filtered_df["Category"].apply(
                        lambda cats: any(
                            cat in st.session_state.selected_categories for cat in cats
                        )
                    )
                ]

            if filtered_df.empty:
                st.info("No data available for the selected filters.")
            else:
                st.subheader("Overall Retention Summary")
                summary_cols = [
                    "Market",
                    "Category",
                    "New Traders (Count)",
                    "Retained at 14d (Count)",
                    "Retention Rate 14d",
                    "Retained at 28d (Count)",
                    "Retention Rate 28d",
                ]
                main_summary_df = filtered_df[summary_cols]
                st.dataframe(
                    main_summary_df.style.format(
                        {"Retention Rate 14d": "{:.2%}", "Retention Rate 28d": "{:.2%}"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                st.subheader("New Traders Lists")
                new_traders_list_df = filtered_df[["Market", "New Traders (List)"]]
                new_traders_list_df = new_traders_list_df[
                    new_traders_list_df["New Traders (List)"].apply(
                        lambda x: len(x) > 0
                    )
                ]
                if not new_traders_list_df.empty:
                    st.dataframe(
                        new_traders_list_df, hide_index=True, use_container_width=True
                    )
                else:
                    st.caption(
                        "No new traders found for the selected market(s) or new trader lists are empty."
                    )

                st.subheader("14-Day Retained User Lists")
                retained_14d_list_df = filtered_df[["Market", "Retained 14d (List)"]]
                retained_14d_list_df = retained_14d_list_df[
                    retained_14d_list_df["Retained 14d (List)"].apply(
                        lambda x: len(x) > 0
                    )
                ]
                if not retained_14d_list_df.empty:
                    st.dataframe(
                        retained_14d_list_df, hide_index=True, use_container_width=True
                    )
                else:
                    st.caption(
                        "No users retained at 14 days for the selected market(s) or lists are empty."
                    )

                st.subheader("28-Day Retained User Lists")
                retained_28d_list_df = filtered_df[["Market", "Retained 28d (List)"]]
                retained_28d_list_df = retained_28d_list_df[
                    retained_28d_list_df["Retained 28d (List)"].apply(
                        lambda x: len(x) > 0
                    )
                ]
                if not retained_28d_list_df.empty:
                    st.dataframe(
                        retained_28d_list_df, hide_index=True, use_container_width=True
                    )
                else:
                    st.caption(
                        "No users retained at 28 days for the selected market(s) or lists are empty."
                    )

                st.subheader("Export Main Summary Data")
                col1, col2 = st.columns(2)

                csv_data = main_summary_df.to_csv(index=False).encode("utf-8")
                col1.download_button(
                    label="Download Summary as CSV",
                    data=csv_data,
                    file_name="user_retention_main_summary.csv",
                    mime="text/csv",
                )

                json_data_str = main_summary_df.to_json(orient="records", indent=4)
                col2.download_button(
                    label="Download Summary as JSON",
                    data=json_data_str,
                    file_name="user_retention_main_summary.json",
                    mime="application/json",
                )
                with st.expander("View Main Summary JSON Data"):
                    st.json(json_data_str)

        except Exception as e:
            st.error(f"An error occurred while processing and displaying the data: {e}")
            import traceback

            st.text(traceback.format_exc())

    elif isinstance(data, list) and not data:
        st.info(
            "No user retention data found. The static summary file might be empty or unformatted."
        )
        if st.button("Reload Data"):
            st.cache_data.clear()
            st.rerun()

    else:
        st.error("Received unexpected data format from the backend.")
        st.write("Data received:", data)
        if st.button("Attempt Reload"):
            st.cache_data.clear()
            st.rerun()

    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    user_retention_summary_page()
