import plotly.express as px
import streamlit as st
import pandas as pd

from lib.api import fetch_cached_data


def health_cached_page():
    health_distribution = fetch_cached_data("health/health_distribution")
    
    # Fetch other data
    most_levered_positions = fetch_cached_data(
        "health/most_levered_perp_positions_above_1m"
    )
    largest_spot_borrows = fetch_cached_data("health/largest_spot_borrows")
    most_levered_borrows = fetch_cached_data(
        "health/most_levered_spot_borrows_above_1m"
    )

    print(health_distribution)

    fig = px.bar(
        health_distribution,
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
        
        largest_perp_positions = fetch_cached_data(
            "health/largest_perp_positions", 
            {"number_of_positions": 100}
        )
        
        # Convert to DataFrame and add pagination
        df = pd.DataFrame(largest_perp_positions)
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
            
        st.markdown("### **Most levered perp positions > $1m:**")
        st.dataframe(most_levered_positions, hide_index=True)

    with spot_col:
        st.markdown("### **Largest spot borrows:**")
        st.dataframe(largest_spot_borrows, hide_index=True)
        st.markdown("### **Most levered spot borrows > $750k:**")
        st.dataframe(most_levered_borrows, hide_index=True)
