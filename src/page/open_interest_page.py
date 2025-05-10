import streamlit as st
import pandas as pd
from lib.api import fetch_api_data
from utils import get_current_slot

def open_interest_page():
    st.title("Open Interest per Authority")

    try:
        # Fetch data from the API endpoint directly
        result = fetch_api_data(
            section="open-interest", 
            path="per-authority",
            retry=True # Enable retry for cache misses/processing states
        )
        
        if result is None:
            st.error("Failed to fetch data from the API.")
            return

        data = result.get("data", [])
        slot = result.get("slot", "N/A")
        current_slot = get_current_slot()

        if slot != "N/A" and current_slot:
            try:
                slot_age = int(current_slot) - int(slot)
                st.info(f"Displaying data for slot {slot} (age: {slot_age} slots)")
            except ValueError:
                st.info(f"Displaying data for slot {slot}. Current slot: {current_slot}")
        else:
            st.info(f"Slot information unavailable. Current slot: {current_slot}")

        if not data:
            st.warning("No open interest data found.")
            return

        df = pd.DataFrame(data)

        if df.empty:
            st.warning("No open interest data to display.")
            return

        # Rename columns for better readability
        df.rename(columns={
            'authority': 'User Authority',
            'total_open_interest_usd': 'Total Open Interest (USD)'
        }, inplace=True)
        # Reorder columns
        df = df[["User Authority", "Total Open Interest (USD)"]]
        # Format USD column with dollar sign and commas
        df["Total Open Interest (USD)"] = df["Total Open Interest (USD)"].apply(lambda x: f"${x:,.2f}")
        
        st.metric("Total Authorities with Open Interest", len(df))
        st.metric("Total Open Interest (USD)", f"{df['Total Open Interest (USD)'].str.replace('$','').str.replace(',','').astype(float).sum():,.2f}")

        st.subheader("Open Interest Details")
        st.dataframe(df, hide_index=True)

    except Exception as e:
        st.error(f"An error occurred while displaying the page: {e}")
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    open_interest_page()
