import driftpy
import streamlit as st
from streamlit.navigation.page import StreamlitPage


def welcome_page():
    st.title("Drift Risk Dashboard")
    st.markdown("Track key risk metrics across Drift through these dashboard pages")
    st.page_link(
        StreamlitPage("page/orderbook.py", url_path="orderbook"),
        label="📈 **Orderbook** - Compare hyperliquid price to drift orderbook price",
    )

    st.page_link(
        StreamlitPage("page/health.py", url_path="health"),
        label="🏥 **Health** - View account health distribution and largest positions",
    )

    st.page_link(
        StreamlitPage("page/price_shock.py", url_path="price-shock"),
        label="⚡ **Price Shock** - Analyze the impact of price changes on the protocol",
    )

    st.page_link(
        StreamlitPage("page/asset_liability.py", url_path="asset-liability-matrix"),
        label="📊 **Asset-Liability Matrix** - Track assets and liabilities across markets and accounts",
    )

    st.page_link(
        StreamlitPage("page/liquidation_curves.py", url_path="liquidation-curves"),
        label="💧 **Liquidations** - Explore liquidation curves and potential risks",
    )
    st.page_link(
        StreamlitPage("page/deposits_page.py", url_path="deposits"),
        label="💰 **Deposits** - Track total deposits across all the protocol",
    )
    st.page_link(
        StreamlitPage("page/pnl.py", url_path="pnl"),
        label="💸 **PnL** - Track top trader PnLs",
    )

    st.markdown("---")
    st.markdown(
        "For more information about Drift Protocol, visit [drift.trade](https://drift.trade)"
    )
    st.text(f"v{driftpy.__version__}")
