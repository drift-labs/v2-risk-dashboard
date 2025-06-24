import os
import sys

# Add the parent directory to the Python path so we can import 'shared'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

from lib.page import header, needs_backend, sidebar
from page.asset_liability import asset_liab_matrix_cached_page
from page.backend import backend_page
from page.deposits_page import deposits_page
from page.health import health_page
from page.liquidation_curves import liquidation_curves_page
from page.market_inspector import market_inspector_page
from page.orderbook import orderbook_page
from page.pnl import pnl_page
from page.price_shock import price_shock_cached_page
from page.swap import show as swap_page
from page.vaults import vaults_page
from page.welcome import welcome_page
from page.market_recommender_page import market_recommender_page
from page.open_interest_page import open_interest_page
from page.high_leverage_page import high_leverage_page
from page.user_retention_summary_page import user_retention_summary_page
from page.user_retention_explorer_page import user_retention_explorer_page

load_dotenv()

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(path) as css:
        custom_css = css.read()

    def apply_custom_css(css):
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": None, "Get help": None, "Report a bug": None},
    )
    apply_custom_css(custom_css)
    header()
    sidebar()

    if os.getenv("DEV"):
        settings_pages = [
            st.Page(
                needs_backend(backend_page),
                url_path="backend",
                title="Control Backend",
                icon="🧪",
            )
        ]
    else:
        settings_pages = []

    welcome_pages = [
        st.Page(
            welcome_page,
            url_path="welcome",
            title="Welcome",
            icon="🏠",
        ),
    ]

    market_pages = [
        st.Page(
            swap_page,
            url_path="swap",
            title="Swap",
            icon="🔄",
        ),
        st.Page(
            orderbook_page,
            url_path="orderbook",
            title="Orderbook",
            icon="📈",
        ),
        st.Page(
            market_inspector_page,
            url_path="market-inspector",
            title="Market Inspector",
            icon="🔎",
        ),
        st.Page(
            needs_backend(deposits_page),
            url_path="deposits",
            title="Deposits",
            icon="💰",
        ),
        st.Page(
            market_recommender_page,
            url_path="market-recommender",
            title="Market Recommender",
            icon="🚀",
        ),
    ]

    risk_pages = [
        st.Page(
            needs_backend(health_page),
            url_path="health",
            title="Health",
            icon="🏥",
        ),
        st.Page(
            price_shock_cached_page,
            url_path="price-shock",
            title="Price Shock",
            icon="💸",
        ),
        st.Page(
            needs_backend(liquidation_curves_page),
            url_path="liquidation-curves",
            title="Liquidation Curves",
            icon="🌊",
        ),
    ]

    analytics_pages = [
        st.Page(
            asset_liab_matrix_cached_page,
            url_path="asset-liability-matrix",
            title="Asset-Liability Matrix",
            icon="📊",
        ),
        st.Page(
            needs_backend(pnl_page),
            url_path="pnl",
            title="PnL",
            icon="💹",
        ),
        st.Page(
            needs_backend(vaults_page),
            url_path="vaults",
            title="Vaults",
            icon="🏦",
        ),
        st.Page(
            open_interest_page,
            url_path="open-interest",
            title="Open Interest",
            icon="💰",
        ),
        st.Page(
            high_leverage_page,
            url_path="high-leverage",
            title="High Leverage",
            icon="⚡",
        ),
        st.Page(
            needs_backend(user_retention_summary_page),
            url_path="user-retention-summary",
            title="User Retention Summary",
            icon="👥",
        ),
        st.Page(
            needs_backend(user_retention_explorer_page),
            url_path="user-retention-explorer",
            title="User Retention Explorer",
            icon="🔍",
        ),
    ]

    pg = st.navigation(
        {
            "Welcome": welcome_pages,
            "Markets": market_pages,
            "Risk Management": risk_pages,
            "Analytics": analytics_pages,
            **({"Settings": settings_pages} if settings_pages else {}),
        }
    )
    pg.run()
