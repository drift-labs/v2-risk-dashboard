import os
import sys

# Add the parent directory to the Python path so we can import 'shared'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from dotenv import load_dotenv

from lib.page import header, needs_backend, sidebar
from page.asset_liability import asset_liab_matrix_cached_page
from page.backend import backend_page
from page.deposits import deposits_page
from page.health import health_page
from page.liquidation_curves import liquidation_curves_page
from page.orderbook import orderbook_page
from page.pnl import pnl_page
from page.price_shock import price_shock_cached_page
from page.welcome import welcome_page
from page.swap import show as swap_page

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

    main_pages = [
        st.Page(
            welcome_page,
            url_path="welcome",
            title="Welcome",
            icon="🏠",
        ),
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
            asset_liab_matrix_cached_page,
            url_path="asset-liability-matrix",
            title="Asset-Liability Matrix",
            icon="📊",
        ),
        st.Page(
            needs_backend(liquidation_curves_page),
            url_path="liquidation-curves",
            title="Liquidation Curves",
            icon="🌊",
        ),
        st.Page(
            needs_backend(deposits_page),
            url_path="deposits",
            title="Deposits",
            icon="💰",
        ),
        st.Page(needs_backend(pnl_page), url_path="pnl", title="PnL", icon="💹"),
    ]

    if os.getenv("DEV"):
        main_pages.append(
            st.Page(
                needs_backend(backend_page),
                url_path="backend",
                title="Control Backend",
                icon="🧪",
            )
        )

    pg = st.navigation(
        {
            "Main": main_pages,
        }
    )
    pg.run()
