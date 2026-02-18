"""
Common page functionality
"""

import os
from typing import Callable

import streamlit as st

from lib.api import fetch_api_data

RPC_STATE_KEY = "rpc_url"
NETWORK_STATE_KEY = "network"
VAT_STATE_KEY = "vat"


def header():
    image_path = os.path.join(os.path.dirname(__file__), "../../images/driftlogo.png")
    st.logo(image=image_path)


def sidebar():
    st.sidebar.header("Data Information")
    st.sidebar.write(
        "Slot information available for price shock and asset liability matrix pages. Data is live otherwise."
    )


def needs_backend(page_callable: Callable):
    """
    Decorator to add a guard to a page function
    """

    def page_with_guard():
        try:
            fetch_api_data("metadata", "")
        except Exception as e:
            print(e)
            st.error(e)
            return

        page_callable()

    return page_with_guard
