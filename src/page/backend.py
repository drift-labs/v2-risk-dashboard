import streamlit as st

from lib.api import fetch_api_data


def backend_page():
    if st.button("Load Pickle"):
        result = fetch_api_data("snapshot", "pickle")
        st.write(result)

    st.write(
        """
    ## Backend API

    - [swagger](http://localhost:8000/docs)

    - [redoc](http://localhost:8000/redoc)
    """
    )

    st.title("Health")
