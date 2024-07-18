from asyncio import AbstractEventLoop
from scenario import get_usermap_df
import time
import streamlit as st
from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat
import pandas as pd


def _load_asset_liab_dfs(
    dc: DriftClient, vat: Vat, loop: AbstractEventLoop
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], list[str]]:
    start = time.time()
    oracle_distort = 0
    (levs_none, levs_init, levs_maint), user_keys = loop.run_until_complete(
        get_usermap_df(
            dc,
            vat.users,
            "margins",
            oracle_distort,
            None,
            "ignore stables",
            n_scenarios=0,
            all_fields=True,
        )
    )
    print(f"Loaded asset/liability data in {time.time() - start:.2f} seconds")
    return (levs_none, levs_init, levs_maint), user_keys


@st.cache_data
def _cache_dataframes(
    dfs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], user_keys: list[str]
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], list[str]]:
    return dfs, user_keys


def get_cached_asset_liab_dfs(
    dc: DriftClient, vat: Vat, loop: AbstractEventLoop
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], list[str]]:
    return _cache_dataframes(*_load_asset_liab_dfs(dc, vat, loop))
