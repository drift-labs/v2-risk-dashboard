import sys
from tokenize import tabsize
import driftpy
import pandas as pd
import numpy as np
import copy
import plotly.express as px

pd.options.plotting.backend = "plotly"
# from datafetch.transaction_fetch import load_token_balance
# from driftpy.constants.config import configs
from anchorpy import Provider, Wallet, AccountClient
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import MemcmpOpts
from driftpy.drift_client import DriftClient
from driftpy.accounts import (
    get_perp_market_account,
    get_spot_market_account,
    get_user_account,
    get_state_account,
)
from driftpy.constants.numeric_constants import *
from driftpy.drift_user import DriftUser, get_token_amount

# from datafetch.transaction_fetch import transaction_history_for_account, load_token_balance
import pickle

from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

import os
import json
import streamlit as st
from driftpy.constants.spot_markets import devnet_spot_market_configs, SpotMarketConfig
from driftpy.constants.perp_markets import devnet_perp_market_configs, PerpMarketConfig
from dataclasses import dataclass
from solders.pubkey import Pubkey

# from helpers import serialize_perp_market_2, serialize_spot_market, all_user_stats, DRIFT_WHALE_LIST_SNAP
from anchorpy import EventParser
import asyncio
from driftpy.math.margin import MarginCategory
import requests
from driftpy.types import InsuranceFundStakeAccount, SpotMarketAccount, OraclePriceData
from driftpy.addresses import *
import time
from driftpy.market_map.market_map_config import WebsocketConfig
from driftpy.user_map.user_map import UserMap, UserMapConfig, PollingConfig
import datetime
import csv

NUMBER_OF_SPOT = len(mainnet_spot_market_configs)
NUMBER_OF_PERP = len(mainnet_perp_market_configs)


def get_init_health(user: DriftUser):
    if user.is_being_liquidated():
        return 0

    total_collateral = user.get_total_collateral(MarginCategory.INITIAL)
    maintenance_margin_req = user.get_margin_requirement(MarginCategory.INITIAL)

    if maintenance_margin_req == 0 and total_collateral >= 0:
        return 100
    elif total_collateral <= 0:
        return 0
    else:
        return round(
            min(100, max(0, (1 - maintenance_margin_req / total_collateral) * 100))
        )


def comb_asset_liab(a_l_tup):
    return a_l_tup[0] - a_l_tup[1]


def get_collateral_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_v = {
        i: comb_asset_liab(
            x.get_spot_market_asset_and_liability_value(i, margin_category)
        )
        / QUOTE_PRECISION
        for i in range(n)
    }
    return net_v


def get_perp_liab_composition(x: DriftUser, margin_category, n):
    # ua = x.get_user_account()
    net_p = {
        i: x.get_perp_market_liability(i, margin_category, signed=True)
        / QUOTE_PRECISION
        for i in range(n)
    }
    return net_p


def get_perp_lp_share_composition(x: DriftUser, n):
    # ua = x.get_user_account()
    def get_lp_shares(x, i):
        res = x.get_perp_position(i)
        if res is not None:
            return res.lp_shares / 1e9
        else:
            return 0

    net_p = {i: get_lp_shares(x, i) for i in range(n)}
    return net_p


async def get_usermap_df(
    _drift_client: DriftClient,
    user_map: UserMap,
    mode: str,
    oracle_distor=0.1,
    only_one_index=None,
    cov_matrix=None,
    n_scenarios=5,
    all_fields=False,
):
    perp_n = NUMBER_OF_PERP
    spot_n = NUMBER_OF_SPOT

    def do_dict(x: DriftUser, margin_category: MarginCategory, oracle_cache=None):
        if oracle_cache is not None:
            x.drift_client.account_subscriber.cache = oracle_cache

        if margin_category == MarginCategory.INITIAL:
            health_func = lambda x: get_init_health(x)
        else:
            health_func = lambda x: x.get_health()

        # user_account = x.get_user_account()
        levs0 = {
            # 'tokens': [x.get_token_amount(i) for i in range(spot_n)],
            "user_key": x.user_public_key,
            "leverage": x.get_leverage() / MARGIN_PRECISION,
            "health": health_func(x),
            "perp_liability": x.get_perp_market_liability(None, margin_category)
            / QUOTE_PRECISION,
            "spot_asset": x.get_spot_market_asset_value(None, margin_category)
            / QUOTE_PRECISION,
            "spot_liability": x.get_spot_market_liability_value(None, margin_category)
            / QUOTE_PRECISION,
            "upnl": x.get_unrealized_pnl(True) / QUOTE_PRECISION,
            "net_usd_value": (
                x.get_net_spot_market_value(None) + x.get_unrealized_pnl(True)
            )
            / QUOTE_PRECISION,
            # 'funding_upnl': x.get_unrealized_funding_pnl() / QUOTE_PRECISION,
            # 'total_collateral': x.get_total_collateral(margin_category or MarginCategory.INITIAL) / QUOTE_PRECISION,
            # 'margin_req': x.get_margin_requirement(margin_category or MarginCategory.INITIAL) / QUOTE_PRECISION,
            # 'net_v': get_collateral_composition(x, margin_category, spot_n),
            # 'net_p': get_perp_liab_composition(x, margin_category, perp_n),
            # 'net_lp': get_perp_lp_share_composition(x, perp_n),
            # 'last_active_slot': user_account.last_active_slot,
            # 'cumulative_perp_funding': user_account.cumulative_perp_funding/QUOTE_PRECISION,
            # 'settled_perp_pnl': user_account.settled_perp_pnl/QUOTE_PRECISION,
            # 'name': bytes(user_account.name).decode('utf-8',  errors='ignore').strip(),
            # 'authority': str(user_account.authority),
            # 'has_open_order': user_account.has_open_order,
            # 'sub_account_id': user_account.sub_account_id,
            # 'next_liquidation_id': user_account.next_liquidation_id,
            # 'cumulative_spot_fees': user_account.cumulative_spot_fees,
            # 'total_deposits': user_account.total_deposits,
            # 'total_withdraws': user_account.total_withdraws,
            # 'total_social_loss': user_account.total_social_loss,
            # 'unsettled_pnl_perp_x': x.get_unrealized_pnl(True, market_index=24) / QUOTE_PRECISION,
        }
        # levs0['net_usd_value'] = levs0['spot_asset'] + levs0['upnl'] - levs0['spot_liability']

        if all_fields:
            levs0["net_v"] = get_collateral_composition(x, margin_category, spot_n)
            levs0["net_p"] = get_perp_liab_composition(x, margin_category, spot_n)

        return levs0

    user_map_result: UserMap = user_map

    user_keys = list(user_map_result.user_map.keys())
    user_vals = list(user_map_result.values())
    if cov_matrix == "ignore stables":
        skipped_oracles = [
            str(x.oracle) for x in mainnet_spot_market_configs if "USD" in x.symbol
        ]
    elif cov_matrix == "sol + lst only":
        skipped_oracles = [
            str(x.oracle) for x in mainnet_spot_market_configs if "SOL" not in x.symbol
        ]
    elif cov_matrix == "sol lst only":
        skipped_oracles = [
            str(x.oracle)
            for x in mainnet_spot_market_configs
            if x.symbol not in ["mSOL", "jitoSOL", "bSOL"]
        ]
    elif cov_matrix == "sol ecosystem only":
        skipped_oracles = [
            str(x.oracle)
            for x in mainnet_spot_market_configs
            if x.symbol not in ["PYTH", "JTO", "WIF", "JUP", "TNSR", "DRIFT"]
        ]
    elif cov_matrix == "meme":
        skipped_oracles = [
            str(x.oracle)
            for x in mainnet_spot_market_configs
            if x.symbol not in ["WIF"]
        ]
    elif cov_matrix == "wrapped only":
        skipped_oracles = [
            str(x.oracle)
            for x in mainnet_spot_market_configs
            if x.symbol not in ["wBTC", "wETH"]
        ]
    elif cov_matrix == "stables only":
        skipped_oracles = [
            str(x.oracle) for x in mainnet_spot_market_configs if "USD" not in x.symbol
        ]

    if only_one_index is None or len(only_one_index) > 12:
        only_one_index_key = only_one_index
    else:
        only_one_index_key = (
            [
                str(x.oracle)
                for x in mainnet_perp_market_configs
                if x.base_asset_symbol == only_one_index
            ]
            + [
                str(x.oracle)
                for x in mainnet_spot_market_configs
                if x.symbol == only_one_index
            ]
        )[0]

    if mode == "margins":
        levs_none = list(do_dict(x, None) for x in user_vals)
        levs_init = list(do_dict(x, MarginCategory.INITIAL) for x in user_vals)
        levs_maint = list(do_dict(x, MarginCategory.MAINTENANCE) for x in user_vals)
        return (levs_none, levs_init, levs_maint), user_keys
    else:
        num_entrs = n_scenarios  # increment to get more steps
        new_oracles_dat_up = []
        new_oracles_dat_down = []

        for i in range(num_entrs):
            new_oracles_dat_up.append({})
            new_oracles_dat_down.append({})

        assert len(new_oracles_dat_down) == num_entrs
        print("skipped oracles:", skipped_oracles)
        distorted_oracles = []
        cache_up = copy.deepcopy(_drift_client.account_subscriber.cache)
        cache_down = copy.deepcopy(_drift_client.account_subscriber.cache)
        for i, (key, val) in enumerate(
            _drift_client.account_subscriber.cache["oracle_price_data"].items()
        ):
            for i in range(num_entrs):
                new_oracles_dat_up[i][key] = copy.deepcopy(val)
                new_oracles_dat_down[i][key] = copy.deepcopy(val)
            if cov_matrix is not None and key in skipped_oracles:
                continue
            if only_one_index is None or only_one_index_key == key:
                distorted_oracles.append(key)
                for i in range(num_entrs):
                    oracle_distort_up = max(1 + oracle_distor * (i + 1), 1)
                    oracle_distort_down = max(1 - oracle_distor * (i + 1), 0)

                    # weird pickle artifact
                    if isinstance(new_oracles_dat_up[i][key], OraclePriceData):
                        new_oracles_dat_up[i][key].price *= oracle_distort_up
                        new_oracles_dat_down[i][key].price *= oracle_distort_down
                    else:
                        new_oracles_dat_up[i][key].data.price *= oracle_distort_up
                        new_oracles_dat_down[i][key].data.price *= oracle_distort_down

        levs_none = list(do_dict(x, None, None) for x in user_vals)
        levs_up = []
        levs_down = []

        for i in range(num_entrs):
            cache_up["oracle_price_data"] = new_oracles_dat_up[i]
            cache_down["oracle_price_data"] = new_oracles_dat_down[i]
            levs_up_i = list(do_dict(x, None, cache_up) for x in user_vals)
            levs_down_i = list(do_dict(x, None, cache_down) for x in user_vals)
            levs_up.append(levs_up_i)
            levs_down.append(levs_down_i)

        return (
            (levs_none, tuple(levs_up), tuple(levs_down)),
            user_keys,
            distorted_oracles,
        )


async def get_new_ff(usermap):
    await usermap.sync()
    usermap.dump()
