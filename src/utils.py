import datetime
import os
from typing import Optional, Any, Union
import pandas as pd
import copy # Added for deepcopy

import requests
from driftpy.decode.utils import decode_name
from driftpy.drift_client import DriftClient
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig
from driftpy.market_map.market_map_config import (
    WebsocketConfig as MarketMapWebsocketConfig,
)
from driftpy.pickle.vat import Vat
from driftpy.types import MarketType, PerpMarketAccount, SpotMarketAccount
from driftpy.user_map.user_map import UserMap
from driftpy.user_map.user_map_config import UserMapConfig, UserStatsMapConfig
from driftpy.user_map.user_map_config import (
    WebsocketConfig as UserMapWebsocketConfig,
)
from driftpy.user_map.userstats_map import UserStatsMap
from solders.pubkey import Pubkey # Import Pubkey

from dotenv import load_dotenv
load_dotenv()

# Helper function to stringify sumtypes, Pubkeys, and other problematic types
def _stringify_value(value: Any) -> Any:
    if hasattr(value, 'kind') and isinstance(value.kind, str):
        # Common pattern for driftpy sumtypes (e.g., MarketType, OracleSource)
        return value.kind
    elif (hasattr(value, '__class__') and
          hasattr(value.__class__, '__module__') and
          'sumtypes' in value.__class__.__module__):
        # Another pattern for sumtypes (e.g., MarketStatus)
        return value.__class__.__name__
    elif isinstance(value, Pubkey):
        return str(value)
    elif isinstance(value, list):
        return [_stringify_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _stringify_value(v) for k, v in value.items()}
    # Add other specific type checks if needed, e.g., for specific complex objects
    # that are not sumtypes but still cause issues with Arrow.
    return value

def _prepare_for_serialization(obj_dict: dict) -> dict:
    prepared_dict = {}
    for key, value in obj_dict.items():
        prepared_dict[key] = _stringify_value(value)
    return prepared_dict

def to_financial(num):
    num_str = str(num)
    decimal_pos = num_str.find(".")
    if decimal_pos != -1:
        return float(num_str[: decimal_pos + 3])
    return num


def load_newest_files(directory: Optional[str] = None) -> dict[str, str]:
    directory = directory or os.getcwd()

    newest_files: dict[str, tuple[str, int]] = {}

    prefixes = ["perp", "perporacles", "spot", "spotoracles", "usermap", "userstats"]

    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and any(
            filename.startswith(prefix + "_") for prefix in prefixes
        ):
            start = filename.index("_") + 1
            prefix = filename[: start - 1]
            end = filename.index(".")
            slot = int(filename[start:end])
            if prefix not in newest_files or slot > newest_files[prefix][1]:
                newest_files[prefix] = (directory + "/" + filename, slot)

    prefix_to_filename = {
        prefix: filename for prefix, (filename, _) in newest_files.items()
    }

    return prefix_to_filename


async def load_vat(dc: DriftClient, pickle_map: dict[str, str]) -> Vat:
    perp = MarketMap(
        MarketMapConfig(
            dc.program,
            MarketType.Perp(),  # type: ignore
            MarketMapWebsocketConfig(),
            dc.connection,
        )
    )

    spot = MarketMap(
        MarketMapConfig(
            dc.program,
            MarketType.Spot(),  # type: ignore
            MarketMapWebsocketConfig(),
            dc.connection,
        )
    )

    user = UserMap(UserMapConfig(dc, UserMapWebsocketConfig()))

    stats = UserStatsMap(UserStatsMapConfig(dc))

    user_filename = pickle_map["usermap"]
    stats_filename = pickle_map["userstats"]
    perp_filename = pickle_map["perp"]
    spot_filename = pickle_map["spot"]
    perp_oracles_filename = pickle_map["perporacles"]
    spot_oracles_filename = pickle_map["spotoracles"]

    vat = Vat(dc, user, stats, spot, perp)

    await vat.unpickle(
        user_filename,
        stats_filename,
        spot_filename,
        perp_filename,
        spot_oracles_filename,
        perp_oracles_filename,
    )

    return vat


def get_current_slot():
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getSlot",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    response = requests.post(
        os.getenv("RPC_URL"), json=payload, headers=headers
    )
    return response.json()["result"]


def human_market_df(df):
    enum_fields = ['status', 'contract_tier', '']
    pure_fields = ['number_of_users', 'market_index', 'next_curve_record_id', 'next_fill_record_id', 'next_funding_rate_record_id']
    pct_fields = ['imf_factor', 'unrealized_pnl_imf_factor', 'liquidator_fee', 'if_liquidation_fee']
    wgt_fields = ['initial_asset_weight', 'maintenance_asset_weight',

    'initial_liability_weight', 'maintenance_liability_weight',
    'unrealized_pnl_initial_asset_weight', 'unrealized_pnl_maintenance_asset_weight']
    margin_fields = ['margin_ratio_initial', 'margin_ratio_maintenance']
    px_fields = [
        'expiry_price',
        'last_oracle_normalised_price',
        'order_tick_size',
        'last_bid_price_twap', 'last_ask_price_twap', 'last_mark_price_twap', 'last_mark_price_twap5min',
    'peg_multiplier',
    'mark_std',
    'oracle_std',
    'last_oracle_price_twap', 'last_oracle_price_twap5min',

    ]
    time_fields = ['last_trade_ts', 'expiry_ts', 'last_revenue_withdraw_ts']
    balance_fields = ['scaled_balance', 'deposit_balance', 'borrow_balance']
    quote_fields = [
        'total_spot_fee',
        'unrealized_pnl_max_imbalance', 'quote_settled_insurance', 'quote_max_insurance',
    'max_revenue_withdraw_per_period', 'revenue_withdraw_since_last_settle', ]
    token_fields = ['borrow_token_twap', 'deposit_token_twap', 'withdraw_guard_threshold', 'max_token_deposits']
    interest_fields = ['cumulative_deposit_interest', 'cumulative_borrow_interest']

    for col in df.columns:
        # if col in enum_fields:
        #     pass
        # elif col in pure_fields:
        #     pass
        if col in pct_fields:
            df[col] /= 1e6
        elif col in px_fields:
            df[col] /= 1e6
        elif col in margin_fields:
            df[col] /= 1e4
        elif col in wgt_fields:
            df[col] /= 1e4
        # elif col in time_fields:
        #     pass
        elif col in quote_fields:
            df[col] /= 1e6
        elif col in balance_fields:
            df[col] /= 1e9
        elif col in interest_fields:
            df[col] /= 1e10
        elif col in token_fields:
            df[col] /= 1e6 #todo

    return df


def human_amm_df(df):
    bool_fields = [ 'last_oracle_valid']
    enum_fields = ['oracle_source']
    pure_fields = ['last_update_slot', 'long_intensity_count', 'short_intensity_count',
    'curve_update_intensity', 'amm_jit_intensity'
    ]
    reserve_fields = [
        'base_asset_reserve', 'quote_asset_reserve', 'min_base_asset_reserve', 'max_base_asset_reserve', 'sqrt_k',
        'ask_base_asset_reserve', 'ask_quote_asset_reserve', 'bid_base_asset_reserve', 'bid_quote_asset_reserve',
        'terminal_quote_asset_reserve', 'base_asset_amount_long', 'base_asset_amount_short', 'base_asset_amount_with_amm', 'base_asset_amount_with_unsettled_lp',
        'user_lp_shares', 'min_order_size', 'max_position_size', 'order_step_size', 'max_open_interest',
        ]

    wgt_fields = ['initial_asset_weight', 'maintenance_asset_weight',

    'initial_liability_weight', 'maintenance_liability_weight',
    'unrealized_pnl_initial_asset_weight', 'unrealized_pnl_maintenance_asset_weight']

    pct_fields = ['base_spread','long_spread', 'short_spread', 'max_spread', 'concentration_coef',
    'last_oracle_reserve_price_spread_pct',
    'last_oracle_conf_pct',
        #spot market ones
    'utilization_twap',

    'imf_factor', 'unrealized_pnl_imf_factor', 'liquidator_fee', 'if_liquidation_fee',
    'optimal_utilization', 'optimal_borrow_rate', 'max_borrow_rate',
    ]

    funding_fields = ['cumulative_funding_rate_long', 'cumulative_funding_rate_short', 'last_funding_rate', 'last_funding_rate_long', 'last_funding_rate_short', 'last24h_avg_funding_rate']
    quote_asset_fields = ['total_fee', 'total_mm_fee', 'total_exchange_fee', 'total_fee_minus_distributions',
    'total_fee_withdrawn', 'total_liquidation_fee', 'cumulative_social_loss', 'net_revenue_since_last_funding',
    'quote_asset_amount_long', 'quote_asset_amount_short', 'quote_entry_amount_long', 'quote_entry_amount_short',
    'volume24h', 'long_intensity_volume', 'short_intensity_volume',
    'total_spot_fee', 'quote_asset_amount',
    'quote_break_even_amount_short', 'quote_break_even_amount_long'
    ]
    time_fields = ['last_trade_ts', 'last_mark_price_twap_ts', 'last_oracle_price_twap_ts', 'last_index_price_twap_ts',]
    duration_fields = ['lp_cooldown_time', 'funding_period']
    px_fields = [
        'last_oracle_normalised_price',
        'order_tick_size',
        'last_bid_price_twap', 'last_ask_price_twap', 'last_mark_price_twap', 'last_mark_price_twap5min',
    'peg_multiplier',
    'mark_std',
    'oracle_std',
    'last_oracle_price_twap', 'last_oracle_price_twap5min',
    'last_oracle_price', 'last_oracle_conf',

    #spot market ones
        'last_index_bid_price', 'last_index_ask_price', 'last_index_price_twap', 'last_index_price_twap5min',

    ]
    token_fields = ['deposit_token_twap', 'borrow_token_twap', 'max_token_deposits', 'withdraw_guard_threshold']
    balance_fields = ['scaled_balance', 'deposit_balance', 'borrow_balance']
    interest_fileds = ['cumulative_deposit_interest', 'cumulative_borrow_interest']
    for col in df.columns:
        # if col in enum_fields or col in bool_fields:
        #     pass
        # else if col in duration_fields:
        #     pass
        # else if col in pure_fields:
        #     pass
        if col in reserve_fields:
            df[col] /= 1e9
        elif col in funding_fields:
            df[col] /= 1e9
        elif col in wgt_fields:
            df[col] /= 1e4
        elif col in quote_asset_fields:
            df[col] /= 1e6
        elif col in pct_fields:
            df[col] /= 1e6
        elif col in px_fields:
            df[col] /= 1e6
        elif col in token_fields:
            z = df['decimals'].values[0]
            df[col] /= (10**z)
        elif col in interest_fileds:
            df[col] /= 1e10
        elif col in time_fields:
            df[col] = [datetime.datetime.fromtimestamp(x) for x in df[col].values]
        elif col in balance_fields:
            df[col] /= 1e9

    return df


def serialize_perp_market(market: PerpMarketAccount):
    # Prepare market data by stringifying sumtypes and Pubkeys
    market_dict_prepared = _prepare_for_serialization(copy.deepcopy(market.__dict__))
    amm_dict_prepared = _prepare_for_serialization(copy.deepcopy(market.amm.__dict__))
    hist_oracle_data_prepared = _prepare_for_serialization(copy.deepcopy(market.amm.historical_oracle_data.__dict__))
    fee_pool_prepared = _prepare_for_serialization(copy.deepcopy(market.amm.fee_pool.__dict__))
    insurance_claim_prepared = _prepare_for_serialization(copy.deepcopy(market.insurance_claim.__dict__))
    pnl_pool_prepared = _prepare_for_serialization(copy.deepcopy(market.pnl_pool.__dict__))

    market_df = pd.json_normalize(market_dict_prepared).drop(['amm', 'insurance_claim', 'pnl_pool'],axis=1, errors='ignore').pipe(human_market_df)
    # 'name' is bytes, decode_name handles it; 'pubkey' is already stringified by _prepare_for_serialization if it was a Pubkey object
    if 'name' in market_df.columns and market_dict_prepared.get('name'): # Check if name exists before decoding
         market_df['name'] = decode_name(market_dict_prepared['name']) # Use original bytes for decode_name
    market_df.columns = ['market.'+col for col in market_df.columns]

    amm_df= pd.json_normalize(amm_dict_prepared).drop(['historical_oracle_data', 'fee_pool'],axis=1, errors='ignore').pipe(human_amm_df)
    amm_df.columns = ['market.amm.'+col for col in amm_df.columns]

    amm_hist_oracle_df= pd.json_normalize(hist_oracle_data_prepared).pipe(human_amm_df)
    amm_hist_oracle_df.columns = ['market.amm.historical_oracle_data.'+col for col in amm_hist_oracle_df.columns]

    market_amm_pool_df = pd.json_normalize(fee_pool_prepared).pipe(human_amm_df)
    market_amm_pool_df.columns = ['market.amm.fee_pool.'+col for col in market_amm_pool_df.columns]

    market_if_df = pd.json_normalize(insurance_claim_prepared).pipe(human_market_df)
    market_if_df.columns = ['market.insurance_claim.'+col for col in market_if_df.columns]

    market_pool_df = pd.json_normalize(pnl_pool_prepared).pipe(human_amm_df)
    market_pool_df.columns = ['market.pnl_pool.'+col for col in market_pool_df.columns]

    result_df = pd.concat([market_df, amm_df, amm_hist_oracle_df, market_amm_pool_df, market_if_df, market_pool_df],axis=1)
    
    # Final conversion of object columns to string for Arrow compatibility
    for col in result_df.columns:
        if result_df[col].dtype == 'object':
            try:
                result_df[col] = result_df[col].astype(str)
            except Exception:
                # Fallback if astype(str) fails for any reason on a column
                result_df[col] = result_df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
    return result_df


def serialize_spot_market(spot_market: SpotMarketAccount):
    # Prepare spot_market data
    spot_market_dict_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.__dict__))
    insurance_fund_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.insurance_fund.__dict__))
    hist_oracle_data_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.historical_oracle_data.__dict__))
    hist_index_data_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.historical_index_data.__dict__))
    revenue_pool_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.revenue_pool.__dict__))
    spot_fee_pool_prepared = _prepare_for_serialization(copy.deepcopy(spot_market.spot_fee_pool.__dict__))

    spot_market_df = pd.json_normalize(spot_market_dict_prepared).drop([
        'historical_oracle_data', 'historical_index_data',
        'insurance_fund', 
        'spot_fee_pool', 'revenue_pool'
        ], axis=1, errors='ignore').pipe(human_amm_df) # Note: using human_amm_df as per original
    
    # 'name' is bytes, decode_name handles it. Other Pubkey fields are stringified.
    if 'name' in spot_market_df.columns and spot_market_dict_prepared.get('name'):
        spot_market_df['name'] = decode_name(spot_market_dict_prepared['name']) # Use original bytes

    spot_market_df.columns = ['spot_market.'+col for col in spot_market_df.columns]

    if_df= pd.json_normalize(insurance_fund_prepared).pipe(human_amm_df)
    if_df.columns = ['spot_market.insurance_fund.'+col for col in if_df.columns]

    hist_oracle_df= pd.json_normalize(hist_oracle_data_prepared).pipe(human_amm_df)
    hist_oracle_df.columns = ['spot_market.historical_oracle_data.'+col for col in hist_oracle_df.columns]

    hist_index_df= pd.json_normalize(hist_index_data_prepared).pipe(human_amm_df)
    hist_index_df.columns = ['spot_market.historical_index_data.'+col for col in hist_index_df.columns]


    market_pool_df = pd.json_normalize(revenue_pool_prepared).pipe(human_amm_df)
    market_pool_df.columns = ['spot_market.revenue_pool.'+col for col in market_pool_df.columns]


    market_fee_df = pd.json_normalize(spot_fee_pool_prepared).pipe(human_amm_df)
    market_fee_df.columns = ['spot_market.spot_fee_pool.'+col for col in market_fee_df.columns]

    result_df = pd.concat([spot_market_df, if_df, hist_oracle_df, hist_index_df, market_pool_df, market_fee_df],axis=1)
    
    # Final conversion of object columns to string for Arrow compatibility
    for col in result_df.columns:
        if result_df[col].dtype == 'object':
            try:
                result_df[col] = result_df[col].astype(str)
            except Exception:
                # Fallback if astype(str) fails for any reason on a column
                result_df[col] = result_df[col].apply(lambda x: str(x) if pd.notnull(x) else x)
    return result_df