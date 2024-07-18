import copy
import streamlit as st
import pandas as pd  # type: ignore

from dataclasses import dataclass
from asyncio import AbstractEventLoop
from typing import Iterator


from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.pickle.vat import Vat
from driftpy.accounts.types import DataAndSlot
from driftpy.types import SpotMarketAccount
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.math.margin import MarginCategory
from driftpy.constants.numeric_constants import (
    SPOT_BALANCE_PRECISION,
    PERCENTAGE_PRECISION,
    QUOTE_PRECISION,
    PRICE_PRECISION,
    BASE_PRECISION,
)

from sections.asset_liab_matrix import get_matrix, NUMBER_OF_SPOT  # type: ignore
from utils import aggregate_perps


@dataclass
class LiquidationInfo:
    spot_market_index: int
    user_public_key: str
    notional_liquidated: float
    spot_asset_scaled_balance: int


spot_fields = [
    "deposit_balance",
    "borrow_balance",
    "initial_asset_weight",
    "maintenance_asset_weight",
    "initial_liability_weight",
    "maintenance_liability_weight",
    "optimal_utilization",
    "optimal_borrow_rate",
    "max_borrow_rate",
    "market_index",
    "scale_initial_asset_weight_start",
]

margin_scalars = {
    "USDC": 1.3,
    "USDT": 1.3,
    "SOL": 1.1,
    "mSOL": 1.1,
    "jitoSOL": 1.1,
    "bSOL": 1.1,
    "INF": 1.1,
    "dSOL": 1.1,
}

DEFAULT_MARGIN_SCALAR = 0.5

stables = [0, 5, 18]
sol_and_lst = [1, 2, 6, 8, 16, 17]
sol_eco = [7, 9, 11, 12, 13, 14, 15, 19]
wrapped = [3, 4]

spot_market_indexes = [market.market_index for market in mainnet_spot_market_configs]

index_options = {}
index_options["All"] = spot_market_indexes
index_options["Stables"] = stables
index_options["Solana and LST"] = sol_and_lst
index_options["Solana Ecosystem"] = sol_eco
index_options["Wrapped"] = wrapped
index_options.update(
    {market.symbol: [market.market_index] for market in mainnet_spot_market_configs}
)


def margin_model(loop: AbstractEventLoop, dc: DriftClient):
    st.header("Margin Model")
    if "vat" not in st.session_state:
        st.write("No Vat loaded.")
        return

    vat: Vat = st.session_state["vat"]

    open_interest = 0
    for perp_market in vat.perp_markets.values():
        oracle_price = (
            vat.perp_oracles.get(perp_market.data.market_index).price / PRICE_PRECISION
        )
        oi_long = perp_market.data.amm.base_asset_amount_long / BASE_PRECISION
        oi_short = abs(perp_market.data.amm.base_asset_amount_short) / BASE_PRECISION
        oi = max(oi_long, oi_short) * oracle_price
        open_interest += oi
    print(open_interest)

    aggregated_users = aggregate_perps(vat, loop)
    # aggregated_users: list[DriftUser]
    # if "agg_perps" not in st.session_state:
    #     aggregated_users = aggregate_perps(vat, loop)
    #     st.session_state["agg_perps"] = aggregated_users
    # else:
    #     aggregated_users = st.session_state["agg_perps"]

    spot_df: pd.DataFrame
    if "spot_df" not in st.session_state:
        spot_df = get_spot_df(vat.spot_markets.values(), vat)
        st.session_state["spot_df"] = spot_df
    else:
        spot_df = st.session_state["spot_df"]

    stable_df = spot_df[spot_df.index.isin(stables)]

    col1, col2, col3 = st.columns([1, 1, 1])
    total_deposits = spot_df["deposit_balance"].sum()
    total_stable_deposits_notional = stable_df["deposit_balance"].sum()
    total_stable_borrows_notional = stable_df["borrow_balance"].sum()
    total_stable_utilization = (
        total_stable_borrows_notional / total_stable_deposits_notional
    )
    total_margin_available = spot_df["max_margin_extended"].sum()
    maximum_exchange_leverage = total_margin_available / total_deposits

    with col1:
        st.markdown(
            f"##### Total Stable Collateral: `${total_stable_deposits_notional:,.2f}` #####"
        )

        st.markdown(
            f"##### Total Stable Liabilities: `${total_stable_borrows_notional:,.2f}` #####"
        )

        st.markdown(
            f"##### Total Stable Utilization: `{(total_stable_utilization * 100):,.2f}%` #####"
        )

    margin_df: pd.DataFrame
    res: pd.DataFrame
    if "margin" not in st.session_state:
        st.session_state["margin"] = get_matrix(loop, vat, dc)
        margin_df = st.session_state["margin"][1]
        res = st.session_state["margin"][0]
    else:
        margin_df = st.session_state["margin"][1]
        res = st.session_state["margin"][0]

    spot_df["all_liabilities"] = spot_df["symbol"].map(res["all_liabilities"])
    spot_df["all_liabilities"] = (
        spot_df["all_liabilities"].str.replace(r"[$,]", "", regex=True).astype(float)
    )

    spot_df.insert(3, "all_liabilities", spot_df.pop("all_liabilities"))

    spot_df["perp_leverage"] = spot_df["all_liabilities"] / spot_df["deposit_balance"]

    spot_df.insert(6, "perp_leverage", spot_df.pop("perp_leverage"))

    spot_df.rename(columns={"leverage": "actual_utilization"}, inplace=True)

    spot_df.insert(4, "optimal_utilization", spot_df.pop("optimal_utilization"))
    total_margin_utilized_notional = sum(
        margin_df[f"spot_{i}_all"].sum() for i in range(NUMBER_OF_SPOT)
    )
    total_margin_utilized = total_margin_utilized_notional / total_margin_available

    # sanity check
    assert float(total_margin_utilized_notional) > float(open_interest)

    actual_exchange_leverage = total_margin_utilized_notional / total_deposits

    with col2:
        st.markdown(
            f"##### Total Margin Extended: `${total_margin_available:,.2f}` #####"
        )

        st.markdown(
            f"##### Total Margin Utilized: `${total_margin_utilized_notional:,.2f}` #####"
        )

        st.markdown(
            f"##### Total Margin Utilized: `{(total_margin_utilized * 100):,.2f}%` #####"
        )

    with col3:
        st.markdown(f"##### Total Collateral: `${total_deposits:,.2f}` #####")

        st.markdown(
            f"##### Maximum Exchange Leverage: `{maximum_exchange_leverage:,.2f}` #####"
        )

        st.markdown(
            f"##### Actual Exchange Leverage: `{actual_exchange_leverage:,.2f}` #####"
        )

    st.markdown("#### Spot Market Overview ####")

    col1, col2 = st.columns([1, 1])

    default_index = list(index_options.keys()).index("All")
    with col1:
        selected_option = st.selectbox(
            "Select Markets", options=list(index_options.keys()), index=default_index
        )

    with col2:
        oracle_warp = st.text_input("Oracle Warp (%, 0-100)", 50)

        liqs_long, liqs_short = get_liquidations(
            aggregated_users, vat, int(oracle_warp), loop
        )

        totals_long = {}
        users_long = {}

        for liquidation_info in liqs_long:
            print(
                f"notional liquidated long: {liquidation_info.notional_liquidated} user public key: {liquidation_info.user_public_key}"
            )
            totals_long[liquidation_info.spot_market_index] = (
                totals_long.get(liquidation_info.spot_market_index, 0)
                + liquidation_info.notional_liquidated
            )
            users_long[liquidation_info.user_public_key] = (
                liquidation_info.notional_liquidated,
                liquidation_info.spot_asset_scaled_balance / SPOT_BALANCE_PRECISION,
            )

        totals_short = {}
        users_short = {}
        print("\n\n\n")

        for liquidation_info in liqs_short:
            print(
                f"notional liquidated short: {liquidation_info.notional_liquidated} user public key: {liquidation_info.user_public_key}"
            )
            totals_short[liquidation_info.spot_market_index] = (
                totals_short.get(liquidation_info.spot_market_index, 0)
                + liquidation_info.notional_liquidated
            )
            users_short[liquidation_info.user_public_key] = (
                liquidation_info.notional_liquidated,
                liquidation_info.spot_asset_scaled_balance / SPOT_BALANCE_PRECISION,
            )

        # this is where we will hit ec2 server for liquidation numbers
        # users_long, users_short = make_request(oracle_warp)

        long_users_list = pd.DataFrame.from_dict(users_long, orient="index")
        short_users_list = pd.DataFrame.from_dict(users_short, orient="index")

        long_users_list.columns = ["liquidation_notional", "scaled_balance"]
        short_users_list.columns = ["liquidation_notional", "scaled_balance"]

        long_liq_notional = pd.Series(totals_long, name="long_liq_notional")
        short_liq_notional = pd.Series(totals_short, name="short_liq_notional")

        long_liq_notional = long_liq_notional.reindex(spot_df.index, fill_value=0)
        short_liq_notional = short_liq_notional.reindex(spot_df.index, fill_value=0)

        spot_df = spot_df.join(long_liq_notional)
        spot_df = spot_df.join(short_liq_notional)

    spot_df.insert(6, "long_liq_notional", spot_df.pop("long_liq_notional"))
    spot_df.insert(7, "short_liq_notional", spot_df.pop("short_liq_notional"))

    selected_indexes = index_options[selected_option]  # type: ignore

    filtered_df = spot_df.loc[spot_df.index.isin(selected_indexes)]

    st.dataframe(display_formatted_df(filtered_df))

    st.markdown("#### Spot Market Analysis ####")

    index_options_list = list(index_options.values())

    all_analytics_df = get_analytics_df(index_options_list[0], spot_df)
    sol_and_lst_basis_df = get_analytics_df(index_options_list[2], spot_df)
    wrapped_basis_df = get_analytics_df(index_options_list[4], spot_df)
    wif_df = get_analytics_df([10], spot_df)

    total_margin_available = all_analytics_df["target_margin_extended"].sum()

    st.markdown(
        f"##### Total Target Margin Extended: `${total_margin_available:,.2f}` #####"
    )

    tabs = st.tabs(list(index_options.keys()))

    def get_recommendations(analytics_df: pd.DataFrame, selected_indexes: list[int]):
        st.write(selected_indexes)

    for idx, tab in enumerate(tabs):
        with tab:
            if idx == 0:
                current_analytics_df = all_analytics_df
                st.dataframe(display_formatted_df(current_analytics_df))
            elif idx == 2:
                st.dataframe(display_formatted_df(sol_and_lst_basis_df))
            elif idx == 4:
                st.dataframe(display_formatted_df(wrapped_basis_df))
            elif set(index_options_list[idx]).issubset(set(sol_and_lst)):
                filtered_df = sol_and_lst_basis_df[
                    sol_and_lst_basis_df.index == index_options_list[idx][0]
                ]
                st.dataframe(display_formatted_df(filtered_df))
            elif set(index_options_list[idx]).issubset(wrapped):
                filtered_df = wrapped_basis_df[
                    wrapped_basis_df.index == index_options_list[idx][0]
                ]
                st.dataframe(display_formatted_df(filtered_df))
            elif set(index_options_list[idx]).issubset(set([10])):  # WIF
                st.dataframe(display_formatted_df(wif_df))
            else:
                analytics_df = get_analytics_df(index_options_list[idx], spot_df)
                st.dataframe(display_formatted_df(analytics_df))

            # if len(index_options_list[idx]) == 1:
            #     get_recommendations(analytics_df, index_options_list[idx])

    (levs_none, _, _) = st.session_state["asset_liab_data"][0]
    user_keys = st.session_state["asset_liab_data"][1]

    df = pd.DataFrame(levs_none, index=user_keys)

    lev, size = get_size_and_lev(df, selected_indexes)

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.expander("users by size (selected overview category)"):
            st.dataframe(size)

    with col2:
        with st.expander("users by lev (selected overview category)"):
            st.dataframe(lev)

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.expander("users long liquidation (offset)"):
            st.dataframe(display_formatted_df(long_users_list))

    with col2:
        with st.expander("users short liquidation (offset)"):
            st.dataframe(display_formatted_df(short_users_list))


def get_size_and_lev(df: pd.DataFrame, market_indexes: list[int]):
    def has_target(net_v, market_indexes):
        if isinstance(net_v, dict):
            return any(net_v.get(idx, 0) != 0 for idx in market_indexes)
        return False

    lev = df.sort_values(by="leverage", ascending=False)

    lev = lev[lev.apply(lambda row: has_target(row["net_v"], market_indexes), axis=1)]

    lev["selected_assets"] = lev["net_v"].apply(
        lambda net_v: {
            idx: net_v.get(idx, 0) for idx in market_indexes if net_v.get(idx, 0) != 0
        }
    )

    size = df.sort_values(by="spot_asset", ascending=False)
    size = size[
        size.apply(lambda row: has_target(row["net_v"], market_indexes), axis=1)
    ]

    size["selected_assets"] = size["net_v"].apply(
        lambda net_v: {
            idx: net_v.get(idx, 0) for idx in market_indexes if net_v.get(idx, 0) != 0
        }
    )

    lev.pop("user_key")
    size.pop("user_key")

    lev.insert(2, "selected_assets", lev.pop("selected_assets"))
    size.insert(2, "selected_assets", size.pop("selected_assets"))

    return lev, size


def get_analytics_df(market_indexes: list[int], spot_df: pd.DataFrame):
    (levs_none, _, _) = st.session_state["asset_liab_data"][0]
    user_keys = st.session_state["asset_liab_data"][1]

    df = pd.DataFrame(levs_none, index=user_keys)

    analytics_df = pd.DataFrame()
    analytics_df.index = spot_df.index
    columns = [
        "symbol",
        "deposit_balance",
        "borrow_balance",
        "all_liabilities",
        "perp_leverage",
        "max_margin_extended",
    ]
    analytics_df[columns] = spot_df[columns]

    def is_basis(market_indexes):
        sol_lst_set = set(sol_and_lst)
        wrapped_set = set(wrapped)
        wif_set = set([10])

        is_sol_lst = market_indexes.issubset(sol_lst_set)
        is_wrapped = market_indexes.issubset(wrapped_set)
        is_wif = market_indexes.issubset(wif_set)

        return is_sol_lst or is_wrapped or is_wif

    if is_basis(set(market_indexes)):
        analytics_df["basis_short"] = spot_df.apply(
            lambda row: get_basis_trade_notional(row, df), axis=1
        )

    def calculate_target_margin(row):
        current_max_margin = row["max_margin_extended"]
        deposit_balance = row["deposit_balance"]
        all_liabilities = row["all_liabilities"]

        target_leverage = 1.0

        if deposit_balance > current_max_margin:
            new_target_margin = deposit_balance
        elif all_liabilities > deposit_balance:
            new_target_margin = all_liabilities / target_leverage
        else:
            new_target_margin = min(deposit_balance, all_liabilities / target_leverage)

        return new_target_margin

    analytics_df["target_margin_extended"] = analytics_df.apply(
        calculate_target_margin, axis=1
    )

    safety_factor = 1.1
    analytics_df["target_margin_extended"] = (
        analytics_df["target_margin_extended"] * safety_factor
    )

    return analytics_df.loc[analytics_df.index.isin(market_indexes)]


def get_perp_short(df, market_index, basis_index):
    new_column_name = f"perp_{basis_index}_short"

    def calculate_perp_short(row):
        net_v = row["net_v"][market_index]
        net_p = row["net_p"][basis_index]
        spot_asset = row["spot_asset"]

        condition = net_v > 0 and net_p < 0
        value = net_v / spot_asset * net_p if condition else 0

        return value

    df[new_column_name] = df.apply(calculate_perp_short, axis=1)

    return df[new_column_name]


def get_basis_trade_notional(row, df):
    basis_index = -1
    market_index = row.name
    if market_index in sol_and_lst and market_index != 0:
        basis_index = 0
    elif market_index == 3:
        basis_index = 1
    elif market_index == 4:
        basis_index = 2

    if basis_index == -1:
        return 0

    perp_short_series: pd.Series
    if f"perp_short_series_{market_index}" in st.session_state:
        perp_short_series = st.session_state[f"perp_short_series_{market_index}"]
    else:
        perp_short_series = get_perp_short(df, market_index, basis_index)
        st.session_state[f"perp_short_series_{market_index}"] = perp_short_series

    total_perp_short = perp_short_series.sum()
    print(
        f"Total perp short for market_index {market_index} basis_index {basis_index}: {total_perp_short}"
    )
    return abs(total_perp_short)


def get_spot_df(accounts: Iterator[DataAndSlot[SpotMarketAccount]], vat: Vat):
    transformations = {
        "deposit_balance": lambda x: x / SPOT_BALANCE_PRECISION,
        "borrow_balance": lambda x: x / SPOT_BALANCE_PRECISION,
        "initial_asset_weight": lambda x: x / PERCENTAGE_PRECISION * 100,
        "maintenance_asset_weight": lambda x: x / PERCENTAGE_PRECISION * 100,
        "initial_liability_weight": lambda x: x / PERCENTAGE_PRECISION * 100,
        "maintenance_liability_weight": lambda x: x / PERCENTAGE_PRECISION * 100,
        "optimal_utilization": lambda x: x / PERCENTAGE_PRECISION,
        "optimal_borrow_rate": lambda x: x / PERCENTAGE_PRECISION,
        "max_borrow_rate": lambda x: x / PERCENTAGE_PRECISION,
        "scale_initial_asset_weight_start": lambda x: x / QUOTE_PRECISION,
    }

    data = [
        {field: getattr(account.data, field) for field in spot_fields}
        for account in accounts
    ]

    df = pd.DataFrame(data)

    if "market_index" in spot_fields:
        df.set_index("market_index", inplace=True)

    for column, transformation in transformations.items():
        if column in df.columns:
            df[column] = df[column].apply(transformation)

    df["leverage"] = df["borrow_balance"] / df["deposit_balance"]

    df["symbol"] = df.index.map(lambda idx: mainnet_spot_market_configs[idx].symbol)

    def notional(row, balance_type):
        market_price = vat.spot_oracles.get(row.name).price / PRICE_PRECISION  # type: ignore
        size = row[balance_type]
        notional_value = size * market_price
        return notional_value

    df["deposit_balance"] = df.apply(notional, balance_type="deposit_balance", axis=1)
    df["borrow_balance"] = df.apply(notional, balance_type="borrow_balance", axis=1)

    df.insert(0, "symbol", df.pop("symbol"))
    df.insert(1, "deposit_balance", df.pop("deposit_balance"))
    df.insert(2, "borrow_balance", df.pop("borrow_balance"))
    df.insert(4, "leverage", df.pop("leverage"))
    df.insert(
        5,
        "scale_initial_asset_weight_start",
        df.pop("scale_initial_asset_weight_start"),
    )

    df.rename(
        columns={"scale_initial_asset_weight_start": "max_margin_extended"},
        inplace=True,
    )
    df = df.sort_index()

    return df


def display_formatted_df(df):
    format_dict = {
        "deposit_balance": "${:,.2f}",
        "borrow_balance": "${:,.2f}",
        "perp_liabilities": "${:,.2f}",
        "initial_asset_weight": "{:.2%}",
        "maintenance_asset_weight": "{:.2%}",
        "initial_liability_weight": "{:.2%}",
        "maintenance_liability_weight": "{:.2%}",
        "optimal_utilization": "{:.2%}",
        "actual_utilization": "{:.2%}",
        "optimal_borrow_rate": "{:.2%}",
        "max_borrow_rate": "{:.2%}",
        "market_index": "{:}",
        "max_margin_extended": "${:,.2f}",
        "perp_leverage": "{:.2f}",
        "target_margin_extended": "${:,.2f}",
        "basis_short": "${:,.2f}",
        "leverage": "{:.2f}",
        "health": "{:.2%}%",
        "short_liq_notional": "${:,.2f}",
        "long_liq_notional": "${:,.2f}",
        "liquidation_notional": "${:,.2f}",
        "scaled_balance": "{:,.2f}",
    }

    df.rename(columns={"all_liabilities": "perp_liabilities"}, inplace=True)

    styled_df = df.style.format(format_dict)

    return styled_df


def get_liquidations(
    aggregated_users: list[DriftUser],
    vat: Vat,
    oracle_warp: int,
    loop: AbstractEventLoop,
) -> tuple[list[LiquidationInfo], list[LiquidationInfo]]:
    long_liquidations: list[LiquidationInfo] = []
    short_liquidations: list[LiquidationInfo] = []

    from driftpy.account_subscription_config import AccountSubscriptionConfig
    from driftpy.user_map.user_map import UserMap
    from driftpy.user_map.user_map_config import UserMapConfig, WebsocketConfig

    usermap = UserMap(UserMapConfig(vat.drift_client, WebsocketConfig()))
    for user in aggregated_users:
        loop.run_until_complete(
            usermap.add_pubkey(
                copy.deepcopy(user.user_public_key),
                copy.deepcopy(user.get_user_account_and_slot()),
            )
        )

    user_copies = list(usermap.values())

    current_sol_perp_price = vat.perp_oracles.get(0).price / PRICE_PRECISION  # type: ignore
    for user in user_copies:
        user_total_spot_value = user.get_spot_market_asset_value()
        print(f"user public key {user.user_public_key}")
        print(f"user total spot value {user_total_spot_value}")
        # ignore users that are bankrupt, of which there should not be many
        if user_total_spot_value <= 0:
            continue

        print(user.get_user_account().spot_positions)
        print(user.get_user_account().perp_positions)

        # save the current user account state
        saved_user_account = copy.deepcopy(user.get_user_account())

        print("\n\n")
        for i, spot_position in enumerate(saved_user_account.spot_positions):
            print(f"spot position: {i}")
            # ignore borrows
            from driftpy.types import is_variant

            if is_variant(spot_position.balance_type, "Borrow"):
                continue
            spot_market_index = spot_position.market_index
            precision = vat.spot_markets.get(spot_market_index).data.decimals  # type: ignore

            # create a copy to force isolated margin upon
            fake_user_account = copy.deepcopy(user.get_user_account())

            # save the current oracle price, s.t. we can reset it after our calculations
            spot_oracle_pubkey = vat.spot_markets.get(spot_position.market_index).data.oracle  # type: ignore
            saved_price_data = vat.spot_oracles.get(spot_position.market_index)
            saved_price = saved_price_data.price
            try:
                # figure out what proportion of the user's collateral is in this spot market
                spot_position_token_amount = user.get_token_amount(
                    spot_position.market_index
                )
                print(f"spot position scaled balance {spot_position.scaled_balance}")
                print(f"spot position token amount {spot_position_token_amount}")
                print(
                    f"spot pos normalized {spot_position_token_amount / (10 ** precision)}"
                )
                collateral_in_spot_asset_usd = (
                    spot_position_token_amount / (10**precision)
                ) * (saved_price / PRICE_PRECISION)
                proportion_of_net_collateral = collateral_in_spot_asset_usd / (
                    user_total_spot_value / QUOTE_PRECISION
                )

                p = [
                    pos
                    for pos in fake_user_account.perp_positions
                    if pos.market_index == 0
                ][0]
                perp_position = copy.deepcopy(p)

                # this shouldn't ever happen, but if it does, we'll skip this user
                if proportion_of_net_collateral > 1:
                    print("proportion of net collateral > 1")
                    continue

                # anything less than 1% of their collateral is dust relative to the rest of the account, so it's negligibly small
                if proportion_of_net_collateral < 0.01:
                    print(
                        f"proportion of net collateral {proportion_of_net_collateral}"
                    )
                    continue

                # scale the perp position size by the proportion of net collateral to mock isolated margin
                perp_position.base_asset_amount = int(
                    perp_position.base_asset_amount * proportion_of_net_collateral
                )
                perp_position.quote_asset_amount = int(
                    perp_position.quote_asset_amount * proportion_of_net_collateral
                )

                # if the position is so small that it's proportionally less than 1e-7 units of the asset, it's dust & negligible
                if abs(perp_position.base_asset_amount) < 100:
                    print(
                        f"perp position base asset amount {perp_position.base_asset_amount}"
                    )
                    continue

                # replace the user's UserAccount with the mocked isolated margin account
                fake_user_account.spot_positions = [copy.deepcopy(spot_position)]
                fake_user_account.perp_positions = [copy.deepcopy(perp_position)]
                user.account_subscriber.user_and_slot.data = fake_user_account

                print(user.get_user_account().spot_positions)
                print(user.get_user_account().perp_positions)

                # set the oracle price to the price after an oracle_warp percent decrease
                # it doesn't make sense to increase the collateral price, because nobody would ever get liquidated if their collateral went up in value
                # our short / long numbers are evaluated based on the type of the perp position, which is...
                shocked_price = int(
                    current_sol_perp_price * (1 - (int(oracle_warp) / 100))
                )
                user.drift_client.account_subscriber.cache["oracle_price_data"][
                    str(spot_oracle_pubkey)
                ].price = int(shocked_price * PRICE_PRECISION)

                # ...evaluated here
                is_short = perp_position.base_asset_amount < 0

                # get the notional value that we would liquidate at the shock_price
                # users are liquidated to 100 "margin ratio units" above their maintenance margin requirements
                shocked_margin_requirement = user.get_margin_requirement(
                    MarginCategory.MAINTENANCE, 100
                )

                shocked_spot_asset_value = user.get_spot_market_asset_value(
                    MarginCategory.MAINTENANCE, include_open_orders=True, strict=False
                )
                shocked_upnl = user.get_unrealized_pnl(
                    True, MarginCategory.MAINTENANCE, strict=False
                )

                print(f"STREAMLIT spot asset value: {shocked_spot_asset_value}")
                print(f"STREAMLIT upnl: {shocked_upnl}")

                shocked_total_collateral = user.get_total_collateral(
                    MarginCategory.MAINTENANCE, False
                )

                # if the user has more collateral than margin required, the position by definition cannot be in liquidation
                if shocked_total_collateral >= shocked_margin_requirement:
                    continue

                shocked_notional = (
                    shocked_margin_requirement - shocked_total_collateral
                ) / QUOTE_PRECISION

                # get the liquidation price of the weighted & aggregated SOL-PERP position after collateral price shock
                print(f"get perp liq price")
                shocked_liquidation_price = user.get_perp_liq_price(0) / PRICE_PRECISION

                # some forced isolated accounts will have such a tiny position that their liquidation price will be some super-tiny negative number
                # in this case, we do not care, because the position size is totally negligible and that liquidation price will never be hit
                if shocked_liquidation_price < 0:
                    continue

                print(f"is short {is_short}")
                print(f"user public key {user.user_public_key}")
                print(f"spot market index {spot_market_index}")
                print(f"margin requirement: {shocked_margin_requirement}")

                print(f"streamlit total collateral: {shocked_total_collateral}")
                print(f"notional: {shocked_notional}")
                print(f"sol perp price {current_sol_perp_price}")
                print(f"liquidation price {shocked_liquidation_price}")
                print(f"proportion of net collateral {proportion_of_net_collateral}")
                print(f"base asset amount {perp_position.base_asset_amount}")
                print(
                    f"perp position value {(perp_position.base_asset_amount / BASE_PRECISION) * current_sol_perp_price}"
                )

                print("\n\n")

                if is_short:
                    # if the position is short, and the liquidation price is lte the current price, the position is in liquidation
                    if shocked_liquidation_price <= current_sol_perp_price:
                        short_liquidations.append(
                            LiquidationInfo(
                                spot_market_index=spot_market_index,
                                user_public_key=user.user_public_key,
                                notional_liquidated=shocked_notional,
                                spot_asset_scaled_balance=spot_position.scaled_balance,
                            )
                        )
                else:
                    # similarly, if the position is long, and the liquidation price is gte the current price, the position is in liquidation
                    if shocked_liquidation_price >= current_sol_perp_price:
                        long_liquidations.append(
                            LiquidationInfo(
                                spot_market_index=spot_market_index,
                                user_public_key=user.user_public_key,
                                notional_liquidated=shocked_notional,
                                spot_asset_scaled_balance=spot_position.scaled_balance,
                            )
                        )
            finally:
                # reset the user object to the original state
                user.drift_client.account_subscriber.cache["oracle_price_data"][
                    str(spot_oracle_pubkey)
                ].price = saved_price
                user.account_subscriber.user_and_slot.data = saved_user_account

    print(len(long_liquidations))
    print(len(short_liquidations))
    return long_liquidations, short_liquidations
