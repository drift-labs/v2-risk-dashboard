from asyncio import AbstractEventLoop
import pandas as pd
import numpy as np

import os
from typing import Optional

from driftpy.drift_client import DriftClient
from driftpy.pickle.vat import Vat

from driftpy.user_map.user_map_config import (
    UserMapConfig,
    UserStatsMapConfig,
    WebsocketConfig as UserMapWebsocketConfig,
)
from driftpy.user_map.user_map import UserMap
from driftpy.user_map.userstats_map import UserStatsMap

from driftpy.market_map.market_map_config import (
    MarketMapConfig,
    WebsocketConfig as MarketMapWebsocketConfig,
)
from driftpy.market_map.market_map import MarketMap

from driftpy.types import MarketType, PerpPosition, PerpMarketAccount
from driftpy.drift_user import DriftUser
from driftpy.math.margin import MarginCategory
from driftpy.constants.numeric_constants import (
    PRICE_PRECISION,
    MARGIN_PRECISION,
    QUOTE_PRECISION,
)
from driftpy.math.margin import calculate_size_premium_liability_weight
from scenario import (
    NUMBER_OF_SPOT,
    get_collateral_composition,
    get_perp_liab_composition,
)


def calculate_market_margin_ratio(
    market: PerpMarketAccount, size, margin_category, custom_margin_ratio=0
):
    market_margin_ratio = (
        market.margin_ratio_initial
        if margin_category == MarginCategory.INITIAL
        else market.margin_ratio_maintenance
    )

    margin_ratio = max(
        calculate_size_premium_liability_weight(
            size, market.imf_factor, market_margin_ratio, MARGIN_PRECISION
        ),
        custom_margin_ratio,
    )

    return margin_ratio


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
            if not prefix in newest_files or slot > newest_files[prefix][1]:
                newest_files[prefix] = (directory + "/" + filename, slot)

    # mapping e.g { 'spotoracles' : 'spotoracles_272636137.pkl' }
    prefix_to_filename = {
        prefix: filename for prefix, (filename, _) in newest_files.items()
    }

    return prefix_to_filename


# function assumes that you have already subscribed
# the use of websocket configs in here doesn't matter because the maps are never subscribed to
async def load_vat(
    dc: DriftClient,
    pickle_map: dict[str, str],
    loop: AbstractEventLoop,
    env: str = "prod",
) -> Vat:
    perp = MarketMap(
        MarketMapConfig(
            dc.program, MarketType.Perp(), MarketMapWebsocketConfig(), dc.connection
        )
    )

    spot = MarketMap(
        MarketMapConfig(
            dc.program, MarketType.Spot(), MarketMapWebsocketConfig(), dc.connection
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

    if env == "dev":
        users = []
        for user in vat.users.values():
            value = user.get_net_spot_market_value(None) + user.get_unrealized_pnl(True)
            users.append(
                (value, user.user_public_key, user.get_user_account_and_slot())
            )
        users.sort(key=lambda x: x[0], reverse=True)
        vat.users.clear()
        for user in users[:100]:
            await vat.users.add_pubkey(user[1], user[2])

    print(vat.users.values())

    return vat


def clear_local_pickles(directory: str):
    for filename in os.listdir(directory):
        os.remove(directory + "/" + filename)


def aggregate_perps(vat: Vat, loop: AbstractEventLoop):
    print("aggregating perps")

    def aggregate_perp(user: DriftUser) -> DriftUser:
        agg_perp = PerpPosition(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        sol_price = vat.perp_oracles.get(0).price
        user_account = user.get_user_account()
        sol_market = vat.perp_markets.get(0).data
        for perp_position in user_account.perp_positions:
            if perp_position.base_asset_amount == 0:
                continue
            asset_price = vat.perp_oracles.get(perp_position.market_index).price  # type: ignore

            market = vat.perp_markets.get(perp_position.market_index)
            # ratio transform
            sol_margin_ratio = calculate_market_margin_ratio(
                sol_market, agg_perp.base_asset_amount, MarginCategory.INITIAL
            )
            margin_ratio = calculate_market_margin_ratio(
                market.data, perp_position.base_asset_amount, MarginCategory.INITIAL
            )
            sol_margin_scalar = 1 / (sol_margin_ratio / MARGIN_PRECISION)
            curr_margin_scalar = 1 / (margin_ratio / MARGIN_PRECISION)

            # simple price conversion
            exchange_rate = sol_price / asset_price
            exchange_rate_normalized = exchange_rate / PRICE_PRECISION
            new_baa = perp_position.base_asset_amount * exchange_rate_normalized

            # apply margin ratio transofmr
            new_baa_adjusted = new_baa * (sol_margin_scalar / curr_margin_scalar)

            # aggregate
            agg_perp.base_asset_amount += new_baa_adjusted
            agg_perp.quote_asset_amount += perp_position.quote_asset_amount * (
                sol_margin_scalar / curr_margin_scalar
            )

        if agg_perp.base_asset_amount == 0:
            return None

        # force use this new fake user account for all sdk functions
        user_account.perp_positions = [agg_perp]
        ds = user.account_subscriber.user_and_slot
        ds.data = user_account
        user.account_subscriber.user_and_slot = ds
        return user

    users_list = list(vat.users.values())

    import copy

    # deep copy usermap
    # required or else aggregation affects vat.users which breaks stuff p bad
    usermap = UserMap(UserMapConfig(vat.drift_client, UserMapWebsocketConfig()))
    for user in users_list:
        loop.run_until_complete(
            usermap.add_pubkey(
                copy.deepcopy(user.user_public_key),
                copy.deepcopy(user.get_user_account_and_slot()),
            )
        )

    aggregated_users = [
        user
        for user in (aggregate_perp(user) for user in usermap.values())
        if user is not None
    ]

    aggregated_users = sorted(
        aggregated_users,
        key=lambda x: x.get_total_collateral(MarginCategory.MAINTENANCE)
        + x.get_total_perp_position_value(MarginCategory.MAINTENANCE),
    )

    return aggregated_users
