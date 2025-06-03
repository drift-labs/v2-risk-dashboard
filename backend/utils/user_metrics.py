import copy
import functools
from typing import List, Optional

from driftpy.accounts.cache import DriftClientCache
from driftpy.constants.numeric_constants import MARGIN_PRECISION, QUOTE_PRECISION
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.drift_client import DriftClient
from driftpy.drift_user import DriftUser
from driftpy.math.margin import MarginCategory
from driftpy.oracles.oracle_id import get_oracle_id
from driftpy.types import OraclePriceData
from driftpy.user_map.user_map import UserMap

from shared.types import PriceShockAssetGroup


def get_init_health(user: DriftUser):
    """
    Returns the initial health of the user.
    """
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


def combine_asset_liability(asset_liability_tuple):
    return asset_liability_tuple[0] - asset_liability_tuple[1]


def get_collateral_composition(user: DriftUser, margin_category, num_markets: int):
    spot_market_net_values = {
        market_index: combine_asset_liability(
            user.get_spot_market_asset_and_liability_value(
                market_index, margin_category
            )
        )
        / QUOTE_PRECISION
        for market_index in range(num_markets)
    }
    return spot_market_net_values


def get_perp_liab_composition(user: DriftUser, margin_category, num_markets: int):
    perp_net_liabilities = {
        market_index: user.get_perp_market_liability(
            market_index, margin_category, signed=True
        )
        / QUOTE_PRECISION
        for market_index in range(num_markets)
    }
    return perp_net_liabilities


@functools.cache
def get_stable_metrics(x: DriftUser):
    unrealized_pnl = x.get_unrealized_pnl(True)
    net_spot_market_value = x.get_net_spot_market_value(None)
    net_usd_value = (net_spot_market_value + unrealized_pnl) / QUOTE_PRECISION
    return {
        "user_key": x.user_public_key,
        "is_high_leverage": x.is_high_leverage_mode(),
        "leverage": x.get_leverage() / MARGIN_PRECISION,
        "upnl": unrealized_pnl / QUOTE_PRECISION,
        "net_usd_value": net_usd_value,
    }


def get_user_metrics_for_asset_liability(
    x: DriftUser,
    margin_category: MarginCategory | None,
):
    """
    Returns a dictionary of the user's health, leverage, and other metrics.
    """

    asset_value, liability_value = x.get_spot_market_asset_and_liability_value(
        None, margin_category
    )
    asset_value = asset_value / QUOTE_PRECISION
    liability_value = liability_value / QUOTE_PRECISION

    perp_liability = (
        x.get_total_perp_position_liability(margin_category) / QUOTE_PRECISION
    )

    metrics_stable = get_stable_metrics(x)
    metrics_unstable = {
        "perp_liability": perp_liability,
        "spot_asset": asset_value,
        "spot_liability": liability_value,
    }

    metrics = {
        **metrics_stable,
        **metrics_unstable,
    }

    if margin_category == MarginCategory.INITIAL:
        metrics["health"] = get_init_health(x)
    else:
        metrics["health"] = x.get_health()

    NUMBER_OF_SPOT = len(mainnet_spot_market_configs)
    NUMBER_OF_PERP = len(mainnet_perp_market_configs)
    metrics["net_v"] = get_collateral_composition(x, margin_category, NUMBER_OF_SPOT)
    metrics["net_p"] = get_perp_liab_composition(x, margin_category, NUMBER_OF_PERP)

    return metrics


def get_user_metrics_for_price_shock(
    x: DriftUser,
    margin_category: MarginCategory | None,
    oracle_cache: Optional[DriftClientCache] = None,
):
    """
    Returns a dictionary of the user's health, leverage, and other metrics.
    """
    if oracle_cache is not None:
        x.drift_client.account_subscriber.cache = oracle_cache

    asset_value = x.get_spot_market_asset_value(None, margin_category) / QUOTE_PRECISION
    liability_value = (
        x.get_spot_market_liability_value(None, margin_category) / QUOTE_PRECISION
    )
    perp_liability = (
        x.get_total_perp_position_liability(margin_category) / QUOTE_PRECISION
    )
    unrealized_pnl = x.get_unrealized_pnl(True)
    net_spot_market_value = x.get_net_spot_market_value(None)
    net_usd_value = (net_spot_market_value + unrealized_pnl) / QUOTE_PRECISION

    metrics_stable = {
        "user_key": x.user_public_key,
        "leverage": x.get_leverage() / MARGIN_PRECISION,
        "upnl": unrealized_pnl / QUOTE_PRECISION,
        "net_usd_value": net_usd_value,
    }
    metrics_unstable = {
        "perp_liability": perp_liability,
        "spot_asset": asset_value,
        "spot_liability": liability_value,
    }

    metrics = {
        **metrics_stable,
        **metrics_unstable,
    }

    if margin_category == MarginCategory.INITIAL:
        metrics["health"] = get_init_health(x)
    else:
        metrics["health"] = x.get_health()

    return metrics


def get_skipped_oracles(asset_group: PriceShockAssetGroup) -> List[str]:
    """
    Determine which asset group of oracles to skip

    NB: Generate a list of oracle addresses to *skip* i.e. not to distort
    """
    all_configs = mainnet_spot_market_configs + mainnet_perp_market_configs
    if asset_group == PriceShockAssetGroup.IGNORE_STABLES:
        usd_markets = [
            get_oracle_id(x.oracle, x.oracle_source)
            for x in all_configs
            if "USD" in x.symbol
        ]
        return usd_markets
    if asset_group == PriceShockAssetGroup.JLP_ONLY:
        non_jlp_markets = [
            get_oracle_id(x.oracle, x.oracle_source)
            for x in all_configs
            if "JLP" not in x.symbol
        ]
        return non_jlp_markets
    else:
        return []


def calculate_leverages_for_asset_liability(
    user_values: list[DriftUser], maintenance_category: MarginCategory | None
):
    """
    Calculate the leverages for all users at a given maintenance category
    """
    return [
        get_user_metrics_for_asset_liability(x, maintenance_category)
        for x in user_values
    ]


def calculate_leverages_for_price_shock(
    user_values: list[DriftUser],
    maintenance_category: MarginCategory | None,
    oracle_cache: Optional[DriftClientCache] = None,
):
    """
    Calculate the leverages for all users at a given maintenance category
    """
    return [
        get_user_metrics_for_price_shock(x, maintenance_category, oracle_cache)
        for x in user_values
    ]


def get_user_metrics_none(user_map: UserMap):
    user_keys = list(user_map.user_map.keys())
    user_values = list(user_map.values())
    metrics_none = calculate_leverages_for_asset_liability(user_values, None)
    return {
        "metrics_none": metrics_none,
        "user_keys": user_keys,
    }


def get_user_metrics_initial(user_map: UserMap):
    user_keys = list(user_map.user_map.keys())
    user_values = list(user_map.values())
    metrics_initial = calculate_leverages_for_asset_liability(
        user_values, MarginCategory.INITIAL
    )
    return {
        "metrics_initial": metrics_initial,
        "user_keys": user_keys,
    }


def get_user_metrics_maintenance(user_map: UserMap):
    user_keys = list(user_map.user_map.keys())
    user_values = list(user_map.values())
    metrics_maintenance = calculate_leverages_for_asset_liability(
        user_values, MarginCategory.MAINTENANCE
    )
    return {
        "metrics_maintenance": metrics_maintenance,
        "user_keys": user_keys,
    }


def get_user_leverages_for_price_shock(
    slot: int,
    drift_client: DriftClient,
    user_map: UserMap,
    oracle_distortion: float = 0.1,
    asset_group: PriceShockAssetGroup = PriceShockAssetGroup.IGNORE_STABLES,
    scenarios: int = 5,
    pool_id: int = None,
):
    user_keys = list(user_map.user_map.keys())
    user_vals = list(user_map.values())
    
    # Filter users by pool_id if specified
    if pool_id is not None:
        filtered_user_vals = []
        filtered_user_keys = []
        pool_id_counts = {}  # Track distribution of pool_ids
        
        for user_key, user in zip(user_keys, user_vals):
            try:
                user_account = user.get_user_account()
                actual_pool_id = user_account.pool_id
                
                # Track pool_id distribution for debugging
                pool_id_counts[actual_pool_id] = pool_id_counts.get(actual_pool_id, 0) + 1
                
                if actual_pool_id == pool_id:
                    filtered_user_vals.append(user)
                    filtered_user_keys.append(user_key)
            except Exception as e:
                print(f"Error checking pool_id for user {user_key}: {e}")
                continue
        
        print(f"Pool ID distribution: {dict(sorted(pool_id_counts.items()))}")
        print(f"Requested pool_id: {pool_id}")
        print(f"Filtered to {len(filtered_user_vals)} users with pool_id {pool_id} (out of {len(user_vals)} total users)")
        
        user_vals = filtered_user_vals
        user_keys = filtered_user_keys
    
    all_configs = mainnet_spot_market_configs + mainnet_perp_market_configs

    print(f"User keys : {len(user_keys)}")
    new_oracles_dat_up = []
    new_oracles_dat_down = []
    skipped_oracles = get_skipped_oracles(asset_group)
    print(
        f"Skipping {len(skipped_oracles)} oracles (from a total of {len(all_configs)}) for asset group {asset_group}"
    )

    for i in range(scenarios):
        new_oracles_dat_up.append({})
        new_oracles_dat_down.append({})

    distorted_oracles = []
    cache_up = copy.deepcopy(drift_client.account_subscriber.cache)
    cache_down = copy.deepcopy(drift_client.account_subscriber.cache)

    for key, val in drift_client.account_subscriber.cache["oracle_price_data"].items():
        for i in range(scenarios):
            new_oracles_dat_up[i][key] = copy.deepcopy(val)
            new_oracles_dat_down[i][key] = copy.deepcopy(val)
        if key in skipped_oracles:
            continue

        distorted_oracles.append(key)
        for i in range(scenarios):
            oracle_distort_up = max(1 + oracle_distortion * (i + 1), 1)
            oracle_distort_down = max(1 - oracle_distortion * (i + 1), 0)

            if isinstance(new_oracles_dat_up[i][key], OraclePriceData):
                new_oracles_dat_up[i][key].price *= oracle_distort_up
                new_oracles_dat_down[i][key].price *= oracle_distort_down
            else:
                new_oracles_dat_up[i][key].data.price *= oracle_distort_up
                new_oracles_dat_down[i][key].data.price *= oracle_distort_down

    leverages_none = calculate_leverages_for_price_shock(user_vals, None)
    leverages_up = []
    leverages_down = []

    for i in range(scenarios):
        cache_up["oracle_price_data"] = new_oracles_dat_up[i]
        cache_down["oracle_price_data"] = new_oracles_dat_down[i]
        leverages_up_i = calculate_leverages_for_price_shock(
            user_vals,
            None,
            cache_up,
        )
        leverages_down_i = calculate_leverages_for_price_shock(
            user_vals,
            None,
            cache_down,
        )
        leverages_up.append(leverages_up_i)
        leverages_down.append(leverages_down_i)

    return {
        "slot": slot,
        "leverages_none": leverages_none,
        "leverages_up": tuple(leverages_up),
        "leverages_down": tuple(leverages_down),
        "user_keys": user_keys,
        "distorted_oracles": distorted_oracles,
    }
