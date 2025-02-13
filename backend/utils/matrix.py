import pandas as pd
from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.pickle.vat import Vat

from backend.utils.user_metrics import get_user_leverages_for_asset_liability


def calculate_effective_leverage(assets: float, liabilities: float) -> float:
    return liabilities / assets if assets != 0 else 0


def format_metric(
    value: float, should_highlight: bool, mode: int, financial: bool = False
) -> str:
    formatted = f"{value:,.2f}" if financial else f"{value:.2f}"
    return f"{formatted} ✅" if should_highlight and mode > 0 else formatted


async def get_matrix(vat: Vat, mode: int = 0, perp_market_index: int = 0):
    NUMBER_OF_SPOT = len(mainnet_spot_market_configs)

    res = get_user_leverages_for_asset_liability(vat.users)
    leverage_data = {
        0: res["leverages_none"],
        1: res["leverages_none"],
        2: [x for x in res["leverages_initial"] if int(x["health"]) <= 10],
        3: [x for x in res["leverages_maintenance"] if int(x["health"]) <= 10],
    }

    user_keys = (
        [x["user_key"] for x in leverage_data[mode]]
        if mode in [2, 3]
        else res["user_keys"]
    )

    df = pd.DataFrame(leverage_data[mode], index=user_keys)

    new_columns = {}
    for i in range(NUMBER_OF_SPOT):
        prefix = f"spot_{i}"
        column_names = [
            f"{prefix}_all_assets",
            f"{prefix}_all",
            f"{prefix}_all_perp",
            f"{prefix}_all_spot",
            f"{prefix}_perp_{perp_market_index}_long",
            f"{prefix}_perp_{perp_market_index}_short",
        ]
        for col in column_names:
            new_columns[col] = pd.Series(0.0, index=df.index)

    for idx, row in df.iterrows():
        spot_asset = row["spot_asset"]

        for market_id, value in row["net_v"].items():
            if value < 0:
                # print(f"value: {value}, type: {type(value)}")
                pass

            if value == 0:
                continue

            base_name = f"spot_{market_id}"

            if row["spot_asset"] == 0:
                continue

            metrics = {
                f"{base_name}_all_assets": value,
                f"{base_name}_all": value
                / spot_asset
                * (row["perp_liability"] + row["spot_liability"]),
                f"{base_name}_all_perp": value / spot_asset * row["perp_liability"],
                f"{base_name}_all_spot": value / spot_asset * row["spot_liability"],
            }

            net_perp = float(row["net_p"][perp_market_index])
            # print(f"net_perp value: {net_perp}")

            if net_perp > 0:
                metrics[f"{base_name}_perp_{perp_market_index}_long"] = (
                    value / spot_asset * net_perp
                )
            if net_perp < 0:
                metrics[f"{base_name}_perp_{perp_market_index}_short"] = (
                    value / spot_asset * net_perp
                )

            for col, val in metrics.items():
                new_columns[col][idx] = val

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df
