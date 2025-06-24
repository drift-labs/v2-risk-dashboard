from driftpy.constants import BASE_PRECISION, PRICE_PRECISION
from driftpy.pickle.vat import Vat
from fastapi import APIRouter

from backend.state import BackendRequest

router = APIRouter()


@router.get("/liquidation-curves")
def get_liquidation_curve(request: BackendRequest, market_index: int):
    vat: Vat = request.state.backend_state.vat
    last_oracle_slot = getattr(request.state.backend_state, "last_oracle_slot", 0)
    # Debug logging to understand the slot issue
    print(f"[DEBUG] Liquidation API - last_oracle_slot: {last_oracle_slot}, pickle_path: {request.state.backend_state.current_pickle_path}")
    liquidations_long: list[tuple[float, float, str]] = []
    liquidations_short: list[tuple[float, float, str]] = []
    market_price = vat.perp_oracles.get(market_index)
    if market_price is None:
        print("Market price is None")
        return {
            "liquidations_long": [], 
            "liquidations_short": [], 
            "market_price_ui": 0,
            "slot": last_oracle_slot
        }
    market_price_ui = market_price.price / PRICE_PRECISION
    for pubkey, user in vat.users.user_map.items():
        perp_position = user.get_perp_position(market_index)
        if perp_position is not None:
            liquidation_price = user.get_perp_liq_price(market_index)
            if liquidation_price is not None:
                liquidation_price_ui = liquidation_price / PRICE_PRECISION
                position_size = abs(perp_position.base_asset_amount) / BASE_PRECISION
                position_notional = position_size * market_price_ui
                is_zero = round(position_notional) == 0
                is_short = perp_position.base_asset_amount < 0
                is_long = perp_position.base_asset_amount > 0
                if is_zero:
                    continue
                if is_short and liquidation_price_ui > market_price_ui:
                    liquidations_short.append(
                        (liquidation_price_ui, position_notional, str(pubkey))
                    )
                elif is_long and liquidation_price_ui < market_price_ui:
                    liquidations_long.append(
                        (liquidation_price_ui, position_notional, str(pubkey))
                    )
                else:
                    pass

    liquidations_long.sort(key=lambda x: x[0])
    liquidations_short.sort(key=lambda x: x[0])

    return {
        "liquidations_long": liquidations_long,
        "liquidations_short": liquidations_short,
        "market_price_ui": market_price_ui,
        "slot": last_oracle_slot,
    }
