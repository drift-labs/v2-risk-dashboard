from driftpy.pickle.vat import Vat
from fastapi import APIRouter

from backend.state import BackendRequest

router = APIRouter()


@router.get("/top_pnl")
def get_top_pnl(request: BackendRequest, limit: int = 1000):
    vat: Vat = request.state.backend_state.vat
    last_oracle_slot = getattr(request.state.backend_state, "last_oracle_slot", 0)

    pnl_data = []
    for user in vat.users.values():
        try:
            realized_pnl = user.get_user_account().settled_perp_pnl / 1e6
            unrealized_pnl = user.get_unrealized_pnl(True) / 1e6
            total_pnl = realized_pnl + unrealized_pnl

            pnl_data.append(
                {
                    "authority": str(user.get_user_account().authority),
                    "user_key": str(user.user_public_key),
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "total_pnl": total_pnl,
                    # "collateral": user.get_total_collateral() / 1e6,
                }
            )
        except Exception as e:
            print(f"Error calculating PnL for {user.user_public_key}: {e}")
            continue

    pnl_data.sort(key=lambda x: x["total_pnl"], reverse=True)
    return {"pnl": pnl_data[:limit], "slot": last_oracle_slot}
