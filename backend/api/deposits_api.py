from typing import Optional

from driftpy.constants import PRICE_PRECISION, SPOT_BALANCE_PRECISION
from driftpy.pickle.vat import Vat
from driftpy.types import is_variant
from driftpy.vaults import get_vaults_program
from fastapi import APIRouter

from backend.state import BackendRequest

router = APIRouter()


@router.get("/deposits")
async def get_deposits(request: BackendRequest, market_index: Optional[int] = None):
    """
    Get all deposits grouped by authority, optionally filtered by market index.

    Args:
        market_index: Optional filter for specific market. If None, returns all markets.

    Returns:
        dict: A dictionary containing deposits with total value and balance info
    """
    vat: Vat = request.state.backend_state.vat
    last_oracle_slot = getattr(request.state.backend_state, "last_oracle_slot", 0)
    deposits = []
    vaults_program = await get_vaults_program(request.state.backend_state.connection)
    vaults = await vaults_program.account["Vault"].all()
    vault_pubkeys = [str(vault.account.pubkey) for vault in vaults]

    for user in vat.users.values():
        for position in user.get_user_account().spot_positions:
            if (
                position.scaled_balance > 0
                and not is_variant(position.balance_type, "Borrow")
                and (market_index is None or position.market_index == market_index)
            ):
                market_price = vat.spot_oracles.get(position.market_index)
                if market_price is not None:
                    market_price_ui = market_price.price / PRICE_PRECISION
                    balance = position.scaled_balance / SPOT_BALANCE_PRECISION
                    value = balance * market_price_ui

                    deposits.append(
                        {
                            "authority": str(user.get_user_account().authority),
                            "user_account": str(user.user_public_key),
                            "market_index": position.market_index,
                            "balance": balance,
                            "value": value,
                        }
                    )

    deposits.sort(key=lambda x: x["value"], reverse=True)
    return {
        "slot": last_oracle_slot,
        "deposits": deposits,
        "vaults": vault_pubkeys,
        "total_value": sum(d["value"] for d in deposits),
        "total_deposits": len(deposits),
    }
