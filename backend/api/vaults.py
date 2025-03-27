import os

from driftpy.vaults import VaultClient
from fastapi import APIRouter
from solana.rpc.async_api import AsyncClient

from backend.state import BackendRequest

router = APIRouter()


async def create_vault_client():
    connection = AsyncClient(os.getenv("RPC_URL"))
    return await VaultClient(connection).initialize()


@router.get("/data")
async def get_vault_data(request: BackendRequest):
    """Fetch all vault data including analytics and depositors in one call"""

    client = await create_vault_client()
    analytics = await client.calculate_analytics()

    vaults = []
    print(analytics["vaults"][0])

    for vault in analytics["vaults"]:
        vaults.append(
            {
                "pubkey": vault["pubkey"],
                "name": vault["name"],
                "total_shares": vault["total_shares"],
                "true_net_deposits": vault["true_net_deposits"],
                "depositor_count": vault["depositor_count"],
            }
        )

    serializable_analytics = {
        "vaults": vaults,
        "total_vaults": analytics["total_vaults"],
        "total_deposits": analytics["total_deposits"],
    }

    vault_depositors = {}
    for vault_info in serializable_analytics["vaults"]:
        raw_depositors = await client.get_vault_depositors_with_stats(
            vault_info["pubkey"]
        )

        serializable_depositors = []
        for dep in raw_depositors:
            serializable_depositors.append(
                {
                    "pubkey": str(dep["depositor"].pubkey),
                    "shares": float(dep["shares"]),
                    "share_percentage": float(dep["share_percentage"]),
                }
            )

        vault_depositors[vault_info["pubkey"]] = serializable_depositors

    return {
        "data": {"analytics": serializable_analytics, "depositors": vault_depositors}
    }
