import requests
import pandas as pd
from enum import Enum
import streamlit as st
from driftpy.constants.perp_markets import mainnet_perp_market_configs
from driftpy.constants.spot_markets import mainnet_spot_market_configs

from lib.api import fetch_cached_data
from utils import get_current_slot

# --- START NEW IMPORTS FOR DRIFTPY SPOT MARKET DATA --- #
import asyncio
import os
from anchorpy import Wallet
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from driftpy.drift_client import DriftClient
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig, WebsocketConfig
from driftpy.types import MarketType
# --- END NEW IMPORTS --- #

options = [0, 1, 2, 3]
labels = [
    "none",
    "liq within 50% of oracle",
    "maint. health < 10%",
    "init. health < 10%",
]

class PriceImpactStatus(str, Enum):
    PASS = "✅"  # Price impact is below threshold
    NO_BALANCE = "ℹ️"  # No balance to check
    QUOTE_TOKEN = "�"  # USDC or quote token
    FAIL = "❌"  # Price impact above threshold

def calculate_effective_leverage(assets: float, liabilities: float) -> float:
    return liabilities / assets if assets != 0 else 0

def format_metric(
    value: float, should_highlight: bool, mode: int, financial: bool = False
) -> str:
    formatted = f"{value:,.2f}" if financial else f"{value:.2f}"
    return f"{formatted} ✅" if should_highlight and mode > 0 else formatted

def get_jupiter_quote(input_mint: str, output_mint: str, amount: int) -> float:
    """Get price impact quote from Jupiter DEX API.
    
    Args:
        input_mint: Token being sold
        output_mint: Token being bought
        amount: Amount in base units (lamports)
    
    Returns:
        Price impact as a decimal (e.g., 0.01 = 1%)
    """
    if input_mint == output_mint:
        return 0
        
    base_url = "https://quote-api.jup.ag/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount),
        "slippageBps": 50  # 0.5% slippage tolerance
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "error" in data:
            return 0
            
        price_impact = float(data.get("priceImpactPct", 0))
        return price_impact
    except requests.RequestException:
        return 0

def get_largest_spot_borrow_per_market():
    """Fetch the largest spot borrow for each market from the API.
    
    Returns:
        Dictionary containing:
        - market_indices: List of market indices
        - scaled_balances: List of borrow amounts in token units
        - values: List of USD values of the borrows
        - public_keys: List of borrower public keys
    """
    try:
        response = fetch_cached_data("health/largest_spot_borrow_per_market")
        result = {
            "market_indices": response["Market Index"],
            "scaled_balances": [float(bal.replace(",", "")) for bal in response["Scaled Balance"]],
            "values": [float(val.replace("$", "").replace(",", "")) for val in response["Value"]],
            "public_keys": response["Public Key"]
        }
        return result
    except Exception:
        return {
            "market_indices": [], 
            "scaled_balances": [],
            "values": [],
            "public_keys": []
        }

@st.cache_data
def load_spot_markets_data() -> dict[int, dict[str, float]]:
    """
    Fetch on-chain spot market data (maintenance_asset_weight, decimals) using driftpy
    and cache the results.
    Returns a dictionary structured as:
        {
            market_index: {
                "maintenance_asset_weight": float,
                "decimals": float
            },
            ...
        }
    """
    rpc_url = os.environ.get("RPC_URL", "https://api.mainnet-beta.solana.com")
    loop = asyncio.new_event_loop()
    data = loop.run_until_complete(_async_load_spot_markets_data(rpc_url))
    loop.close()
    return data

async def _async_load_spot_markets_data(rpc_url: str) -> dict[int, dict[str, float]]:
    """
    Asynchronous function to load spot markets from driftpy
    and retrieve maintenance_asset_weight plus decimals.
    """
    # Create ephemeral keypair for reading
    keypair = Keypair()
    wallet = Wallet(keypair)
    connection = AsyncClient(rpc_url)
    
    drift_client = DriftClient(connection, wallet)
    spot_market_map = MarketMap(
        MarketMapConfig(
            drift_client.program,
            MarketType.Spot(),
            WebsocketConfig(resub_timeout_ms=10000),
            connection
        )
    )

    await spot_market_map.pre_dump()

    ret = {}
    for market in spot_market_map.values():
        market_index = market.data.market_index
        maint_raw = market.data.maintenance_asset_weight  # e.g. 9000 => 0.9
        decimals = market.data.decimals or 6
        ret[market_index] = {
            "maintenance_asset_weight": float(maint_raw)/1e4,
            "decimals": float(decimals)
        }

    await connection.close()
    return ret

def get_maintenance_asset_weight(market_index: int) -> float:
    """
    Get the maintenance asset weight for a spot market from the cached dictionary.
    Falls back to 0.9 if not found.
    """
    data = load_spot_markets_data()
    return data.get(market_index, {}).get("maintenance_asset_weight", 0.9)

def get_spot_market_decimals(market_index: int) -> int:
    """
    Get the decimals for the specified spot market from the cached dictionary.
    Defaults to 6 if not found.
    """
    data = load_spot_markets_data()
    return int(data.get(market_index, {}).get("decimals"))

def check_price_impact(market_index: int, scaled_balance: float) -> PriceImpactStatus:
    """Check if liquidating a position would have too much price impact using the new driftpy-based data.
    
    1. Gets the market config for the user-specified index.
    2. Checks special cases (USDC, zero balance).
    3. Calculates the maximum allowed price impact using maintenance_asset_weight.
    4. Gets a quote from Jupiter for the swap.
    5. Compares the price impact to the threshold.
    """
    try:
        # Get market config for the mint address
        market_config = next(
            (m for m in mainnet_spot_market_configs if m.market_index == market_index),
            None
        )
        if not market_config:
            return PriceImpactStatus.PASS  # if unknown, skip

        # Special handling for USDC market (quote currency)
        if market_config.symbol == "USDC":
            return PriceImpactStatus.QUOTE_TOKEN

        # Skip if no balance to check
        if scaled_balance == 0:
            return PriceImpactStatus.NO_BALANCE

        # Get USDC market config for quote currency
        usdc_config = next(
            (m for m in mainnet_spot_market_configs if m.symbol == "USDC"),
            None
        )
        if not usdc_config:
            return PriceImpactStatus.PASS

        # Maximum allowed price impact: (1 - maintenance_asset_weight)
        maint_asset_weight = get_maintenance_asset_weight(market_index)
        threshold = 1 - maint_asset_weight

        # Get decimals from on-chain data
        decimals = get_spot_market_decimals(market_index)

        # Convert token amount to base units
        amount = int(scaled_balance * (10 ** decimals))

        # Get price impact from Jupiter
        price_impact = get_jupiter_quote(
            market_config.mint,
            usdc_config.mint,
            amount
        )

        # Compare price impact to threshold
        if price_impact < threshold:
            return PriceImpactStatus.PASS
        else:
            return PriceImpactStatus.FAIL
    except Exception:
        return PriceImpactStatus.PASS

def generate_summary_data(
    df: pd.DataFrame, mode: int, perp_market_index: int
) -> pd.DataFrame:
    summary_data = {}
    for i in range(len(mainnet_spot_market_configs)):
        prefix = f"spot_{i}"
        assets = df[f"{prefix}_all_assets"].sum()
        liabilities = df[f"{prefix}_all"].sum()

        summary_data[f"spot{i} ({mainnet_spot_market_configs[i].symbol})"] = {
            "all_assets": assets,
            "all_liabilities": format_metric(
                liabilities, 0 < liabilities < 1_000_000, mode, financial=True
            ),
            "effective_leverage": format_metric(
                calculate_effective_leverage(assets, liabilities),
                0 < calculate_effective_leverage(assets, liabilities) < 2,
                mode,
            ),
            "all_spot": df[f"{prefix}_all_spot"].sum(),
            "all_perp": df[f"{prefix}_all_perp"].sum(),
            f"perp_{perp_market_index}_long": df[
                f"{prefix}_perp_{perp_market_index}_long"
            ].sum(),
            f"perp_{perp_market_index}_short": df[
                f"{prefix}_perp_{perp_market_index}_short"
            ].sum(),
        }
    return pd.DataFrame(summary_data).T

def asset_liab_matrix_cached_page():
    if "only_high_leverage_mode_users" not in st.session_state:
        st.session_state.only_high_leverage_mode_users = False
    if "min_leverage" not in st.session_state:
        st.session_state.min_leverage = 0.0

    params = st.query_params
    mode = int(params.get("mode", 0))
    perp_market_index = int(params.get("perp_market_index", 0))

    mode = st.selectbox(
        "Options", options, format_func=lambda x: labels[x], index=options.index(mode)
    )
    st.query_params.update({"mode": str(mode)})

    perp_market_index = st.selectbox(
        "Market index",
        [x.market_index for x in mainnet_perp_market_configs],
        index=[x.market_index for x in mainnet_perp_market_configs].index(
            perp_market_index
        ),
        format_func=lambda x: f"{x} ({mainnet_perp_market_configs[int(x)].symbol})",
    )
    st.query_params.update({"perp_market_index": str(perp_market_index)})

    result = fetch_cached_data(
        "asset-liability/matrix",
        _params={"mode": mode, "perp_market_index": perp_market_index},
        key=f"asset-liability/matrix_{mode}_{perp_market_index}",
    )
    df = pd.DataFrame(result["df"])

    if st.session_state.only_high_leverage_mode_users:
        df = df[df["is_high_leverage"]]

    filtered_df = df[df["leverage"] >= st.session_state.min_leverage].sort_values(
        "leverage", ascending=False
    )

    summary_df = generate_summary_data(filtered_df, mode, perp_market_index)
    slot = result["slot"]
    current_slot = get_current_slot()

    st.info(
        f"This data is for slot {slot}, which is now {int(current_slot) - int(slot)} slots old"
    )
    st.write(f"{df.shape[0]} users")
    st.checkbox(
        "Only show high leverage mode users", key="only_high_leverage_mode_users"
    )
    st.slider(
        "Filter by minimum leverage",
        0.0,
        110.0,
        key="min_leverage",
    )
    st.write(summary_df)

    tabs = st.tabs(["FULL"] + [x.symbol for x in mainnet_spot_market_configs])

    with tabs[0]:
        if st.session_state.only_high_leverage_mode_users:
            st.write(
                f"There are **{len(filtered_df)}** users with high leverage mode and {st.session_state.min_leverage}x leverage or more"
            )
        else:
            st.write(
                f"There are **{len(filtered_df)}** users with this **{st.session_state.min_leverage}x** leverage or more"
            )
        st.write(f"Total USD value: **{filtered_df['net_usd_value'].sum():,.2f}**")
        st.write(f"Total collateral: **{filtered_df['spot_asset'].sum():,.2f}**")
        st.write(f"Total liabilities: **{filtered_df['spot_liability'].sum():,.2f}**")
        st.dataframe(filtered_df, hide_index=True)

    for idx, tab in enumerate(tabs[1:]):
        important_cols = [x for x in filtered_df.columns if "spot_" + str(idx) in x]
        filtered_df["Link"] = filtered_df["user_key"].apply(
            lambda x: f"https://app.drift.trade/overview?userAccount={x}"
        )
        toshow = filtered_df[
            ["user_key", "Link", "spot_asset", "net_usd_value"] + important_cols
        ]
        toshow = toshow[toshow[important_cols].abs().sum(axis=1) != 0].sort_values(
            by="spot_" + str(idx) + "_all", ascending=False
        )
        tab.write(
            f"{len(toshow)} users with this asset to cover liabilities (with {st.session_state.min_leverage}x leverage or more)"
        )
        tab.dataframe(
            toshow,
            hide_index=True,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="View"),
            },
        )