import os
import asyncio
import base58
import streamlit as st
import inspect

from anchorpy import Wallet
from dotenv import load_dotenv
from driftpy.drift_client import DriftClient
from driftpy.market_map.market_map import MarketMap
from driftpy.market_map.market_map_config import MarketMapConfig, WebsocketConfig
from driftpy.types import MarketType, PerpMarketAccount, SpotMarketAccount
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair #type: ignore

from driftpy.constants.spot_markets import mainnet_spot_market_configs
from driftpy.constants.perp_markets import mainnet_perp_market_configs

# ---------------------------------------------------------
# Utility Functions (adapted from the old script)
# ---------------------------------------------------------

def format_market_name(name_bytes) -> str:
    """Convert market name bytes to string."""
    return bytes(name_bytes).decode('utf-8').strip()

def format_pubkey(pubkey: str) -> str:
    """Truncate pubkey for display."""
    return f"{str(pubkey)[:10]}...{str(pubkey)[-10:]}"

def format_number(number: int, decimals=6) -> str:
    """Format large numbers for better readability."""
    return f"{number / (10 ** decimals):,.6f}"

def get_all_attributes_from_class(cls):
    """
    Dynamically extract all attributes from a class (PerpMarketAccount or SpotMarketAccount)
    including nested attributes.
    Returns a list of attribute paths.
    """
    attrs = []
    
    # Get all class attributes (fields) from the dataclass
    fields = getattr(cls, '__dataclass_fields__', {})
    
    # Add direct attributes
    for field_name in fields.keys():
        # Skip the padding field
        if field_name == 'padding':
            continue
        attrs.append(field_name)
    
    # Add nested attributes for complex fields
    for field_name in fields.keys():
        # Check if this is a nested dataclass
        if field_name in ['amm', 'historical_oracle_data', 'historical_index_data', 'insurance_claim', 'insurance_fund']:
            nested_cls = fields[field_name].type
            if hasattr(nested_cls, '__dataclass_fields__'):
                nested_fields = getattr(nested_cls, '__dataclass_fields__', {})
                for nested_field in nested_fields.keys():
                    if nested_field != 'padding':
                        attrs.append(f"{field_name}.{nested_field}")
    
    return attrs

def get_perp_market_attributes():
    """Dynamically retrieve all attributes from PerpMarketAccount."""
    return get_all_attributes_from_class(PerpMarketAccount)

def get_spot_market_attributes():
    """Dynamically retrieve all attributes from SpotMarketAccount."""
    return get_all_attributes_from_class(SpotMarketAccount)

def extract_nested_attribute(market_data, attr_path: str):
    """
    Given a MarketAccount data object and a dotted attribute path
    (e.g. "amm.base_spread"), traverse and return the final attribute value if exists.
    """
    parts = attr_path.split('.')
    current = market_data
    for part in parts:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current

def display_attribute(market_data, attr_path: str):
    """
    Return a string that nicely formats the nested attribute for display.
    """
    val = extract_nested_attribute(market_data, attr_path)
    if val is None:
        return f"{attr_path}: N/A"

    # Special handling for name fields
    if "name" in attr_path.lower() and isinstance(val, list):
        # Attempt to interpret as bytes
        val = format_market_name(val)
        return f"{attr_path}: {val}"

    # If it's a Pubkey or similar
    if hasattr(val, '__class__') and val.__class__.__name__ == 'Pubkey':
        return f"{attr_path}: {format_pubkey(val)}"

    # If it's an int that might represent a 'price' or 'reserve'
    if isinstance(val, int):
        # Heuristic: if 'reserve', 'ratio', or 'weight' in attr_path -> fewer decimals
        if any(x in attr_path.lower() for x in ["reserve", "ratio", "weight"]):
            return f"{attr_path}: {format_number(val, 4)}"
        # or if 'price', 'amount', 'balance'
        if any(x in attr_path.lower() for x in ["price", "amount", "balance"]):
            return f"{attr_path}: {format_number(val, 6)}"

    # Default formatting
    return f"{attr_path}: {str(val)}"


# ---------------------------------------------------------
# Async/Cache Loading of Market Maps
# ---------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_market_maps():
    """
    Use st.cache_resource for unserializable objects like MarketMap.
    We'll return the MarketMap objects so they aren't reinitialized each time.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_market_maps())
    finally:
        loop.close()

async def _fetch_market_maps():
    load_dotenv()
    rpc_url = os.environ.get("RPC_URL", "https://api.mainnet-beta.solana.com")

    # Generate ephemeral keypair for demonstration
    kp = Keypair()
    wallet = Wallet(kp)
    connection = AsyncClient(rpc_url)

    drift_client = DriftClient(connection, wallet)

    # Create and fetch for perp
    perp_market_map = MarketMap(
        MarketMapConfig(
            drift_client.program,
            MarketType.Perp(),
            WebsocketConfig(resub_timeout_ms=10000),
            connection,
        )
    )
    await perp_market_map.pre_dump()

    # Create and fetch for spot
    spot_market_map = MarketMap(
        MarketMapConfig(
            drift_client.program,
            MarketType.Spot(),
            WebsocketConfig(resub_timeout_ms=10000),
            connection,
        )
    )
    await spot_market_map.pre_dump()

    await connection.close()
    return (spot_market_map, perp_market_map)


# ---------------------------------------------------------
# Main Page
# ---------------------------------------------------------

def market_inspector_page():
    st.title("Market Inspector")

    # 1) Load the maps
    spot_market_map, perp_market_map = load_market_maps()

    # 2) Let user choose Spot or Perp
    market_type_choice = st.radio("Select Market Type:", ["Spot", "Perp"], horizontal=True)

    if market_type_choice == "Spot":
        markets = sorted(spot_market_map.values(), key=lambda m: m.data.market_index)
        display_name = lambda m: f"Spot {m.data.market_index} - {format_market_name(m.data.name)}"
        available_attrs = get_spot_market_attributes()
    else:
        markets = sorted(perp_market_map.values(), key=lambda m: m.data.market_index)
        display_name = lambda m: f"Perp {m.data.market_index} - {format_market_name(m.data.name)}"
        available_attrs = get_perp_market_attributes()

    # 3) Let user select which market
    selected_market = st.selectbox(
        "Select Market:",
        markets,
        format_func=display_name,
    )

    # 4) Let user pick which attributes to show (multi-select)
    st.write("Select which attributes you would like to see:")
    selected_attrs = st.multiselect(
        "Attributes",
        available_attrs,
        default=[],
    )

    # 5) Display results
    if not selected_attrs:
        st.info("Please select at least one attribute to display.")
        return

    st.write(f"**Market Index:** {selected_market.data.market_index}")
    st.write(f"**Name:** {format_market_name(selected_market.data.name)}")

    st.markdown("---")
    st.subheader("Selected Attributes:")
    for attr in selected_attrs:
        line = display_attribute(selected_market.data, attr)
        st.write(line)