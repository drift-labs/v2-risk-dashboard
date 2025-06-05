import os
import asyncio
import base58
from src.utils import serialize_perp_market, serialize_spot_market
import streamlit as st
import inspect
import pandas as pd

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

def get_debug_info(attr_path: str, val) -> list[str]:
    """Get detailed debug information for any attribute value."""
    debug_info = [
        f"{attr_path} (debug):",
        f"  • raw value: {val}",
        f"  • type: {type(val)}",
        f"  • dir: {dir(val)}",
        f"  • repr: {repr(val)}",
        f"  • str: {str(val)}",
        f"  • has __dict__: {hasattr(val, '__dict__')}",
        f"  • has kind: {hasattr(val, 'kind')}"
    ]

    if hasattr(val, '__dict__'):
        debug_info.append(f"  • __dict__: {val.__dict__}")
    if hasattr(val, 'kind'):
        debug_info.append(f"  • kind value: {val.kind}")
    if hasattr(val, '__class__'):
        debug_info.append(f"  • class name: {val.__class__.__name__}")
        debug_info.append(f"  • class module: {val.__class__.__module__}")

    return debug_info

def is_sumtype(val):
    """Check if a value is a sumtype instance (like asset_tier, oracle_source, etc.)"""
    return (hasattr(val, '__class__') and
            hasattr(val.__class__, '__module__') and
            'sumtypes' in val.__class__.__module__)

def get_sumtype_variant_name(val):
    """Extract the variant name from a sumtype value"""
    if is_sumtype(val):
        return val.__class__.__name__
    return None

def display_attribute(market_data, attr_path: str, debug_mode: bool = False):
    """
    Return a string that nicely formats the nested attribute for display.
    """
    val = extract_nested_attribute(market_data, attr_path)
    if val is None:
        return f"{attr_path}: N/A"

    # If in debug mode, return detailed debug information
    if debug_mode:
        return "\n".join(get_debug_info(attr_path, val))

    # Special handling for name fields
    if "name" in attr_path.lower() and isinstance(val, list):
        # Attempt to interpret as bytes
        val = format_market_name(val)
        return f"{attr_path}: {val}"

    # Handle sumtypes generically
    if is_sumtype(val):
        variant_name = get_sumtype_variant_name(val)
        return f"{attr_path}: {variant_name}"

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

    # Handle complex objects (dataclasses, etc.)
    if (hasattr(val, '__class__')
        and not isinstance(val, (str, int, float, bool, list, dict))
        and hasattr(val, '__dict__')):
        # This is likely a complex object with attributes
        return format_complex_object(attr_path, val)

    # Default formatting
    return f"{attr_path}: {str(val)}"

def format_complex_object(attr_name, obj):
    """
    Format a complex object (like a dataclass) by extracting and displaying
    its attributes in an indented, readable format.
    """
    result = [f"{attr_name}:"]

    # Handle sumtypes generically in complex objects
    if is_sumtype(obj):
        variant_name = get_sumtype_variant_name(obj)
        result = [f"{attr_name}: {variant_name}"]
        return "\n".join(result)

    # Special handling for enum objects
    if hasattr(obj, "kind"):
        enum_kind = obj.kind
        result.append(f"  • kind: {enum_kind}")

        # Add any additional attributes specific to this variant
        if hasattr(obj, enum_kind) and getattr(obj, enum_kind) is not None:
            variant_data = getattr(obj, enum_kind)
            result.append(f"  • {enum_kind}: {variant_data}")

        # Add any other attributes
        for attr_name in dir(obj):
            if (not attr_name.startswith("_") and
                attr_name != "kind" and
                attr_name != enum_kind and
                not callable(getattr(obj, attr_name))):
                attr_val = getattr(obj, attr_name)
                # Format the value appropriately
                if isinstance(attr_val, (int, float)) and any(x in attr_name.lower() for x in ["price", "amount", "balance"]):
                    formatted_value = format_number(attr_val, 6)
                elif hasattr(attr_val, '__class__') and attr_val.__class__.__name__ == 'Pubkey':
                    formatted_value = format_pubkey(attr_val)
                else:
                    formatted_value = str(attr_val)
                result.append(f"  • {attr_name}: {formatted_value}")

        return "\n".join(result)

    # Get all attributes of the object
    attributes = {}

    # Try to get dataclass fields first
    if hasattr(obj, '__dataclass_fields__'):
        attributes = {field: getattr(obj, field) for field in obj.__dataclass_fields__}
    # Fallback to __dict__ for regular objects
    elif hasattr(obj, '__dict__'):
        attributes = obj.__dict__

    # Format each attribute
    for name, value in sorted(attributes.items()):
        # Skip internal attributes
        if name.startswith('_') or name == 'padding':
            continue

        # Format value based on type
        if isinstance(value, (int, float)) and any(x in name.lower() for x in ["price", "amount", "balance"]):
            formatted_value = format_number(value, 6)
        elif hasattr(value, '__class__') and value.__class__.__name__ == 'Pubkey':
            formatted_value = format_pubkey(value)
        elif (hasattr(value, '__class__')
              and not isinstance(value, (str, int, float, bool, list, dict))
              and hasattr(value, '__dict__')):
            # This is a nested complex object, format it recursively
            nested_format = format_complex_object(name, value)
            # Indent the nested format one level further
            formatted_value = "\n    " + nested_format.replace("\n", "\n    ")
            result.append(formatted_value)
            continue
        else:
            formatted_value = str(value)

        # Add indented line
        result.append(f"  • {name}: {formatted_value}")

    return "\n".join(result)


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

    # Add debug mode toggle in sidebar
    with st.sidebar:
        st.write("---")
        st.write("Debug Options")
        debug_mode = st.toggle("Enable Debug Mode", value=False,
                             help="Show detailed debug information for all attributes")

    # 1) Load the maps
    spot_market_map, perp_market_map = load_market_maps()

    # 2) Let user choose Spot or Perp
    # Initialize previous market type in session state if not exists
    if "previous_market_type" not in st.session_state:
        st.session_state.previous_market_type = "Spot"  # Default value

    market_type_choice = st.radio("Select Market Type:", ["Spot", "Perp"], horizontal=True)

    # Clear selections when switching market types
    if market_type_choice != st.session_state.previous_market_type:
        if "selected_attrs" in st.session_state:
            st.session_state.selected_attrs = []
        st.session_state.previous_market_type = market_type_choice

    if market_type_choice == "Spot":
        markets = sorted(spot_market_map.values(), key=lambda m: m.data.market_index)
        display_name = lambda m: f"Spot {m.data.market_index} - {format_market_name(m.data.name)}"
        available_attrs = get_spot_market_attributes()
    else:
        markets = sorted(perp_market_map.values(), key=lambda m: m.data.market_index)
        display_name = lambda m: f"Perp {m.data.market_index} - {format_market_name(m.data.name)}"
        available_attrs = get_perp_market_attributes()

    # 3) Let user select which market
    market_indices = [m.data.market_index for m in markets]

    if not market_indices:
        st.warning(f"No {market_type_choice} markets available to display.")
        # Ensure selected_attrs is cleared if no markets are available for the chosen type
        if "selected_attrs" in st.session_state: # Reset selected attributes if market type changes or no markets
            st.session_state.selected_attrs = []
        return # Stop further processing for this render if no markets

    market_lookup = {m.data.market_index: m for m in markets}

    # Use a unique key for the selectbox based on market_type_choice
    # to ensure it resets if the options change significantly (e.g., switching market types)
    selectbox_key = f"market_selectbox_{market_type_choice}"

    selected_market_index = st.selectbox(
        "Select Market:",
        market_indices,
        format_func=lambda index: display_name(market_lookup[index]),
        key=selectbox_key
    )

    selected_market = market_lookup.get(selected_market_index) # Reassign selected_market

    if not selected_market:
        # This case should ideally not be hit if market_indices is not empty and selectbox works as expected
        st.error("Failed to retrieve selected market data. Please ensure a market is selected or try refreshing.")
        return

    # 4) Let user pick which attributes to show (multi-select)
    st.write("Select which attributes you would like to see:")

    # Initialize session state for selected attributes if not exists
    if "selected_attrs" not in st.session_state:
        st.session_state.selected_attrs = []

    # Define callback functions for multiselect changes
    def on_attribute_selection_change():
        # No additional processing needed as the multiselect directly updates session_state
        pass

    # Add Select All and Clear Selection buttons in columns
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All Attributes"):
            st.session_state.selected_attrs = available_attrs
    with col2:
        if st.button("Clear Selection"):
            st.session_state.selected_attrs = []

    # Use the session state directly with the key parameter instead of default + updating afterward
    selected_attrs = st.multiselect(
        "Attributes",
        options=available_attrs,
        key="selected_attrs",
        on_change=on_attribute_selection_change,
    )

    # 5) Display results
    if not selected_attrs:
        st.info("Please select at least one attribute to display.")
        with st.expander("All markets (Serialized Data)"): # Clarified title
            st.write("Perp markets:")
            perp_markets_list = sorted(perp_market_map.values(), key=lambda m: m.data.market_index)
            if perp_markets_list:
                try:
                    perp_data_for_df = [serialize_perp_market(m.data) for m in perp_markets_list]
                    if perp_data_for_df: # Ensure list is not empty before concat
                        df_perp = pd.concat(perp_data_for_df, axis=0).reset_index(drop=True)
                        if 'market_index' in df_perp.columns:
                            # Ensure market_index is the first column for better readability
                            df_perp = df_perp[['market_index'] + [col for col in df_perp.columns if col != 'market_index']]
                        st.dataframe(df_perp)
                    else:
                        st.write("No data to display for perp markets.")                        
                except Exception as e:
                    st.error(f"Error displaying perp markets table: {e}")
                    st.caption("Raw data might contain non-serializable fields or other issues.")
            else:
                st.write("No perp markets found.")

            st.write("Spot markets:")
            spot_markets_list = sorted(spot_market_map.values(), key=lambda m: m.data.market_index)
            if spot_markets_list:
                try:
                    spot_data_for_df = [serialize_spot_market(m.data) for m in spot_markets_list]
                    if spot_data_for_df: # Ensure list is not empty before concat
                        df_spot = pd.concat(spot_data_for_df, axis=0).reset_index(drop=True)
                        if 'market_index' in df_spot.columns:
                            # Ensure market_index is the first column for better readability
                            df_spot = df_spot[['market_index'] + [col for col in df_spot.columns if col != 'market_index']]
                        st.dataframe(df_spot)
                    else:
                        st.write("No data to display for spot markets.")
                except Exception as e:
                    st.error(f"Error displaying spot markets table: {e}")
                    st.caption("Raw data might contain non-serializable fields or other issues.")
            else:
                st.write("No spot markets found.")
        return

    st.write(f"**Market Index:** {selected_market.data.market_index}")
    st.write(f"**Name:** {format_market_name(selected_market.data.name)}")

    st.markdown("---")
    st.subheader("Selected Attributes:")

    # First, organize attributes to handle parent/child relationships
    parent_attrs = set()
    child_attrs = set()

    # Identify parent and child attributes
    for attr in selected_attrs:
        if '.' in attr:
            parent = attr.split('.')[0]
            if parent in selected_attrs:
                child_attrs.add(attr)
            else:
                parent_attrs.add(attr)
        else:
            parent_attrs.add(attr)

    # Display all attributes in code blocks with consistent formatting
    # Only process parent attributes and child attributes whose parents aren't selected
    for attr in sorted(parent_attrs):
        val = extract_nested_attribute(selected_market.data, attr)

        if debug_mode:
            # In debug mode, show debug info for all attributes
            debug_info = get_debug_info(attr, val)
            st.markdown(f"```\n" + "\n".join(debug_info) + "\n```")
            continue

        # Check if this is a complex object that needs expanded display
        is_complex = (hasattr(val, '__class__')
                    and not isinstance(val, (str, int, float, bool, list, dict))
                    and hasattr(val, '__dict__')
                    and not is_sumtype(val))  # Don't treat sumtypes as complex objects

        if is_complex:
            # Format as a complex object with sub-attributes
            formatted_output = format_complex_object(attr, val)
            st.markdown(f"```\n{formatted_output}\n```")
        else:
            # Format simple attributes in the same style as complex ones for consistency
            formatted_line = display_attribute(selected_market.data, attr)
            # Wrap in code block
            st.markdown(f"```\n{formatted_line}\n```")

    with st.expander("All markets (Serialized Data)"): # Clarified title and reused logic
        st.write("Perp markets:")
        # Use a different variable name to avoid conflicts if any part of the script is re-run in a weird way
        perp_markets_list_bottom = sorted(perp_market_map.values(), key=lambda m: m.data.market_index)
        if perp_markets_list_bottom:
            try:
                perp_data_for_df_bottom = [serialize_perp_market(m.data) for m in perp_markets_list_bottom]
                if perp_data_for_df_bottom: # Ensure list is not empty before concat
                    df_perp_bottom = pd.concat(perp_data_for_df_bottom, axis=0).reset_index(drop=True)
                    if 'market_index' in df_perp_bottom.columns:
                        # Ensure market_index is the first column
                        df_perp_bottom = df_perp_bottom[['market_index'] + [col for col in df_perp_bottom.columns if col != 'market_index']]
                    st.dataframe(df_perp_bottom)
                else:
                    st.write("No data to display for perp markets.")
            except Exception as e:
                st.error(f"Error displaying perp markets table (bottom): {e}")
                st.caption("Raw data might contain non-serializable fields or other issues.")
        else:
            st.write("No perp markets found.")

        st.write("Spot markets:")
        spot_markets_list_bottom = sorted(spot_market_map.values(), key=lambda m: m.data.market_index)
        if spot_markets_list_bottom:
            try:
                spot_data_for_df_bottom = [serialize_spot_market(m.data) for m in spot_markets_list_bottom]
                if spot_data_for_df_bottom: # Ensure list is not empty before concat
                    df_spot_bottom = pd.concat(spot_data_for_df_bottom, axis=0).reset_index(drop=True)
                    if 'market_index' in df_spot_bottom.columns:
                        # Ensure market_index is the first column
                        df_spot_bottom = df_spot_bottom[['market_index'] + [col for col in df_spot_bottom.columns if col != 'market_index']]
                    st.dataframe(df_spot_bottom)
                else:
                    st.write("No data to display for spot markets.")
            except Exception as e:
                st.error(f"Error displaying spot markets table (bottom): {e}")
                st.caption("Raw data might contain non-serializable fields or other issues.")
        else:
            st.write("No spot markets found.")