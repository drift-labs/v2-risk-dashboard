from fastapi import APIRouter
from driftpy.constants.numeric_constants import PRICE_PRECISION
from driftpy.constants.config import mainnet_perp_market_configs

from backend.state import BackendRequest

router = APIRouter()

async def _get_open_interest_per_authority(request: BackendRequest) -> dict:
    vat = request.state.backend_state.vat
    slot = request.state.backend_state.last_oracle_slot

    oi_per_authority = {} 

    for user_data in vat.users.values():
        user_account = user_data.get_user_account()
        
        if user_account is None:
            # Optionally log this: print(f"Warning: Skipping user_data as get_user_account() returned None.")
            continue

        authority = str(user_account.authority)

        current_oi_for_authority = oi_per_authority.get(authority, {
            'total_open_interest_usd': 0.0, 
            'authority': authority
        })

        for position in user_account.perp_positions:
            if position.base_asset_amount == 0:
                continue

            market_index = position.market_index
            oracle_price_data = vat.perp_oracles.get(market_index)
            
            if oracle_price_data is None:
                print(f"Warning: Missing oracle price data for market_index {market_index} for authority {authority}. Skipping position.")
                continue
            
            try:
                oracle_price = float(oracle_price_data.price) / PRICE_PRECISION
                # All perpetual markets use BASE_PRECISION (10^9) for base asset amounts.
                decimals = 9 

                base_asset_amount_val = position.base_asset_amount
                if base_asset_amount_val is None: # Should not happen with base_asset_amount == 0 check, but good for safety
                    print(f"Warning: Position base_asset_amount is None for authority {authority}, market {market_index}. Skipping position.")
                    continue

                position_value_usd = (abs(base_asset_amount_val) / (10**decimals)) * oracle_price
                current_oi_for_authority['total_open_interest_usd'] += position_value_usd
            except (TypeError, ValueError) as e:
                base_val_repr = repr(getattr(position, 'base_asset_amount', 'N/A'))
                oracle_price_repr = repr(getattr(oracle_price_data, 'price', 'N/A'))
                print(f"Error calculating position_value_usd for authority {authority}, market {market_index}: {e}. Base: {base_val_repr}, OraclePriceRaw: {oracle_price_repr}. Skipping position.")
                continue
        
        if current_oi_for_authority['total_open_interest_usd'] > 0:
             oi_per_authority[authority] = current_oi_for_authority
    
    # Filtered values are now implicitly handled by only adding to oi_per_authority if OI > 0
    result_list = sorted(list(oi_per_authority.values()), key=lambda x: x['total_open_interest_usd'], reverse=True)
    
    return {
        "slot": slot,
        "data": result_list, 
    }

async def _get_open_interest_per_account(request: BackendRequest) -> dict:
    vat = request.state.backend_state.vat
    slot = request.state.backend_state.last_oracle_slot

    oi_per_account = {}

    for user_data in vat.users.values(): # user_data is of type UserMapItem (based on health.py it should have user_public_key)
        user_account = user_data.get_user_account()
        
        if user_account is None:
            # Optionally log this: print(f"Warning: Skipping user_data as get_user_account() returned None.")
            continue

        user_public_key = str(user_data.user_public_key) # Get user_public_key from user_data
        authority = str(user_account.authority) # Get authority from user_account

        current_oi_for_account = oi_per_account.get(user_public_key, {
            'total_open_interest_usd': 0.0,
            'user_public_key': user_public_key, # Store user_public_key in the dict
            'authority': authority # Add authority field
        })

        for position in user_account.perp_positions:
            if position.base_asset_amount == 0:
                continue

            market_index = position.market_index
            oracle_price_data = vat.perp_oracles.get(market_index)
            
            if oracle_price_data is None:
                print(f"Warning: Missing oracle price data for market_index {market_index} for user {user_public_key}. Skipping position.")
                continue
            
            try:
                oracle_price = float(oracle_price_data.price) / PRICE_PRECISION
                # All perpetual markets use BASE_PRECISION (10^9) for base asset amounts.
                decimals = 9

                base_asset_amount_val = position.base_asset_amount
                if base_asset_amount_val is None: # Should not happen with base_asset_amount == 0 check, but good for safety
                    print(f"Warning: Position base_asset_amount is None for user {user_public_key}, market {market_index}. Skipping position.")
                    continue

                position_value_usd = (abs(base_asset_amount_val) / (10**decimals)) * oracle_price
                current_oi_for_account['total_open_interest_usd'] += position_value_usd
            except (TypeError, ValueError) as e:
                base_val_repr = repr(getattr(position, 'base_asset_amount', 'N/A'))
                oracle_price_repr = repr(getattr(oracle_price_data, 'price', 'N/A'))
                print(f"Error calculating position_value_usd for user {user_public_key}, market {market_index}: {e}. Base: {base_val_repr}, OraclePriceRaw: {oracle_price_repr}. Skipping position.")
                continue
        
        if current_oi_for_account['total_open_interest_usd'] > 0:
             oi_per_account[user_public_key] = current_oi_for_account
    
    result_list = sorted(list(oi_per_account.values()), key=lambda x: x['total_open_interest_usd'], reverse=True)
    
    return {
        "slot": slot,
        "data": result_list,
    }

async def _get_open_positions_detailed(request: BackendRequest) -> dict:
    vat = request.state.backend_state.vat
    slot = request.state.backend_state.last_oracle_slot
    
    detailed_positions = []
    decimals = 9 # Constant for perpetual markets base asset amount

    for user_data in vat.users.values():
        user_account = user_data.get_user_account()
        
        if user_account is None:
            continue

        user_public_key = str(user_data.user_public_key)
        authority = str(user_account.authority)

        for position in user_account.perp_positions:
            if position.base_asset_amount == 0:
                continue

            market_index = position.market_index
            oracle_price_data = vat.perp_oracles.get(market_index)
            
            if oracle_price_data is None:
                print(f"Warning: Missing oracle price data for market_index {market_index} for user {user_public_key}. Skipping position detail.")
                continue
            
            try:
                base_asset_amount_val = position.base_asset_amount
                if base_asset_amount_val is None:
                    print(f"Warning: Position base_asset_amount is None for user {user_public_key}, market {market_index}. Skipping position detail.")
                    continue

                oracle_price = float(oracle_price_data.price) / PRICE_PRECISION
                position_value_usd = (abs(base_asset_amount_val) / (10**decimals)) * oracle_price
                base_asset_amount_ui = base_asset_amount_val / (10**decimals)

                # Get market symbol
                market_symbol = 'N/A'
                try:
                    market_symbol = mainnet_perp_market_configs[market_index].symbol
                except (IndexError, AttributeError, KeyError) as symbol_e:
                     print(f"Warning: Could not find symbol for market_index {market_index}. Error: {symbol_e}")

                detailed_positions.append({
                    'market_index': market_index,
                    'market_symbol': market_symbol,
                    'base_asset_amount_ui': base_asset_amount_ui,
                    'position_value_usd': position_value_usd,
                    'user_public_key': user_public_key,
                    'authority': authority
                })

            except (TypeError, ValueError) as e:
                base_val_repr = repr(getattr(position, 'base_asset_amount', 'N/A'))
                oracle_price_repr = repr(getattr(oracle_price_data, 'price', 'N/A'))
                print(f"Error calculating detailed position data for user {user_public_key}, market {market_index}: {e}. Base: {base_val_repr}, OraclePriceRaw: {oracle_price_repr}. Skipping position detail.")
                continue
                
    # Sort by notional value descending
    sorted_positions = sorted(detailed_positions, key=lambda x: x['position_value_usd'], reverse=True)
    
    return {
        "slot": slot,
        "data": sorted_positions,
    }

@router.get("/per-authority")
async def get_open_interest_per_authority(request: BackendRequest):
    return await _get_open_interest_per_authority(request)

@router.get("/per-account")
async def get_open_interest_per_account(request: BackendRequest):
    return await _get_open_interest_per_account(request)

@router.get("/detailed-positions")
async def get_open_positions_detailed(request: BackendRequest):
    return await _get_open_positions_detailed(request)
