from fastapi import APIRouter
from backend.state import BackendRequest
import logging
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/debug")
async def debug_state(request: BackendRequest) -> Dict[str, Any]:
    """
    Debug endpoint to check state details.
    """
    state = request.backend_state
    
    if not state:
        return {
            "status": "error",
            "error": "no_state",
            "details": "No backend state found in request"
        }
    
    components = {
        "ready": state.ready,
        "vat_exists": state.vat is not None,
        "dc_exists": state.dc is not None,
        "connection_exists": state.connection is not None,
        "spot_map_exists": state.spot_markets is not None,
        "perp_map_exists": state.perp_markets is not None,
        "user_map_exists": state.user_map is not None,
        "stats_map_exists": state.stats_map is not None
    }
    
    try:
        if not state.is_ready:
            components["state_error"] = "State is not ready"
            return {
                "status": "error",
                "components": components,
                "is_ready": False
            }
        
        if state.vat:
            vat = state.vat
            
            # Check spot markets
            if hasattr(vat, "spot_markets"):
                spot_markets = vat.spot_markets
                if hasattr(spot_markets, 'market_map'):
                    market_map = spot_markets.market_map
                elif hasattr(spot_markets, 'markets'):
                    market_map = spot_markets.markets
                
                if market_map:
                    try:
                        # Get market keys safely
                        market_keys = []
                        for key in market_map:
                            market_keys.append(key)
                        components["spot_markets_count"] = len(market_keys)
                        components["spot_markets_indices"] = market_keys[:5]
                    except Exception as e:
                        logger.error(f"Error getting spot markets count: {str(e)}")
                        components["spot_markets_error"] = str(e)
                else:
                    components["spot_markets_error"] = "No market map found"
            
            # Check perp markets
            if hasattr(vat, "perp_markets"):
                perp_markets = vat.perp_markets
                if hasattr(perp_markets, 'market_map'):
                    market_map = perp_markets.market_map
                elif hasattr(perp_markets, 'markets'):
                    market_map = perp_markets.markets
                
                if market_map:
                    try:
                        # Get market keys safely
                        market_keys = []
                        for key in market_map:
                            market_keys.append(key)
                        components["perp_markets_count"] = len(market_keys)
                        components["perp_markets_indices"] = market_keys[:5]
                    except Exception as e:
                        logger.error(f"Error getting perp markets count: {str(e)}")
                        components["perp_markets_error"] = str(e)
                else:
                    components["perp_markets_error"] = "No market map found"
            
            # Check users
            if hasattr(vat, "users"):
                users = vat.users
                if hasattr(users, 'users'):
                    user_map = users.users
                elif hasattr(users, 'user_map'):
                    user_map = users.user_map
                
                if user_map:
                    try:
                        # Get user keys safely
                        user_keys = []
                        for key in user_map:
                            user_keys.append(key)
                        components["users_count"] = len(user_keys)
                        components["sample_users"] = [str(key) for key in user_keys[:5]]
                    except Exception as e:
                        logger.error(f"Error getting users count: {str(e)}")
                        components["users_error"] = str(e)
                else:
                    components["users_error"] = "No users found"
        else:
            components["vat_error"] = "VAT is not initialized"
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        components["error"] = str(e)
    
    return {
        "status": "success",
        "components": components,
        "is_ready": state.is_ready
    } 