from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Request
import httpx
import logging
from driftpy.math.margin import MarginCategory
import asyncio

from backend.state import BackendRequest
from backend.utils.risk_metrics import calculate_target_scale_iaw

router = APIRouter()
logger = logging.getLogger(__name__)

async def get_jupiter_price_impact(
    input_mint: str,
    output_mint: str,
    amount: int
) -> float:
    """
    Get price impact from Jupiter API for a given swap.
    """
    url = "https://quote-api.jup.ag/v6/quote"
    params = {
        "inputMint": input_mint,
        "outputMint": output_mint,
        "amount": str(amount)  # Convert to string for older httpx
    }
    
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        http2=False  # Disable HTTP/2 for older httpx
    )
    try:
        logger.info(f"Requesting Jupiter quote for {amount} {input_mint} -> {output_mint}")
        response = await client.get(url, params=params)
        if response.status_code != 200:
            error_msg = f"Jupiter API error: {response.text}"
            logger.error(error_msg)
            raise HTTPException(status_code=502, detail=error_msg)
        data = response.json()
        price_impact = float(data.get("priceImpactPct", 0))
        logger.info(f"Got price impact: {price_impact}")
        return price_impact
    except Exception as e:
        logger.error(f"Error getting Jupiter price impact: {str(e)}")
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        await client.aclose()

async def wait_for_backend_state(request: Request, max_retries: int = 5, delay: float = 1.0) -> bool:
    """
    Wait for the backend state to be ready.
    
    Args:
        request: FastAPI request object
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
    
    Returns:
        bool: True if backend state is ready, False otherwise
    """
    for i in range(max_retries):
        if hasattr(request.state, "backend_state") and request.state.backend_state.is_ready:
            return True
        logger.info(f"Waiting for backend state (attempt {i+1}/{max_retries})")
        await asyncio.sleep(delay)
    return False

@router.get("/target_scale_iaw")
async def get_target_scale_iaw(
    request: BackendRequest,
    market_index: int
) -> Dict[str, Any]:
    """
    Get Target Scale IAW and criteria results for all users in a market.
    
    Returns:
        Dict with user public keys as keys and their Target Scale IAW metrics as values
    """
    logger.info(f"Processing target_scale_iaw request for market {market_index}")
    
    try:
        # Wait for backend state to be ready
        if not await wait_for_backend_state(request):
            logger.error("Backend state not ready after retries")
            return {"result": "miss", "reason": "state_not_ready_after_retries"}
            
        state = request.backend_state
        vat = state.vat
        
        if not hasattr(vat, "spot_markets"):
            logger.warning("Spot markets not available")
            return {
                "result": "miss",
                "reason": "spot_markets_not_available",
            }
            
        results = {}
        
        # Get market configuration
        try:
            market = None
            market_map = vat.spot_markets.market_map if hasattr(vat.spot_markets, 'market_map') else vat.spot_markets.markets
            
            if not market_map:
                logger.error("Market map is empty")
                return {"result": "miss", "reason": "market_map_empty"}
            
            # Find market by index
            for m in market_map.values():
                if hasattr(m, 'market_index') and m.market_index == market_index:
                    market = m
                    break
                elif hasattr(m, 'data') and m.data.market_index == market_index:
                    market = m.data
                    break
                    
            if not market:
                logger.error(f"Market index {market_index} not found")
                return {"result": "miss", "reason": f"market_index_{market_index}_not_found"}
                
            # Get insurance fund balance safely
            insurance_fund_balance = 0
            if hasattr(market, 'insurance_fund'):
                if hasattr(market.insurance_fund, 'total_shares'):
                    insurance_fund_balance = market.insurance_fund.total_shares
                elif hasattr(market.insurance_fund, 'shares'):
                    insurance_fund_balance = market.insurance_fund.shares
                    
            logger.info(f"Processing market {market_index} with insurance fund balance: {insurance_fund_balance}")
            
            # Get market mint safely
            market_mint = None
            if hasattr(market, 'mint'):
                market_mint = str(market.mint)
            elif hasattr(market, 'oracle_source'):
                market_mint = str(market.oracle_source)
            
            if not market_mint:
                logger.error(f"Market {market_index} mint not found")
                return {"result": "miss", "reason": f"market_{market_index}_mint_not_found"}
            
        except Exception as e:
            logger.error(f"Error getting market {market_index}: {str(e)}")
            return {"result": "miss", "reason": f"market_error: {str(e)}"}
        
        user_count = 0
        if state.user_map and hasattr(state.user_map, 'users'):
            users = state.user_map.users
            for user in users.values():
                try:
                    # Get largest position for price impact calculation
                    position_value = user.get_spot_market_asset_value(market_index, MarginCategory.MAINTENANCE)
                    logger.debug(f"User {user.user_public_key} position value: {position_value}")
                    
                    if position_value <= 0:
                        continue
                    
                    # Get price impact from Jupiter for the position
                    try:
                        price_impact = await get_jupiter_price_impact(
                            input_mint=market_mint,
                            output_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC mint
                            amount=int(position_value)
                        )
                    except Exception as e:
                        logger.error(f"Jupiter API error for user {user.user_public_key}: {e}")
                        continue
                    
                    # Calculate Target Scale IAW and get criteria results
                    target_scale, criteria = calculate_target_scale_iaw(
                        user=user,
                        market_index=market_index,
                        price_impact=price_impact,
                        insurance_fund_balance=insurance_fund_balance
                    )
                    
                    results[str(user.user_public_key)] = {
                        "target_scale_iaw": target_scale,
                        "criteria_results": criteria
                    }
                    user_count += 1
                    
                except Exception as e:
                    logger.error(f"Error calculating Target Scale IAW for user {user.user_public_key}: {e}")
                    continue
        
        logger.info(f"Processed {user_count} users for market {market_index}")
        
        if not results:
            logger.warning(f"No results found for market {market_index}")
            return {"result": "miss", "reason": "no_results_found"}
            
        return {"result": "success", "data": results}
        
    except Exception as e:
        logger.error(f"Unexpected error in get_target_scale_iaw: {str(e)}")
        return {"result": "miss", "reason": f"unexpected_error: {str(e)}"}

@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Check the health of the risk metrics service.
    """
    state_ready = hasattr(request.state, "backend_state") and request.state.backend_state.is_ready
    vat_ready = hasattr(request.state, "backend_state") and request.state.backend_state.vat is not None
    
    logger.info(f"Health check - State ready: {state_ready}, VAT ready: {vat_ready}")
    
    return {
        "status": "healthy" if state_ready and vat_ready else "initializing",
        "state_ready": state_ready,
        "vat_ready": vat_ready
    }

@router.get("/debug")
async def debug_state(request: Request) -> Dict[str, Any]:
    """
    Debug endpoint to check state details.
    """
    state = getattr(request.state, "backend_state", None)
    
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
        "spot_markets_exists": hasattr(state.vat, "spot_markets") and state.vat.spot_markets is not None,
        "perp_map_exists": state.perp_map is not None,
        "user_map_exists": state.user_map is not None,
        "stats_map_exists": state.stats_map is not None,
        "current_pickle_path": state.current_pickle_path,
        "last_oracle_slot": state.last_oracle_slot
    }
    
    # Check if state is initialized
    try:
        logger.info("Checking state initialization")
        if not state.is_ready:
            logger.warning("State is not ready")
            components["state_error"] = "State is not ready"
            return {
                "status": "error",
                "components": components,
                "is_ready": False
            }
        
        # Check VAT details
        if state.vat:
            logger.info("Checking VAT details")
            vat = state.vat
            
            # Debug VAT structure
            logger.info(f"VAT type: {type(vat)}")
            logger.info(f"VAT dir: {dir(vat)}")
            
            # Check spot markets
            if vat.spot_markets:
                logger.info("Checking spot markets")
                logger.info(f"Spot markets type: {type(vat.spot_markets)}")
                logger.info(f"Spot markets dir: {dir(vat.spot_markets)}")
                try:
                    # Try to get spot markets count
                    if hasattr(vat.spot_markets, 'size'):
                        spot_markets_count = vat.spot_markets.size()
                        components["spot_markets_count"] = spot_markets_count
                    elif hasattr(vat.spot_markets, 'markets'):
                        spot_markets_count = len(vat.spot_markets.markets)
                        components["spot_markets_count"] = spot_markets_count
                    else:
                        logger.warning("Could not determine spot markets count")
                        components["spot_markets_error"] = "Could not determine count"
                    
                    # Try to get market details
                    spot_markets = []
                    if hasattr(vat.spot_markets, 'markets'):
                        for market_data in list(vat.spot_markets.markets.values())[:5]:
                            if hasattr(market_data, 'data'):
                                spot_markets.append({
                                    "market_index": market_data.data.market_index,
                                    "oracle_source": str(market_data.data.oracle_source)
                                })
                    elif hasattr(vat.spot_markets, 'values'):
                        for market_data in list(vat.spot_markets.values())[:5]:
                            if hasattr(market_data, 'data'):
                                spot_markets.append({
                                    "market_index": market_data.data.market_index,
                                    "oracle_source": str(market_data.data.oracle_source)
                                })
                    
                    if spot_markets:
                        components["spot_markets"] = spot_markets
                    else:
                        logger.warning("No spot markets found")
                        components["spot_markets_error"] = "No markets found"
                        
                except Exception as e:
                    logger.error(f"Error getting spot markets: {str(e)}")
                    components["spot_markets_error"] = str(e)
            
            # Check perp markets
            if vat.perp_markets:
                logger.info("Checking perp markets")
                logger.info(f"Perp markets type: {type(vat.perp_markets)}")
                logger.info(f"Perp markets dir: {dir(vat.perp_markets)}")
                try:
                    # Try to get perp markets count
                    if hasattr(vat.perp_markets, 'size'):
                        perp_markets_count = vat.perp_markets.size()
                        components["perp_markets_count"] = perp_markets_count
                    elif hasattr(vat.perp_markets, 'markets'):
                        perp_markets_count = len(vat.perp_markets.markets)
                        components["perp_markets_count"] = perp_markets_count
                    else:
                        logger.warning("Could not determine perp markets count")
                        components["perp_markets_error"] = "Could not determine count"
                    
                    # Try to get market details
                    perp_markets = []
                    if hasattr(vat.perp_markets, 'markets'):
                        for market_data in list(vat.perp_markets.markets.values())[:5]:
                            if hasattr(market_data, 'data'):
                                perp_markets.append({
                                    "market_index": market_data.data.market_index,
                                    "oracle_source": str(market_data.data.oracle_source)
                                })
                    elif hasattr(vat.perp_markets, 'values'):
                        for market_data in list(vat.perp_markets.values())[:5]:
                            if hasattr(market_data, 'data'):
                                perp_markets.append({
                                    "market_index": market_data.data.market_index,
                                    "oracle_source": str(market_data.data.oracle_source)
                                })
                    
                    if perp_markets:
                        components["perp_markets"] = perp_markets
                    else:
                        logger.warning("No perp markets found")
                        components["perp_markets_error"] = "No markets found"
                        
                except Exception as e:
                    logger.error(f"Error getting perp markets: {str(e)}")
                    components["perp_markets_error"] = str(e)
            
            # Check users
            if vat.users:
                logger.info("Checking users")
                logger.info(f"Users type: {type(vat.users)}")
                logger.info(f"Users dir: {dir(vat.users)}")
                try:
                    # Try to get users count
                    if hasattr(vat.users, 'size'):
                        users_count = vat.users.size()
                        components["users_count"] = users_count
                    elif hasattr(vat.users, 'user_map'):
                        users_count = len(vat.users.user_map)
                        components["users_count"] = users_count
                    else:
                        logger.warning("Could not determine users count")
                        components["users_error"] = "Could not determine count"
                    
                    # Try to get user details
                    users = []
                    if hasattr(vat.users, 'user_map'):
                        for user_data in list(vat.users.user_map.values())[:5]:
                            if hasattr(user_data, 'user_public_key'):
                                users.append({
                                    "public_key": user_data.user_public_key
                                })
                    elif hasattr(vat.users, 'values'):
                        for user_data in list(vat.users.values())[:5]:
                            if hasattr(user_data, 'user_public_key'):
                                users.append({
                                    "public_key": user_data.user_public_key
                                })
                    
                    if users:
                        components["users"] = users
                    else:
                        logger.warning("No users found")
                        components["users_error"] = "No users found"
                        
                except Exception as e:
                    logger.error(f"Error getting users: {str(e)}")
                    components["users_error"] = str(e)
            
            # Check user stats
            if vat.user_stats:
                logger.info("Checking user stats")
                logger.info(f"User stats type: {type(vat.user_stats)}")
                logger.info(f"User stats dir: {dir(vat.user_stats)}")
                try:
                    # Try to get stats count
                    if hasattr(vat.user_stats, 'size'):
                        stats_count = vat.user_stats.size()
                        components["stats_count"] = stats_count
                    elif hasattr(vat.user_stats, 'stats'):
                        stats_count = len(vat.user_stats.stats)
                        components["stats_count"] = stats_count
                    else:
                        logger.warning("Could not determine stats count")
                        components["stats_error"] = "Could not determine count"
                except Exception as e:
                    logger.error(f"Error getting user stats: {str(e)}")
                    components["stats_error"] = str(e)
            
        else:
            logger.warning("VAT is not initialized")
            components["vat_error"] = "VAT is not initialized"
        
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        components["error"] = str(e)
    
    logger.info(f"Debug state check: {components}")
    
    return {
        "status": "success",
        "components": components,
        "is_ready": state.is_ready
    }

@router.get("/vat-state")
async def get_vat_state(request: BackendRequest) -> Dict[str, Any]:
    """
    Get the current state of the VAT object.
    """
    logger.info("Processing vat-state request")
    
    try:
        # Wait for backend state to be ready
        if not await wait_for_backend_state(request):
            logger.error("Backend state not ready after retries")
            return {"result": "miss", "reason": "state_not_ready_after_retries"}
            
        state = request.backend_state
        vat = state.vat
        
        if not hasattr(vat, "spot_markets"):
            logger.warning("Spot markets not available")
            return {
                "result": "miss",
                "reason": "spot_markets_not_available",
            }
            
        # Get spot markets count safely
        spot_markets = {}
        if hasattr(vat.spot_markets, 'markets'):
            market_map = vat.spot_markets.markets
            for market_index, market in market_map.items():
                market_info = {
                    "market_index": market_index,
                    "insurance_fund_balance": 0,  # Default value
                    "mint": None  # Default value
                }
                
                # Get insurance fund balance safely
                if hasattr(market, 'insurance_fund'):
                    if hasattr(market.insurance_fund, 'total_shares'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.total_shares / 1e6  # Convert from lamports
                    elif hasattr(market.insurance_fund, 'shares'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.shares / 1e6  # Convert from lamports
                    elif hasattr(market.insurance_fund, 'balance'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.balance / 1e6  # Convert from lamports
                
                # Get market mint safely
                if hasattr(market, 'mint'):
                    market_info["mint"] = str(market.mint)
                elif hasattr(market, 'oracle_source'):
                    market_info["mint"] = str(market.oracle_source)
                
                spot_markets[str(market_index)] = market_info
        
        # Get perp markets count safely
        perp_markets = {}
        if hasattr(vat.perp_markets, 'markets'):
            market_map = vat.perp_markets.markets
            for market_index, market in market_map.items():
                market_info = {
                    "market_index": market_index,
                    "insurance_fund_balance": 0  # Default value
                }
                
                # Get insurance fund balance safely
                if hasattr(market, 'insurance_fund'):
                    if hasattr(market.insurance_fund, 'total_shares'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.total_shares / 1e6  # Convert from lamports
                    elif hasattr(market.insurance_fund, 'shares'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.shares / 1e6  # Convert from lamports
                    elif hasattr(market.insurance_fund, 'balance'):
                        market_info["insurance_fund_balance"] = market.insurance_fund.balance / 1e6  # Convert from lamports
                
                perp_markets[str(market_index)] = market_info
        
        return {
            "result": "success",
            "data": {
                "spot_markets": spot_markets,
                "perp_markets": perp_markets,
                "last_update": state.last_oracle_slot
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting VAT state: {str(e)}")
        return {"result": "miss", "reason": f"error: {str(e)}"}