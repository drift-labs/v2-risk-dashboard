import asyncio
import glob
import hashlib
import json
import logging
import os
from typing import Callable, Dict, List

from fastapi import BackgroundTasks, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.state import BackendRequest, BackendState


class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, state: BackendState, cache_dir: str = "cache"):
        super().__init__(app)
        self.state = state
        self.cache_dir = cache_dir  # Normal cache for responses (tied to pickle path)
        self.ucache_dir = "ucache"  # This is the generated cache folder for asset liability and price shock
        self.revalidation_locks: Dict[str, asyncio.Lock] = {}
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.ucache_dir):
            os.makedirs(self.ucache_dir)

    async def dispatch(self, request: BackendRequest, call_next: Callable):
        if request.url.path.startswith("/api/ucache"):
            return await call_next(request)
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        # Add special logging for largest_perp_positions endpoint
        is_perp_positions = request.url.path == "/api/health/largest_perp_positions"
        if is_perp_positions:
            logging.info(f"Processing request for largest_perp_positions with query: {request.url.query}")

        current_pickle = self.state.current_pickle_path
        previous_pickles = self._get_previous_pickles(4)  # Get last 4 pickles

        current_cache_key = self._generate_cache_key(request, current_pickle)
        current_cache_file = os.path.join(self.cache_dir, f"{current_cache_key}.json")
        
        if is_perp_positions:
            logging.info(f"Current cache file for largest_perp_positions: {current_cache_file}")
            logging.info(f"File exists: {os.path.exists(current_cache_file)}")

        if os.path.exists(current_cache_file):
            if is_perp_positions:
                logging.info(f"Serving fresh cache for largest_perp_positions from: {current_cache_file}")
            return self._serve_cached_response(current_cache_file, "Fresh", request.url.path)

        for previous_pickle in previous_pickles:
            previous_cache_key = self._generate_cache_key(request, previous_pickle)
            previous_cache_file = os.path.join(
                self.cache_dir, f"{previous_cache_key}.json"
            )
            
            if is_perp_positions:
                logging.info(f"Checking previous cache file: {previous_cache_file}")
                logging.info(f"File exists: {os.path.exists(previous_cache_file)}")

            if os.path.exists(previous_cache_file):
                if is_perp_positions:
                    logging.info(f"Serving stale cache for largest_perp_positions from: {previous_cache_file}")
                return await self._serve_stale_response(
                    previous_cache_file,
                    request,
                    call_next,
                    current_cache_key,
                    current_cache_file,
                )

        if is_perp_positions:
            logging.info(f"No cache found for largest_perp_positions, serving miss response")
        return await self._serve_miss_response(
            request, call_next, current_cache_key, current_cache_file
        )

    def _serve_cached_response(self, cache_file: str, cache_status: str, request_path=None):
        logging.info(f"Serving {cache_status.lower()} data from {cache_file}")
        with open(cache_file, "r") as f:
            response_data = json.load(f)

        content = json.dumps(response_data["content"]).encode("utf-8")
        headers = {
            k: v
            for k, v in response_data["headers"].items()
            if k.lower() != "content-length"
        }
        headers["Content-Length"] = str(len(content))
        headers["X-Cache-Status"] = cache_status

        # Log detailed information for largest_perp_positions endpoint
        try:
            # Check both the cache file path and the request path
            is_perp_positions = "largest_perp_positions" in cache_file
            if request_path:
                is_perp_positions = is_perp_positions or request_path == "/api/health/largest_perp_positions"
                
            if is_perp_positions:
                # Direct print statements guaranteed to show in the terminal
                print(f"\n======== CACHED POSITIONS ({cache_status}) ========")
                print(f"Cache file: {cache_file}")
                
                data = response_data["content"]
                positions_returned = len(data.get("Market Index", []))
                print(f"Total positions returned: {positions_returned}")
                
                if positions_returned > 0:
                    market_indices = data.get("Market Index", [])
                    values = data.get("Value", [])
                    base_amounts = data.get("Base Asset Amount", [])
                    public_keys = data.get("Public Key", [])
                    
                    # Log positions in the correct order (1 to N)
                    for idx in range(positions_returned):
                        market_idx = market_indices[idx] if idx < len(market_indices) else "N/A"
                        value = values[idx] if idx < len(values) else "N/A"
                        base_amt = base_amounts[idx] if idx < len(base_amounts) else "N/A"
                        pub_key = public_keys[idx] if idx < len(public_keys) else "N/A"
                        
                        # Positions are already sorted, so we can log them as is
                        position_num = idx + 1
                        print(f"Position {position_num}:")
                        print(f"    Market Index: {market_idx}")
                        print(f"    Value: {value}")
                        print(f"    Base Asset Amount: {base_amt}")
                        print(f"    Public Key: {pub_key}")
                else:
                    print("No positions found in cached response")
                    
                print("======== END CACHED POSITIONS ========\n")
                
                # Also log to standard logger for Loki
                logger = logging.getLogger("backend.api.health")
                logger.info("[CACHED] BEGIN POSITION DETAILS ----------------------------------------")
                for idx in range(positions_returned):
                    market_idx = market_indices[idx] if idx < len(market_indices) else "N/A"
                    value = values[idx] if idx < len(values) else "N/A"
                    base_amt = base_amounts[idx] if idx < len(base_amounts) else "N/A"
                    pub_key = public_keys[idx] if idx < len(public_keys) else "N/A"
                    
                    position_num = idx + 1
                    logger.info(
                        f"Position {position_num}:\n"
                        f"    Market Index: {market_idx}\n"
                        f"    Value: {value}\n"
                        f"    Base Asset Amount: {base_amt}\n"
                        f"    Public Key: {pub_key}"
                    )
                logger.info("[CACHED] END POSITION DETAILS ----------------------------------------")
                
        except Exception as e:
            print(f"Error printing cached position details: {str(e)}")
            import traceback
            traceback.print_exc()

        self.cleanup_old_cache_files()

        return Response(
            content=content,
            status_code=response_data["status_code"],
            headers=headers,
            media_type="application/json",
        )

    async def _serve_stale_response(
        self,
        cache_file: str,
        request: BackendRequest,
        call_next: Callable,
        current_cache_key: str,
        current_cache_file: str,
    ):
        logging.info(f"Serving stale response from {cache_file} for path: {request.url.path}")
        
        # Check if this is the largest_perp_positions endpoint and log it
        if request.url.path == "/api/health/largest_perp_positions":
            logging.info(f"Stale cache detected for largest_perp_positions with query: {request.url.query}")
            
        response = self._serve_cached_response(cache_file, "Stale", request.url.path)
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            self._fetch_and_cache,
            request,
            call_next,
            current_cache_key,
            current_cache_file,
        )
        response.background = background_tasks
        return response

    async def _serve_miss_response(
        self,
        request: BackendRequest,
        call_next: Callable,
        cache_key: str,
        cache_file: str,
    ):
        logging.info(f"No data available for {request.url.path}")
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            self._fetch_and_cache,
            request,
            call_next,
            cache_key,
            cache_file,
        )
        content = json.dumps({"result": "miss"}).encode("utf-8")

        response = Response(
            content=content,
            status_code=200,
            headers={"X-Cache-Status": "Miss", "Content-Length": str(len(content))},
            media_type="application/json",
        )
        response.background = background_tasks
        return response

    async def _fetch_and_cache(
        self,
        request: BackendRequest,
        call_next: Callable,
        cache_key: str,
        cache_file: str,
    ):
        if cache_key not in self.revalidation_locks:
            self.revalidation_locks[cache_key] = asyncio.Lock()

        async with self.revalidation_locks[cache_key]:
            try:
                response = await call_next(request)

                if response.status_code == 200:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk

                    body_content = json.loads(response_body.decode())
                    response_data = {
                        "content": body_content,
                        "status_code": response.status_code,
                        "headers": {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() != "content-length"
                        },
                    }

                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, "w") as f:
                        json.dump(response_data, f)

                    logging.info(
                        f"Cached fresh data for {request.url.path} with query {request.url.query}"
                    )
                    
                    # Special handling for largest_perp_positions endpoint
                    if request.url.path == "/api/health/largest_perp_positions":
                        print(f"\n======== CACHING NEW POSITION DATA ========")
                        print(f"Path: {request.url.path}?{request.url.query}")
                        print(f"Cache file: {cache_file}")
                        
                        if "Market Index" in body_content:
                            positions_returned = len(body_content.get("Market Index", []))
                            print(f"Total positions being cached: {positions_returned}")
                        else:
                            print(f"Warning: No Market Index found in response data. Keys: {body_content.keys()}")
                        
                        print("======== END CACHING NEW POSITION DATA ========\n")
                    
                else:
                    logging.warning(
                        f"Failed to cache data for {request.url.path}. Status code: {response.status_code}"
                    )
            except Exception as e:
                logging.error(
                    f"Error in background task for {request.url.path}: {str(e)}"
                )
                import traceback
                traceback.print_exc()

    def _generate_cache_key(self, request: BackendRequest, pickle_path: str) -> str:
        hash_input = (
            f"{pickle_path}:{request.method}:{request.url.path}:{request.url.query}"
        )
        logging.info("Hash input: %s", hash_input)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_previous_pickles(self, num_pickles: int = 4) -> List[str]:
        logging.info(f"Attempting to get previous {num_pickles} pickles")
        _pickle_paths = glob.glob(f"{self.state.current_pickle_path}/../*")
        pickle_paths = sorted(
            [os.path.realpath(dir) for dir in _pickle_paths], reverse=True
        )
        logging.info("Pickle paths: %s", pickle_paths)

        previous_pickles = pickle_paths[1 : num_pickles + 1]
        logging.info(f"Previous {len(previous_pickles)} pickles: %s", previous_pickles)
        return previous_pickles

    def cleanup_old_cache_files(self, keep_newest: int = 30):
        """Delete all but the newest N cache files"""
        for cache_dir in [self.cache_dir, self.ucache_dir]:
            if not os.path.exists(cache_dir):
                continue
            files = glob.glob(f"{cache_dir}/*.json")
            files.sort(key=os.path.getmtime, reverse=True)
            for old_file in files[keep_newest:]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    logging.error(f"Failed to remove {old_file}: {e}")
