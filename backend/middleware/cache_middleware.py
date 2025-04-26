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
        # Skip caching for ucache and non-API endpoints
        if request.url.path.startswith("/api/ucache"):
            return await call_next(request)
        if not request.url.path.startswith("/api"):
            return await call_next(request)
            
        # Skip caching for health check endpoints
        if request.url.path == "/api/health/" or request.url.path.startswith("/api/backend-health"):
            return await call_next(request)
            
        # Skip caching if bypass_cache=true is in the query parameters
        if "bypass_cache=true" in request.url.query or "bypass_cache=True" in request.url.query:
            logging.info(f"Bypassing cache for {request.url.path} due to bypass_cache parameter")
            return await call_next(request)

        # Add special logging for largest_perp_positions endpoint (for debugging purposes)
        is_perp_positions = request.url.path == "/api/health/largest_perp_positions"
        if is_perp_positions:
            logging.info(f"Processing request for largest_perp_positions with query: {request.url.query}")

        # Defensive check to ensure state is initialized
        if not hasattr(self.state, 'current_pickle_path') or self.state.current_pickle_path is None:
            logging.error(f"State not properly initialized, missing current_pickle_path. Bypassing cache for {request.url.path}")
            return await call_next(request)
            
        current_pickle = self.state.current_pickle_path
        previous_pickles = self._get_previous_pickles(4)  # Get last 4 pickles

        current_cache_key = self._generate_cache_key(request, current_pickle)
        current_cache_file = os.path.join(self.cache_dir, f"{current_cache_key}.json")
        
        if is_perp_positions:
            logging.info(f"Current cache file for largest_perp_positions: {current_cache_file}")
            logging.info(f"File exists: {os.path.exists(current_cache_file)}")

        # 1. Check for fresh cache
        if os.path.exists(current_cache_file):
            if is_perp_positions:
                logging.info(f"Serving fresh cache for largest_perp_positions from: {current_cache_file}")
            return self._serve_cached_response(current_cache_file, "Fresh", request.url.path)

        # 2. Check for stale cache
        for previous_pickle in previous_pickles:
            previous_cache_key = self._generate_cache_key(request, previous_pickle)
            previous_cache_file = os.path.join(
                self.cache_dir, f"{previous_cache_key}.json"
            )
            
            if is_perp_positions:
                logging.info(f"Checking previous cache file: {previous_cache_file}")
                logging.info(f"File exists: {os.path.exists(previous_cache_file)}")

            if os.path.exists(previous_cache_file):
                logging.info(f"Serving stale response from {previous_cache_file} for path: {request.url.path}")
                if is_perp_positions:
                    logging.info(f"Stale cache detected for largest_perp_positions with query: {request.url.query}")
                    
                response = self._serve_cached_response(previous_cache_file, "Stale", request.url.path)
                
                # Check lock before scheduling background task for stale revalidation
                lock = self.revalidation_locks.setdefault(current_cache_key, asyncio.Lock())
                if not lock.locked():
                    logging.info(f"Scheduling background fetch for stale key: {current_cache_key}")
                    background_tasks = BackgroundTasks()
                    background_tasks.add_task(
                        self._fetch_and_cache,
                        request,
                        call_next,
                        current_cache_key, # Use the *current* key for the new cache file
                        current_cache_file,
                    )
                    response.background = background_tasks
                else:
                     logging.info(f"Background fetch already in progress for key: {current_cache_key}, skipping duplicate stale revalidation task.")
                     
                return response

        # 3. Handle cache miss
        logging.info(f"Cache miss for {request.url.path} with key {current_cache_key}")
        if is_perp_positions:
            logging.info(f"No cache found for largest_perp_positions, serving miss response and triggering fetch")

        # Check lock before scheduling background task for miss
        lock = self.revalidation_locks.setdefault(current_cache_key, asyncio.Lock())
        background_tasks = BackgroundTasks()
        
        if not lock.locked():
            # Lock is free, schedule the background task
            logging.info(f"Scheduling background fetch for missed key: {current_cache_key}")
            background_tasks.add_task(
                self._fetch_and_cache,
                request,
                call_next,
                current_cache_key,
                current_cache_file,
            )
            # Serve a "miss" response immediately, indicating process start
            message = "Data is being generated in the background. Please try again shortly."
            result = "miss"
        else:
            # Lock is held, another task is already processing this key
            logging.info(f"Background fetch already in progress for key: {current_cache_key}, skipping duplicate miss fetch task.")
            # Serve a "processing" response immediately
            message = "Data generation already in progress from a previous request. Please try again shortly."
            result = "processing"
            
        # Serve a 202 Accepted response immediately in both cases (lock held or task just started)
        content = json.dumps({"result": result, "message": message}).encode("utf-8")
        response = Response(
            content=content,
            status_code=202, # Accepted: request accepted, processing potentially in background
            headers={"X-Cache-Status": "Miss", "Content-Length": str(len(content))},
            media_type="application/json",
        )
        # Attach background tasks (will be empty if lock was held, non-empty otherwise)
        response.background = background_tasks 
        return response


    def _serve_cached_response(self, cache_file: str, cache_status: str, request_path=None):
        """Serves a response from a cache file with the specified cache status."""
        logging.info(f"Serving {cache_status.lower()} data from {cache_file}")
        try:
            with open(cache_file, "r") as f:
                response_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
             logging.error(f"Error reading cache file {cache_file}: {e}. Returning cache miss.")
             # Simulate a miss scenario if cache file is bad
             content = json.dumps({"result": "error", "message": "Failed to read cache file."}).encode("utf-8")
             return Response(
                 content=content,
                 status_code=500, # Internal Server Error
                 headers={"X-Cache-Status": "Error", "Content-Length": str(len(content))},
                 media_type="application/json",
             )

        content = json.dumps(response_data.get("content", {})).encode("utf-8") # Add default for content
        headers = {
            k: v
            for k, v in response_data.get("headers", {}).items() # Add default for headers
            if k.lower() not in ["content-length", "x-cache-status"] # Exclude old status and length
        }
        headers["Content-Length"] = str(len(content))
        headers["X-Cache-Status"] = cache_status

        # Log detailed information for largest_perp_positions endpoint (optional debugging)
        try:
            is_perp_positions = "largest_perp_positions" in cache_file or request_path == "/api/health/largest_perp_positions"
            if is_perp_positions:
                self._log_perp_positions_details(response_data.get("content"), cache_status, cache_file, request_path)
        except Exception as e:
            logging.error(f"Error during specific logging for perp positions: {e}")
            # Continue serving the response even if logging fails

        self.cleanup_old_cache_files()

        return Response(
            content=content,
            status_code=response_data.get("status_code", 200), # Add default for status_code
            headers=headers,
            media_type="application/json",
        )
        
    def _log_perp_positions_details(self, data, cache_status, cache_file, request_path):
         """Helper method to log details specifically for the largest_perp_positions endpoint."""
         # Direct print statements guaranteed to show in the terminal
         print(f"\n======== CACHED POSITIONS ({cache_status}) ========")
         print(f"Cache file: {cache_file}")
         
         positions_returned = 0
         # Handle potential variations in response structure (list vs dict)
         if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
              positions_returned = len(data)
              print(f"Total positions returned (list format): {positions_returned}")
              for idx, pos in enumerate(data):
                  position_num = idx + 1
                  print(f"Position {position_num}: {pos}")

         elif isinstance(data, dict) and "Market Index" in data:
             positions_returned = len(data.get("Market Index", []))
             print(f"Total positions returned (dict format): {positions_returned}")
             
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
                  print("No positions found in cached dictionary response")
         else:
             print(f"Warning: Unrecognized format for position data or empty data. Data: {data}")

         print("======== END CACHED POSITIONS ========\n")
         
         # Also log to standard logger for Loki
         logger = logging.getLogger("backend.api.health") # Use appropriate logger name if different
         logger.info(f"[CACHED {cache_status}] BEGIN POSITION DETAILS ({request_path})----------------------------------------")
         if isinstance(data, list) and positions_returned > 0:
              for idx, pos in enumerate(data):
                  logger.info(f"Position {idx + 1}: {pos}")
         elif isinstance(data, dict) and positions_returned > 0:
              market_indices = data.get("Market Index", [])
              values = data.get("Value", [])
              base_amounts = data.get("Base Asset Amount", [])
              public_keys = data.get("Public Key", [])
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
         else:
              logger.info("No positions logged.")
         logger.info(f"[CACHED {cache_status}] END POSITION DETAILS ({request_path}) ----------------------------------------")


    async def _fetch_and_cache(
        self,
        request: BackendRequest,
        call_next: Callable,
        cache_key: str,
        cache_file: str,
    ):
        """Fetches data using call_next and caches it. Assumes lock is acquired before calling."""
        lock = self.revalidation_locks.get(cache_key)
        if not lock:
             logging.error(f"Lock not found for cache key {cache_key} in _fetch_and_cache. Aborting fetch.")
             return
             
        # Use the lock to ensure atomicity of the fetch and cache write
        # The lock *must* be acquired here to prevent race conditions during fetch/write
        async with lock:
            # Double-check if cache file was created while waiting for the lock
            if os.path.exists(cache_file):
                 logging.info(f"Cache file {cache_file} created while waiting for lock. Aborting redundant fetch.")
                 return
                 
            try:
                logging.info(f"Starting locked background fetch for {request.url.path} with key {cache_key}")
                response = await call_next(request)

                if response.status_code == 200:
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk

                    # Ensure response_body is not empty before trying to decode
                    if not response_body:
                         logging.warning(f"Empty response body received for {request.url.path}. Cannot cache.")
                         return # Exit the 'async with lock' block, releasing the lock

                    try:
                        body_content = json.loads(response_body.decode())
                    except json.JSONDecodeError as e:
                         logging.error(f"Failed to decode JSON response for {request.url.path}: {e}. Response body: {response_body[:500]}")
                         return # Exit the 'async with lock' block

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
                        f"Cached fresh data for {request.url.path} with query {request.url.query} to {cache_file}"
                    )
                    
                    # Special handling for largest_perp_positions endpoint
                    if request.url.path == "/api/health/largest_perp_positions":
                         try:
                              self._log_perp_positions_details(body_content, "FETCHED", cache_file, request.url.path)
                         except Exception as log_e:
                              logging.error(f"Error logging fetched perp positions details: {log_e}")
                
                else:
                    logging.warning(
                        f"Failed to cache data for {request.url.path}. Status code: {response.status_code}"
                    )
            except Exception as e:
                logging.error(
                    f"Error in background task for {request.url.path} under lock {cache_key}: {str(e)}"
                )
                import traceback
                traceback.print_exc()
            # Lock is automatically released here by 'async with'

    def _generate_cache_key(self, request: BackendRequest, pickle_path: str) -> str:
        """Generates an MD5 cache key based on pickle path basename and request details."""
        pickle_path_str = str(pickle_path)
        pickle_identifier = os.path.basename(pickle_path_str)
        
        hash_input = (
            f"{pickle_identifier}:{request.method}:{request.url.path}:{request.url.query}"
        )
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_previous_pickles(self, num_pickles: int = 4) -> List[str]:
        """Gets the paths of the N most recent pickle directories, excluding the current one."""
        # logging.info(f"Attempting to get previous {num_pickles} pickles") # Verbose logging
        if not hasattr(self.state, 'current_pickle_path') or self.state.current_pickle_path is None:
            logging.error("Cannot get previous pickles: current_pickle_path is not initialized")
            return []
            
        parent_dir = os.path.dirname(self.state.current_pickle_path)
        if not parent_dir or not os.path.exists(parent_dir):
             logging.warning(f"Parent directory '{parent_dir}' for pickles not found or invalid.")
             parent_dir = "pickles" # Fallback assumption
             if not os.path.exists(parent_dir):
                  logging.error(f"Pickle directory '{parent_dir}' not found. Cannot find previous pickles.")
                  return []

        try:
             all_pickle_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        except OSError as e:
             logging.error(f"Error listing pickle directories in '{parent_dir}': {e}")
             return []

        try:
            current_real_path = os.path.realpath(self.state.current_pickle_path)
            # Sort by modification time for robustness against non-timestamp names
            sorted_paths = sorted(
                [os.path.realpath(p) for p in all_pickle_dirs if os.path.realpath(p) != current_real_path],
                 key=os.path.getmtime, # Sort by modification time
                 reverse=True
            )
        except Exception as e:
            logging.error(f"Error sorting pickle paths: {e}. Paths found: {all_pickle_dirs}")
            return []

        previous_pickles = sorted_paths[:num_pickles]
        # logging.info(f"Previous {len(previous_pickles)} pickles found: %s", previous_pickles) # Verbose logging
        return previous_pickles


    def cleanup_old_cache_files(self, keep_newest: int = 30):
        """Deletes all but the newest N cache files based on modification time."""
        # logging.debug(f"Running cache cleanup, keeping newest {keep_newest} files.") # Optional debug log
        for cache_dir in [self.cache_dir, self.ucache_dir]:
            if not os.path.exists(cache_dir):
                continue
            try:
                 files = glob.glob(f"{cache_dir}/*.json")
                 files.sort(key=os.path.getmtime, reverse=True)
                 
                 files_to_delete = files[keep_newest:]
                 if files_to_delete:
                      # logging.info(f"Removing {len(files_to_delete)} old cache files from {cache_dir}.") # Optional info log
                      for old_file in files_to_delete:
                          try:
                              os.remove(old_file)
                              # logging.debug(f"Removed old cache file: {old_file}") # Optional debug log
                          except Exception as e:
                              logging.error(f"Failed to remove {old_file}: {e}")
            except Exception as e:
                 logging.error(f"Error during cache cleanup for {cache_dir}: {e}")