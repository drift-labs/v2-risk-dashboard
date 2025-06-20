import asyncio
import glob
import hashlib
import json
import logging
import os
import time
from typing import Callable, Dict, List

from fastapi import BackgroundTasks, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.state import BackendRequest, BackendState

logger = logging.getLogger(__name__)

# Define placeholder content
PROCESSING_PLACEHOLDER = {"result": "processing", "message": "Data generation in progress. Please try again shortly."}
ERROR_PLACEHOLDER = {"result": "error", "message": "Failed to generate data in the background."}

class CacheMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, state: BackendState, cache_dir: str = "cache"):
        super().__init__(app)
        self.state = state
        self.cache_dir = cache_dir  # Normal cache for responses (tied to pickle path)
        self.ucache_dir = "ucache"  # This is the generated cache folder for asset liability and price shock
        # Locks are per-worker, used for intra-worker coordination before file check
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
            logger.info(f"Bypassing cache for {request.url.path} due to bypass_cache parameter")
            return await call_next(request)

        # Add special logging for largest_perp_positions endpoint (for debugging purposes)
        is_perp_positions = request.url.path == "/api/health/largest_perp_positions"
        if is_perp_positions:
            logger.info(f"Processing request for largest_perp_positions with query: {request.url.query}")

        # Defensive check to ensure state is initialized
        if not hasattr(self.state, 'current_pickle_path') or self.state.current_pickle_path is None:
            logger.error(f"State not properly initialized, missing current_pickle_path. Bypassing cache for {request.url.path}")
            return await call_next(request)

        current_pickle = self.state.current_pickle_path
        previous_pickles = self._get_previous_pickles(4)  # Get last 4 pickles

        current_cache_key = self._generate_cache_key(request, current_pickle)
        current_cache_file = os.path.join(self.cache_dir, f"{current_cache_key}.json")

        if is_perp_positions:
            logger.info(f"Current cache file for largest_perp_positions: {current_cache_file}")
            logger.info(f"File exists: {os.path.exists(current_cache_file)}")

        # 1. Check for fresh cache
        if os.path.exists(current_cache_file):
            if is_perp_positions:
                logger.info(f"Serving fresh cache for largest_perp_positions from: {current_cache_file}")
            # DO NOT run cleanup on every cache hit
            # self.cleanup_old_cache_files() # REMOVED FROM HERE
            return self._serve_cached_response(current_cache_file, "Fresh", request.url.path)

        # 2. Check for stale cache
        for previous_pickle in previous_pickles:
            previous_cache_key = self._generate_cache_key(request, previous_pickle)
            previous_cache_file = os.path.join(
                self.cache_dir, f"{previous_cache_key}.json"
            )

            if is_perp_positions:
                logger.info(f"Checking previous cache file: {previous_cache_file}")
                logger.info(f"File exists: {os.path.exists(previous_cache_file)}")

            if os.path.exists(previous_cache_file):
                logger.info(f"Serving stale response from {previous_cache_file} for path: {request.url.path}")
                if is_perp_positions:
                    logger.info(f"Stale cache detected for largest_perp_positions with query: {request.url.query}")

                response = self._serve_cached_response(previous_cache_file, "Stale", request.url.path)

                # Revalidate stale cache in the background *if* the current cache doesn't exist yet
                # This uses the placeholder/lock mechanism below to avoid duplicate stale fetches
                if not os.path.exists(current_cache_file):
                    lock = self.revalidation_locks.setdefault(current_cache_key, asyncio.Lock())
                    # Schedule background task only if lock is free AND placeholder doesn't exist
                    if not lock.locked():
                        async with lock: # Acquire lock before file check/write
                            if not os.path.exists(current_cache_file):
                                try:
                                    # Create placeholder to prevent other workers/requests
                                    os.makedirs(os.path.dirname(current_cache_file), exist_ok=True)
                                    with open(current_cache_file, "w") as f:
                                        json.dump({
                                            "content": PROCESSING_PLACEHOLDER,
                                            "status_code": 202,
                                            "headers": {}
                                        }, f)
                                    logger.info(f"Created placeholder for stale revalidation: {current_cache_file}")

                                    # Schedule the actual fetch
                                    logger.info(f"Scheduling background fetch for stale key: {current_cache_key}")
                                    background_tasks = BackgroundTasks()
                                    background_tasks.add_task(
                                        self._fetch_and_cache,
                                        request,
                                        call_next,
                                        current_cache_key,
                                        current_cache_file,
                                    )
                                    response.background = background_tasks # Attach task to the *stale* response

                                except Exception as e:
                                    logger.error(f"Error creating placeholder or scheduling stale fetch for {current_cache_key}: {e}")
                                    # Clean up potentially broken placeholder
                                    if os.path.exists(current_cache_file):
                                        try: os.remove(current_cache_file)
                                        except OSError: pass

                            else:
                                logger.info(f"Placeholder/Cache file {current_cache_file} already exists. Skipping duplicate stale revalidation task.")
                    else:
                       logger.info(f"Background fetch lock held for key: {current_cache_key}, skipping duplicate stale revalidation task.")

                # DO NOT run cleanup on every stale cache hit
                # self.cleanup_old_cache_files() # REMOVED FROM HERE
                return response # Return the stale response immediately

        # 3. Handle cache miss (current cache file does not exist)
        logger.info(f"Cache miss for {request.url.path} with key {current_cache_key}")
        if is_perp_positions:
            logger.info(f"No cache found for largest_perp_positions, attempting to generate.")

        lock = self.revalidation_locks.setdefault(current_cache_key, asyncio.Lock())
        background_tasks = BackgroundTasks()
        response_content = None
        response_status = 202 # Default to Accepted

        async with lock: # Use async with lock to ensure it's released
            # Double-check file existence *inside the lock*
            if not os.path.exists(current_cache_file):
                # File still doesn't exist, this worker should create placeholder and schedule task
                logger.info(f"Lock acquired for {current_cache_key}. Creating placeholder and scheduling fetch.")
                try:
                    # Create placeholder atomically (as best as possible without external locks)
                    os.makedirs(os.path.dirname(current_cache_file), exist_ok=True)
                    with open(current_cache_file, "w") as f:
                        json.dump({
                            "content": PROCESSING_PLACEHOLDER,
                            "status_code": 202,
                            "headers": {}
                        }, f)
                    logger.info(f"Created placeholder file: {current_cache_file}")

                    # Schedule the background task
                    background_tasks.add_task(
                        self._fetch_and_cache,
                        request,
                        call_next,
                        current_cache_key,
                        current_cache_file,
                    )
                    response_content = PROCESSING_PLACEHOLDER # Initial response content
                    response_status = 202
                    logger.info(f"Scheduled background fetch for new key: {current_cache_key}")

                except Exception as e:
                    logger.error(f"Error creating placeholder or scheduling fetch for {current_cache_key}: {e}")
                    # Clean up potentially broken placeholder before releasing lock
                    if os.path.exists(current_cache_file):
                        try: os.remove(current_cache_file)
                        except OSError: pass
                    response_content = ERROR_PLACEHOLDER
                    response_status = 500 # Internal server error

            else:
                # Placeholder/Cache file was created by another worker/request while waiting for the lock
                logger.info(f"Placeholder/Cache file {current_cache_file} found after acquiring lock. Checking status.")
                # Check if it's a processing placeholder or actual data (should be placeholder)
                try:
                    with open(current_cache_file, "r") as f:
                        cached_data = json.load(f)
                    # Check if the *content* field specifically matches the placeholder structure
                    if cached_data.get("content", {}).get("result") == "processing":
                        logger.info(f"Cache file indicates processing for {current_cache_key}. Returning 202.")
                        response_content = PROCESSING_PLACEHOLDER
                        response_status = 202
                    else:
                        # It's actual data - serve it (though this path is less likely for a pure miss)
                        logger.warning(f"Found completed data in {current_cache_file} unexpectedly after miss lock. Serving.")
                        # We need to re-serve using the standard method structure
                        # Note: This bypasses the _serve_cached_response method for simplicity here
                        response_content = cached_data.get("content", {})
                        response_status = cached_data.get("status_code", 200)
                        # Need to construct headers properly if taking this path
                        headers = {k: v for k, v in cached_data.get("headers", {}).items()}
                        headers["X-Cache-Status"] = "Miss-Locked-Found" # Custom status
                        content_bytes = json.dumps(response_content).encode("utf-8")
                        headers["Content-Length"] = str(len(content_bytes))
                        return Response(
                             content=content_bytes,
                             status_code=response_status,
                             headers=headers,
                             media_type="application/json",
                             background=None # No background task needed here
                         )

                except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                    logger.error(f"Error reading existing cache file {current_cache_file} within lock: {e}")
                    response_content = ERROR_PLACEHOLDER
                    response_status = 500


        # Lock is released automatically here
        # Return the 202 Accepted response (either for 'miss' or 'processing')
        final_content = json.dumps(response_content or {}).encode("utf-8") # Ensure content is not None
        response = Response(
            content=final_content,
            status_code=response_status,
            headers={"X-Cache-Status": "Miss" if response_status == 202 and response_content==PROCESSING_PLACEHOLDER else "Processing",
                     "Content-Length": str(len(final_content))},
            media_type="application/json",
        )
        # Attach background tasks (will be empty if placeholder existed)
        response.background = background_tasks
        return response


    def _serve_cached_response(self, cache_file: str, cache_status: str, request_path=None):
        """Serves a response from a cache file with the specified cache status."""
        logger.info(f"Serving {cache_status.lower()} data from {cache_file}")
        try:
            with open(cache_file, "r") as f:
                response_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            logger.error(f"Error reading cache file {cache_file}: {e}. Returning cache miss.")
            # Simulate a miss scenario if cache file is bad
            content = json.dumps({"result": "error", "message": "Failed to read cache file."}).encode("utf-8")
            return Response(
                content=content,
                status_code=500, # Internal Server Error
                headers={"X-Cache-Status": "Error", "Content-Length": str(len(content))},
                media_type="application/json",
            )

        # Check if the cached content is a placeholder
        content_data = response_data.get("content", {})
        if isinstance(content_data, dict) and content_data.get("result") == "processing":
            logger.info(f"Cache file {cache_file} contains 'processing' placeholder. Returning 202.")
            content_bytes = json.dumps(content_data).encode("utf-8")
            return Response(
                content=content_bytes,
                status_code=202, # Accepted
                headers={"X-Cache-Status": "Processing", "Content-Length": str(len(content_bytes))},
                media_type="application/json",
            )
        elif isinstance(content_data, dict) and content_data.get("result") == "error":
            logger.warning(f"Cache file {cache_file} contains 'error' placeholder. Returning 500.")
            content_bytes = json.dumps(content_data).encode("utf-8")
            return Response(
                content=content_bytes,
                status_code=500, # Internal Server Error
                headers={"X-Cache-Status": "Error", "Content-Length": str(len(content_bytes))},
                media_type="application/json",
            )

        # --- Serve actual cached data ---
        content = json.dumps(content_data).encode("utf-8") # Add default for content
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
                self._log_perp_positions_details(content_data, cache_status, cache_file, request_path)
        except Exception as e:
            logger.error(f"Error during specific logging for perp positions: {e}")
            # Continue serving the response even if logging fails

        # DO NOT run cleanup on every cache hit
        # self.cleanup_old_cache_files() # MOVED

        return Response(
            content=content,
            status_code=response_data.get("status_code", 200), # Add default for status_code
            headers=headers,
            media_type="application/json",
        )

    def _log_perp_positions_details(self, data, cache_status, cache_file, request_path):
         """Helper method to log details specifically for the largest_perp_positions endpoint."""
         positions_returned = 0
         if isinstance(data, list):
             positions_returned = len(data)
         elif isinstance(data, dict) and "Market Index" in data:
             positions_returned = len(data.get("Market Index", []))

         summary_message = f"[{cache_status}] ({request_path}): Found {positions_returned} cached positions. Details suppressed to reduce log noise."
         
         # Log summary to console
         print(f"\n======== CACHED POSITIONS SUMMARY ({cache_status}) ========")
         print(summary_message)
         print(f"Cache file: {cache_file}")
         print("======== END CACHED POSITIONS SUMMARY ========\n")
         
         # Log summary to standard logger
         log = logging.getLogger("backend.api.health")
         log.info(summary_message)


    async def _fetch_and_cache(
        self,
        request: BackendRequest,
        call_next: Callable,
        cache_key: str,
        cache_file: str,
    ):
        """Fetches data using call_next and caches it. Should be called after a placeholder is written."""
        logger.info(f"Starting background fetch for {request.url.path} with key {cache_key}")
        try:
            response = await call_next(request) # This is the actual call to the API route handler

            if response.status_code == 200:
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                # Ensure response_body is not empty before trying to decode
                if not response_body:
                    logger.warning(f"Empty response body received for {request.url.path}. Cannot cache.")
                    # Overwrite placeholder with error state
                    with open(cache_file, "w") as f:
                        json.dump({"content": ERROR_PLACEHOLDER, "status_code": 500, "headers": {}}, f)
                    return

                try:
                    body_content = json.loads(response_body.decode())
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response for {request.url.path}: {e}. Response body: {response_body[:500]}")
                    # Overwrite placeholder with error state
                    with open(cache_file, "w") as f:
                        json.dump({"content": ERROR_PLACEHOLDER, "status_code": 500, "headers": {}}, f)
                    return

                response_data = {
                    "content": body_content,
                    "status_code": response.status_code,
                    "headers": {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() != "content-length"
                    },
                }

                # Overwrite placeholder with actual data
                with open(cache_file, "w") as f:
                    json.dump(response_data, f)

                logger.info(
                    f"Cached fresh data for {request.url.path} with query {request.url.query} to {cache_file}"
                )

                # Special handling for largest_perp_positions endpoint
                if request.url.path == "/api/health/largest_perp_positions":
                    try:
                        self._log_perp_positions_details(body_content, "FETCHED", cache_file, request.url.path)
                    except Exception as log_e:
                        logger.error(f"Error logging fetched perp positions details: {log_e}")

            else:
                logger.warning(
                    f"Failed to fetch data for {request.url.path}. Status code: {response.status_code}"
                )
                # Overwrite placeholder with error state or remove? Let's overwrite.
                with open(cache_file, "w") as f:
                    json.dump({"content": ERROR_PLACEHOLDER, "status_code": response.status_code, "headers": {}}, f)

            # Run cleanup AFTER successfully writing new cache or error placeholder
            self.cleanup_old_cache_files()

        except Exception as e:
            logger.error(
                f"Error in background task for {request.url.path} under key {cache_key}: {str(e)}"
            )
            # Attempt to overwrite placeholder with error state
            try:
                with open(cache_file, "w") as f:
                     json.dump({"content": ERROR_PLACEHOLDER, "status_code": 500, "headers": {}}, f)
            except Exception as write_e:
                 logger.error(f"Failed to write error placeholder for {cache_file}: {write_e}")
            import traceback
            traceback.print_exc()
        finally:
             # Optional: Release lock if it was passed or managed differently
             pass


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
        # logger.info(f"Attempting to get previous {num_pickles} pickles") # Verbose logging
        if not hasattr(self.state, 'current_pickle_path') or self.state.current_pickle_path is None:
            logger.error("Cannot get previous pickles: current_pickle_path is not initialized")
            return []

        parent_dir = os.path.dirname(self.state.current_pickle_path)
        if not parent_dir or not os.path.exists(parent_dir):
            logger.warning(f"Parent directory '{parent_dir}' for pickles not found or invalid.")
            parent_dir = "pickles" # Fallback assumption
            if not os.path.exists(parent_dir):
                logger.error(f"Pickle directory '{parent_dir}' not found. Cannot find previous pickles.")
                return []

        try:
            all_pickle_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        except OSError as e:
            logger.error(f"Error listing pickle directories in '{parent_dir}': {e}")
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
            logger.error(f"Error sorting pickle paths: {e}. Paths found: {all_pickle_dirs}")
            return []

        previous_pickles = sorted_paths[:num_pickles]
        # logger.info(f"Previous {len(previous_pickles)} pickles found: %s", previous_pickles) # Verbose logging
        return previous_pickles


    def cleanup_old_cache_files(self, keep_newest: int = 30):
        """Deletes all but the newest N cache files based on modification time."""
        logger.debug(f"Running cache cleanup, keeping newest {keep_newest} files.") # Optional debug log
        for cache_dir in [self.cache_dir, self.ucache_dir]:
            if not os.path.exists(cache_dir):
                continue
            try:
                files = glob.glob(f"{cache_dir}/*.json")
                # Filter out placeholder files older than, say, 1 hour to prevent them lingering
                placeholder_expiry = time.time() - 3600
                files_to_consider = []
                for f_path in files:
                    try:
                        mtime = os.path.getmtime(f_path)
                        # Check if it might be a placeholder and is expired
                        is_potentially_stale_placeholder = False
                        if mtime < placeholder_expiry:
                            try:
                                with open(f_path, "r") as f_content:
                                    # Load the *entire* structure written to the file
                                    data_in_file = json.load(f_content)
                                    # Check if the 'content' field *is* the processing placeholder dict
                                    if isinstance(data_in_file.get("content"), dict) and \
                                       data_in_file["content"].get("result") == "processing":
                                        is_potentially_stale_placeholder = True
                            except (json.JSONDecodeError, OSError, FileNotFoundError, KeyError):
                                # If we can't read it properly, or it doesn't match expected structure,
                                # treat it as potentially stale/corrupt if old
                                logger.warning(f"Could not properly read or parse potentially stale file {f_path}. Considering for removal.")
                                is_potentially_stale_placeholder = True # Mark for removal if old

                        if is_potentially_stale_placeholder:
                            logger.warning(f"Removing potentially stale placeholder cache file: {f_path}")
                            os.remove(f_path)
                        else:
                             files_to_consider.append(f_path)
                    except OSError as e:
                        logger.error(f"Error accessing file during cleanup check {f_path}: {e}")
                        # Decide whether to include potentially problematic files for deletion sort
                        # files_to_consider.append(f_path) # Or skip

                # Sort the remaining valid files by modification time
                files_to_consider.sort(key=os.path.getmtime, reverse=True)

                files_to_delete = files_to_consider[keep_newest:]
                if files_to_delete:
                    logger.info(f"Removing {len(files_to_delete)} old cache files from {cache_dir}.") # Optional info log
                    for old_file in files_to_delete:
                        try:
                            os.remove(old_file)
                            logger.debug(f"Removed old cache file: {old_file}") # Optional debug log
                        except Exception as e:
                            logger.error(f"Failed to remove {old_file}: {e}")
            except Exception as e:
                logger.error(f"Error during cache cleanup for {cache_dir}: {e}", exc_info=True) # Add exc_info