import asyncio
import time
from functools import wraps
from threading import Lock

_cache = {}
_lock = Lock()


def cached_response(ttl_seconds: int = 300):
    """Simple TTL cache decorator for FastAPI route handlers.

    Caches based on the full URL path + query string.
    Returns cached result if within TTL, otherwise calls the handler.
    Works with both sync and async route handlers.
    """

    def decorator(func):
        is_async = asyncio.iscoroutinefunction(func)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            request = kwargs.get("request") or (args[0] if args else None)
            if request is None or not hasattr(request, "method"):
                # No request object (e.g. parameterless handler) — use function name
                key = f"fn:{func.__module__}.{func.__qualname__}"
            else:
                key = f"{request.method}:{request.url.path}:{request.url.query}"
            now = time.time()

            with _lock:
                if key in _cache:
                    result, cached_at = _cache[key]
                    if now - cached_at < ttl_seconds:
                        return result

            result = await func(*args, **kwargs)

            with _lock:
                _cache[key] = (result, now)
                _evict_expired(now, ttl_seconds)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            request = kwargs.get("request") or (args[0] if args else None)
            if request is None or not hasattr(request, "method"):
                key = f"fn:{func.__module__}.{func.__qualname__}"
            else:
                key = f"{request.method}:{request.url.path}:{request.url.query}"
            now = time.time()

            with _lock:
                if key in _cache:
                    result, cached_at = _cache[key]
                    if now - cached_at < ttl_seconds:
                        return result

            result = func(*args, **kwargs)

            with _lock:
                _cache[key] = (result, now)
                _evict_expired(now, ttl_seconds)

            return result

        return async_wrapper if is_async else sync_wrapper

    return decorator


def _evict_expired(now: float, ttl_seconds: int):
    if len(_cache) > 200:
        expired = [k for k, (_, t) in _cache.items() if now - t > ttl_seconds]
        for k in expired:
            del _cache[k]
