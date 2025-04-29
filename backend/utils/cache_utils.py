from functools import wraps
import time
from typing import Any, Callable, Dict, Optional, TypeVar
import logging

RT = TypeVar('RT')  # Return Type

class TTLCache:
    """Time-based cache implementation."""
    
    def __init__(self, ttl_seconds: int = 43200):  # Default 12 hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger(__name__)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                self.hits += 1
                self.logger.info(f"🟢 Cache HIT for key: {key[:100]}...")
                self.logger.debug(f"Cache stats - Hits: {self.hits}, Misses: {self.misses}, Hit Rate: {(self.hits/(self.hits+self.misses))*100:.1f}%")
                return entry['value']
            else:
                del self.cache[key]
                self.logger.info(f"🟡 Cache EXPIRED for key: {key[:100]}...")
        self.misses += 1
        self.logger.info(f"🔴 Cache MISS for key: {key[:100]}...")
        self.logger.debug(f"Cache stats - Hits: {self.hits}, Misses: {self.misses}, Hit Rate: {(self.hits/(self.hits+self.misses))*100:.1f}%")
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self.logger.info(f"🔵 Cache SET for key: {key[:100]}... (TTL: {self.ttl_seconds}s)")

def ttl_cache(ttl_seconds: int = 43200) -> Callable:
    """
    Decorator that implements a TTL cache for function results.
    
    Args:
        ttl_seconds (int): Time to live in seconds (default 12 hours)
    """
    cache = TTLCache(ttl_seconds)
    
    def decorator(func: Callable[..., RT]) -> Callable[..., RT]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> RT:
            # Create a cache key from function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Try to get from cache
            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value
            
            # If not in cache or expired, compute and store
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
            
        return wrapper
    return decorator 