"""
Cache Manager - Simple caching utility for expensive operations.

This module provides basic caching functionality for video analysis
and other expensive operations to improve performance.
"""

import time
from typing import Any, Optional, Dict, Callable
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Simple in-memory cache manager with TTL support.
    
    In a production environment, this could be extended to use
    Redis or other persistent caching solutions.
    """
    
    def __init__(self, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"CacheManager initialized with default TTL: {default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if time.time() > entry['expires_at']:
            del self._cache[key]
            return None
        
        logger.debug(f"Cache hit for key: {key}")
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
        
        logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key existed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Deleted cache key: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        active_entries = sum(
            1 for entry in self._cache.values()
            if current_time <= entry['expires_at']
        )
        expired_entries = len(self._cache) - active_entries
        
        return {
            'total_entries': len(self._cache),
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'default_ttl': self.default_ttl
        }
    
    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a deterministic string from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def cached(self, ttl: Optional[int] = None, key_func: Optional[Callable] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time-to-live for cached result
            key_func: Function to generate cache key (uses default if None)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}_{self.generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager