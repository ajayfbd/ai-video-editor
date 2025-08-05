"""
CacheManager - Manages caching for expensive operations and API responses.

This module provides intelligent caching for expensive operations including
API responses, keyword research, and processing results to optimize
performance and reduce costs.
"""

import json
import logging
import hashlib
import pickle
import time
from typing import Any, Dict, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
import threading
from collections import OrderedDict

from .content_context import ContentContext
from .exceptions import ContentContextError


logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None,
                 tags: Optional[List[str]] = None):
        """
        Initialize cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds
            tags: Optional tags for cache invalidation
        """
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
        self.tags = tags or []
        self.access_count = 0
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def access(self) -> Any:
        """Access cached value and update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'ttl': self.ttl,
            'tags': self.tags,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary."""
        entry = cls(
            key=data['key'],
            value=data['value'],
            ttl=data.get('ttl'),
            tags=data.get('tags', [])
        )
        entry.created_at = datetime.fromisoformat(data['created_at'])
        entry.access_count = data.get('access_count', 0)
        entry.last_accessed = datetime.fromisoformat(data.get('last_accessed', entry.created_at.isoformat()))
        return entry


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not entry.is_expired:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    return entry
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
    
    def put(self, key: str, entry: CacheEntry):
        """Put cache entry."""
        with self.lock:
            if key in self.cache:
                # Update existing entry
                self.cache[key] = entry
                self.cache.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = entry
                
                # Remove oldest entries if over capacity
                while len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
    
    def remove(self, key: str) -> bool:
        """Remove cache entry by key."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self.lock:
            return list(self.cache.keys())


class CacheManager:
    """
    Manages caching for expensive operations and API responses.
    
    Provides intelligent caching with TTL, LRU eviction, tag-based invalidation,
    and persistent storage for cache entries.
    """
    
    def __init__(self, cache_dir: str = "temp/cache", max_memory_entries: int = 1000):
        """
        Initialize CacheManager.
        
        Args:
            cache_dir: Directory for persistent cache storage
            max_memory_entries: Maximum entries in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed items
        self.memory_cache = LRUCache(max_memory_entries)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0,
            'evictions': 0,
            'api_cost_saved': 0.0
        }
        
        # Default TTL values for different cache types
        self.default_ttls = {
            'api_response': 3600,  # 1 hour
            'keyword_research': 86400,  # 24 hours
            'competitor_analysis': 604800,  # 7 days
            'content_analysis': 3600,  # 1 hour
            'processing_result': 86400,  # 24 hours
            'thumbnail_template': None,  # No expiration
            'user_preference': None  # No expiration
        }
        
        logger.info(f"CacheManager initialized with cache directory: {self.cache_dir}")
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from prefix and arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a deterministic hash from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
    
    def _get_persistent_path(self, key: str) -> Path:
        """Get persistent storage path for cache key."""
        # Sanitize key for use as a filename
        safe_key = key.replace(":", "_")
        
        # Use first two characters of key for directory structure
        subdir = safe_key[:2] if len(safe_key) >= 2 else "00"
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        
        return cache_subdir / f"{safe_key}.cache"
    
    def _load_from_persistent(self, key: str) -> Optional[CacheEntry]:
        """Load cache entry from persistent storage."""
        try:
            cache_file = self._get_persistent_path(key)
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                entry_data = pickle.load(f)
            
            entry = CacheEntry.from_dict(entry_data)
            
            # Check if expired
            if entry.is_expired:
                cache_file.unlink()  # Remove expired file
                return None
            
            return entry
            
        except Exception as e:
            logger.warning(f"Failed to load cache entry from persistent storage: {str(e)}")
            return None
    
    def _save_to_persistent(self, entry: CacheEntry):
        """Save cache entry to persistent storage."""
        try:
            cache_file = self._get_persistent_path(entry.key)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry.to_dict(), f)
                
        except Exception as e:
            logger.warning(f"Failed to save cache entry to persistent storage: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        # Try memory cache first
        entry = self.memory_cache.get(key)
        
        if entry is None:
            # Try persistent storage
            entry = self._load_from_persistent(key)
            
            if entry is not None:
                # Load into memory cache
                self.memory_cache.put(key, entry)
        
        if entry is not None:
            self.stats['hits'] += 1
            return entry.access()
        else:
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, 
            tags: Optional[List[str]] = None, persist: bool = True):
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            tags: Optional tags for invalidation
            persist: Whether to save to persistent storage
        """
        entry = CacheEntry(key, value, ttl, tags)
        
        # Store in memory cache
        self.memory_cache.put(key, entry)
        
        # Store in persistent cache if requested
        if persist:
            self._save_to_persistent(entry)
        
        self.stats['puts'] += 1
        logger.debug(f"Cached value with key: {key}")
    
    def cache_api_response(self, service: str, endpoint: str, params: Dict[str, Any],
                          response: Any, cost: float = 0.0):
        """
        Cache API response with automatic key generation.
        
        Args:
            service: API service name (e.g., 'gemini', 'imagen')
            endpoint: API endpoint
            params: API parameters
            response: API response to cache
            cost: API call cost for savings tracking
        """
        key = self._generate_key(f"api:{service}:{endpoint}", **params)
        ttl = self.default_ttls.get('api_response', 3600)
        tags = [f"service:{service}", f"endpoint:{endpoint}"]
        
        self.put(key, response, ttl=ttl, tags=tags)
        
        # Track cost savings
        self.stats['api_cost_saved'] += cost
        
        logger.debug(f"Cached API response for {service}:{endpoint}")
    
    def get_api_response(self, service: str, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached API response.
        
        Args:
            service: API service name
            endpoint: API endpoint
            params: API parameters
            
        Returns:
            Cached response or None if not found
        """
        key = self._generate_key(f"api:{service}:{endpoint}", **params)
        return self.get(key)
    
    def cache_keyword_research(self, concepts: List[str], content_type: str, 
                             research_result: Any):
        """
        Cache keyword research results.
        
        Args:
            concepts: Content concepts used for research
            content_type: Type of content
            research_result: Research results to cache
        """
        key = self._generate_key("keyword_research", concepts=concepts, content_type=content_type)
        ttl = self.default_ttls.get('keyword_research', 86400)
        tags = ["keyword_research", f"content_type:{content_type}"]
        
        self.put(key, research_result, ttl=ttl, tags=tags)
        logger.debug(f"Cached keyword research for {content_type} content")
    
    def get_keyword_research(self, concepts: List[str], content_type: str) -> Optional[Any]:
        """
        Get cached keyword research results.
        
        Args:
            concepts: Content concepts
            content_type: Type of content
            
        Returns:
            Cached research results or None if not found
        """
        key = self._generate_key("keyword_research", concepts=concepts, content_type=content_type)
        return self.get(key)
    
    def cache_processing_result(self, context_id: str, module_name: str, 
                              stage: str, result: Any):
        """
        Cache processing result for a specific context and module.
        
        Args:
            context_id: ContentContext project ID
            module_name: Name of processing module
            stage: Processing stage
            result: Processing result to cache
        """
        key = self._generate_key("processing", context_id=context_id, 
                                module=module_name, stage=stage)
        ttl = self.default_ttls.get('processing_result', 86400)
        tags = [f"context:{context_id}", f"module:{module_name}", f"stage:{stage}"]
        
        self.put(key, result, ttl=ttl, tags=tags)
        logger.debug(f"Cached processing result for {module_name}:{stage}")
    
    def get_processing_result(self, context_id: str, module_name: str, 
                            stage: str) -> Optional[Any]:
        """
        Get cached processing result.
        
        Args:
            context_id: ContentContext project ID
            module_name: Name of processing module
            stage: Processing stage
            
        Returns:
            Cached result or None if not found
        """
        key = self._generate_key("processing", context_id=context_id, 
                                module=module_name, stage=stage)
        return self.get(key)
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        invalidated_count = 0
        
        # Invalidate from memory cache
        keys_to_remove = []
        for key in self.memory_cache.keys():
            entry = self.memory_cache.get(key)
            if entry and tag in entry.tags:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.memory_cache.remove(key)
            invalidated_count += 1
        
        # Invalidate from persistent storage
        try:
            for cache_file in self.cache_dir.rglob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    if tag in entry_data.get('tags', []):
                        cache_file.unlink()
                        invalidated_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to check cache file {cache_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to invalidate persistent cache by tag: {str(e)}")
        
        logger.info(f"Invalidated {invalidated_count} cache entries with tag: {tag}")
        return invalidated_count
    
    def invalidate_context(self, context_id: str) -> int:
        """
        Invalidate all cache entries for a specific context.
        
        Args:
            context_id: ContentContext project ID
            
        Returns:
            Number of entries invalidated
        """
        return self.invalidate_by_tag(f"context:{context_id}")
    
    def clear_expired(self) -> int:
        """
        Clear all expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        cleared_count = 0
        
        # Clear from memory cache
        keys_to_remove = []
        for key in self.memory_cache.keys():
            entry = self.memory_cache.get(key)
            if entry and entry.is_expired:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.memory_cache.remove(key)
            cleared_count += 1
        
        # Clear from persistent storage
        try:
            for cache_file in self.cache_dir.rglob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry_data = pickle.load(f)
                    
                    entry = CacheEntry.from_dict(entry_data)
                    if entry.is_expired:
                        cache_file.unlink()
                        cleared_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to check cache file {cache_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to clear expired persistent cache: {str(e)}")
        
        logger.info(f"Cleared {cleared_count} expired cache entries")
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0.0
        
        return {
            'memory_cache_size': self.memory_cache.size(),
            'memory_cache_max_size': self.memory_cache.max_size,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'puts': self.stats['puts'],
            'evictions': self.stats['evictions'],
            'api_cost_saved': self.stats['api_cost_saved'],
            'cache_directory': str(self.cache_dir)
        }
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get cache storage usage information.
        
        Returns:
            Dictionary with storage usage statistics
        """
        try:
            total_size = 0
            file_count = 0
            
            for cache_file in self.cache_dir.rglob("*.cache"):
                total_size += cache_file.stat().st_size
                file_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'file_count': file_count,
                'cache_directory': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {str(e)}")
            return {'error': str(e)}
    
    def cleanup_storage(self, max_age_days: int = 30) -> int:
        """
        Clean up old cache files from persistent storage.
        
        Args:
            max_age_days: Maximum age in days for cache files
            
        Returns:
            Number of files cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            for cache_file in self.cache_dir.rglob("*.cache"):
                try:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        cache_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to clean cache file {cache_file}: {str(e)}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} old cache files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup storage: {str(e)}")
            return 0


def cached(cache_manager: CacheManager, ttl: Optional[int] = None, 
          tags: Optional[List[str]] = None, key_prefix: str = "func"):
    """
    Decorator for caching function results.
    
    Args:
        cache_manager: CacheManager instance
        ttl: Time to live in seconds
        tags: Cache tags
        key_prefix: Prefix for cache key
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_manager._generate_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for function {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(key, result, ttl=ttl, tags=tags)
            
            logger.debug(f"Cached result for function {func.__name__}")
            return result
        
        return wrapper
    return decorator