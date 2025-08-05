"""
Unit tests for CacheManager.

Tests the CacheManager class for caching expensive operations and API responses
with comprehensive mocking strategies.
"""

import pytest
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from ai_video_editor.core.cache_manager import (
    CacheManager, CacheEntry, LRUCache, cached
)


class TestCacheEntry:
    """Test CacheEntry functionality."""
    
    def test_cache_entry_creation(self):
        """Test creating a CacheEntry."""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test_value"},
            ttl=3600,
            tags=["test", "cache"]
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test_value"}
        assert entry.ttl == 3600
        assert entry.tags == ["test", "cache"]
        assert entry.access_count == 0
        assert isinstance(entry.created_at, datetime)
        assert entry.last_accessed == entry.created_at
    
    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Create entry with short TTL
        entry = CacheEntry("test_key", "test_value", ttl=1)
        
        # Should not be expired immediately
        assert not entry.is_expired
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        assert entry.is_expired
    
    def test_cache_entry_no_expiration(self):
        """Test cache entry without TTL (no expiration)."""
        entry = CacheEntry("test_key", "test_value", ttl=None)
        
        # Should never expire
        assert not entry.is_expired
        
        # Even after time passes
        time.sleep(0.1)
        assert not entry.is_expired
    
    def test_cache_entry_access(self):
        """Test accessing cache entry."""
        entry = CacheEntry("test_key", "test_value")
        
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count
        
        # Access the entry
        time.sleep(0.01)  # Small delay to ensure time difference
        value = entry.access()
        
        assert value == "test_value"
        assert entry.access_count == initial_count + 1
        assert entry.last_accessed > initial_access_time
    
    def test_cache_entry_serialization(self):
        """Test cache entry serialization."""
        entry = CacheEntry(
            key="test_key",
            value={"complex": "data", "number": 42},
            ttl=3600,
            tags=["test"]
        )
        
        # Access to update stats
        entry.access()
        
        # Test to_dict
        entry_dict = entry.to_dict()
        
        assert entry_dict["key"] == "test_key"
        assert entry_dict["value"] == {"complex": "data", "number": 42}
        assert entry_dict["ttl"] == 3600
        assert entry_dict["tags"] == ["test"]
        assert entry_dict["access_count"] == 1
        
        # Test from_dict
        restored_entry = CacheEntry.from_dict(entry_dict)
        
        assert restored_entry.key == entry.key
        assert restored_entry.value == entry.value
        assert restored_entry.ttl == entry.ttl
        assert restored_entry.tags == entry.tags
        assert restored_entry.access_count == entry.access_count


class TestLRUCache:
    """Test LRUCache functionality."""
    
    def test_lru_cache_creation(self):
        """Test creating LRUCache."""
        cache = LRUCache(max_size=100)
        
        assert cache.max_size == 100
        assert cache.size() == 0
        assert len(cache.keys()) == 0
    
    def test_lru_cache_put_and_get(self):
        """Test putting and getting from LRU cache."""
        cache = LRUCache(max_size=3)
        
        entry1 = CacheEntry("key1", "value1")
        entry2 = CacheEntry("key2", "value2")
        
        # Put entries
        cache.put("key1", entry1)
        cache.put("key2", entry2)
        
        assert cache.size() == 2
        
        # Get entries
        retrieved1 = cache.get("key1")
        retrieved2 = cache.get("key2")
        
        assert retrieved1 is entry1
        assert retrieved2 is entry2
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction when over capacity."""
        cache = LRUCache(max_size=2)
        
        entry1 = CacheEntry("key1", "value1")
        entry2 = CacheEntry("key2", "value2")
        entry3 = CacheEntry("key3", "value3")
        
        # Fill cache to capacity
        cache.put("key1", entry1)
        cache.put("key2", entry2)
        assert cache.size() == 2
        
        # Add third entry (should evict oldest)
        cache.put("key3", entry3)
        assert cache.size() == 2
        
        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is entry2
        assert cache.get("key3") is entry3
    
    def test_lru_cache_access_order(self):
        """Test LRU cache access order."""
        cache = LRUCache(max_size=2)
        
        entry1 = CacheEntry("key1", "value1")
        entry2 = CacheEntry("key2", "value2")
        entry3 = CacheEntry("key3", "value3")
        
        # Fill cache
        cache.put("key1", entry1)
        cache.put("key2", entry2)
        
        # Access key1 (makes it most recently used)
        cache.get("key1")
        
        # Add key3 (should evict key2, not key1)
        cache.put("key3", entry3)
        
        assert cache.get("key1") is entry1  # Should still be there
        assert cache.get("key2") is None    # Should be evicted
        assert cache.get("key3") is entry3  # Should be there
    
    def test_lru_cache_expired_entries(self):
        """Test LRU cache with expired entries."""
        cache = LRUCache(max_size=10)
        
        # Create entry with short TTL
        entry = CacheEntry("key1", "value1", ttl=1)
        cache.put("key1", entry)
        
        # Should be retrievable immediately
        assert cache.get("key1") is entry
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should return None for expired entry
        assert cache.get("key1") is None
        
        # Entry should be removed from cache
        assert cache.size() == 0
    
    def test_lru_cache_remove(self):
        """Test removing entries from LRU cache."""
        cache = LRUCache(max_size=10)
        
        entry = CacheEntry("key1", "value1")
        cache.put("key1", entry)
        
        assert cache.size() == 1
        
        # Remove existing entry
        success = cache.remove("key1")
        assert success is True
        assert cache.size() == 0
        assert cache.get("key1") is None
        
        # Remove non-existent entry
        success = cache.remove("nonexistent")
        assert success is False
    
    def test_lru_cache_clear(self):
        """Test clearing LRU cache."""
        cache = LRUCache(max_size=10)
        
        # Add multiple entries
        for i in range(5):
            entry = CacheEntry(f"key{i}", f"value{i}")
            cache.put(f"key{i}", entry)
        
        assert cache.size() == 5
        
        # Clear cache
        cache.clear()
        
        assert cache.size() == 0
        assert len(cache.keys()) == 0


class TestCacheManager:
    """Test CacheManager functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager with temporary directory."""
        return CacheManager(cache_dir=temp_cache_dir, max_memory_entries=10)
    
    def test_cache_manager_initialization(self, temp_cache_dir):
        """Test CacheManager initialization."""
        manager = CacheManager(cache_dir=temp_cache_dir, max_memory_entries=5)
        
        assert manager.cache_dir.exists()
        assert manager.memory_cache.max_size == 5
        assert manager.stats['hits'] == 0
        assert manager.stats['misses'] == 0
    
    def test_cache_manager_put_and_get(self, cache_manager):
        """Test basic put and get operations."""
        test_data = {"key": "value", "number": 42}
        
        # Put data in cache
        cache_manager.put("test_key", test_data, ttl=3600, tags=["test"])
        
        # Get data from cache
        retrieved_data = cache_manager.get("test_key")
        
        assert retrieved_data == test_data
        assert cache_manager.stats['hits'] == 1
        assert cache_manager.stats['misses'] == 0
        assert cache_manager.stats['puts'] == 1
    
    def test_cache_manager_miss(self, cache_manager):
        """Test cache miss."""
        retrieved_data = cache_manager.get("nonexistent_key")
        
        assert retrieved_data is None
        assert cache_manager.stats['hits'] == 0
        assert cache_manager.stats['misses'] == 1
    
    def test_cache_manager_ttl_expiration(self, cache_manager):
        """Test TTL expiration."""
        # Put data with short TTL
        cache_manager.put("short_ttl_key", "test_value", ttl=1)
        
        # Should be retrievable immediately
        assert cache_manager.get("short_ttl_key") == "test_value"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should return None after expiration
        assert cache_manager.get("short_ttl_key") is None
    
    def test_cache_api_response(self, cache_manager):
        """Test caching API responses."""
        service = "gemini"
        endpoint = "analyze_content"
        params = {"text": "test content", "language": "en"}
        response = {"analysis": "test result", "confidence": 0.95}
        cost = 0.25
        
        # Cache API response
        cache_manager.cache_api_response(service, endpoint, params, response, cost)
        
        # Retrieve API response
        retrieved_response = cache_manager.get_api_response(service, endpoint, params)
        
        assert retrieved_response == response
        assert cache_manager.stats['api_cost_saved'] == cost
    
    def test_cache_api_response_different_params(self, cache_manager):
        """Test API response caching with different parameters."""
        service = "gemini"
        endpoint = "analyze_content"
        
        params1 = {"text": "content1"}
        params2 = {"text": "content2"}
        
        response1 = {"result": "analysis1"}
        response2 = {"result": "analysis2"}
        
        # Cache responses with different params
        cache_manager.cache_api_response(service, endpoint, params1, response1)
        cache_manager.cache_api_response(service, endpoint, params2, response2)
        
        # Should retrieve correct responses
        assert cache_manager.get_api_response(service, endpoint, params1) == response1
        assert cache_manager.get_api_response(service, endpoint, params2) == response2
    
    def test_cache_keyword_research(self, cache_manager):
        """Test caching keyword research results."""
        concepts = ["finance", "investment", "education"]
        content_type = "educational"
        research_result = {
            "primary_keywords": ["financial education", "investment basics"],
            "trending_hashtags": ["#finance", "#investing"]
        }
        
        # Cache keyword research
        cache_manager.cache_keyword_research(concepts, content_type, research_result)
        
        # Retrieve keyword research
        retrieved_result = cache_manager.get_keyword_research(concepts, content_type)
        
        assert retrieved_result == research_result
    
    def test_cache_processing_result(self, cache_manager):
        """Test caching processing results."""
        context_id = "test_context_123"
        module_name = "audio_analysis"
        stage = "transcription"
        result = {"transcript": "test transcript", "confidence": 0.92}
        
        # Cache processing result
        cache_manager.cache_processing_result(context_id, module_name, stage, result)
        
        # Retrieve processing result
        retrieved_result = cache_manager.get_processing_result(context_id, module_name, stage)
        
        assert retrieved_result == result
    
    def test_invalidate_by_tag(self, cache_manager):
        """Test invalidating cache entries by tag."""
        # Cache entries with different tags
        cache_manager.put("key1", "value1", tags=["tag1", "common"])
        cache_manager.put("key2", "value2", tags=["tag2", "common"])
        cache_manager.put("key3", "value3", tags=["tag3"])
        
        # Verify all entries are cached
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"
        assert cache_manager.get("key3") == "value3"
        
        # Invalidate entries with "common" tag
        invalidated_count = cache_manager.invalidate_by_tag("common")
        
        assert invalidated_count >= 2
        
        # Entries with "common" tag should be invalidated
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        
        # Entry without "common" tag should remain
        assert cache_manager.get("key3") == "value3"
    
    def test_invalidate_context(self, cache_manager):
        """Test invalidating cache entries for a specific context."""
        context_id = "test_context_123"
        
        # Cache processing results for the context
        cache_manager.cache_processing_result(context_id, "module1", "stage1", "result1")
        cache_manager.cache_processing_result(context_id, "module2", "stage2", "result2")
        cache_manager.cache_processing_result("other_context", "module1", "stage1", "result3")
        
        # Verify all results are cached
        assert cache_manager.get_processing_result(context_id, "module1", "stage1") == "result1"
        assert cache_manager.get_processing_result(context_id, "module2", "stage2") == "result2"
        assert cache_manager.get_processing_result("other_context", "module1", "stage1") == "result3"
        
        # Invalidate context
        invalidated_count = cache_manager.invalidate_context(context_id)
        
        assert invalidated_count >= 2
        
        # Context entries should be invalidated
        assert cache_manager.get_processing_result(context_id, "module1", "stage1") is None
        assert cache_manager.get_processing_result(context_id, "module2", "stage2") is None
        
        # Other context should remain
        assert cache_manager.get_processing_result("other_context", "module1", "stage1") == "result3"
    
    def test_clear_expired(self, cache_manager):
        """Test clearing expired cache entries."""
        # Add entries with different TTLs
        cache_manager.put("persistent", "value1", ttl=None)  # No expiration
        cache_manager.put("long_ttl", "value2", ttl=3600)    # 1 hour
        cache_manager.put("short_ttl", "value3", ttl=1)      # 1 second
        
        # Wait for short TTL to expire
        time.sleep(1.1)
        
        # Clear expired entries
        cleared_count = cache_manager.clear_expired()
        
        assert cleared_count == 1
        
        # Check remaining entries
        assert cache_manager.get("persistent") == "value1"
        assert cache_manager.get("long_ttl") == "value2"
        assert cache_manager.get("short_ttl") is None
    
    def test_get_stats(self, cache_manager):
        """Test getting cache statistics."""
        # Perform some cache operations
        cache_manager.put("key1", "value1")
        cache_manager.put("key2", "value2")
        cache_manager.get("key1")  # Hit
        cache_manager.get("key1")  # Hit
        cache_manager.get("nonexistent")  # Miss
        
        stats = cache_manager.get_stats()
        
        assert stats['memory_cache_size'] == 2
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['puts'] == 2
        assert stats['hit_rate'] == 2/3  # 2 hits out of 3 total requests
        assert 'cache_directory' in stats
    
    def test_get_storage_usage(self, cache_manager):
        """Test getting storage usage statistics."""
        # Add some entries to create persistent storage
        cache_manager.put("key1", "value1", persist=True)
        cache_manager.put("key2", "value2", persist=True)
        
        usage = cache_manager.get_storage_usage()
        
        assert 'total_size_bytes' in usage
        assert 'total_size_mb' in usage
        assert 'file_count' in usage
        assert 'cache_directory' in usage
        
        assert usage['total_size_bytes'] > 0
        assert usage['file_count'] >= 2
    
    def test_cleanup_storage(self, cache_manager):
        """Test cleaning up old storage files."""
        # Add entries to create files
        cache_manager.put("key1", "value1", persist=True)
        time.sleep(0.1)  # Ensure different modification times
        cache_manager.put("key2", "value2", persist=True)
        
        # Cleanup with 0 max age (should remove all files)
        cleaned_count = cache_manager.cleanup_storage(max_age_days=0)
        
        assert cleaned_count >= 2
        
        # Storage usage should be reduced
        usage = cache_manager.get_storage_usage()
        assert usage['file_count'] == 0
    
    def test_memory_and_persistent_cache_interaction(self, cache_manager):
        """Test interaction between memory and persistent cache."""
        test_data = {"complex": "data", "with": ["nested", "structures"]}
        
        # Put data in cache (both memory and persistent)
        cache_manager.put("test_key", test_data, ttl=3600, persist=True)
        
        # Should be in memory cache
        assert cache_manager.get("test_key") == test_data
        
        # Clear memory cache
        cache_manager.memory_cache.clear()
        
        # Should still be retrievable from persistent storage
        assert cache_manager.get("test_key") == test_data
        
        # Should now be back in memory cache
        assert cache_manager.memory_cache.get("test_key") is not None
    
    def test_concurrent_access(self, cache_manager):
        """Test concurrent access to cache manager."""
        results = []
        errors = []
        
        def cache_worker(worker_id):
            try:
                # Each worker caches and retrieves data
                key = f"worker_{worker_id}"
                value = f"data_{worker_id}"
                
                cache_manager.put(key, value)
                retrieved = cache_manager.get(key)
                
                if retrieved == value:
                    results.append(worker_id)
                else:
                    errors.append(f"Worker {worker_id}: expected {value}, got {retrieved}")
                    
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert set(results) == set(range(10))


class TestCachedDecorator:
    """Test cached decorator functionality."""
    
    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create CacheManager for decorator testing."""
        return CacheManager(cache_dir=str(tmp_path), max_memory_entries=10)
    
    def test_cached_decorator_basic(self, cache_manager):
        """Test basic cached decorator functionality."""
        call_count = 0
        
        @cached(cache_manager, ttl=3600, key_prefix="test_func")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Call with different args should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_cached_decorator_with_kwargs(self, cache_manager):
        """Test cached decorator with keyword arguments."""
        call_count = 0
        
        @cached(cache_manager, ttl=3600, key_prefix="test_kwargs")
        def function_with_kwargs(a, b=10, c=20):
            nonlocal call_count
            call_count += 1
            return a + b + c
        
        # Calls with same effective arguments should use cache
        result1 = function_with_kwargs(a=1, b=10, c=20)
        result2 = function_with_kwargs(a=1, b=10, c=20)
        result3 = function_with_kwargs(a=1)  # Uses defaults, but should be a different call
        
        assert result1 == result2 == 31
        assert result3 == 31
        assert call_count == 2  # Should execute twice because of kwarg ambiguity
    
    def test_cached_decorator_ttl_expiration(self, cache_manager):
        """Test cached decorator with TTL expiration."""
        call_count = 0
        
        @cached(cache_manager, ttl=1, key_prefix="test_ttl")
        def function_with_ttl(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = function_with_ttl(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call immediately (should use cache)
        result2 = function_with_ttl(5)
        assert result2 == 10
        assert call_count == 1
        
        # Wait for TTL expiration
        time.sleep(1.1)
        
        # Third call after expiration (should execute function)
        result3 = function_with_ttl(5)
        assert result3 == 10
        assert call_count == 2
    
    def test_cached_decorator_with_tags(self, cache_manager):
        """Test cached decorator with tags."""
        @cached(cache_manager, ttl=3600, tags=["test_tag"], key_prefix="test_tags")
        def tagged_function(x):
            return x ** 2
        
        # Call function to cache result
        result = tagged_function(4)
        assert result == 16
        
        # Verify result is cached
        cached_result = tagged_function(4)
        assert cached_result == 16
        
        # Invalidate by tag
        cache_manager.invalidate_by_tag("test_tag")
        
        # Function should be called again after invalidation
        # (We can't easily test this without modifying the function,
        # but we can verify the cache entry is gone)
        stats_before = cache_manager.get_stats()
        tagged_function(4)  # This should be a cache miss now
        stats_after = cache_manager.get_stats()
        
        assert stats_after['misses'] > stats_before['misses']