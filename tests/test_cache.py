"""
Unit tests for SimpleCache.
"""

import pytest
import time
from code.utils.simple_cache import SimpleCache, get_cache


class TestSimpleCache:
    """Test suite for SimpleCache."""
    
    def test_cache_initialization(self):
        """Test cache initializes with correct parameters."""
        cache = SimpleCache(max_size=100, ttl_seconds=300)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_set_and_get(self):
        """Test basic cache set/get operations."""
        cache = SimpleCache()
        
        session_id = "test_123"
        question = "จดทะเบียนร้านอาหารต้องทำอย่างไร"
        value = {"answer": "ทำตามขั้นตอน...", "tokens": 1234}
        
        # Set value
        cache.set(session_id, question, value)
        
        # Get value
        result = cache.get(session_id, question)
        
        assert result is not None
        assert result["answer"] == "ทำตามขั้นตอน..."
        assert result["tokens"] == 1234
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache returns None for non-existent keys."""
        cache = SimpleCache()
        
        result = cache.get("nonexistent", "question")
        
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_cache_ttl_expiration(self):
        """Test cache entries expire after TTL."""
        cache = SimpleCache(ttl_seconds=1)  # 1 second TTL
        
        cache.set("session_1", "question", {"answer": "test"})
        
        # Immediate get should work
        result = cache.get("session_1", "question")
        assert result is not None
        
        # Wait for TTL to expire
        time.sleep(1.1)
        
        # Should be expired now
        result = cache.get("session_1", "question")
        assert result is None
    
    def test_cache_different_personas(self):
        """Test cache differentiates between personas."""
        cache = SimpleCache()
        
        session_id = "session_1"
        question = "same question"
        
        cache.set(session_id, question, {"answer": "practical"}, persona="practical")
        cache.set(session_id, question, {"answer": "academic"}, persona="academic")
        
        practical_result = cache.get(session_id, question, persona="practical")
        academic_result = cache.get(session_id, question, persona="academic")
        
        assert practical_result["answer"] == "practical"
        assert academic_result["answer"] == "academic"
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when max_size is reached."""
        cache = SimpleCache(max_size=2)
        
        cache.set("s1", "q1", {"a": "1"})
        cache.set("s1", "q2", {"a": "2"})
        cache.set("s1", "q3", {"a": "3"})  # Should evict q1
        
        # q1 should be evicted
        assert cache.get("s1", "q1") is None
        
        # q2 and q3 should still exist
        assert cache.get("s1", "q2") is not None
        assert cache.get("s1", "q3") is not None
        
        assert cache.evictions == 1
    
    def test_cache_clear(self):
        """Test cache clear functionality."""
        cache = SimpleCache()
        
        cache.set("s1", "q1", {"a": "1"})
        cache.set("s1", "q2", {"a": "2"})
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        cache = SimpleCache(max_size=100, ttl_seconds=3600)
        
        cache.set("s1", "q1", {"a": "1"})
        cache.set("s1", "q2", {"a": "2"})
        
        cache.get("s1", "q1")  # hit
        cache.get("s1", "q2")  # hit
        cache.get("s1", "q3")  # miss
        
        stats = cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.667  # 2/3
        assert stats["ttl_seconds"] == 3600
    
    def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = SimpleCache(ttl_seconds=1)
        
        cache.set("s1", "q1", {"a": "1"})
        cache.set("s1", "q2", {"a": "2"})
        
        time.sleep(1.1)
        
        removed = cache.cleanup_expired()
        
        assert removed == 2
        assert cache.get_stats()["size"] == 0
    
    def test_global_cache_singleton(self):
        """Test get_cache returns singleton instance."""
        cache1 = get_cache()
        cache2 = get_cache()
        
        assert cache1 is cache2
        
        # Test that data persists across calls
        cache1.set("s1", "q1", {"a": "1"})
        result = cache2.get("s1", "q1")
        
        assert result is not None
        assert result["a"] == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
