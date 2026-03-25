"""
Unit tests for RateLimiter.
"""

import pytest
import time
from code.utils.rate_limiter import RateLimiter, get_rate_limiter


class TestRateLimiter:
    """Test suite for RateLimiter."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        assert limiter.max_requests == 10
        assert limiter.window_seconds == 60
        assert limiter.total_requests == 0
        assert limiter.blocked_requests == 0
    
    def test_allow_under_limit(self):
        """Test requests are allowed when under limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for i in range(5):
            allowed, info = limiter.is_allowed("user_1")
            assert allowed is True
            assert info["remaining"] == 4 - i
            assert info["allowed"] is True
    
    def test_block_over_limit(self):
        """Test requests are blocked when over limit."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        # Fill the limit
        for _ in range(3):
            allowed, info = limiter.is_allowed("user_1")
            assert allowed is True
        
        # Next request should be blocked
        allowed, info = limiter.is_allowed("user_1")
        assert allowed is False
        assert info["allowed"] is False
        assert info["remaining"] == 0
        assert "retry_after" in info
        assert limiter.blocked_requests == 1
    
    def test_sliding_window(self):
        """Test sliding window behavior."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        
        # Make 2 requests (fill limit)
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_1")
        
        # Should be blocked
        allowed, info = limiter.is_allowed("user_1")
        assert allowed is False
        
        # Wait for window to slide
        time.sleep(1.1)
        
        # Should be allowed again
        allowed, info = limiter.is_allowed("user_1")
        assert allowed is True
    
    def test_different_identifiers(self):
        """Test rate limits are separate per identifier."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # User 1 makes 2 requests
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_1")
        
        # User 1 blocked
        allowed, _ = limiter.is_allowed("user_1")
        assert allowed is False
        
        # User 2 should still be allowed
        allowed, _ = limiter.is_allowed("user_2")
        assert allowed is True
    
    def test_reset_identifier(self):
        """Test resetting rate limit for specific identifier."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        # Fill limit
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_1")
        
        # Blocked
        allowed, _ = limiter.is_allowed("user_1")
        assert allowed is False
        
        # Reset
        limiter.reset("user_1")
        
        # Should be allowed again
        allowed, _ = limiter.is_allowed("user_1")
        assert allowed is True
    
    def test_clear_all(self):
        """Test clearing all rate limit data."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_2")
        limiter.is_allowed("user_3")
        
        stats = limiter.get_stats()
        assert stats["total_requests"] == 3
        assert stats["active_identifiers"] == 3
        
        limiter.clear_all()
        
        stats = limiter.get_stats()
        assert stats["total_requests"] == 0
        assert stats["active_identifiers"] == 0
    
    def test_get_stats(self):
        """Test statistics tracking."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_1")
        limiter.is_allowed("user_1")  # blocked
        limiter.is_allowed("user_2")
        
        stats = limiter.get_stats()
        
        assert stats["total_requests"] == 4
        assert stats["blocked_requests"] == 1
        assert stats["block_rate"] == 0.25  # 1/4
        assert stats["active_identifiers"] == 2
        assert stats["max_requests"] == 2
        assert stats["window_seconds"] == 60
    
    def test_retry_after_calculation(self):
        """Test retry_after is calculated correctly."""
        limiter = RateLimiter(max_requests=1, window_seconds=10)
        
        # First request
        limiter.is_allowed("user_1")
        
        # Second request (blocked)
        allowed, info = limiter.is_allowed("user_1")
        
        assert allowed is False
        assert "retry_after" in info
        assert 0 < info["retry_after"] <= 10
    
    def test_global_rate_limiter_singleton(self):
        """Test get_rate_limiter returns singleton instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
        
        # Test that state persists
        limiter1.is_allowed("user_1")
        stats = limiter2.get_stats()
        
        assert stats["total_requests"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
