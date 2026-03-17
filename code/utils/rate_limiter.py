"""
Rate limiter to prevent API abuse and excessive costs.

Uses sliding window algorithm to limit requests per session/IP.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Tuple
import threading


class RateLimiter:
    """
    Sliding window rate limiter.
    
    Example:
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        if limiter.is_allowed("session_123"):
            # Process request
            pass
        else:
            # Reject with 429 Too Many Requests
            raise HTTPException(429, "Rate limit exceeded")
    """
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window (default: 10)
            window_seconds: Time window in seconds (default: 60 seconds)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Store: {identifier: deque([timestamp1, timestamp2, ...])}
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        
        # Metrics
        self.total_requests = 0
        self.blocked_requests = 0
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed for given identifier.
        
        Args:
            identifier: Unique identifier (session_id, IP, user_id, etc.)
        
        Returns:
            (allowed: bool, info: dict)
            
        Example:
            allowed, info = limiter.is_allowed("session_123")
            # info = {"remaining": 7, "reset_in": 42, "limit": 10}
        """
        current_time = time.time()
        
        with self._lock:
            self.total_requests += 1
            
            # Get request timestamps for this identifier
            timestamps = self._requests[identifier]
            
            # Remove expired timestamps (outside window)
            cutoff_time = current_time - self.window_seconds
            while timestamps and timestamps[0] < cutoff_time:
                timestamps.popleft()
            
            # Check if under limit
            if len(timestamps) < self.max_requests:
                timestamps.append(current_time)
                
                # Calculate reset time (when oldest request expires)
                reset_in = 0
                if timestamps:
                    oldest = timestamps[0]
                    reset_in = int(self.window_seconds - (current_time - oldest))
                
                return True, {
                    "allowed": True,
                    "remaining": self.max_requests - len(timestamps),
                    "reset_in": max(0, reset_in),
                    "limit": self.max_requests,
                    "window": self.window_seconds
                }
            else:
                self.blocked_requests += 1
                
                # Calculate when oldest request will expire
                oldest = timestamps[0]
                reset_in = int(self.window_seconds - (current_time - oldest) + 1)
                
                return False, {
                    "allowed": False,
                    "remaining": 0,
                    "reset_in": max(0, reset_in),
                    "limit": self.max_requests,
                    "window": self.window_seconds,
                    "retry_after": max(1, reset_in)
                }
    
    def reset(self, identifier: str) -> None:
        """Reset rate limit for specific identifier."""
        with self._lock:
            if identifier in self._requests:
                del self._requests[identifier]
    
    def clear_all(self) -> None:
        """Clear all rate limit data."""
        with self._lock:
            self._requests.clear()
            self.total_requests = 0
            self.blocked_requests = 0
    
    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            {
                "total_requests": 1234,
                "blocked_requests": 56,
                "block_rate": 0.045,
                "active_identifiers": 42,
                "max_requests": 10,
                "window_seconds": 60
            }
        """
        with self._lock:
            block_rate = 0.0
            if self.total_requests > 0:
                block_rate = self.blocked_requests / self.total_requests
            
            return {
                "total_requests": self.total_requests,
                "blocked_requests": self.blocked_requests,
                "block_rate": round(block_rate, 3),
                "active_identifiers": len(self._requests),
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds
            }


# Global rate limiter instance
_global_limiter: RateLimiter = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance (singleton)."""
    global _global_limiter
    if _global_limiter is None:
        # Default: 10 requests per minute per session
        _global_limiter = RateLimiter(max_requests=10, window_seconds=60)
    return _global_limiter
