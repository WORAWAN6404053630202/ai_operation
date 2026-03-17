"""
Redis-based distributed caching for Thai Regulatory AI.

Features:
- Distributed caching across multiple instances
- Connection pooling
- TTL management
- Cache warming strategies
- Pub/sub for cache invalidation
- Fallback to in-memory cache

Usage:
    from code.utils.redis_cache import RedisCache, get_redis_cache
    
    # Initialize
    cache = RedisCache(host="localhost", port=6379)
    
    # Set value
    cache.set("key", "value", ttl=3600)
    
    # Get value
    value = cache.get("key")
    
    # Delete
    cache.delete("key")
    
    # Pattern matching
    keys = cache.keys("user:*")
"""

import os
import json
import logging
import hashlib
from typing import Any, Optional, List, Dict
from datetime import timedelta
import redis
from redis import ConnectionPool, Redis
from redis.exceptions import RedisError, ConnectionError


logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based distributed cache.
    
    Provides distributed caching with automatic failover to in-memory cache.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = True,
        enable_fallback: bool = True
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            decode_responses: Decode responses to strings
            enable_fallback: Enable in-memory fallback
        """
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.enable_fallback = enable_fallback
        
        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "fallback_uses": 0
        }
        
        # In-memory fallback
        self._fallback_cache: Dict[str, Any] = {}
        
        # Initialize connection pool
        try:
            self.pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=max_connections,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                decode_responses=decode_responses
            )
            
            self.client = Redis(connection_pool=self.pool)
            
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected: {self.host}:{self.port}/{self.db}")
            self.is_available = True
            
        except (RedisError, ConnectionError) as e:
            logger.warning(f"Redis connection failed: {e}")
            self.client = None
            self.is_available = False
            
            if not enable_fallback:
                raise
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            if self.is_available and self.client:
                value = self.client.get(key)
                if value is not None:
                    self.stats["hits"] += 1
                    # Try to deserialize JSON
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value
                else:
                    self.stats["misses"] += 1
                    return default
            else:
                # Fallback to in-memory cache
                self.stats["fallback_uses"] += 1
                if key in self._fallback_cache:
                    self.stats["hits"] += 1
                    return self._fallback_cache[key]
                else:
                    self.stats["misses"] += 1
                    return default
                    
        except RedisError as e:
            logger.error(f"Redis get error: {e}")
            self.stats["errors"] += 1
            self._fallback_cache.get(key, default) if self.enable_fallback else default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if self.is_available and self.client:
                result = self.client.set(
                    key,
                    value,
                    ex=ttl,
                    nx=nx,
                    xx=xx
                )
                self.stats["sets"] += 1
                return bool(result)
            else:
                # Fallback to in-memory cache
                self.stats["fallback_uses"] += 1
                self._fallback_cache[key] = value
                self.stats["sets"] += 1
                return True
                
        except RedisError as e:
            logger.error(f"Redis set error: {e}")
            self.stats["errors"] += 1
            if self.enable_fallback:
                self._fallback_cache[key] = value
            return False
    
    def delete(self, *keys: str) -> int:
        """
        Delete keys from cache.
        
        Args:
            *keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        try:
            if self.is_available and self.client:
                count = self.client.delete(*keys)
                self.stats["deletes"] += count
                return count
            else:
                # Fallback
                self.stats["fallback_uses"] += 1
                count = sum(1 for key in keys if self._fallback_cache.pop(key, None) is not None)
                self.stats["deletes"] += count
                return count
                
        except RedisError as e:
            logger.error(f"Redis delete error: {e}")
            self.stats["errors"] += 1
            return 0
    
    def exists(self, *keys: str) -> int:
        """
        Check if keys exist.
        
        Args:
            *keys: Keys to check
            
        Returns:
            Number of existing keys
        """
        try:
            if self.is_available and self.client:
                return self.client.exists(*keys)
            else:
                return sum(1 for key in keys if key in self._fallback_cache)
        except RedisError as e:
            logger.error(f"Redis exists error: {e}")
            return 0
    
    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            List of matching keys
        """
        try:
            if self.is_available and self.client:
                return [k.decode() if isinstance(k, bytes) else k 
                       for k in self.client.keys(pattern)]
            else:
                import re
                pattern_re = pattern.replace("*", ".*").replace("?", ".")
                return [k for k in self._fallback_cache.keys() 
                       if re.match(pattern_re, k)]
        except RedisError as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    def flush(self, pattern: Optional[str] = None) -> int:
        """
        Flush cache.
        
        Args:
            pattern: Only flush keys matching pattern
            
        Returns:
            Number of keys deleted
        """
        try:
            if pattern:
                keys = self.keys(pattern)
                return self.delete(*keys) if keys else 0
            else:
                if self.is_available and self.client:
                    self.client.flushdb()
                    return -1  # All keys
                else:
                    count = len(self._fallback_cache)
                    self._fallback_cache.clear()
                    return count
        except RedisError as e:
            logger.error(f"Redis flush error: {e}")
            return 0
    
    def ttl(self, key: str) -> int:
        """
        Get time to live for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds (-1 if no expiry, -2 if key doesn't exist)
        """
        try:
            if self.is_available and self.client:
                return self.client.ttl(key)
            else:
                return -1 if key in self._fallback_cache else -2
        except RedisError as e:
            logger.error(f"Redis ttl error: {e}")
            return -2
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter.
        
        Args:
            key: Counter key
            amount: Increment amount
            
        Returns:
            New value
        """
        try:
            if self.is_available and self.client:
                return self.client.incrby(key, amount)
            else:
                current = self._fallback_cache.get(key, 0)
                new_value = int(current) + amount
                self._fallback_cache[key] = new_value
                return new_value
        except RedisError as e:
            logger.error(f"Redis increment error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        total_ops = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_ops * 100) if total_ops > 0 else 0
        
        stats = {
            **self.stats,
            "hit_rate": round(hit_rate, 2),
            "is_available": self.is_available,
            "fallback_cache_size": len(self._fallback_cache)
        }
        
        if self.is_available and self.client:
            try:
                info = self.client.info()
                stats.update({
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_human": info.get("used_memory_human"),
                    "total_connections_received": info.get("total_connections_received"),
                    "total_commands_processed": info.get("total_commands_processed")
                })
            except RedisError:
                pass
        
        return stats
    
    def warm_cache(self, data: Dict[str, Any], ttl: int = 3600):
        """
        Warm cache with initial data.
        
        Args:
            data: Dict of key-value pairs
            ttl: Time to live in seconds
        """
        logger.info(f"Warming cache with {len(data)} entries")
        for key, value in data.items():
            self.set(key, value, ttl=ttl)
    
    def close(self):
        """Close Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


# Global cache instance
_redis_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """
    Get global Redis cache instance.
    
    Returns:
        RedisCache instance
    """
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache()
    
    return _redis_cache


def create_cache_key(*parts: str) -> str:
    """
    Create cache key from parts.
    
    Args:
        *parts: Key parts
        
    Returns:
        Cache key string
    """
    combined = ":".join(str(p) for p in parts)
    # For very long keys, use hash
    if len(combined) > 200:
        hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"{parts[0]}:{hash_suffix}"
    return combined


if __name__ == "__main__":
    # Example usage
    cache = RedisCache(enable_fallback=True)
    
    # Set values
    cache.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=60)
    cache.set("counter", 0)
    
    # Get values
    user = cache.get("user:123")
    print(f"User: {user}")
    
    # Increment
    cache.increment("counter", 5)
    counter = cache.get("counter")
    print(f"Counter: {counter}")
    
    # Pattern matching
    keys = cache.keys("user:*")
    print(f"User keys: {keys}")
    
    # Stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Close
    cache.close()
