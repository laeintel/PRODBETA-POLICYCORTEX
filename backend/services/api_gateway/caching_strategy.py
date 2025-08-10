"""
Comprehensive caching strategy with TTL, invalidation, and multi-tier caching
"""

import asyncio
import json
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from functools import wraps

# Redis for distributed caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory cache")

# Local in-memory caching
from cachetools import TTLCache, LRUCache
import cachetools.func

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels for multi-tier caching"""
    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"    # Redis distributed cache
    L3_DATABASE = "l3_database"  # Database cache (for computed results)


class CacheStrategy(Enum):
    """Cache strategies"""
    CACHE_ASIDE = "cache_aside"  # Load on miss
    READ_THROUGH = "read_through"  # Automatic loading
    WRITE_THROUGH = "write_through"  # Write to cache and store
    WRITE_BEHIND = "write_behind"  # Write to cache, async to store
    REFRESH_AHEAD = "refresh_ahead"  # Proactive refresh before expiry


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    ttl: int
    created_at: datetime
    accessed_at: datetime
    access_count: int
    tags: Set[str]
    version: int
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl <= 0:
            return False
        return (datetime.utcnow() - self.created_at).seconds > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "key": self.key,
            "value": self.value,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "tags": list(self.tags),
            "version": self.version
        }


class CacheKeyGenerator:
    """Generate cache keys with consistent hashing"""
    
    @staticmethod
    def generate(
        prefix: str,
        params: Dict[str, Any],
        version: str = "v1"
    ) -> str:
        """Generate cache key from parameters"""
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        
        # Create hash of parameters
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        
        # Build cache key
        return f"{prefix}:{version}:{param_hash}"
    
    @staticmethod
    def generate_pattern(prefix: str, pattern: str = "*") -> str:
        """Generate pattern for key matching"""
        return f"{prefix}:{pattern}"


class InMemoryCache:
    """In-memory L1 cache implementation"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache = TTLCache(maxsize=max_size, ttl=default_ttl)
        self.tags_index: Dict[str, Set[str]] = {}  # tag -> keys
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.cache[key]
            self.stats["hits"] += 1
            return value
        except KeyError:
            self.stats["misses"] += 1
            return None
            
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in cache"""
        self.cache[key] = value
        
        # Update tags index
        if tags:
            for tag in tags:
                if tag not in self.tags_index:
                    self.tags_index[tag] = set()
                self.tags_index[tag].add(key)
                
        return True
        
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            del self.cache[key]
            
            # Remove from tags index
            for tag_keys in self.tags_index.values():
                tag_keys.discard(key)
                
            return True
        except KeyError:
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.cache
        
    async def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
        self.tags_index.clear()
        
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with tag"""
        if tag not in self.tags_index:
            return 0
            
        keys = self.tags_index[tag].copy()
        for key in keys:
            await self.delete(key)
            
        return len(keys)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / max(1, total_requests)
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hit_rate": hit_rate
        }


class RedisCache:
    """Redis L2 cache implementation"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl: int = 3600,
        key_prefix: str = "policycortex"
    ):
        if not REDIS_AVAILABLE:
            raise RuntimeError("Redis client not available")
            
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.client: Optional[redis.Redis] = None
        
    async def connect(self):
        """Connect to Redis"""
        self.client = await redis.from_url(self.redis_url)
        await self.client.ping()
        logger.info("Connected to Redis cache")
        
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            
    def _make_key(self, key: str) -> str:
        """Make full Redis key with prefix"""
        return f"{self.key_prefix}:{key}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.client:
            return None
            
        full_key = self._make_key(key)
        value = await self.client.get(full_key)
        
        if value:
            try:
                return pickle.loads(value)
            except Exception as e:
                logger.error(f"Error deserializing cache value: {e}")
                return None
                
        return None
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Set value in Redis"""
        if not self.client:
            return False
            
        full_key = self._make_key(key)
        serialized = pickle.dumps(value)
        
        # Set with TTL
        ttl = ttl or self.default_ttl
        await self.client.setex(full_key, ttl, serialized)
        
        # Add to tags
        if tags:
            for tag in tags:
                tag_key = self._make_key(f"tag:{tag}")
                await self.client.sadd(tag_key, full_key)
                await self.client.expire(tag_key, ttl)
                
        return True
        
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.client:
            return False
            
        full_key = self._make_key(key)
        result = await self.client.delete(full_key)
        return result > 0
        
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.client:
            return False
            
        full_key = self._make_key(key)
        return await self.client.exists(full_key) > 0
        
    async def clear(self, pattern: str = "*") -> None:
        """Clear cache keys matching pattern"""
        if not self.client:
            return
            
        pattern_key = self._make_key(pattern)
        cursor = 0
        
        while True:
            cursor, keys = await self.client.scan(cursor, match=pattern_key)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break
                
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all keys with tag"""
        if not self.client:
            return 0
            
        tag_key = self._make_key(f"tag:{tag}")
        keys = await self.client.smembers(tag_key)
        
        if keys:
            await self.client.delete(*keys)
            await self.client.delete(tag_key)
            return len(keys)
            
        return 0
        
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        if not self.client:
            return -1
            
        full_key = self._make_key(key)
        return await self.client.ttl(full_key)
        
    async def extend_ttl(self, key: str, ttl: int) -> bool:
        """Extend TTL for key"""
        if not self.client:
            return False
            
        full_key = self._make_key(key)
        return await self.client.expire(full_key, ttl)


class MultiTierCache:
    """Multi-tier caching system with L1 (memory) and L2 (Redis)"""
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_ttl: int = 300,
        l2_ttl: int = 3600,
        redis_url: Optional[str] = None
    ):
        # L1 cache (in-memory)
        self.l1_cache = InMemoryCache(max_size=l1_max_size, default_ttl=l1_ttl)
        
        # L2 cache (Redis)
        self.l2_cache = None
        if redis_url and REDIS_AVAILABLE:
            self.l2_cache = RedisCache(redis_url=redis_url, default_ttl=l2_ttl)
            
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "misses": 0
        }
        
    async def initialize(self):
        """Initialize cache connections"""
        if self.l2_cache:
            await self.l2_cache.connect()
            
    async def close(self):
        """Close cache connections"""
        if self.l2_cache:
            await self.l2_cache.disconnect()
            
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks L1, then L2)"""
        # Check L1
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value
            
        # Check L2
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                self.stats["l2_hits"] += 1
                
                # Promote to L1
                await self.l1_cache.set(key, value)
                
                return value
                
        self.stats["misses"] += 1
        return None
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        levels: Optional[List[CacheLevel]] = None
    ) -> bool:
        """Set value in cache"""
        levels = levels or [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        success = True
        
        # Set in L1
        if CacheLevel.L1_MEMORY in levels:
            success &= await self.l1_cache.set(key, value, ttl, tags)
            
        # Set in L2
        if CacheLevel.L2_REDIS in levels and self.l2_cache:
            success &= await self.l2_cache.set(key, value, ttl, tags)
            
        return success
        
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        l1_deleted = await self.l1_cache.delete(key)
        l2_deleted = await self.l2_cache.delete(key) if self.l2_cache else False
        
        return l1_deleted or l2_deleted
        
    async def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with tag"""
        count = await self.l1_cache.invalidate_by_tag(tag)
        
        if self.l2_cache:
            count += await self.l2_cache.invalidate_by_tag(tag)
            
        return count
        
    async def clear(self) -> None:
        """Clear all cache levels"""
        await self.l1_cache.clear()
        
        if self.l2_cache:
            await self.l2_cache.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"]
        total_requests = total_hits + self.stats["misses"]
        hit_rate = total_hits / max(1, total_requests)
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "l1_stats": self.l1_cache.get_stats()
        }


class CacheDecorator:
    """Decorators for caching function results"""
    
    def __init__(self, cache: MultiTierCache):
        self.cache = cache
        
    def cached(
        self,
        prefix: str,
        ttl: int = 300,
        tags: Optional[List[str]] = None,
        key_func: Optional[Callable] = None
    ):
        """Decorator to cache function results"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    params = {"args": args, "kwargs": kwargs}
                    cache_key = CacheKeyGenerator.generate(prefix, params)
                    
                # Check cache
                cached_value = await self.cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_value
                    
                # Call function
                result = await func(*args, **kwargs)
                
                # Store in cache
                await self.cache.set(cache_key, result, ttl, tags)
                
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, use asyncio.run
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    def invalidate(self, pattern: str = None, tags: List[str] = None):
        """Decorator to invalidate cache after function execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                # Invalidate cache
                if tags:
                    for tag in tags:
                        await self.cache.invalidate_by_tag(tag)
                        
                if pattern:
                    await self.cache.clear(pattern)
                    
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # Invalidate cache
                loop = asyncio.get_event_loop()
                if tags:
                    for tag in tags:
                        loop.run_until_complete(self.cache.invalidate_by_tag(tag))
                        
                if pattern:
                    loop.run_until_complete(self.cache.clear(pattern))
                    
                return result
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class CacheWarmer:
    """Proactively warm cache with frequently accessed data"""
    
    def __init__(self, cache: MultiTierCache):
        self.cache = cache
        self.warming_tasks: List[Callable] = []
        
    def register_warming_task(self, task: Callable):
        """Register a cache warming task"""
        self.warming_tasks.append(task)
        
    async def warm_cache(self):
        """Execute all cache warming tasks"""
        logger.info("Starting cache warming")
        
        for task in self.warming_tasks:
            try:
                await task(self.cache)
            except Exception as e:
                logger.error(f"Cache warming task failed: {e}")
                
        logger.info("Cache warming completed")
        
    async def schedule_periodic_warming(self, interval_seconds: int = 300):
        """Schedule periodic cache warming"""
        while True:
            await self.warm_cache()
            await asyncio.sleep(interval_seconds)


# Global cache instance
cache_manager = None

def initialize_cache(
    l1_max_size: int = 1000,
    l1_ttl: int = 300,
    l2_ttl: int = 3600,
    redis_url: Optional[str] = "redis://localhost:6379"
) -> MultiTierCache:
    """Initialize global cache manager"""
    global cache_manager
    
    cache_manager = MultiTierCache(
        l1_max_size=l1_max_size,
        l1_ttl=l1_ttl,
        l2_ttl=l2_ttl,
        redis_url=redis_url
    )
    
    logger.info("Cache manager initialized")
    
    return cache_manager


# Export key components
__all__ = [
    "CacheLevel",
    "CacheStrategy",
    "CacheEntry",
    "CacheKeyGenerator",
    "InMemoryCache",
    "RedisCache",
    "MultiTierCache",
    "CacheDecorator",
    "CacheWarmer",
    "initialize_cache",
    "cache_manager"
]