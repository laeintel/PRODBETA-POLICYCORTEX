"""
Rate Limiting and Circuit Breaker Middleware for PolicyCortex
Protects API endpoints from abuse and manages service degradation
"""

import os
import time
import asyncio
import logging
from typing import Dict, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import hashlib
import json

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
ENABLE_CIRCUIT_BREAKER = os.getenv("ENABLE_CIRCUIT_BREAKER", "true").lower() == "true"

# Rate limit defaults (can be overridden per endpoint)
DEFAULT_RATE_LIMIT = int(os.getenv("DEFAULT_RATE_LIMIT", "100"))  # requests
DEFAULT_RATE_WINDOW = int(os.getenv("DEFAULT_RATE_WINDOW", "60"))  # seconds
DEFAULT_BURST_LIMIT = int(os.getenv("DEFAULT_BURST_LIMIT", "20"))  # burst requests

# Circuit breaker defaults
CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5"))
CIRCUIT_RECOVERY_TIMEOUT = int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "60"))
CIRCUIT_EXPECTED_EXCEPTION = HTTPException

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests: int = DEFAULT_RATE_LIMIT
    window: int = DEFAULT_RATE_WINDOW
    burst: int = DEFAULT_BURST_LIMIT
    key_func: Optional[Callable] = None
    error_message: str = "Rate limit exceeded. Please try again later."
    
@dataclass 
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD
    recovery_timeout: int = CIRCUIT_RECOVERY_TIMEOUT
    expected_exception: type = CIRCUIT_EXPECTED_EXCEPTION
    fallback_func: Optional[Callable] = None

class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_buckets: Dict[str, Dict] = defaultdict(lambda: {
            "tokens": DEFAULT_RATE_LIMIT,
            "last_update": time.time()
        })
        self.use_redis = False
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not ENABLE_RATE_LIMITING:
            return
            
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            await self.redis_client.ping()
            self.use_redis = True
            logger.info("Rate limiter connected to Redis")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis, using local rate limiting: {e}")
            self.use_redis = False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    def _get_key(self, request: Request, config: RateLimitConfig) -> str:
        """Generate rate limit key for request"""
        if config.key_func:
            return config.key_func(request)
        
        # Default: use IP + path
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        # Try to get user ID from auth if available
        user_id = None
        if hasattr(request.state, "auth") and request.state.auth:
            user_id = request.state.auth.user_id
        
        if user_id:
            key = f"rate_limit:{user_id}:{path}"
        else:
            key = f"rate_limit:{client_ip}:{path}"
        
        return key
    
    async def _check_redis(self, key: str, config: RateLimitConfig) -> bool:
        """Check rate limit using Redis"""
        try:
            pipe = self.redis_client.pipeline()
            now = time.time()
            window_start = now - config.window
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, config.window)
            
            results = await pipe.execute()
            request_count = results[1]
            
            # Check burst limit for recent requests
            recent_key = f"{key}:burst"
            await self.redis_client.incr(recent_key)
            await self.redis_client.expire(recent_key, 1)
            burst_count = int(await self.redis_client.get(recent_key) or 0)
            
            if burst_count > config.burst:
                return False
            
            return request_count < config.requests
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True
    
    def _check_local(self, key: str, config: RateLimitConfig) -> bool:
        """Check rate limit using local token bucket"""
        now = time.time()
        bucket = self.local_buckets[key]
        
        # Refill tokens
        time_passed = now - bucket["last_update"]
        tokens_to_add = time_passed * (config.requests / config.window)
        bucket["tokens"] = min(config.requests, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
        
        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False
    
    async def check_limit(self, request: Request, config: RateLimitConfig) -> bool:
        """Check if request is within rate limit"""
        if not ENABLE_RATE_LIMITING:
            return True
        
        key = self._get_key(request, config)
        
        if self.use_redis:
            return await self._check_redis(key, config)
        else:
            return self._check_local(key, config)
    
    async def get_remaining(self, request: Request, config: RateLimitConfig) -> Dict[str, int]:
        """Get remaining requests and reset time"""
        if not ENABLE_RATE_LIMITING:
            return {"remaining": config.requests, "reset": 0}
        
        key = self._get_key(request, config)
        
        if self.use_redis:
            try:
                now = time.time()
                window_start = now - config.window
                count = await self.redis_client.zcount(key, window_start, now)
                remaining = max(0, config.requests - count)
                reset = int(now + config.window)
                return {"remaining": remaining, "reset": reset}
            except:
                pass
        
        # Local fallback
        bucket = self.local_buckets.get(key, {"tokens": config.requests})
        return {
            "remaining": int(bucket.get("tokens", 0)),
            "reset": int(time.time() + config.window)
        }

class CircuitBreaker:
    """Circuit breaker for service protection"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.success_count = 0
        self.lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not ENABLE_CIRCUIT_BREAKER:
            return await func(*args, **kwargs)
        
        async with self.lock:
            # Check circuit state
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                    self.success_count = 0
                else:
                    if self.config.fallback_func:
                        return await self.config.fallback_func(*args, **kwargs)
                    raise HTTPException(503, "Service temporarily unavailable")
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            
            # Record success
            async with self.lock:
                if self.state == "half_open":
                    self.success_count += 1
                    if self.success_count >= 3:  # Required successes to close
                        self.state = "closed"
                        self.failure_count = 0
                        logger.info(f"Circuit breaker closed for {func.__name__}")
                elif self.state == "closed":
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            async with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.config.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened for {func.__name__}")
                
                # If half-open, go back to open
                if self.state == "half_open":
                    self.state = "open"
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if not self.last_failure_time:
            return True
        
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout

# Global instances
rate_limiter = RateLimiter()
circuit_breakers: Dict[str, CircuitBreaker] = {}

def rate_limit(
    requests: int = DEFAULT_RATE_LIMIT,
    window: int = DEFAULT_RATE_WINDOW,
    burst: int = DEFAULT_BURST_LIMIT,
    key_func: Optional[Callable] = None,
    error_message: str = "Rate limit exceeded"
):
    """Decorator for rate limiting endpoints"""
    config = RateLimitConfig(
        requests=requests,
        window=window,
        burst=burst,
        key_func=key_func,
        error_message=error_message
    )
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Check rate limit
            if not await rate_limiter.check_limit(request, config):
                # Get remaining info for headers
                remaining_info = await rate_limiter.get_remaining(request, config)
                
                # Create response with rate limit headers
                response = JSONResponse(
                    status_code=429,
                    content={"error": config.error_message},
                    headers={
                        "X-RateLimit-Limit": str(config.requests),
                        "X-RateLimit-Remaining": str(remaining_info["remaining"]),
                        "X-RateLimit-Reset": str(remaining_info["reset"]),
                        "Retry-After": str(config.window)
                    }
                )
                raise HTTPException(429, detail=config.error_message)
            
            # Execute function
            result = await func(request, *args, **kwargs)
            
            # Add rate limit headers to response
            if isinstance(result, Response):
                remaining_info = await rate_limiter.get_remaining(request, config)
                result.headers["X-RateLimit-Limit"] = str(config.requests)
                result.headers["X-RateLimit-Remaining"] = str(remaining_info["remaining"])
                result.headers["X-RateLimit-Reset"] = str(remaining_info["reset"])
            
            return result
        
        return wrapper
    return decorator

def circuit_breaker(
    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
    recovery_timeout: int = CIRCUIT_RECOVERY_TIMEOUT,
    fallback_func: Optional[Callable] = None
):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        # Create circuit breaker for this function
        breaker_key = f"{func.__module__}.{func.__name__}"
        if breaker_key not in circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                fallback_func=fallback_func
            )
            circuit_breakers[breaker_key] = CircuitBreaker(config)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            breaker = circuit_breakers[breaker_key]
            return await breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator

class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""
    
    def __init__(self):
        self.load_history = deque(maxlen=60)  # Last 60 seconds
        self.adjustment_factor = 1.0
        self.last_adjustment = time.time()
        
    async def update_load(self, response_time: float, error_rate: float):
        """Update system load metrics"""
        self.load_history.append({
            "time": time.time(),
            "response_time": response_time,
            "error_rate": error_rate
        })
        
        # Adjust every 10 seconds
        if time.time() - self.last_adjustment > 10:
            await self._adjust_limits()
    
    async def _adjust_limits(self):
        """Adjust rate limits based on system load"""
        if len(self.load_history) < 10:
            return
        
        # Calculate average metrics
        recent = list(self.load_history)[-10:]
        avg_response_time = sum(h["response_time"] for h in recent) / len(recent)
        avg_error_rate = sum(h["error_rate"] for h in recent) / len(recent)
        
        # Adjust factor based on metrics
        if avg_response_time > 1.0 or avg_error_rate > 0.1:
            # High load - reduce limits
            self.adjustment_factor = max(0.5, self.adjustment_factor * 0.9)
        elif avg_response_time < 0.5 and avg_error_rate < 0.01:
            # Low load - increase limits
            self.adjustment_factor = min(2.0, self.adjustment_factor * 1.1)
        
        self.last_adjustment = time.time()
        logger.info(f"Adjusted rate limit factor to {self.adjustment_factor}")
    
    def get_adjusted_limit(self, base_limit: int) -> int:
        """Get adjusted rate limit"""
        return int(base_limit * self.adjustment_factor)

# Global adaptive limiter
adaptive_limiter = AdaptiveRateLimiter()

# Middleware for automatic rate limiting
async def rate_limit_middleware(request: Request, call_next):
    """Middleware to apply default rate limiting to all endpoints"""
    if not ENABLE_RATE_LIMITING:
        return await call_next(request)
    
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/api/v1/health"]:
        return await call_next(request)
    
    # Apply default rate limit
    config = RateLimitConfig()
    
    # Adjust limits based on load
    config.requests = adaptive_limiter.get_adjusted_limit(config.requests)
    
    # Check rate limit
    if not await rate_limiter.check_limit(request, config):
        remaining_info = await rate_limiter.get_remaining(request, config)
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers={
                "X-RateLimit-Limit": str(config.requests),
                "X-RateLimit-Remaining": str(remaining_info["remaining"]),
                "X-RateLimit-Reset": str(remaining_info["reset"]),
                "Retry-After": str(config.window)
            }
        )
    
    # Process request and measure performance
    start_time = time.time()
    response = await call_next(request)
    response_time = time.time() - start_time
    
    # Update adaptive limiter
    error_rate = 0.01 if response.status_code >= 500 else 0.0
    await adaptive_limiter.update_load(response_time, error_rate)
    
    # Add rate limit headers
    remaining_info = await rate_limiter.get_remaining(request, config)
    response.headers["X-RateLimit-Limit"] = str(config.requests)
    response.headers["X-RateLimit-Remaining"] = str(remaining_info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(remaining_info["reset"])
    
    return response

# Export main components
__all__ = [
    "RateLimiter",
    "CircuitBreaker",
    "AdaptiveRateLimiter",
    "rate_limit",
    "circuit_breaker",
    "rate_limit_middleware",
    "rate_limiter",
    "adaptive_limiter",
    "RateLimitConfig",
    "CircuitBreakerConfig"
]