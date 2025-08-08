"""
Rate limiter for API Gateway using Redis-based sliding window algorithm.
Provides flexible rate limiting with different strategies and user/IP-based limits.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as redis
import structlog
from shared.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class RateLimiter:
    """Redis-based rate limiter with sliding window algorithm."""

    def __init__(self):
        self.redis_client = None
        self.settings = settings

    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for rate limiting."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True,
            )
        return self.redis_client

    async def is_allowed(
        self, key: str, limit: int, window: int, burst: Optional[int] = None
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Check if request is allowed based on rate limits.

        Args:
            key: Unique identifier for the rate limit (user, IP, etc.)
            limit: Number of requests allowed in the window
            window: Time window in seconds
            burst: Optional burst limit for initial requests

        Returns:
            Tuple of (is_allowed, reset_time)
        """
        try:
            redis_client = await self._get_redis_client()
            current_time = int(time.time())
            window_start = current_time - window

            # Redis key for this rate limit
            rate_limit_key = f"rate_limit:{key}:{window}"

            # Use Lua script for atomic operations
            lua_script = """
            local key = KEYS[1]
            local window_start = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])
            local window = tonumber(ARGV[4])

            -- Remove expired entries
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

            -- Count current requests
            local current_count = redis.call('ZCARD', key)

            -- Check if limit exceeded
            if current_count >= limit then
                local ttl = redis.call('TTL', key)
                return {0, ttl}
            end

            -- Add current request
            redis.call('ZADD', key, current_time, current_time)
            redis.call('EXPIRE', key, window)

            -- Return success with remaining count
            return {1, limit - current_count - 1}
            """

            result = await redis_client.eval(
                lua_script, 1, rate_limit_key, window_start, current_time, limit, window
            )

            is_allowed = bool(result[0])
            reset_time = None

            if not is_allowed:
                # Calculate reset time
                reset_time = datetime.fromtimestamp(current_time + result[1])

            # Log rate limit check
            logger.debug(
                "rate_limit_check",
                key=key,
                limit=limit,
                window=window,
                allowed=is_allowed,
                reset_time=reset_time.isoformat() if reset_time else None,
            )

            return is_allowed, reset_time

        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e), key=key)
            # Fail open - allow request if rate limiter fails
            return True, None

    async def get_limit_info(self, key: str, window: int) -> Dict[str, Any]:
        """Get current rate limit information for a key."""
        try:
            redis_client = await self._get_redis_client()
            current_time = int(time.time())
            window_start = current_time - window

            rate_limit_key = f"rate_limit:{key}:{window}"

            # Remove expired entries and count current
            await redis_client.zremrangebyscore(rate_limit_key, 0, window_start)
            current_count = await redis_client.zcard(rate_limit_key)

            # Get TTL for reset time
            ttl = await redis_client.ttl(rate_limit_key)
            reset_time = datetime.fromtimestamp(current_time + ttl) if ttl > 0 else None

            return {
                "current_count": current_count,
                "window_seconds": window,
                "reset_time": reset_time.isoformat() if reset_time else None,
            }

        except Exception as e:
            logger.error("get_limit_info_failed", error=str(e), key=key)
            return {"current_count": 0, "window_seconds": window, "reset_time": None}

    async def reset_limit(self, key: str, window: int) -> bool:
        """Reset rate limit for a specific key."""
        try:
            redis_client = await self._get_redis_client()
            rate_limit_key = f"rate_limit:{key}:{window}"

            await redis_client.delete(rate_limit_key)

            logger.info("rate_limit_reset", key=key, window=window)
            return True

        except Exception as e:
            logger.error("rate_limit_reset_failed", error=str(e), key=key)
            return False

    async def check_user_rate_limit(
        self, user_id: str, endpoint: str, method: str = "GET"
    ) -> Tuple[bool, Optional[datetime]]:
        """Check rate limit for a specific user and endpoint."""

        # Different limits for different types of operations
        limits = {
            "GET": {"limit": 1000, "window": 3600},  # 1000 GET requests per hour
            "POST": {"limit": 100, "window": 3600},  # 100 POST requests per hour
            "PUT": {"limit": 50, "window": 3600},  # 50 PUT requests per hour
            "DELETE": {"limit": 20, "window": 3600},  # 20 DELETE requests per hour
            "ai": {"limit": 50, "window": 3600},  # 50 AI requests per hour
            "chat": {"limit": 200, "window": 3600},  # 200 chat messages per hour
        }

        # Determine rate limit based on endpoint and method
        if "/ai/" in endpoint:
            limit_config = limits["ai"]
        elif "/chat/" in endpoint:
            limit_config = limits["chat"]
        else:
            limit_config = limits.get(method.upper(), limits["GET"])

        key = f"user:{user_id}:{method.upper()}:{endpoint}"

        return await self.is_allowed(
            key=key, limit=limit_config["limit"], window=limit_config["window"]
        )

    async def check_ip_rate_limit(
        self, ip_address: str, endpoint: str = None
    ) -> Tuple[bool, Optional[datetime]]:
        """Check rate limit for IP address."""

        # Global IP rate limit
        global_limit = 10000  # 10000 requests per hour per IP
        global_window = 3600

        # More restrictive for unauthenticated requests
        if endpoint and any(path in endpoint for path in ["/auth/", "/public/"]):
            global_limit = 100  # 100 auth requests per hour per IP

        key = f"ip:{ip_address}"

        return await self.is_allowed(key=key, limit=global_limit, window=global_window)

    async def check_service_rate_limit(
        self, service_name: str, user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[datetime]]:
        """Check rate limit for service-specific operations."""

        service_limits = {
            "azure-integration": {"limit": 1000, "window": 3600},
            "ai-engine": {"limit": 100, "window": 3600},
            "conversation": {"limit": 500, "window": 3600},
            "data-processing": {"limit": 200, "window": 3600},
            "notification": {"limit": 1000, "window": 3600},
        }

        limit_config = service_limits.get(service_name, {"limit": 100, "window": 3600})

        if user_id:
            key = f"service:{service_name}:user:{user_id}"
        else:
            key = f"service:{service_name}:global"

        return await self.is_allowed(
            key=key, limit=limit_config["limit"], window=limit_config["window"]
        )

    async def check_burst_limit(
        self, key: str, burst_limit: int = 10, burst_window: int = 60
    ) -> Tuple[bool, Optional[datetime]]:
        """Check burst rate limit for preventing rapid successive requests."""

        burst_key = f"burst:{key}"

        return await self.is_allowed(key=burst_key, limit=burst_limit, window=burst_window)

    async def get_user_quota_info(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive quota information for a user."""
        try:
            quota_info = {}

            # Check different types of limits
            limit_types = [("GET", 3600), ("POST", 3600), ("ai", 3600), ("chat", 3600)]

            for limit_type, window in limit_types:
                key = f"user:{user_id}:{limit_type}"
                info = await self.get_limit_info(key, window)
                quota_info[limit_type] = info

            return quota_info

        except Exception as e:
            logger.error("get_user_quota_info_failed", error=str(e), user_id=user_id)
            return {}

    async def cleanup_expired_limits(self) -> None:
        """Cleanup expired rate limit entries (background task)."""
        try:
            redis_client = await self._get_redis_client()

            # This would typically be run as a background task
            # to clean up expired rate limit entries

            logger.info("rate_limit_cleanup_completed")

        except Exception as e:
            logger.error("rate_limit_cleanup_failed", error=str(e))
