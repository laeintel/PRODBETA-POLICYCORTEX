"""
Unit tests for RateLimiter functionality.
"""

import time
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from backend.services.api_gateway.rate_limiter import RateLimiter
from backend.services.api_gateway.rate_limiter import RateLimitExceeded


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        rl = RateLimiter()

        assert rl.redis_client is not None
        assert hasattr(rl, "logger")

    @pytest.mark.asyncio
    async def test_is_allowed_within_limit(self, mock_redis):
        """Test rate limiting when within limit."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for rate limiting
        mock_redis.get.return_value = None  # No existing count
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed("test_key", limit=100, window=60)

        assert allowed is True
        assert reset_time is None
        mock_redis.incr.assert_called_once_with("test_key")
        mock_redis.expire.assert_called_once_with("test_key", 60)

    @pytest.mark.asyncio
    async def test_is_allowed_exceeds_limit(self, mock_redis):
        """Test rate limiting when limit is exceeded."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for exceeded limit
        mock_redis.get.return_value = "100"  # At limit
        mock_redis.incr.return_value = 101  # Would exceed
        mock_redis.ttl.return_value = 30  # 30 seconds remaining

        allowed, reset_time = await rl.is_allowed("test_key", limit=100, window=60)

        assert allowed is False
        assert reset_time is not None
        mock_redis.incr.assert_called_once_with("test_key")
        mock_redis.ttl.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_is_allowed_first_request(self, mock_redis):
        """Test rate limiting for first request."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for first request
        mock_redis.get.return_value = None
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed("new_key", limit=10, window=60)

        assert allowed is True
        assert reset_time is None
        mock_redis.incr.assert_called_once_with("new_key")
        mock_redis.expire.assert_called_once_with("new_key", 60)

    @pytest.mark.asyncio
    async def test_is_allowed_at_limit(self, mock_redis):
        """Test rate limiting when exactly at limit."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for at limit
        mock_redis.get.return_value = "9"  # One below limit
        mock_redis.incr.return_value = 10  # Exactly at limit
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed("test_key", limit=10, window=60)

        assert allowed is True
        assert reset_time is None

    @pytest.mark.asyncio
    async def test_is_allowed_redis_error(self, mock_redis):
        """Test rate limiting with Redis error."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis error
        mock_redis.get.side_effect = Exception("Redis connection error")

        with patch.object(rl, "logger") as mock_logger:
            allowed, reset_time = await rl.is_allowed("test_key", limit=10, window=60)

            # Should allow request on Redis error (fail open)
            assert allowed is True
            assert reset_time is None
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_remaining_requests(self, mock_redis):
        """Test getting remaining requests."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = "7"  # 7 requests used

        remaining = await rl.get_remaining_requests("test_key", limit=10)

        assert remaining == 3  # 10 - 7 = 3
        mock_redis.get.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_remaining_requests_no_usage(self, mock_redis):
        """Test getting remaining requests with no usage."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis response for no usage
        mock_redis.get.return_value = None

        remaining = await rl.get_remaining_requests("test_key", limit=10)

        assert remaining == 10  # Full limit available

    @pytest.mark.asyncio
    async def test_get_reset_time(self, mock_redis):
        """Test getting reset time."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis response
        mock_redis.ttl.return_value = 45  # 45 seconds remaining

        reset_time = await rl.get_reset_time("test_key")

        assert reset_time is not None
        mock_redis.ttl.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_get_reset_time_no_expiry(self, mock_redis):
        """Test getting reset time with no expiry."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis response for no expiry
        mock_redis.ttl.return_value = -1  # No expiry set

        reset_time = await rl.get_reset_time("test_key")

        assert reset_time is None

    @pytest.mark.asyncio
    async def test_reset_limit(self, mock_redis):
        """Test resetting rate limit."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis response
        mock_redis.delete.return_value = True

        success = await rl.reset_limit("test_key")

        assert success is True
        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_reset_limit_redis_error(self, mock_redis):
        """Test resetting rate limit with Redis error."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis error
        mock_redis.delete.side_effect = Exception("Redis error")

        with patch.object(rl, "logger") as mock_logger:
            success = await rl.reset_limit("test_key")

            assert success is False
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_usage_stats(self, mock_redis):
        """Test getting usage statistics."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses
        mock_redis.get.return_value = "7"
        mock_redis.ttl.return_value = 30

        stats = await rl.get_usage_stats("test_key", limit=10)

        assert stats["current_usage"] == 7
        assert stats["remaining"] == 3
        assert stats["limit"] == 10
        assert stats["reset_time"] is not None

    @pytest.mark.asyncio
    async def test_sliding_window_rate_limiter(self, mock_redis):
        """Test sliding window rate limiter."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for sliding window
        current_time = int(time.time())
        window_start = current_time - 60

        mock_redis.zremrangebyscore.return_value = 5  # Removed 5 old entries
        mock_redis.zcard.return_value = 8  # 8 entries in window
        mock_redis.zadd.return_value = 1  # Added 1 new entry
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed_sliding_window("test_key", limit=10, window=60)

        assert allowed is True
        assert reset_time is None
        mock_redis.zremrangebyscore.assert_called_once()
        mock_redis.zcard.assert_called_once()
        mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_sliding_window_exceeds_limit(self, mock_redis):
        """Test sliding window when limit is exceeded."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for exceeded limit
        mock_redis.zremrangebyscore.return_value = 0  # No old entries
        mock_redis.zcard.return_value = 10  # Already at limit

        allowed, reset_time = await rl.is_allowed_sliding_window("test_key", limit=10, window=60)

        assert allowed is False
        assert reset_time is not None
        mock_redis.zremrangebyscore.assert_called_once()
        mock_redis.zcard.assert_called_once()

    @pytest.mark.asyncio
    async def test_distributed_rate_limiter(self, mock_redis):
        """Test distributed rate limiter."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for distributed limiting
        mock_redis.eval.return_value = [1, 30]  # Allowed, 30 seconds to reset

        allowed, reset_time = await rl.is_allowed_distributed("test_key", limit=10, window=60)

        assert allowed is True
        assert reset_time is not None
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiter_with_burst_capacity(self, mock_redis):
        """Test rate limiter with burst capacity."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses
        mock_redis.get.return_value = "8"  # 8 requests used
        mock_redis.incr.return_value = 9
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed_with_burst(
            "test_key", limit=10, burst=15, window=60
        )

        assert allowed is True
        assert reset_time is None

    @pytest.mark.asyncio
    async def test_hierarchical_rate_limiter(self, mock_redis):
        """Test hierarchical rate limiter."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses for multiple levels
        mock_redis.get.side_effect = ["5", "50"]  # User: 5, Global: 50
        mock_redis.incr.side_effect = [6, 51]
        mock_redis.expire.return_value = True

        allowed, reset_time = await rl.is_allowed_hierarchical(
            user_key="user:123", global_key="global", user_limit=10, global_limit=100, window=60
        )

        assert allowed is True
        assert reset_time is None

    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup(self, mock_redis):
        """Test rate limiter cleanup of expired keys."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        # Mock Redis responses
        mock_redis.scan_iter.return_value = ["rate_limit:key1", "rate_limit:key2"]
        mock_redis.ttl.side_effect = [-2, 30]  # key1 expired, key2 valid
        mock_redis.delete.return_value = 1

        cleaned = await rl.cleanup_expired_keys("rate_limit:*")

        assert cleaned == 1
        mock_redis.delete.assert_called_once_with("rate_limit:key1")

    def test_rate_limit_exceeded_exception(self):
        """Test RateLimitExceeded exception."""
        reset_time = datetime.now() + timedelta(seconds=30)

        exc = RateLimitExceeded(key="test_key", limit=10, window=60, reset_time=reset_time)

        assert exc.key == "test_key"
        assert exc.limit == 10
        assert exc.window == 60
        assert exc.reset_time == reset_time
        assert "Rate limit exceeded" in str(exc)

    @pytest.mark.asyncio
    async def test_rate_limiter_with_custom_key_generator(self, mock_redis):
        """Test rate limiter with custom key generator."""
        rl = RateLimiter()
        rl.redis_client = mock_redis

        def custom_key_generator(user_id, ip_address):
            return f"custom:{user_id}:{ip_address}"

        # Mock Redis responses
        mock_redis.get.return_value = "5"
        mock_redis.incr.return_value = 6
        mock_redis.expire.return_value = True

        key = custom_key_generator("user123", "192.168.1.1")
        allowed, reset_time = await rl.is_allowed(key, limit=10, window=60)

        assert allowed is True
        assert "custom:user123:192.168.1.1" in str(mock_redis.incr.call_args)
