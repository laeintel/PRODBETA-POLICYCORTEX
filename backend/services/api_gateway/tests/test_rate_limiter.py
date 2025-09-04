"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Comprehensive tests for rate limiting and circuit breaker
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import redis.asyncio as redis
import sys
import os
from fastapi import HTTPException

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rate_limiter import (
    RateLimiter,
    CircuitBreaker,
    AdaptiveRateLimiter,
    RateLimitConfig,
    CircuitBreakerConfig,
    rate_limit,
    circuit_breaker,
    rate_limit_middleware
)

class TestRateLimiter:
    """Test RateLimiter class"""
    
    @pytest.mark.asyncio
    async def test_initialize_with_redis(self):
        """Test initializing rate limiter with Redis connection"""
        limiter = RateLimiter()
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_redis.return_value = mock_client
            
            await limiter.initialize()
            
            assert limiter.use_redis
            assert limiter.redis_client == mock_client
    
    @pytest.mark.asyncio
    async def test_initialize_without_redis(self):
        """Test initializing rate limiter without Redis (fallback to local)"""
        limiter = RateLimiter()
        
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            await limiter.initialize()
            
            assert not limiter.use_redis
            assert limiter.redis_client is None
    
    def test_get_key_with_user_auth(self):
        """Test generating rate limit key with authenticated user"""
        limiter = RateLimiter()
        
        request = Mock()
        request.url.path = "/api/test"
        request.state.auth = Mock()
        request.state.auth.user_id = "user-123"
        
        config = RateLimitConfig()
        
        key = limiter._get_key(request, config)
        assert key == "rate_limit:user-123:/api/test"
    
    def test_get_key_with_ip(self):
        """Test generating rate limit key with IP address"""
        limiter = RateLimiter()
        
        request = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.state = Mock()
        request.state.auth = None
        
        config = RateLimitConfig()
        
        key = limiter._get_key(request, config)
        assert key == "rate_limit:192.168.1.1:/api/test"
    
    def test_get_key_with_custom_func(self):
        """Test generating rate limit key with custom function"""
        limiter = RateLimiter()
        
        request = Mock()
        
        def custom_key_func(req):
            return "custom_key"
        
        config = RateLimitConfig(key_func=custom_key_func)
        
        key = limiter._get_key(request, config)
        assert key == "custom_key"
    
    @pytest.mark.asyncio
    async def test_check_redis_within_limit(self):
        """Test checking rate limit with Redis (within limit)"""
        limiter = RateLimiter()
        limiter.use_redis = True
        
        mock_client = AsyncMock()
        mock_pipeline = AsyncMock()
        
        # Mock Redis pipeline operations
        mock_client.pipeline = Mock(return_value=mock_pipeline)
        mock_pipeline.zremrangebyscore = AsyncMock()
        mock_pipeline.zcard = AsyncMock()
        mock_pipeline.zadd = AsyncMock()
        mock_pipeline.expire = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 5, None, None])
        
        mock_client.incr = AsyncMock(return_value=3)
        mock_client.expire = AsyncMock()
        mock_client.get = AsyncMock(return_value="3")
        
        limiter.redis_client = mock_client
        
        config = RateLimitConfig(requests=10, window=60, burst=5)
        
        result = await limiter._check_redis("test_key", config)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_redis_exceeded_limit(self):
        """Test checking rate limit with Redis (limit exceeded)"""
        limiter = RateLimiter()
        limiter.use_redis = True
        
        mock_client = AsyncMock()
        mock_pipeline = AsyncMock()
        
        # Mock Redis pipeline operations - return count exceeding limit
        mock_client.pipeline = Mock(return_value=mock_pipeline)
        mock_pipeline.zremrangebyscore = AsyncMock()
        mock_pipeline.zcard = AsyncMock()
        mock_pipeline.zadd = AsyncMock()
        mock_pipeline.expire = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[None, 15, None, None])
        
        mock_client.incr = AsyncMock(return_value=10)
        mock_client.expire = AsyncMock()
        mock_client.get = AsyncMock(return_value="10")
        
        limiter.redis_client = mock_client
        
        config = RateLimitConfig(requests=10, window=60, burst=5)
        
        result = await limiter._check_redis("test_key", config)
        assert result is False
    
    def test_check_local_within_limit(self):
        """Test checking rate limit locally (within limit)"""
        limiter = RateLimiter()
        
        config = RateLimitConfig(requests=10, window=60)
        key = "test_key"
        
        # Initialize bucket with tokens
        limiter.local_buckets[key] = {
            "tokens": 5.0,
            "last_update": time.time()
        }
        
        result = limiter._check_local(key, config)
        assert result is True
        assert limiter.local_buckets[key]["tokens"] == 4.0
    
    def test_check_local_no_tokens(self):
        """Test checking rate limit locally (no tokens)"""
        limiter = RateLimiter()
        
        config = RateLimitConfig(requests=10, window=60)
        key = "test_key"
        
        # Initialize bucket with no tokens
        limiter.local_buckets[key] = {
            "tokens": 0.0,
            "last_update": time.time()
        }
        
        result = limiter._check_local(key, config)
        assert result is False
    
    def test_check_local_token_refill(self):
        """Test token bucket refill logic"""
        limiter = RateLimiter()
        
        config = RateLimitConfig(requests=10, window=60)
        key = "test_key"
        
        # Initialize bucket with old timestamp
        old_time = time.time() - 30  # 30 seconds ago
        limiter.local_buckets[key] = {
            "tokens": 0.0,
            "last_update": old_time
        }
        
        result = limiter._check_local(key, config)
        
        # Should have refilled some tokens (5 tokens in 30 seconds)
        assert result is True
        assert limiter.local_buckets[key]["tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_check_limit_enabled(self):
        """Test check_limit when rate limiting is enabled"""
        limiter = RateLimiter()
        limiter.use_redis = False
        
        request = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        config = RateLimitConfig(requests=10, window=60)
        
        with patch.dict(os.environ, {'ENABLE_RATE_LIMITING': 'true'}):
            # First request should pass
            result = await limiter.check_limit(request, config)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_limit_disabled(self):
        """Test check_limit when rate limiting is disabled"""
        limiter = RateLimiter()
        
        request = Mock()
        config = RateLimitConfig()
        
        with patch.dict(os.environ, {'ENABLE_RATE_LIMITING': 'false'}):
            result = await limiter.check_limit(request, config)
            assert result is True

class TestCircuitBreaker:
    """Test CircuitBreaker class"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60
        )
        
        breaker = CircuitBreaker(config)
        
        assert breaker.config == config
        assert breaker.failure_count == 0
        assert breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls"""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker(config)
        
        async def successful_func():
            return "success"
        
        with patch.dict(os.environ, {'ENABLE_CIRCUIT_BREAKER': 'true'}):
            result = await breaker.call(successful_func)
            assert result == "success"
            assert breaker.state == "closed"
            assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_opens(self):
        """Test circuit breaker opens after threshold failures"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        async def failing_func():
            raise HTTPException(500, "Server error")
        
        with patch.dict(os.environ, {'ENABLE_CIRCUIT_BREAKER': 'true'}):
            # Fail 3 times to open the circuit
            for _ in range(3):
                with pytest.raises(HTTPException):
                    await breaker.call(failing_func)
            
            assert breaker.state == "open"
            assert breaker.failure_count >= 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects(self):
        """Test circuit breaker rejects calls when open"""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.state = "open"
        breaker.last_failure_time = time.time()
        
        async def some_func():
            return "should not execute"
        
        with patch.dict(os.environ, {'ENABLE_CIRCUIT_BREAKER': 'true'}):
            with pytest.raises(HTTPException) as exc_info:
                await breaker.call(some_func)
            
            assert "Service temporarily unavailable" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open(self):
        """Test circuit breaker half-open state"""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0  # Immediate recovery for testing
        )
        breaker = CircuitBreaker(config)
        breaker.state = "open"
        breaker.last_failure_time = time.time() - 100  # Old failure
        
        async def successful_func():
            return "success"
        
        with patch.dict(os.environ, {'ENABLE_CIRCUIT_BREAKER': 'true'}):
            # Should attempt in half-open state
            result = await breaker.call(successful_func)
            assert result == "success"
            
            # After 3 successes, should close
            breaker.success_count = 3
            result = await breaker.call(successful_func)
            assert breaker.state == "closed"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(self):
        """Test circuit breaker with fallback function"""
        async def fallback_func(*args, **kwargs):
            return "fallback response"
        
        config = CircuitBreakerConfig(
            failure_threshold=1,
            fallback_func=fallback_func
        )
        breaker = CircuitBreaker(config)
        breaker.state = "open"
        breaker.last_failure_time = time.time()
        
        async def some_func():
            return "should not execute"
        
        with patch.dict(os.environ, {'ENABLE_CIRCUIT_BREAKER': 'true'}):
            result = await breaker.call(some_func)
            assert result == "fallback response"
    
    def test_should_attempt_reset_timeout_not_reached(self):
        """Test should_attempt_reset when timeout not reached"""
        config = CircuitBreakerConfig(recovery_timeout=60)
        breaker = CircuitBreaker(config)
        breaker.last_failure_time = time.time()
        
        assert not breaker._should_attempt_reset()
    
    def test_should_attempt_reset_timeout_reached(self):
        """Test should_attempt_reset when timeout reached"""
        config = CircuitBreakerConfig(recovery_timeout=60)
        breaker = CircuitBreaker(config)
        breaker.last_failure_time = time.time() - 120  # 2 minutes ago
        
        assert breaker._should_attempt_reset()

class TestAdaptiveRateLimiter:
    """Test AdaptiveRateLimiter class"""
    
    def test_adaptive_limiter_initialization(self):
        """Test adaptive rate limiter initialization"""
        limiter = AdaptiveRateLimiter()
        
        assert limiter.adjustment_factor == 1.0
        assert len(limiter.load_history) == 0
    
    @pytest.mark.asyncio
    async def test_update_load_normal(self):
        """Test updating load metrics"""
        limiter = AdaptiveRateLimiter()
        
        await limiter.update_load(response_time=0.5, error_rate=0.01)
        
        assert len(limiter.load_history) == 1
        assert limiter.load_history[0]["response_time"] == 0.5
        assert limiter.load_history[0]["error_rate"] == 0.01
    
    @pytest.mark.asyncio
    async def test_adjust_limits_high_load(self):
        """Test adjusting limits under high load"""
        limiter = AdaptiveRateLimiter()
        
        # Add high load metrics
        for _ in range(10):
            limiter.load_history.append({
                "time": time.time(),
                "response_time": 2.0,  # High response time
                "error_rate": 0.2  # High error rate
            })
        
        limiter.last_adjustment = time.time() - 20  # Force adjustment
        
        await limiter._adjust_limits()
        
        # Factor should decrease under high load
        assert limiter.adjustment_factor < 1.0
    
    @pytest.mark.asyncio
    async def test_adjust_limits_low_load(self):
        """Test adjusting limits under low load"""
        limiter = AdaptiveRateLimiter()
        
        # Add low load metrics
        for _ in range(10):
            limiter.load_history.append({
                "time": time.time(),
                "response_time": 0.3,  # Low response time
                "error_rate": 0.005  # Low error rate
            })
        
        limiter.last_adjustment = time.time() - 20  # Force adjustment
        
        await limiter._adjust_limits()
        
        # Factor should increase under low load
        assert limiter.adjustment_factor > 1.0
    
    def test_get_adjusted_limit(self):
        """Test getting adjusted rate limit"""
        limiter = AdaptiveRateLimiter()
        limiter.adjustment_factor = 1.5
        
        base_limit = 100
        adjusted = limiter.get_adjusted_limit(base_limit)
        
        assert adjusted == 150
    
    def test_get_adjusted_limit_min_max(self):
        """Test adjusted limit calculation with different factors"""
        limiter = AdaptiveRateLimiter()
        
        # Test with low factor - no clamping in get_adjusted_limit
        limiter.adjustment_factor = 0.3
        adjusted = limiter.get_adjusted_limit(100)
        assert adjusted == 30  # Direct multiplication
        
        # Test with high factor - no clamping in get_adjusted_limit
        limiter.adjustment_factor = 3.0
        adjusted = limiter.get_adjusted_limit(100)
        assert adjusted == 300  # Direct multiplication

class TestRateLimitDecorator:
    """Test rate_limit decorator"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_decorator_within_limit(self):
        """Test rate limit decorator when within limit"""
        @rate_limit(requests=10, window=60)
        async def protected_endpoint(request):
            return {"status": "success"}
        
        request = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        with patch('rate_limiter.rate_limiter') as mock_limiter:
            mock_limiter.check_limit = AsyncMock(return_value=True)
            mock_limiter.get_remaining = AsyncMock(return_value={
                "remaining": 9,
                "reset": int(time.time()) + 60
            })
            
            result = await protected_endpoint(request)
            assert result == {"status": "success"}
    
    @pytest.mark.asyncio
    async def test_rate_limit_decorator_exceeded(self):
        """Test rate limit decorator when limit exceeded"""
        @rate_limit(requests=10, window=60)
        async def protected_endpoint(request):
            return {"status": "success"}
        
        request = Mock()
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        with patch('rate_limiter.rate_limiter') as mock_limiter:
            mock_limiter.check_limit = AsyncMock(return_value=False)
            mock_limiter.get_remaining = AsyncMock(return_value={
                "remaining": 0,
                "reset": int(time.time()) + 60
            })
            
            with pytest.raises(HTTPException) as exc_info:
                await protected_endpoint(request)
            
            assert exc_info.value.status_code == 429

class TestCircuitBreakerDecorator:
    """Test circuit_breaker decorator"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_success(self):
        """Test circuit breaker decorator with successful call"""
        @circuit_breaker(failure_threshold=3)
        async def protected_function():
            return "success"
        
        result = await protected_function()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_failure(self):
        """Test circuit breaker decorator with failing call"""
        @circuit_breaker(failure_threshold=1)
        async def failing_function():
            raise HTTPException(500, "Server error")
        
        with pytest.raises(HTTPException):
            await failing_function()

class TestRateLimitMiddleware:
    """Test rate limit middleware"""
    
    @pytest.mark.asyncio
    async def test_middleware_health_check_bypass(self):
        """Test middleware bypasses health check endpoints"""
        request = Mock()
        request.url.path = "/health"
        
        async def call_next(req):
            return Mock(status_code=200, headers={})
        
        with patch.dict(os.environ, {'ENABLE_RATE_LIMITING': 'true'}):
            response = await rate_limit_middleware(request, call_next)
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_middleware_applies_rate_limit(self):
        """Test middleware applies rate limiting"""
        request = Mock()
        request.url.path = "/api/endpoint"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        response_mock = Mock(status_code=200, headers={})
        
        async def call_next(req):
            return response_mock
        
        with patch('rate_limiter.rate_limiter') as mock_limiter:
            mock_limiter.check_limit = AsyncMock(return_value=True)
            mock_limiter.get_remaining = AsyncMock(return_value={
                "remaining": 99,
                "reset": int(time.time()) + 60
            })
            
            with patch('rate_limiter.adaptive_limiter') as mock_adaptive:
                mock_adaptive.get_adjusted_limit = Mock(return_value=100)
                mock_adaptive.update_load = AsyncMock()
                
                with patch.dict(os.environ, {'ENABLE_RATE_LIMITING': 'true'}):
                    response = await rate_limit_middleware(request, call_next)
                    
                    assert response.status_code == 200
                    assert "X-RateLimit-Limit" in response.headers
                    assert "X-RateLimit-Remaining" in response.headers
    
    @pytest.mark.asyncio
    async def test_middleware_rate_limit_exceeded(self):
        """Test middleware when rate limit is exceeded"""
        request = Mock()
        request.url.path = "/api/endpoint"
        request.client = Mock()
        request.client.host = "192.168.1.1"
        
        async def call_next(req):
            return Mock(status_code=200)
        
        with patch('rate_limiter.rate_limiter') as mock_limiter:
            mock_limiter.check_limit = AsyncMock(return_value=False)
            mock_limiter.get_remaining = AsyncMock(return_value={
                "remaining": 0,
                "reset": int(time.time()) + 60
            })
            
            with patch('rate_limiter.adaptive_limiter') as mock_adaptive:
                mock_adaptive.get_adjusted_limit = Mock(return_value=100)
                
                with patch.dict(os.environ, {'ENABLE_RATE_LIMITING': 'true'}):
                    response = await rate_limit_middleware(request, call_next)
                    
                    assert response.status_code == 429
                    assert response.headers["X-RateLimit-Remaining"] == "0"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])