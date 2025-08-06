"""
Circuit breaker pattern implementation for API Gateway.
Provides resilience and fault tolerance for downstream service calls.
"""

import asyncio
import time
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Callable
from typing import Optional

import structlog

from .models import CircuitBreakerState

logger = structlog.get_logger(__name__)


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Circuit breaker exception."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience.

    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Service is failing, requests are blocked
    - HALF_OPEN: Testing if service has recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 3,
        failure_timeout: float = 30.0
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time in seconds to wait before transitioning to half-open
            success_threshold: Number of successes needed in half-open to close
            failure_timeout: Timeout for individual service calls
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_timeout = failure_timeout

        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_start_time = None

        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []

        # Synchronization
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            if not self.can_execute():
                raise CircuitBreakerError(
                    f"Circuit breaker is {self.state}, request blocked"
                )

            self.total_requests += 1

        try:
            # Execute function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.failure_timeout
            )

            # Record success
            await self.record_success()
            return result

        except asyncio.TimeoutError:
            await self.record_failure()
            raise CircuitBreakerError("Service call timed out")
        except Exception as e:
            await self.record_failure()
            raise e

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and (current_time - self.last_failure_time) >= self.timeout:
                # Transition to half-open
                self._transition_to_half_open()
                return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False

    async def record_success(self) -> None:
        """Record successful service call."""
        async with self._lock:
            self.total_successes += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    async def record_failure(self) -> None:
        """Record failed service call."""
        async with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state transitions back to open
                self._transition_to_open()

    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state."""
        previous_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0

        self._log_state_change(previous_state, self.state)
        logger.warning(
            "circuit_breaker_opened",
            failure_count=self.failure_count,
            failure_threshold=self.failure_threshold
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state."""
        previous_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.half_open_start_time = time.time()

        self._log_state_change(previous_state, self.state)
        logger.info("circuit_breaker_half_opened")

    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state."""
        previous_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        self._log_state_change(previous_state, self.state)
        logger.info(
            "circuit_breaker_closed",
            success_count=self.success_count,
            success_threshold=self.success_threshold
        )

    def _log_state_change(
        self,
        from_state: CircuitBreakerState,
        to_state: CircuitBreakerState
    ) -> None:
        """Log state change for monitoring."""
        change = {
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": datetime.utcnow().isoformat(),
            "failure_count": self.failure_count,
            "success_count": self.success_count
        }
        self.state_changes.append(change)

        # Keep only last 10 state changes
        if len(self.state_changes) > 10:
            self.state_changes = self.state_changes[-10:]

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        current_time = time.time()

        stats = {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": self.total_failures / self.total_requests if self.total_requests > 0 else 0,
            "last_failure_time": self.last_failure_time,
            "state_changes": self.state_changes[-5:],  # Last 5 state changes
        }

        if self.state == CircuitBreakerState.OPEN and self.last_failure_time:
            stats["time_until_half_open"] = max(
                0, self.timeout - (current_time - self.last_failure_time)
            )

        if self.state == CircuitBreakerState.HALF_OPEN and self.half_open_start_time:
            stats["half_open_duration"] = current_time - self.half_open_start_time

        return stats

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_start_time = None

        logger.info("circuit_breaker_reset")


class AdvancedCircuitBreaker(CircuitBreaker):
    """
    Advanced circuit breaker with additional features like:
    - Exponential backoff
    - Different thresholds for different error types
    - Custom fallback functions
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 3,
        failure_timeout: float = 30.0,
        exponential_backoff: bool = True,
        max_timeout: int = 300,
        fallback_function: Optional[Callable] = None
    ):
        super().__init__(failure_threshold, timeout, success_threshold, failure_timeout)

        self.exponential_backoff = exponential_backoff
        self.max_timeout = max_timeout
        self.fallback_function = fallback_function
        self.backoff_multiplier = 2
        self.current_timeout = timeout

    async def call_with_fallback(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker and fallback support."""
        try:
            return await self.call(func, *args, **kwargs)
        except CircuitBreakerError:
            if self.fallback_function:
                logger.info("circuit_breaker_fallback_executed")
                return await self.fallback_function(*args, **kwargs)
            raise

    def _transition_to_open(self) -> None:
        """Enhanced transition with exponential backoff."""
        super()._transition_to_open()

        if self.exponential_backoff:
            self.current_timeout = min(
                self.current_timeout * self.backoff_multiplier,
                self.max_timeout
            )
            logger.info(
                "circuit_breaker_backoff_increased",
                new_timeout=self.current_timeout
            )

    def _transition_to_closed(self) -> None:
        """Enhanced transition with timeout reset."""
        super()._transition_to_closed()

        # Reset timeout on successful recovery
        self.current_timeout = self.timeout
        logger.info("circuit_breaker_timeout_reset")

    def can_execute(self) -> bool:
        """Enhanced execution check with current timeout."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and
                (current_time - self.last_failure_time) >= self.current_timeout:
                self._transition_to_half_open()
                return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True

        return False


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self.breakers = {}

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        success_threshold: int = 3,
        failure_timeout: float = 30.0
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                timeout=timeout,
                success_threshold=success_threshold,
                failure_timeout=failure_timeout
            )
        return self.breakers[name]

    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("all_circuit_breakers_reset")


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()
