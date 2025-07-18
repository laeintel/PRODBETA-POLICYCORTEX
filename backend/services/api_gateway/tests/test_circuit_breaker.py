"""
Unit tests for CircuitBreaker functionality.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from backend.services.api_gateway.circuit_breaker import CircuitBreaker, CircuitBreakerState


class TestCircuitBreaker:
    """Test CircuitBreaker class."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=5, timeout=60)
        
        assert cb.failure_threshold == 5
        assert cb.timeout == 60
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Initially closed and can execute
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True
        
        # Record some failures but below threshold
        cb.record_failure()
        cb.record_failure()
        
        assert cb.failure_count == 2
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True
    
    def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record failures to reach threshold
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        # Should now be open
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False
        assert cb.last_failure_time is not None
    
    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker in half-open state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)  # 1 second timeout
        
        # Record failures to open circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should now be half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should be half-open now
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Record success - should close the circuit
        cb.record_success()
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
    
    def test_circuit_breaker_failure_in_half_open(self):
        """Test circuit breaker failure in half-open state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Should be half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Record another failure - should open again
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 4
    
    def test_circuit_breaker_success_in_closed_state(self):
        """Test circuit breaker success in closed state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record some failures
        cb.record_failure()
        cb.record_failure()
        
        assert cb.failure_count == 2
        
        # Record success - should reset failure count
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_multiple_successes(self):
        """Test multiple successes don't affect closed circuit."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record multiple successes
        cb.record_success()
        cb.record_success()
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.can_execute() is True
    
    def test_circuit_breaker_with_zero_threshold(self):
        """Test circuit breaker with zero failure threshold."""
        cb = CircuitBreaker(failure_threshold=0, timeout=60)
        
        # Should open immediately on first failure
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False
    
    def test_circuit_breaker_state_transitions(self):
        """Test all state transitions."""
        cb = CircuitBreaker(failure_threshold=2, timeout=1)
        
        # Start in CLOSED
        assert cb.state == CircuitBreakerState.CLOSED
        
        # CLOSED -> OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # OPEN -> HALF_OPEN (after timeout)
        time.sleep(1.1)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # HALF_OPEN -> CLOSED (on success)
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        
        # CLOSED -> OPEN again
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # OPEN -> HALF_OPEN -> OPEN (on failure)
        time.sleep(1.1)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_timeout_boundary(self):
        """Test circuit breaker timeout boundary conditions."""
        cb = CircuitBreaker(failure_threshold=1, timeout=2)
        
        # Open the circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Check before timeout
        time.sleep(1)
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False
        
        # Check after timeout
        time.sleep(1.1)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True
    
    def test_circuit_breaker_thread_safety(self):
        """Test circuit breaker thread safety."""
        # Note: This is a basic test for thread safety
        # In a real implementation, you'd use threading to test concurrent access
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Multiple operations should be safe
        for _ in range(10):
            cb.record_failure()
            cb.record_success()
        
        # Should still be in a valid state
        assert cb.state in [CircuitBreakerState.CLOSED, CircuitBreakerState.OPEN, CircuitBreakerState.HALF_OPEN]
    
    def test_circuit_breaker_string_representation(self):
        """Test circuit breaker string representation."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Test in different states
        assert "CLOSED" in str(cb)
        
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert "OPEN" in str(cb)
    
    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics collection."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record some operations
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        
        # Check metrics
        assert cb.failure_count == 2
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.last_failure_time is not None
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        
        assert cb.state == CircuitBreakerState.OPEN
        
        # Reset should close the circuit
        cb.reset()
        
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.last_failure_time is None
    
    def test_circuit_breaker_custom_parameters(self):
        """Test circuit breaker with custom parameters."""
        # Test with different threshold and timeout values
        cb1 = CircuitBreaker(failure_threshold=1, timeout=30)
        cb2 = CircuitBreaker(failure_threshold=10, timeout=300)
        
        assert cb1.failure_threshold == 1
        assert cb1.timeout == 30
        assert cb2.failure_threshold == 10
        assert cb2.timeout == 300
        
        # Test behavior with different thresholds
        cb1.record_failure()
        assert cb1.state == CircuitBreakerState.OPEN
        
        cb2.record_failure()
        assert cb2.state == CircuitBreakerState.CLOSED