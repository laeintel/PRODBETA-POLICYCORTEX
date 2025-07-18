"""
Pydantic models for API Gateway service.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class CircuitBreakerState(str, Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class ServiceRoute(BaseModel):
    """Service route configuration model."""
    name: str = Field(..., description="Service name")
    url: str = Field(..., description="Service URL")
    health_path: str = Field(..., description="Health check path")
    timeout: int = Field(..., description="Request timeout in seconds")
    status: ServiceStatus = Field(..., description="Current service status")
    circuit_breaker_state: CircuitBreakerState = Field(..., description="Circuit breaker state")


class GatewayMetrics(BaseModel):
    """Gateway metrics model."""
    total_requests: int = Field(..., description="Total number of requests")
    service_requests: Dict[str, int] = Field(..., description="Requests per service")
    circuit_breaker_states: Dict[str, CircuitBreakerState] = Field(..., description="Circuit breaker states")
    uptime_seconds: Optional[int] = Field(None, description="Gateway uptime in seconds")


class AuthRequest(BaseModel):
    """Authentication request model."""
    username: Optional[str] = Field(None, description="Username")
    email: Optional[str] = Field(None, description="Email address")
    password: str = Field(..., description="Password")
    tenant_id: Optional[str] = Field(None, description="Azure tenant ID")


class AuthResponse(BaseModel):
    """Authentication response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(..., description="User information")


class TokenRefreshRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., description="Refresh token")


class UserInfo(BaseModel):
    """User information model."""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="Email address")
    name: str = Field(..., description="Full name")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    tenant_id: Optional[str] = Field(None, description="Azure tenant ID")
    subscription_ids: List[str] = Field(default_factory=list, description="Azure subscription IDs")


class RateLimitInfo(BaseModel):
    """Rate limit information model."""
    limit: int = Field(..., description="Rate limit threshold")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Rate limit reset time")
    window_seconds: int = Field(..., description="Rate limit window in seconds")


class ProxyRequest(BaseModel):
    """Proxy request model."""
    service: str = Field(..., description="Target service name")
    path: str = Field(..., description="Request path")
    method: str = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    query_params: Optional[Dict[str, str]] = Field(None, description="Query parameters")
    body: Optional[Dict[str, Any]] = Field(None, description="Request body")
    timeout: Optional[int] = Field(None, description="Request timeout")


class ProxyResponse(BaseModel):
    """Proxy response model."""
    status_code: int = Field(..., description="HTTP status code")
    headers: Dict[str, str] = Field(..., description="Response headers")
    body: Optional[Dict[str, Any]] = Field(None, description="Response body")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    service: str = Field(..., description="Source service name")
    request_id: str = Field(..., description="Request identifier") 