"""
API Gateway Service for PolicyCortex.
Central entry point for all microservices with authentication, routing, and monitoring.
"""

import os
import sys
import time
import uuid
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import Optional

import httpx
import structlog
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

# Add the backend directory to Python path for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from shared.config import get_settings
    from shared.database import DatabaseUtils
    from shared.database import get_async_db

    from .auth import AuthManager
    from .circuit_breaker import CircuitBreaker
    from .models import APIResponse
    from .models import ErrorResponse
    from .models import GatewayMetrics
    from .models import HealthResponse
    from .models import ServiceRoute
    from .rate_limiter import RateLimiter
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Configuration
try:
    settings = get_settings()
    logger = structlog.get_logger(__name__)
    logger.info("API Gateway starting up", environment=settings.environment.value)
except Exception as e:
    print(f"Failed to initialize settings: {e}")
    sys.exit(1)

# Metrics
REQUEST_COUNT = Counter(
    "api_gateway_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram("api_gateway_request_duration_seconds", "Request duration")
SERVICE_REQUESTS = Counter(
    "api_gateway_service_requests_total", "Service requests", ["service", "status"]
)

# FastAPI app
app = FastAPI(
    title="PolicyCortex API Gateway",
    description="Central API Gateway for PolicyCortex microservices",
    version=settings.service.service_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Security
security = HTTPBearer(auto_error=False)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=settings.security.cors_methods,
    allow_headers=settings.security.cors_headers,
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Service Registry
SERVICE_REGISTRY = {
    "azure-integration": {
        "url": settings.azure_integration_url,
        "health_path": "/health",
        "timeout": 30,
        "circuit_breaker": CircuitBreaker(failure_threshold=5, timeout=60),
    },
    "ai-engine": {
        "url": settings.ai_engine_url,
        "health_path": "/health",
        "timeout": 60,
        "circuit_breaker": CircuitBreaker(failure_threshold=3, timeout=120),
    },
    "data-processing": {
        "url": settings.data_processing_url,
        "health_path": "/health",
        "timeout": 45,
        "circuit_breaker": CircuitBreaker(failure_threshold=5, timeout=60),
    },
    "conversation": {
        "url": settings.conversation_url,
        "health_path": "/health",
        "timeout": 30,
        "circuit_breaker": CircuitBreaker(failure_threshold=3, timeout=60),
    },
    "notification": {
        "url": settings.notification_url,
        "health_path": "/health",
        "timeout": 15,
        "circuit_breaker": CircuitBreaker(failure_threshold=5, timeout=30),
    },
}

# Global components
auth_manager = AuthManager()
rate_limiter = RateLimiter()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Add request ID to headers
        request.state.request_id = request_id

        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method, endpoint=request.url.path, status=response.status_code
            ).inc()
            REQUEST_DURATION.observe(duration)

            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Log error
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
            )

            # Update error metrics
            REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=500).inc()

            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)


async def verify_authentication(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Verify authentication for protected endpoints."""

    # Skip authentication for health checks and public endpoints
    if request.url.path in ["/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )

    try:
        user_info = await auth_manager.verify_token(credentials.credentials)
        request.state.user = user_info
        return user_info
    except Exception as e:
        logger.error("authentication_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials"
        )


async def check_rate_limit(request: Request) -> None:
    """Check rate limiting for the request."""
    client_ip = request.client.host if request.client else "unknown"
    user_id = getattr(request.state, "user", {}).get("id", client_ip)

    is_allowed, reset_time = await rate_limiter.is_allowed(
        key=f"user:{user_id}", limit=settings.security.rate_limit_per_minute, window=60
    )

    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Reset at {reset_time}",
        )


async def proxy_request(
    service_name: str,
    path: str,
    request: Request,
    method: str = "GET",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Proxy request to downstream service."""

    if service_name not in SERVICE_REGISTRY:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Service '{service_name}' not found"
        )

    service_config = SERVICE_REGISTRY[service_name]
    circuit_breaker = service_config["circuit_breaker"]
    service_timeout = timeout or service_config["timeout"]

    # Check circuit breaker
    if not circuit_breaker.can_execute():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service_name}' is currently unavailable",
        )

    try:
        url = f"{service_config['url']}{path}"

        # Prepare headers
        headers = dict(request.headers)
        headers["X-Request-ID"] = request.state.request_id
        if hasattr(request.state, "user"):
            headers["X-User-ID"] = request.state.user.get("id", "")

        # Get request body if present
        body = None
        if method.upper() in ["POST", "PUT", "PATCH"]:
            body = await request.body()

        async with httpx.AsyncClient(timeout=service_timeout) as client:
            response = await client.request(
                method=method, url=url, headers=headers, content=body, params=request.query_params
            )

        # Record success
        circuit_breaker.record_success()
        SERVICE_REQUESTS.labels(service=service_name, status="success").inc()

        # Return response
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content": response.content,
            "text": response.text if response.status_code != 204 else None,
        }

    except Exception as e:
        # Record failure
        circuit_breaker.record_failure()
        SERVICE_REQUESTS.labels(service=service_name, status="error").inc()

        logger.error(
            "service_request_failed",
            service=service_name,
            error=str(e),
            request_id=request.state.request_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service '{service_name}' request failed: {str(e)}",
        )


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="api-gateway",
        version=settings.service.service_version,
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    # Check downstream services
    healthy_services = 0
    total_services = len(SERVICE_REGISTRY)

    for service_name, config in SERVICE_REGISTRY.items():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{config['url']}{config['health_path']}")
                if response.status_code == 200:
                    healthy_services += 1
        except Exception:
            pass

    if healthy_services == total_services:
        return HealthResponse(
            status="ready",
            timestamp=datetime.utcnow(),
            service="api-gateway",
            version=settings.service.service_version,
            details={"healthy_services": f"{healthy_services}/{total_services}"},
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Only {healthy_services}/{total_services} services are healthy",
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# Authentication endpoints
@app.post("/auth/login")
async def login(request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)):
    """Login endpoint."""
    return await proxy_request("azure-integration", "/auth/login", request, "POST")


@app.post("/auth/logout")
async def logout(request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)):
    """Logout endpoint."""
    return await proxy_request("azure-integration", "/auth/logout", request, "POST")


@app.post("/auth/refresh")
async def refresh_token(request: Request):
    """Refresh token endpoint."""
    return await proxy_request("azure-integration", "/auth/refresh", request, "POST")


# Azure Integration routes
@app.api_route("/api/v1/azure/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def azure_proxy(
    path: str, request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Proxy requests to Azure Integration service."""
    await check_rate_limit(request)
    response_data = await proxy_request(
        "azure-integration", f"/api/v1/{path}", request, request.method
    )
    return Response(
        content=response_data["content"],
        status_code=response_data["status_code"],
        headers=response_data["headers"],
    )


# AI Engine routes
@app.api_route("/api/v1/ai/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def ai_proxy(
    path: str, request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Proxy requests to AI Engine service."""
    await check_rate_limit(request)
    response_data = await proxy_request(
        "ai-engine", f"/api/v1/{path}", request, request.method, timeout=120
    )
    return Response(
        content=response_data["content"],
        status_code=response_data["status_code"],
        headers=response_data["headers"],
    )


# Conversation routes
@app.api_route("/api/v1/chat/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def conversation_proxy(
    path: str, request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Proxy requests to Conversation service."""
    await check_rate_limit(request)
    response_data = await proxy_request("conversation", f"/api/v1/{path}", request, request.method)
    return Response(
        content=response_data["content"],
        status_code=response_data["status_code"],
        headers=response_data["headers"],
    )


# Data Processing routes
@app.api_route("/api/v1/data/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def data_proxy(
    path: str, request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Proxy requests to Data Processing service."""
    await check_rate_limit(request)
    response_data = await proxy_request(
        "data-processing", f"/api/v1/{path}", request, request.method
    )
    return Response(
        content=response_data["content"],
        status_code=response_data["status_code"],
        headers=response_data["headers"],
    )


# Notification routes
@app.api_route(
    "/api/v1/notifications/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def notification_proxy(
    path: str, request: Request, user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Proxy requests to Notification service."""
    await check_rate_limit(request)
    response_data = await proxy_request("notification", f"/api/v1/{path}", request, request.method)
    return Response(
        content=response_data["content"],
        status_code=response_data["status_code"],
        headers=response_data["headers"],
    )


# Gateway management endpoints
@app.get("/api/v1/gateway/services", response_model=Dict[str, ServiceRoute])
async def get_service_registry(user: Optional[Dict[str, Any]] = Depends(verify_authentication)):
    """Get service registry information."""
    services = {}
    for name, config in SERVICE_REGISTRY.items():
        circuit_breaker = config["circuit_breaker"]
        services[name] = ServiceRoute(
            name=name,
            url=config["url"],
            health_path=config["health_path"],
            timeout=config["timeout"],
            status="healthy" if circuit_breaker.can_execute() else "unhealthy",
            circuit_breaker_state=circuit_breaker.state,
        )

    return services


@app.get("/api/v1/gateway/metrics", response_model=GatewayMetrics)
async def get_gateway_metrics(user: Optional[Dict[str, Any]] = Depends(verify_authentication)):
    """Get gateway metrics."""
    return GatewayMetrics(
        total_requests=REQUEST_COUNT._value.sum(),
        service_requests={
            service: SERVICE_REQUESTS.labels(service=service, status="success")._value.get() or 0
            for service in SERVICE_REGISTRY.keys()
        },
        circuit_breaker_states={
            service: config["circuit_breaker"].state for service, config in SERVICE_REGISTRY.items()
        },
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        request_id=getattr(request.state, "request_id", "unknown"),
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message=str(exc) if settings.debug else "An unexpected error occurred",
            request_id=getattr(request.state, "request_id", "unknown"),
        ).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.service.service_host,
        port=settings.service.service_port,
        workers=1,  # Single worker for development
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower(),
    )
