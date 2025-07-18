"""
Azure authentication middleware for the Azure Integration service.
"""

import time
from typing import Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from ...shared.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class AzureAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for Azure authentication and context management."""
    
    def __init__(self, app):
        super().__init__(app)
        self.public_paths = [
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/refresh"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request with Azure authentication context."""
        
        # Skip authentication for public endpoints
        if any(request.url.path.startswith(path) for path in self.public_paths):
            response = await call_next(request)
            return response
        
        try:
            # Get user information from request state (set by API Gateway)
            user_info = getattr(request.state, "user", None)
            if not user_info:
                # Try to get from headers (internal service calls)
                user_id = request.headers.get("X-User-ID")
                if user_id:
                    user_info = {
                        "id": user_id,
                        "tenant_id": request.headers.get("X-Tenant-ID"),
                        "subscription_ids": request.headers.get("X-Subscription-IDs", "").split(",")
                    }
                    request.state.user = user_info
            
            # Get Azure context from headers or user info
            if user_info:
                azure_context = {
                    "tenant_id": user_info.get("tenant_id") or settings.azure.tenant_id,
                    "subscription_ids": user_info.get("subscription_ids", []),
                    "user_id": user_info.get("id"),
                    "request_id": getattr(request.state, "request_id", None)
                }
                request.state.azure_context = azure_context
                
                logger.info(
                    "azure_context_set",
                    user_id=azure_context["user_id"],
                    tenant_id=azure_context["tenant_id"],
                    subscription_count=len(azure_context["subscription_ids"])
                )
            
            # Process request
            response = await call_next(request)
            
            # Add Azure context headers to response
            if hasattr(request.state, "azure_context"):
                response.headers["X-Azure-Tenant-ID"] = request.state.azure_context["tenant_id"]
            
            return response
            
        except Exception as e:
            logger.error(
                "azure_auth_middleware_error",
                error=str(e),
                request_id=getattr(request.state, "request_id", "unknown")
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Azure authentication error"
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting Azure API calls."""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limits = {}
        self.window_seconds = 60
        self.max_requests = 100  # Per minute
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to Azure API calls."""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            response = await call_next(request)
            return response
        
        # Get user identifier
        user_info = getattr(request.state, "user", {})
        user_id = user_info.get("id", "anonymous")
        
        # Check rate limit
        current_time = time.time()
        user_key = f"user:{user_id}"
        
        if user_key not in self.rate_limits:
            self.rate_limits[user_key] = {
                "requests": 0,
                "window_start": current_time
            }
        
        user_limit = self.rate_limits[user_key]
        
        # Reset window if expired
        if current_time - user_limit["window_start"] > self.window_seconds:
            user_limit["requests"] = 0
            user_limit["window_start"] = current_time
        
        # Check if rate limit exceeded
        if user_limit["requests"] >= self.max_requests:
            remaining_time = self.window_seconds - (current_time - user_limit["window_start"])
            logger.warning(
                "rate_limit_exceeded",
                user_id=user_id,
                requests=user_limit["requests"],
                remaining_seconds=remaining_time
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {int(remaining_time)} seconds."
            )
        
        # Increment request count
        user_limit["requests"] += 1
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(self.max_requests - user_limit["requests"])
        response.headers["X-RateLimit-Reset"] = str(int(user_limit["window_start"] + self.window_seconds))
        
        return response


class AzureResourceCacheMiddleware(BaseHTTPMiddleware):
    """Middleware for caching Azure resource responses."""
    
    def __init__(self, app):
        super().__init__(app)
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.cacheable_paths = [
            "/api/v1/policies",
            "/api/v1/rbac/roles",
            "/api/v1/resources/groups",
            "/api/v1/networks"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Cache Azure resource responses for performance."""
        
        # Only cache GET requests
        if request.method != "GET":
            response = await call_next(request)
            return response
        
        # Check if path is cacheable
        is_cacheable = any(request.url.path.startswith(path) for path in self.cacheable_paths)
        if not is_cacheable:
            response = await call_next(request)
            return response
        
        # Generate cache key
        cache_key = f"{request.method}:{request.url.path}:{request.url.query}"
        
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data:
            cache_age = time.time() - cached_data["timestamp"]
            if cache_age < self.cache_ttl:
                logger.info(
                    "cache_hit",
                    path=request.url.path,
                    cache_age=cache_age
                )
                
                # Return cached response
                return Response(
                    content=cached_data["content"],
                    status_code=cached_data["status_code"],
                    headers=cached_data["headers"],
                    media_type=cached_data["media_type"]
                )
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Cache response data
            self.cache[cache_key] = {
                "content": body,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "media_type": response.media_type,
                "timestamp": time.time()
            }
            
            # Return new response with cached body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        return response