"""
Azure Integration Service for PolicyCortex.
Provides Azure SDK integrations for policy, RBAC, cost, network, and resource management.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

import structlog
from fastapi import FastAPI, Request, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from shared.config import get_settings
from shared.database import get_async_db, DatabaseUtils
    HealthResponse,
    APIResponse,
    ErrorResponse,
    PolicyResponse,
    PolicyRequest,
    RBACResponse,
    RBACRequest,
    CostResponse,
    NetworkResponse,
    ResourceResponse,
    AzureAuthRequest,
    AzureAuthResponse
)
    PolicyManagementService,
    RBACManagementService,
    CostManagementService,
    NetworkManagementService,
    ResourceManagementService,
    AzureAuthService
)
from services.azure_integration.services.event_collector import AzureEventCollector
    from services.azure_integration.middleware import AzureAuthMiddleware

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'azure_integration_requests_total',
    'Total API requests',
    ['method',
    'endpoint',
    'status']
)
REQUEST_DURATION = Histogram('azure_integration_request_duration_seconds', 'Request duration')
AZURE_API_CALLS = Counter(
    'azure_api_calls_total',
    'Azure API calls',
    ['service',
    'operation',
    'status']
)
AZURE_API_DURATION = Histogram('azure_api_duration_seconds', 'Azure API call duration', ['service'])

# FastAPI app
app = FastAPI(
    title="PolicyCortex Azure Integration Service",
    description="Azure SDK integration service for PolicyCortex",
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
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Service instances
policy_service = PolicyManagementService()
rbac_service = RBACManagementService()
cost_service = CostManagementService()
network_service = NetworkManagementService()
resource_service = ResourceManagementService()
auth_service = AzureAuthService()
event_collector = AzureEventCollector(settings)


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
            client_ip=request.client.host if request.client else None
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            REQUEST_DURATION.observe(duration)

            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
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
                duration_ms=round(duration * 1000, 2)
            )

            # Update error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()

            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AzureAuthMiddleware)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current user from request state."""
    return getattr(request.state, "user", None)


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="azure-integration",
        version=settings.service.service_version
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    # Check Azure connectivity
    try:
        is_connected = await auth_service.verify_azure_connection()
        if is_connected:
            return HealthResponse(
                status="ready",
                timestamp=datetime.utcnow(),
                service="azure-integration",
                version=settings.service.service_version,
                details={"azure_connected": True}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Azure connection not available"
            )
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# Authentication endpoints
@app.post("/auth/login", response_model=AzureAuthResponse)
async def azure_login(auth_request: AzureAuthRequest):
    """Azure AD authentication endpoint."""
    try:
        auth_response = await auth_service.authenticate(
            tenant_id=auth_request.tenant_id,
            client_id=auth_request.client_id,
            client_secret=auth_request.client_secret
        )
        return auth_response
    except Exception as e:
        logger.error("azure_login_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}"
        )


@app.post("/auth/refresh", response_model=AzureAuthResponse)
async def refresh_token(refresh_token: str):
    """Refresh authentication token."""
    try:
        auth_response = await auth_service.refresh_token(refresh_token)
        return auth_response
    except Exception as e:
        logger.error("token_refresh_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {str(e)}"
        )


# Policy Management endpoints
@app.get("/api/v1/policies", response_model=List[PolicyResponse])
async def list_policies(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    resource_group: Optional[str] = Query(None, description="Filter by resource group"),
    policy_type: Optional[str] = Query(None, description="Filter by policy type"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List all Azure policies in a subscription."""
    try:
        start_time = time.time()
        policies = await policy_service.list_policies(
            subscription_id=subscription_id,
            resource_group=resource_group,
            policy_type=policy_type
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="list", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return policies
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="list", status="error").inc()
        logger.error("list_policies_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list policies: {str(e)}"
        )


@app.get("/api/v1/policies/{policy_id}", response_model=PolicyResponse)
async def get_policy(
    policy_id: str,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific Azure policy."""
    try:
        start_time = time.time()
        policy = await policy_service.get_policy(
            subscription_id=subscription_id,
            policy_id=policy_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="get", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return policy
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="get", status="error").inc()
        logger.error("get_policy_failed", error=str(e), policy_id=policy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get policy: {str(e)}"
        )


@app.post("/api/v1/policies", response_model=PolicyResponse)
async def create_policy(
    policy_request: PolicyRequest,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new Azure policy."""
    try:
        start_time = time.time()
        policy = await policy_service.create_policy(
            subscription_id=subscription_id,
            policy_data=policy_request.dict()
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="create", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return policy
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="create", status="error").inc()
        logger.error("create_policy_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create policy: {str(e)}"
        )


@app.put("/api/v1/policies/{policy_id}", response_model=PolicyResponse)
async def update_policy(
    policy_id: str,
    policy_request: PolicyRequest,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update an existing Azure policy."""
    try:
        start_time = time.time()
        policy = await policy_service.update_policy(
            subscription_id=subscription_id,
            policy_id=policy_id,
            policy_data=policy_request.dict()
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="update", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return policy
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="update", status="error").inc()
        logger.error("update_policy_failed", error=str(e), policy_id=policy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update policy: {str(e)}"
        )


@app.delete("/api/v1/policies/{policy_id}")
async def delete_policy(
    policy_id: str,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete an Azure policy."""
    try:
        start_time = time.time()
        await policy_service.delete_policy(
            subscription_id=subscription_id,
            policy_id=policy_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="delete", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return {"message": f"Policy {policy_id} deleted successfully"}
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="delete", status="error").inc()
        logger.error("delete_policy_failed", error=str(e), policy_id=policy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete policy: {str(e)}"
        )


@app.get("/api/v1/policies/{policy_id}/compliance", response_model=Dict[str, Any])
async def get_policy_compliance(
    policy_id: str,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get compliance status for a policy."""
    try:
        start_time = time.time()
        compliance = await policy_service.get_policy_compliance(
            subscription_id=subscription_id,
            policy_id=policy_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="policy", operation="compliance", status="success").inc()
        AZURE_API_DURATION.labels(service="policy").observe(time.time() - start_time)

        return compliance
    except Exception as e:
        AZURE_API_CALLS.labels(service="policy", operation="compliance", status="error").inc()
        logger.error("get_policy_compliance_failed", error=str(e), policy_id=policy_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get policy compliance: {str(e)}"
        )


# RBAC Management endpoints
@app.get("/api/v1/rbac/roles", response_model=List[RBACResponse])
async def list_roles(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    scope: Optional[str] = Query(None, description="Scope for role assignments"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List all RBAC roles."""
    try:
        start_time = time.time()
        roles = await rbac_service.list_roles(
            subscription_id=subscription_id,
            scope=scope
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="rbac", operation="list_roles", status="success").inc()
        AZURE_API_DURATION.labels(service="rbac").observe(time.time() - start_time)

        return roles
    except Exception as e:
        AZURE_API_CALLS.labels(service="rbac", operation="list_roles", status="error").inc()
        logger.error("list_roles_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list roles: {str(e)}"
        )


@app.get("/api/v1/rbac/assignments", response_model=List[Dict[str, Any]])
async def list_role_assignments(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    principal_id: Optional[str] = Query(None, description="Filter by principal ID"),
    scope: Optional[str] = Query(None, description="Filter by scope"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List role assignments."""
    try:
        start_time = time.time()
        assignments = await rbac_service.list_role_assignments(
            subscription_id=subscription_id,
            principal_id=principal_id,
            scope=scope
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="rbac", operation="list_assignments", status="success").inc()
        AZURE_API_DURATION.labels(service="rbac").observe(time.time() - start_time)

        return assignments
    except Exception as e:
        AZURE_API_CALLS.labels(service="rbac", operation="list_assignments", status="error").inc()
        logger.error("list_role_assignments_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list role assignments: {str(e)}"
        )


@app.post("/api/v1/rbac/assignments", response_model=Dict[str, Any])
async def create_role_assignment(
    rbac_request: RBACRequest,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a new role assignment."""
    try:
        start_time = time.time()
        assignment = await rbac_service.create_role_assignment(
            subscription_id=subscription_id,
            assignment_data=rbac_request.dict()
        )

        # Update metrics
        AZURE_API_CALLS.labels(
            service="rbac",
            operation="create_assignment",
            status="success").inc(
        )
        AZURE_API_DURATION.labels(service="rbac").observe(time.time() - start_time)

        return assignment
    except Exception as e:
        AZURE_API_CALLS.labels(service="rbac", operation="create_assignment", status="error").inc()
        logger.error("create_role_assignment_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create role assignment: {str(e)}"
        )


@app.delete("/api/v1/rbac/assignments/{assignment_id}")
async def delete_role_assignment(
    assignment_id: str,
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a role assignment."""
    try:
        start_time = time.time()
        await rbac_service.delete_role_assignment(
            subscription_id=subscription_id,
            assignment_id=assignment_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(
            service="rbac",
            operation="delete_assignment",
            status="success").inc(
        )
        AZURE_API_DURATION.labels(service="rbac").observe(time.time() - start_time)

        return {"message": f"Role assignment {assignment_id} deleted successfully"}
    except Exception as e:
        AZURE_API_CALLS.labels(service="rbac", operation="delete_assignment", status="error").inc()
        logger.error("delete_role_assignment_failed", error=str(e), assignment_id=assignment_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete role assignment: {str(e)}"
        )


# Cost Management endpoints
@app.get("/api/v1/costs/usage", response_model=CostResponse)
async def get_cost_usage(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    granularity: Optional[str] = Query("Daily", description="Granularity: Daily, Monthly"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get cost usage data."""
    try:
        start_time = time.time()
        cost_data = await cost_service.get_usage_details(
            subscription_id=subscription_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="cost", operation="usage", status="success").inc()
        AZURE_API_DURATION.labels(service="cost").observe(time.time() - start_time)

        return cost_data
    except Exception as e:
        AZURE_API_CALLS.labels(service="cost", operation="usage", status="error").inc()
        logger.error("get_cost_usage_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost usage: {str(e)}"
        )


@app.get("/api/v1/costs/forecast", response_model=CostResponse)
async def get_cost_forecast(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    forecast_days: int = Query(30, description="Number of days to forecast"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get cost forecast."""
    try:
        start_time = time.time()
        forecast = await cost_service.get_cost_forecast(
            subscription_id=subscription_id,
            forecast_days=forecast_days
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="cost", operation="forecast", status="success").inc()
        AZURE_API_DURATION.labels(service="cost").observe(time.time() - start_time)

        return forecast
    except Exception as e:
        AZURE_API_CALLS.labels(service="cost", operation="forecast", status="error").inc()
        logger.error("get_cost_forecast_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost forecast: {str(e)}"
        )


@app.get("/api/v1/costs/budgets", response_model=List[Dict[str, Any]])
async def list_budgets(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List cost budgets."""
    try:
        start_time = time.time()
        budgets = await cost_service.list_budgets(subscription_id=subscription_id)

        # Update metrics
        AZURE_API_CALLS.labels(service="cost", operation="list_budgets", status="success").inc()
        AZURE_API_DURATION.labels(service="cost").observe(time.time() - start_time)

        return budgets
    except Exception as e:
        AZURE_API_CALLS.labels(service="cost", operation="list_budgets", status="error").inc()
        logger.error("list_budgets_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list budgets: {str(e)}"
        )


@app.get("/api/v1/costs/recommendations", response_model=List[Dict[str, Any]])
async def get_cost_recommendations(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get cost optimization recommendations."""
    try:
        start_time = time.time()
        recommendations = await cost_service.get_cost_recommendations(
            subscription_id=subscription_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="cost", operation="recommendations", status="success").inc()
        AZURE_API_DURATION.labels(service="cost").observe(time.time() - start_time)

        return recommendations
    except Exception as e:
        AZURE_API_CALLS.labels(service="cost", operation="recommendations", status="error").inc()
        logger.error("get_cost_recommendations_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cost recommendations: {str(e)}"
        )


# Network Management endpoints
@app.get("/api/v1/networks", response_model=List[NetworkResponse])
async def list_networks(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    resource_group: Optional[str] = Query(None, description="Filter by resource group"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List virtual networks."""
    try:
        start_time = time.time()
        networks = await network_service.list_virtual_networks(
            subscription_id=subscription_id,
            resource_group=resource_group
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="network", operation="list", status="success").inc()
        AZURE_API_DURATION.labels(service="network").observe(time.time() - start_time)

        return networks
    except Exception as e:
        AZURE_API_CALLS.labels(service="network", operation="list", status="error").inc()
        logger.error("list_networks_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list networks: {str(e)}"
        )


@app.get("/api/v1/networks/security-groups", response_model=List[Dict[str, Any]])
async def list_network_security_groups(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    resource_group: Optional[str] = Query(None, description="Filter by resource group"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List network security groups."""
    try:
        start_time = time.time()
        nsgs = await network_service.list_network_security_groups(
            subscription_id=subscription_id,
            resource_group=resource_group
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="network", operation="list_nsgs", status="success").inc()
        AZURE_API_DURATION.labels(service="network").observe(time.time() - start_time)

        return nsgs
    except Exception as e:
        AZURE_API_CALLS.labels(service="network", operation="list_nsgs", status="error").inc()
        logger.error("list_network_security_groups_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list network security groups: {str(e)}"
        )


@app.get("/api/v1/networks/security-analysis", response_model=Dict[str, Any])
async def analyze_network_security(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    resource_group: Optional[str] = Query(None, description="Analyze specific resource group"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze network security configuration."""
    try:
        start_time = time.time()
        analysis = await network_service.analyze_network_security(
            subscription_id=subscription_id,
            resource_group=resource_group
        )

        # Update metrics
        AZURE_API_CALLS.labels(
            service="network",
            operation="security_analysis",
            status="success").inc(
        )
        AZURE_API_DURATION.labels(service="network").observe(time.time() - start_time)

        return analysis
    except Exception as e:
        AZURE_API_CALLS.labels(
            service="network",
            operation="security_analysis",
            status="error").inc(
        )
        logger.error("analyze_network_security_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze network security: {str(e)}"
        )


# Resource Management endpoints
@app.get("/api/v1/resources", response_model=List[ResourceResponse])
async def list_resources(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    resource_group: Optional[str] = Query(None, description="Filter by resource group"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    tags: Optional[Dict[str, str]] = Query(None, description="Filter by tags"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List Azure resources."""
    try:
        start_time = time.time()
        resources = await resource_service.list_resources(
            subscription_id=subscription_id,
            resource_group=resource_group,
            resource_type=resource_type,
            tags=tags
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="resource", operation="list", status="success").inc()
        AZURE_API_DURATION.labels(service="resource").observe(time.time() - start_time)

        return resources
    except Exception as e:
        AZURE_API_CALLS.labels(service="resource", operation="list", status="error").inc()
        logger.error("list_resources_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list resources: {str(e)}"
        )


@app.get("/api/v1/resources/{resource_id}", response_model=ResourceResponse)
async def get_resource(
    resource_id: str,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific resource."""
    try:
        start_time = time.time()
        resource = await resource_service.get_resource(resource_id=resource_id)

        # Update metrics
        AZURE_API_CALLS.labels(service="resource", operation="get", status="success").inc()
        AZURE_API_DURATION.labels(service="resource").observe(time.time() - start_time)

        return resource
    except Exception as e:
        AZURE_API_CALLS.labels(service="resource", operation="get", status="error").inc()
        logger.error("get_resource_failed", error=str(e), resource_id=resource_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource: {str(e)}"
        )


@app.get("/api/v1/resources/groups", response_model=List[Dict[str, Any]])
async def list_resource_groups(
    subscription_id: str = Query(..., description="Azure subscription ID"),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """List resource groups."""
    try:
        start_time = time.time()
        resource_groups = await resource_service.list_resource_groups(
            subscription_id=subscription_id
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="resource", operation="list_groups", status="success").inc()
        AZURE_API_DURATION.labels(service="resource").observe(time.time() - start_time)

        return resource_groups
    except Exception as e:
        AZURE_API_CALLS.labels(service="resource", operation="list_groups", status="error").inc()
        logger.error("list_resource_groups_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list resource groups: {str(e)}"
        )


@app.post("/api/v1/resources/tags/{resource_id}")
async def update_resource_tags(
    resource_id: str,
    tags: Dict[str, str],
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Update resource tags."""
    try:
        start_time = time.time()
        await resource_service.update_resource_tags(
            resource_id=resource_id,
            tags=tags
        )

        # Update metrics
        AZURE_API_CALLS.labels(service="resource", operation="update_tags", status="success").inc()
        AZURE_API_DURATION.labels(service="resource").observe(time.time() - start_time)

        return {"message": f"Tags updated successfully for resource {resource_id}"}
    except Exception as e:
        AZURE_API_CALLS.labels(service="resource", operation="update_tags", status="error").inc()
        logger.error("update_resource_tags_failed", error=str(e), resource_id=resource_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update resource tags: {str(e)}"
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        request_id=getattr(request.state, "request_id", "unknown")
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message=str(exc) if settings.debug else "An unexpected error occurred",
            request_id=getattr(request.state, "request_id", "unknown")
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.service.service_host,
        port=8001,  # Azure Integration service runs on port 8001
        workers=1,  # Single worker for development
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )
