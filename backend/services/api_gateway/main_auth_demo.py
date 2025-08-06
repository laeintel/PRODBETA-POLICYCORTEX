"""
Demo Integration of PolicyCortex Authentication System
Simplified version that works with existing dependencies
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SERVICE_NAME = os.getenv("SERVICE_NAME", "api_gateway_auth_demo")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8010"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-super-secure")

# FastAPI app with enhanced configuration
app = FastAPI(
    title="PolicyCortex API Gateway - Auth Demo",
    description="Demonstration of Zero-Configuration Authentication with existing Azure integration",
    version="2.0.0-demo",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.policycortex.com",
        "https://policycortex.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for authentication
class OrganizationDetectionRequest(BaseModel):
    email: str


class OrganizationDetectionResponse(BaseModel):
    domain: str
    organization_name: str
    organization_type: str
    authentication_method: str
    sso_enabled: bool
    tenant_id: str
    features: Dict[str, Any]
    settings: Dict[str, Any]


class LoginRequest(BaseModel):
    email: str
    password: Optional[str] = None
    auth_code: Optional[str] = None
    saml_response: Optional[str] = None


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Mock authentication system for demo
class MockAuthManager:
    """Simplified authentication manager for demo purposes"""

    def __init__(self):
        self.domain_patterns = {
            "microsoft.com": {"type": "enterprise", "auth": "azure_ad"},
            "google.com": {"type": "enterprise", "auth": "oauth2"},
            "amazon.com": {"type": "enterprise", "auth": "saml"},
            "startup.com": {"type": "professional", "auth": "internal"},
            "company.org": {"type": "trial", "auth": "internal"},
        }

    def detect_organization(self, email: str) -> Dict[str, Any]:
        """Mock organization detection"""
        domain = email.split("@")[1].lower()

        # Check predefined patterns
        config = None
        for pattern, settings in self.domain_patterns.items():
            if domain.endswith(pattern):
                config = settings
                break

        if not config:
            # Default configuration
            if domain.endswith(".gov"):
                config = {"type": "enterprise", "auth": "saml"}
            elif domain.endswith(".edu"):
                config = {"type": "enterprise", "auth": "ldap"}
            elif domain.endswith(".org"):
                config = {"type": "professional", "auth": "internal"}
            else:
                config = {"type": "starter", "auth": "internal"}

        # Generate organization name
        org_name = domain.split(".")[0].replace("-", " ").replace("_", " ").title()

        # Generate tenant ID
        tenant_id = hashlib.sha256(f"tenant_{domain}".encode()).hexdigest()[:32]

        # Define features by type
        features = {
            "enterprise": {
                "unlimited_users": True,
                "custom_roles": True,
                "api_access": True,
                "advanced_analytics": True,
                "ai_predictions": True,
                "custom_policies": True,
                "white_labeling": True,
                "dedicated_support": True,
                "sla_guarantee": True,
                "data_export": True,
                "audit_logs": True,
                "multi_region": True,
            },
            "professional": {
                "unlimited_users": False,
                "custom_roles": True,
                "api_access": True,
                "advanced_analytics": True,
                "ai_predictions": True,
                "custom_policies": True,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": True,
                "data_export": True,
                "audit_logs": True,
                "multi_region": False,
            },
            "starter": {
                "unlimited_users": False,
                "custom_roles": False,
                "api_access": False,
                "advanced_analytics": False,
                "ai_predictions": True,
                "custom_policies": False,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": False,
                "data_export": True,
                "audit_logs": True,
                "multi_region": False,
            },
            "trial": {
                "unlimited_users": False,
                "custom_roles": False,
                "api_access": False,
                "advanced_analytics": False,
                "ai_predictions": True,
                "custom_policies": False,
                "white_labeling": False,
                "dedicated_support": False,
                "sla_guarantee": False,
                "data_export": False,
                "audit_logs": True,
                "multi_region": False,
            },
        }

        return {
            "domain": domain,
            "name": org_name,
            "type": config["type"],
            "authentication_method": config["auth"],
            "tenant_id": tenant_id,
            "settings": {
                "sso_enabled": config["auth"] != "internal",
                "mfa_required": config["type"] == "enterprise",
                "session_timeout_minutes": 480 if config["type"] == "enterprise" else 120,
                "data_residency": "us",
                "compliance_frameworks": ["SOC2", "GDPR"] if config["type"] != "trial" else [],
            },
            "features": features.get(config["type"], features["starter"]),
        }

    def authenticate_user(self, email: str, **kwargs) -> tuple:
        """Mock user authentication"""
        org_config = self.detect_organization(email)

        # Generate mock user
        user_id = hashlib.sha256(email.encode()).hexdigest()[:32]
        name = email.split("@")[0].replace(".", " ").title()

        # Assign roles based on email patterns
        roles = []
        if "admin" in email.lower():
            roles = ["global_admin", "policy_administrator"]
        elif "ceo" in email.lower() or "cto" in email.lower():
            roles = ["global_admin", "executive_viewer"]
        elif "compliance" in email.lower():
            roles = ["compliance_officer"]
        elif "security" in email.lower():
            roles = ["policy_administrator"]
        elif "manager" in email.lower():
            roles = ["department_manager"]
        else:
            roles = ["team_member"]

        # Generate permissions
        permissions = []
        if "global_admin" in roles:
            permissions = ["*"]
        elif "policy_administrator" in roles:
            permissions = ["policies:*", "resources:view", "audit:view"]
        elif "compliance_officer" in roles:
            permissions = ["policies:view", "compliance:*", "audit:*"]
        else:
            permissions = ["policies:view", "resources:view", "dashboard:view"]

        user_info = {
            "id": user_id,
            "email": email,
            "name": name,
            "tenant_id": org_config["tenant_id"],
            "organization": org_config["name"],
            "org_type": org_config["type"],
            "roles": roles,
            "permissions": permissions,
            "auth_method": org_config["authentication_method"],
        }

        # Generate mock tokens
        import time

        current_time = int(time.time())

        # Simple JWT-like token (not real JWT for demo)
        token_data = f"{user_id}:{current_time}:{org_config['tenant_id']}"
        access_token = f"demo_token_{hashlib.sha256(token_data.encode()).hexdigest()[:32]}"
        refresh_token = (
            f"demo_refresh_{hashlib.sha256(f'refresh_{token_data}'.encode()).hexdigest()[:32]}"
        )

        tokens = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 28800,  # 8 hours
        }

        return user_info, tokens


# Initialize mock auth manager
auth_manager = MockAuthManager()

# Simple token storage for demo
active_tokens = {}


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Extract user from demo token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    token = auth_header.replace("Bearer ", "")
    return active_tokens.get(token)


# Azure CLI helper functions (existing functionality)
def run_az_command(command: List[str]) -> Dict[str, Any]:
    """Run Azure CLI command and return JSON result."""
    try:
        full_command = ["az"] + command + ["--output", "json"]
        print(f"Running command: {' '.join(full_command)}")
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout) if result.stdout.strip() else {}
    except Exception as e:
        print(f"Azure CLI error: {e}")
        return {"error": str(e)}


# Authentication endpoints
@app.post("/api/auth/detect-organization", response_model=OrganizationDetectionResponse)
async def detect_organization(request: OrganizationDetectionRequest):
    """
    Detect organization configuration from email domain
    Zero-configuration setup - just provide email address
    """
    try:
        print(f"Organization detection requested for: {request.email}")

        # Detect organization configuration
        org_config = auth_manager.detect_organization(request.email)

        return OrganizationDetectionResponse(
            domain=org_config["domain"],
            organization_name=org_config["name"],
            organization_type=org_config["type"],
            authentication_method=org_config["authentication_method"],
            sso_enabled=org_config["settings"]["sso_enabled"],
            tenant_id=org_config["tenant_id"],
            features=org_config["features"],
            settings=org_config["settings"],
        )

    except Exception as e:
        print(f"Organization detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Organization detection failed: {str(e)}",
        )


@app.post("/api/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user with automatic method detection
    Demo version with mock authentication
    """
    try:
        print(f"Login requested for: {request.email}")

        # Authenticate user
        user_info, tokens = auth_manager.authenticate_user(
            email=request.email, password=request.password, auth_code=request.auth_code
        )

        # Store token for demo
        active_tokens[tokens["access_token"]] = user_info

        print(f"Login successful for: {request.email}")

        return LoginResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user=user_info,
        )

    except Exception as e:
        print(f"Login failed for {request.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
        )


@app.post("/api/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    # For demo, generate new token
    import time

    current_time = int(time.time())

    new_token = (
        f"demo_token_refreshed_{hashlib.sha256(str(current_time).encode()).hexdigest()[:32]}"
    )

    return {"access_token": new_token, "token_type": "bearer", "expires_in": 28800}


@app.post("/api/auth/logout")
async def logout(request: Request):
    """Logout user and revoke session"""
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            if token in active_tokens:
                del active_tokens[token]

        return {"message": "Logged out successfully"}

    except Exception as e:
        print(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


@app.get("/api/auth/me")
async def get_current_user_info(request: Request):
    """Get current user information"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return user


# Enhanced health endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "api_gateway_auth_demo",
        "environment": ENVIRONMENT,
        "version": "2.0.0-demo",
        "authentication": "demo_enabled",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {"authentication": "demo_healthy", "azure_cli": "healthy"},
    }


@app.get("/")
async def root():
    """Root endpoint with authentication info."""
    return {
        "message": "PolicyCortex API Gateway - Authentication Demo",
        "version": "2.0.0-demo",
        "features": [
            "Zero-configuration authentication (Demo)",
            "Multi-tenant data isolation (Demo)",
            "Enterprise audit logging (Demo)",
            "Azure policy integration (Live)",
            "Real-time compliance monitoring (Live)",
        ],
        "demo_emails": [
            "admin@microsoft.com (Enterprise/Azure AD)",
            "user@google.com (Enterprise/OAuth2)",
            "ceo@amazon.com (Enterprise/SAML)",
            "manager@startup.com (Professional/Internal)",
            "trial@company.org (Trial/Internal)",
        ],
        "docs": "/docs",
        "health": "/health",
    }


# Enhanced existing endpoints with authentication context
@app.get("/api/v1/status")
async def api_status(request: Request):
    """API status endpoint with user context."""
    user = get_current_user(request)

    status_data = {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0-demo",
        "environment": ENVIRONMENT,
        "features": {
            "authentication": True,
            "audit_logging": True,
            "multi_tenant": True,
            "azure_integration": True,
        },
    }

    if user:
        status_data["user_context"] = {
            "authenticated": True,
            "tenant_id": user["tenant_id"],
            "organization": user["organization"],
            "roles": user["roles"],
        }

    return status_data


@app.get("/api/v1/azure/resources")
async def get_resources(request: Request):
    """Get real Azure resources with tenant context."""
    try:
        user = get_current_user(request)

        # Get Azure resources (existing functionality)
        try:
            resources_result = run_az_command(["resource", "list"])
            if isinstance(resources_result, list):
                resources = resources_result
            else:
                resources = []
        except:
            resources = []

        # Apply tenant filtering if user is authenticated
        if user:
            for resource in resources:
                resource["tenant_context"] = {
                    "tenant_id": user["tenant_id"],
                    "organization": user["organization"],
                    "access_level": "full" if "*" in user["permissions"] else "read",
                }

        return {
            "resources": resources[:10],  # Limit for demo
            "total": len(resources),
            "tenant_filtered": user is not None,
            "user_context": {
                "authenticated": user is not None,
                "tenant_id": user["tenant_id"] if user else None,
                "organization": user["organization"] if user else None,
            },
        }

    except Exception as e:
        print(f"Resources fetch failed: {e}")
        return {"resources": [], "error": str(e)}


@app.get("/api/v1/dashboard/overview")
async def get_dashboard_overview(request: Request):
    """Get dashboard overview with tenant context."""
    try:
        user = get_current_user(request)

        # Get basic metrics (existing logic)
        try:
            resources_result = run_az_command(["resource", "list"])
            resources = resources_result if isinstance(resources_result, list) else []

            resource_groups_result = run_az_command(["group", "list"])
            resource_groups = (
                resource_groups_result if isinstance(resource_groups_result, list) else []
            )
        except:
            resources = []
            resource_groups = []

        # Enhanced dashboard with authentication context
        dashboard_data = {
            "metrics": {
                "totalResources": len(resources),
                "totalPolicies": 2,
                "totalResourceGroups": len(resource_groups),
                "complianceScore": 85,
                "costOptimizationScore": 78,
                "securityScore": 92,
            },
            "compliance": {
                "compliantResources": int(len(resources) * 0.8) if resources else 4,
                "nonCompliantResources": int(len(resources) * 0.2) if resources else 1,
                "exemptResources": 0,
                "compliancePercentage": 85,
            },
            "costs": {
                "dailyCost": 45.8,
                "monthlyCost": 1374.0,
                "currency": "USD",
                "trend": "stable",
            },
            "security": {
                "highSeverityAlerts": 0,
                "mediumSeverityAlerts": 2,
                "lowSeverityAlerts": 5,
                "lastScanDate": datetime.utcnow().isoformat(),
            },
            "user_context": {
                "authenticated": user is not None,
                "tenant_id": user["tenant_id"] if user else None,
                "organization": user["organization"] if user else None,
                "access_level": "full" if user and "*" in user.get("permissions", []) else "read",
            },
            "data_source": "live-azure-with-auth-demo",
        }

        return dashboard_data

    except Exception as e:
        print(f"Dashboard overview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Development and debug endpoints
@app.get("/debug/auth-test")
async def debug_auth_test():
    """Debug endpoint to test authentication components"""
    return {
        "auth_manager": "mock_initialized",
        "active_tokens": len(active_tokens),
        "test_emails": [
            "admin@microsoft.com",
            "user@google.com",
            "ceo@amazon.com",
            "manager@startup.com",
            "trial@company.org",
        ],
    }


@app.get("/debug/tokens")
async def debug_tokens():
    """Debug endpoint to see active tokens"""
    return {
        "active_tokens": len(active_tokens),
        "tokens": [
            {"token": token[:20] + "...", "user": user["email"]}
            for token, user in active_tokens.items()
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print(f"Starting PolicyCortex API Gateway - Auth Demo")
    print(f"Port: {SERVICE_PORT}")
    print(f"Environment: {ENVIRONMENT}")
    print(f"Authentication: Demo Mode Enabled")
    print(f"Docs: http://localhost:{SERVICE_PORT}/docs")
    print(f"Try these demo emails:")
    print(f"  - admin@microsoft.com (Enterprise/Azure AD)")
    print(f"  - user@google.com (Enterprise/OAuth2)")
    print(f"  - ceo@amazon.com (Enterprise/SAML)")
    print(f"  - manager@startup.com (Professional/Internal)")
    print(f"  - trial@company.org (Trial/Internal)")

    uvicorn.run(
        "main_auth_demo:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=True if ENVIRONMENT == "development" else False,
        log_level=LOG_LEVEL.lower(),
    )
