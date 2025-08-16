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
Enhanced Authentication and Authorization Middleware for PolicyCortex
Implements proper token validation, tenant isolation, and resource-level authorization
"""

import os
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from functools import wraps
import asyncio

from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
import hashlib
import json

logger = logging.getLogger(__name__)

# Security settings
security = HTTPBearer(auto_error=False)

# Configuration
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID") or os.getenv("NEXT_PUBLIC_AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID") or os.getenv("NEXT_PUBLIC_AZURE_CLIENT_ID")
API_AUDIENCE = os.getenv("API_AUDIENCE") or os.getenv("NEXT_PUBLIC_API_AUDIENCE")
ALLOWED_AUDIENCES = [a for a in [API_AUDIENCE, AZURE_CLIENT_ID] if a]
REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"
ENFORCE_TENANT_ISOLATION = os.getenv("ENFORCE_TENANT_ISOLATION", "true").lower() == "true"
ENABLE_RESOURCE_AUTHZ = os.getenv("ENABLE_RESOURCE_AUTHZ", "true").lower() == "true"

# Cache for JWKS keys
JWKS_CACHE: Dict[str, Any] = {}
JWKS_CACHE_TTL = timedelta(hours=24)
JWKS_CACHE_TIME: Dict[str, datetime] = {}

# Cache for user permissions
PERMISSION_CACHE: Dict[str, Dict] = {}
PERMISSION_CACHE_TTL = timedelta(minutes=15)

class AuthContext:
    """Authentication context for the request"""
    def __init__(self, claims: Dict[str, Any]):
        self.claims = claims
        self.user_id = claims.get("oid") or claims.get("sub") or "anonymous"
        self.tenant_id = claims.get("tid") or "default"
        self.email = claims.get("email") or claims.get("preferred_username") or ""
        self.name = claims.get("name") or ""
        self.roles = self._extract_roles(claims)
        self.scopes = self._extract_scopes(claims)
        self.is_authenticated = bool(claims)
        self.is_admin = self._check_admin()
        self.session_id = self._generate_session_id()
        
    def _extract_roles(self, claims: Dict) -> Set[str]:
        """Extract roles from claims"""
        roles = set()
        
        # App roles
        if "roles" in claims:
            if isinstance(claims["roles"], list):
                roles.update(claims["roles"])
            elif isinstance(claims["roles"], str):
                roles.add(claims["roles"])
        
        # Group membership (could map to roles)
        if "groups" in claims:
            groups = claims.get("groups", [])
            if isinstance(groups, list):
                # Map known groups to roles
                for group in groups:
                    if "admin" in group.lower():
                        roles.add("admin")
                    elif "viewer" in group.lower():
                        roles.add("viewer")
                    elif "contributor" in group.lower():
                        roles.add("contributor")
        
        # Default role if authenticated
        if self.is_authenticated and not roles:
            roles.add("user")
            
        return roles
    
    def _extract_scopes(self, claims: Dict) -> Set[str]:
        """Extract scopes from claims"""
        scopes = set()
        
        # OAuth2 scopes
        scp = claims.get("scp", "")
        if isinstance(scp, str):
            scopes.update(scp.split(" "))
        
        # Application permissions
        if "permissions" in claims:
            perms = claims.get("permissions", [])
            if isinstance(perms, list):
                scopes.update(perms)
                
        return scopes
    
    def _check_admin(self) -> bool:
        """Check if user has admin privileges"""
        admin_indicators = ["admin", "administrator", "global_admin"]
        
        # Check roles
        for role in self.roles:
            if any(ind in role.lower() for ind in admin_indicators):
                return True
        
        # Check specific admin scope
        admin_scopes = ["admin.all", "directory.readwrite.all"]
        for scope in self.scopes:
            if any(s in scope.lower() for s in admin_scopes):
                return True
                
        return False
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID for correlation"""
        data = f"{self.user_id}:{self.tenant_id}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def has_role(self, role: str) -> bool:
        """Check if user has a specific role"""
        return role.lower() in {r.lower() for r in self.roles}
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if user has any of the specified roles"""
        user_roles = {r.lower() for r in self.roles}
        required_roles = {r.lower() for r in roles}
        return bool(user_roles & required_roles)
    
    def has_scope(self, scope: str) -> bool:
        """Check if user has a specific scope"""
        return scope in self.scopes
    
    def can_access_tenant(self, tenant_id: str) -> bool:
        """Check if user can access a specific tenant"""
        if self.is_admin:
            return True
        return tenant_id == self.tenant_id
    
    def can_access_resource(self, resource: Dict[str, Any]) -> bool:
        """Check if user can access a specific resource"""
        if not ENABLE_RESOURCE_AUTHZ:
            return True
            
        if self.is_admin:
            return True
        
        # Check tenant match
        resource_tenant = resource.get("tenant_id") or resource.get("tenantId")
        if resource_tenant and resource_tenant != self.tenant_id:
            return False
        
        # Check owner
        resource_owner = resource.get("owner_id") or resource.get("created_by")
        if resource_owner == self.user_id:
            return True
        
        # Check resource-specific permissions
        if "permissions" in resource:
            perms = resource["permissions"]
            if isinstance(perms, dict):
                # Check user permissions
                if self.user_id in perms.get("users", []):
                    return True
                # Check role permissions
                for role in self.roles:
                    if role in perms.get("roles", []):
                        return True
        
        # Default: check if user has read scope
        return self.has_scope("read") or self.has_role("viewer")

async def _get_jwks(tenant_id: str) -> Dict[str, Any]:
    """Get JWKS keys for tenant with caching"""
    # Check cache
    if tenant_id in JWKS_CACHE:
        cache_time = JWKS_CACHE_TIME.get(tenant_id)
        if cache_time and datetime.utcnow() - cache_time < JWKS_CACHE_TTL:
            return JWKS_CACHE[tenant_id]
    
    # Fetch new keys
    url = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            if resp.status != 200:
                raise HTTPException(500, f"Failed to fetch JWKS: HTTP {resp.status}")
            data = await resp.json()
            
            # Update cache
            JWKS_CACHE[tenant_id] = data
            JWKS_CACHE_TIME[tenant_id] = datetime.utcnow()
            
            return data

def _get_rsa_key(token: str, jwks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get RSA key for token verification"""
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return {
                    "kty": key.get("kty"),
                    "kid": key.get("kid"),
                    "use": key.get("use"),
                    "n": key.get("n"),
                    "e": key.get("e"),
                }
    except Exception as e:
        logger.error(f"Error getting RSA key: {e}")
    
    return None

async def validate_token(token: str) -> Dict[str, Any]:
    """Validate Azure AD JWT token"""
    if not AZURE_TENANT_ID or not ALLOWED_AUDIENCES:
        raise HTTPException(500, "Authentication not properly configured")
    
    issuer = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"
    
    # Get JWKS keys
    jwks = await _get_jwks(AZURE_TENANT_ID)
    rsa_key = _get_rsa_key(token, jwks)
    
    if not rsa_key:
        # Refresh keys once and retry
        jwks = await _get_jwks(AZURE_TENANT_ID)
        rsa_key = _get_rsa_key(token, jwks)
        if not rsa_key:
            raise HTTPException(401, "Unable to verify token signature")
    
    try:
        # Validate token
        claims = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=ALLOWED_AUDIENCES if len(ALLOWED_AUDIENCES) > 1 else ALLOWED_AUDIENCES[0],
            issuer=issuer,
            options={"verify_aud": True, "verify_signature": True, "verify_exp": True}
        )
        
        return claims
        
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(401, "Invalid or expired token")
    except Exception as e:
        logger.error(f"Unexpected error during token validation: {e}")
        raise HTTPException(500, "Authentication validation failed")

async def get_auth_context(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthContext:
    """Get authentication context from request"""
    
    # Check if auth is required
    if not REQUIRE_AUTH:
        # Return anonymous context
        return AuthContext({})
    
    # Extract token
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif "Authorization" in request.headers:
        auth_header = request.headers["Authorization"]
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
    
    if not token:
        raise HTTPException(401, "Authentication required")
    
    # Validate token
    try:
        claims = await validate_token(token)
        return AuthContext(claims)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(401, "Authentication failed")

def require_auth(func):
    """Decorator to require authentication"""
    @wraps(func)
    async def wrapper(*args, auth: AuthContext = Depends(get_auth_context), **kwargs):
        if not auth.is_authenticated:
            raise HTTPException(401, "Authentication required")
        return await func(*args, auth=auth, **kwargs)
    return wrapper

def require_roles(*roles):
    """Decorator to require specific roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, auth: AuthContext = Depends(get_auth_context), **kwargs):
            if not auth.is_authenticated:
                raise HTTPException(401, "Authentication required")
            if not auth.has_any_role(roles) and not auth.is_admin:
                raise HTTPException(403, f"Required role(s): {', '.join(roles)}")
            return await func(*args, auth=auth, **kwargs)
        return wrapper
    return decorator

def require_admin(func):
    """Decorator to require admin privileges"""
    @wraps(func)
    async def wrapper(*args, auth: AuthContext = Depends(get_auth_context), **kwargs):
        if not auth.is_authenticated:
            raise HTTPException(401, "Authentication required")
        if not auth.is_admin:
            raise HTTPException(403, "Admin privileges required")
        return await func(*args, auth=auth, **kwargs)
    return wrapper

class TenantIsolation:
    """Middleware for tenant isolation in database queries"""
    
    @staticmethod
    def apply_filter(query, auth: AuthContext, tenant_column="tenant_id"):
        """Apply tenant filter to SQLAlchemy query"""
        if not ENFORCE_TENANT_ISOLATION:
            return query
        
        if auth.is_admin:
            # Admins can see all tenants
            return query
        
        # Filter by user's tenant
        return query.filter(tenant_column == auth.tenant_id)
    
    @staticmethod
    def validate_resource(resource: Dict[str, Any], auth: AuthContext) -> bool:
        """Validate that resource belongs to user's tenant"""
        if not ENFORCE_TENANT_ISOLATION:
            return True
        
        if auth.is_admin:
            return True
        
        resource_tenant = resource.get("tenant_id") or resource.get("tenantId")
        if not resource_tenant:
            # No tenant specified, allow for backward compatibility
            return True
        
        return resource_tenant == auth.tenant_id
    
    @staticmethod
    def inject_tenant(data: Dict[str, Any], auth: AuthContext) -> Dict[str, Any]:
        """Inject tenant_id into data"""
        if ENFORCE_TENANT_ISOLATION and not auth.is_admin:
            data["tenant_id"] = auth.tenant_id
        return data

class ResourceAuthorization:
    """Resource-level authorization checks"""
    
    @staticmethod
    async def check_read(resource_id: str, auth: AuthContext, db: AsyncSession) -> bool:
        """Check if user can read a resource"""
        if not ENABLE_RESOURCE_AUTHZ:
            return True
        
        if auth.is_admin:
            return True
        
        # Add specific resource checks here
        return auth.has_scope("read") or auth.has_role("viewer")
    
    @staticmethod
    async def check_write(resource_id: str, auth: AuthContext, db: AsyncSession) -> bool:
        """Check if user can write to a resource"""
        if not ENABLE_RESOURCE_AUTHZ:
            return True
        
        if auth.is_admin:
            return True
        
        # Add specific resource checks here
        return auth.has_scope("write") or auth.has_role("contributor")
    
    @staticmethod
    async def check_delete(resource_id: str, auth: AuthContext, db: AsyncSession) -> bool:
        """Check if user can delete a resource"""
        if not ENABLE_RESOURCE_AUTHZ:
            return True
        
        if auth.is_admin:
            return True
        
        # Add specific resource checks here
        return auth.has_scope("delete") or auth.has_role("owner")

# Export main components
__all__ = [
    "AuthContext",
    "get_auth_context",
    "require_auth",
    "require_roles",
    "require_admin",
    "TenantIsolation",
    "ResourceAuthorization",
    "validate_token"
]