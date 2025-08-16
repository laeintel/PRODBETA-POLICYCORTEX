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
Comprehensive tests for authentication and authorization middleware
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from jose import jwt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth_middleware import (
    AuthContext,
    validate_token,
    get_auth_context,
    TenantIsolation,
    ResourceAuthorization,
    require_auth,
    require_roles,
    require_admin
)

# Test configuration
TEST_TENANT_ID = "test-tenant-123"
TEST_CLIENT_ID = "test-client-456"
TEST_AUDIENCE = "api://test-audience"
TEST_USER_ID = "user-789"
TEST_SECRET = "test-secret-key"

class TestAuthContext:
    """Test AuthContext class"""
    
    def test_create_auth_context_from_valid_claims(self):
        """Test creating auth context from valid JWT claims"""
        claims = {
            "tid": TEST_TENANT_ID,
            "oid": TEST_USER_ID,
            "sub": TEST_USER_ID,
            "email": "test@example.com",
            "name": "Test User",
            "roles": ["admin", "user"],
            "scp": "read write delete",
            "groups": ["admin-group"]
        }
        
        context = AuthContext(claims)
        
        assert context.tenant_id == TEST_TENANT_ID
        assert context.user_id == TEST_USER_ID
        assert context.email == "test@example.com"
        assert context.name == "Test User"
        assert context.is_authenticated
        assert context.is_admin
        assert "admin" in context.roles
        assert "user" in context.roles
        assert "read" in context.scopes
        assert "write" in context.scopes
    
    def test_create_auth_context_anonymous(self):
        """Test creating anonymous auth context"""
        context = AuthContext({})
        
        assert context.user_id == "anonymous"
        assert context.tenant_id == "default"
        assert not context.is_authenticated
        assert not context.is_admin
        assert len(context.roles) == 0
        assert len(context.scopes) == 0
    
    def test_has_role(self):
        """Test role checking"""
        claims = {
            "roles": ["contributor", "viewer"]
        }
        context = AuthContext(claims)
        
        assert context.has_role("contributor")
        assert context.has_role("viewer")
        assert not context.has_role("admin")
        assert context.has_any_role(["admin", "contributor"])
        assert not context.has_any_role(["admin", "owner"])
    
    def test_has_scope(self):
        """Test scope checking"""
        claims = {
            "scp": "read write",
            "permissions": ["delete", "update"]
        }
        context = AuthContext(claims)
        
        assert context.has_scope("read")
        assert context.has_scope("write")
        assert context.has_scope("delete")
        assert context.has_scope("update")
        assert not context.has_scope("admin")
    
    def test_can_access_tenant(self):
        """Test tenant access control"""
        claims = {
            "tid": TEST_TENANT_ID,
            "roles": ["user"]
        }
        context = AuthContext(claims)
        
        assert context.can_access_tenant(TEST_TENANT_ID)
        assert not context.can_access_tenant("other-tenant")
        
        # Admin can access any tenant
        admin_claims = {
            "tid": TEST_TENANT_ID,
            "roles": ["admin"]
        }
        admin_context = AuthContext(admin_claims)
        assert admin_context.can_access_tenant("other-tenant")
    
    def test_can_access_resource(self):
        """Test resource access control"""
        claims = {
            "tid": TEST_TENANT_ID,
            "oid": TEST_USER_ID,
            "roles": ["viewer"]
        }
        context = AuthContext(claims)
        
        # Can access resource in same tenant
        resource = {
            "tenant_id": TEST_TENANT_ID,
            "owner_id": "other-user"
        }
        assert context.can_access_resource(resource)
        
        # Cannot access resource in different tenant
        resource = {
            "tenant_id": "other-tenant",
            "owner_id": "other-user"
        }
        assert not context.can_access_resource(resource)
        
        # Can access owned resource
        resource = {
            "tenant_id": "other-tenant",
            "owner_id": TEST_USER_ID
        }
        assert context.can_access_resource(resource)

class TestTokenValidation:
    """Test JWT token validation"""
    
    @pytest.mark.asyncio
    async def test_validate_token_with_valid_signature(self):
        """Test validating token with valid signature"""
        # Create a test token
        claims = {
            "tid": TEST_TENANT_ID,
            "oid": TEST_USER_ID,
            "aud": TEST_AUDIENCE,
            "iss": f"https://login.microsoftonline.com/{TEST_TENANT_ID}/v2.0",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow()
        }
        
        token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
        
        # Mock JWKS response
        with patch('auth_middleware._get_jwks') as mock_jwks:
            mock_jwks.return_value = {
                "keys": [{
                    "kty": "RSA",
                    "kid": "test-key",
                    "use": "sig",
                    "n": "test-n",
                    "e": "test-e"
                }]
            }
            
            with patch('jose.jwt.decode') as mock_decode:
                mock_decode.return_value = claims
                
                # Should not raise exception with proper config
                with patch.dict(os.environ, {
                    'AZURE_TENANT_ID': TEST_TENANT_ID,
                    'API_AUDIENCE': TEST_AUDIENCE
                }):
                    result = await validate_token(token)
                    assert result == claims
    
    @pytest.mark.asyncio
    async def test_validate_token_expired(self):
        """Test validating expired token"""
        claims = {
            "tid": TEST_TENANT_ID,
            "oid": TEST_USER_ID,
            "aud": TEST_AUDIENCE,
            "iss": f"https://login.microsoftonline.com/{TEST_TENANT_ID}/v2.0",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired
            "iat": datetime.utcnow() - timedelta(hours=2),
            "nbf": datetime.utcnow() - timedelta(hours=2)
        }
        
        token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
        
        with patch('auth_middleware._get_jwks') as mock_jwks:
            mock_jwks.return_value = {"keys": []}
            
            with pytest.raises(Exception):
                await validate_token(token)
    
    @pytest.mark.asyncio
    async def test_validate_token_wrong_audience(self):
        """Test validating token with wrong audience"""
        claims = {
            "tid": TEST_TENANT_ID,
            "oid": TEST_USER_ID,
            "aud": "wrong-audience",
            "iss": f"https://login.microsoftonline.com/{TEST_TENANT_ID}/v2.0",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow()
        }
        
        token = jwt.encode(claims, TEST_SECRET, algorithm="HS256")
        
        with patch('auth_middleware._get_jwks') as mock_jwks:
            mock_jwks.return_value = {"keys": []}
            
            with patch.dict(os.environ, {
                'AZURE_TENANT_ID': TEST_TENANT_ID,
                'API_AUDIENCE': TEST_AUDIENCE
            }):
                with pytest.raises(Exception):
                    await validate_token(token)

class TestGetAuthContext:
    """Test get_auth_context function"""
    
    @pytest.mark.asyncio
    async def test_get_auth_context_with_valid_token(self):
        """Test getting auth context with valid bearer token"""
        request = Mock()
        request.headers = {"Authorization": f"Bearer valid-token"}
        
        credentials = Mock()
        credentials.credentials = "valid-token"
        
        with patch('auth_middleware.validate_token') as mock_validate:
            mock_validate.return_value = {
                "tid": TEST_TENANT_ID,
                "oid": TEST_USER_ID
            }
            
            with patch.dict(os.environ, {'REQUIRE_AUTH': 'true'}):
                context = await get_auth_context(request, credentials)
                
                assert context.tenant_id == TEST_TENANT_ID
                assert context.user_id == TEST_USER_ID
                assert context.is_authenticated
    
    @pytest.mark.asyncio
    async def test_get_auth_context_without_auth_required(self):
        """Test getting auth context when auth is not required"""
        request = Mock()
        request.headers = {}
        
        with patch.dict(os.environ, {'REQUIRE_AUTH': 'false'}):
            context = await get_auth_context(request, None)
            
            assert not context.is_authenticated
            assert context.user_id == "anonymous"
    
    @pytest.mark.asyncio
    async def test_get_auth_context_missing_token(self):
        """Test getting auth context with missing token when auth required"""
        request = Mock()
        request.headers = {}
        
        with patch.dict(os.environ, {'REQUIRE_AUTH': 'true'}):
            with pytest.raises(Exception) as exc_info:
                await get_auth_context(request, None)
            
            assert "Authentication required" in str(exc_info.value)

class TestTenantIsolation:
    """Test tenant isolation functionality"""
    
    def test_apply_filter_regular_user(self):
        """Test applying tenant filter for regular user"""
        auth = AuthContext({
            "tid": TEST_TENANT_ID,
            "roles": ["user"]
        })
        
        # Mock query
        query = Mock()
        query.filter = Mock(return_value=query)
        
        result = TenantIsolation.apply_filter(query, auth)
        
        query.filter.assert_called_once()
        assert result == query
    
    def test_apply_filter_admin_user(self):
        """Test applying tenant filter for admin user"""
        auth = AuthContext({
            "tid": TEST_TENANT_ID,
            "roles": ["admin"]
        })
        
        # Mock query
        query = Mock()
        
        result = TenantIsolation.apply_filter(query, auth)
        
        # Admin should not have filter applied
        query.filter.assert_not_called()
        assert result == query
    
    def test_validate_resource_same_tenant(self):
        """Test validating resource in same tenant"""
        auth = AuthContext({
            "tid": TEST_TENANT_ID,
            "roles": ["user"]
        })
        
        resource = {"tenant_id": TEST_TENANT_ID}
        assert TenantIsolation.validate_resource(resource, auth)
        
        resource = {"tenant_id": "other-tenant"}
        assert not TenantIsolation.validate_resource(resource, auth)
    
    def test_inject_tenant(self):
        """Test injecting tenant ID into data"""
        auth = AuthContext({
            "tid": TEST_TENANT_ID,
            "roles": ["user"]
        })
        
        data = {"name": "test"}
        result = TenantIsolation.inject_tenant(data, auth)
        
        assert result["tenant_id"] == TEST_TENANT_ID
        assert result["name"] == "test"

class TestResourceAuthorization:
    """Test resource-level authorization"""
    
    @pytest.mark.asyncio
    async def test_check_read_permission(self):
        """Test checking read permission"""
        auth = AuthContext({
            "roles": ["viewer"],
            "scp": "read"
        })
        
        db = Mock()
        
        # User with viewer role or read scope can read
        assert await ResourceAuthorization.check_read("resource-1", auth, db)
        
        # User without proper permissions cannot read
        auth = AuthContext({
            "roles": [],
            "scp": ""
        })
        assert not await ResourceAuthorization.check_read("resource-1", auth, db)
    
    @pytest.mark.asyncio
    async def test_check_write_permission(self):
        """Test checking write permission"""
        auth = AuthContext({
            "roles": ["contributor"],
            "scp": "write"
        })
        
        db = Mock()
        
        # User with contributor role or write scope can write
        assert await ResourceAuthorization.check_write("resource-1", auth, db)
        
        # User with only viewer role cannot write
        auth = AuthContext({
            "roles": ["viewer"],
            "scp": "read"
        })
        assert not await ResourceAuthorization.check_write("resource-1", auth, db)
    
    @pytest.mark.asyncio
    async def test_check_delete_permission(self):
        """Test checking delete permission"""
        auth = AuthContext({
            "roles": ["owner"],
            "scp": "delete"
        })
        
        db = Mock()
        
        # User with owner role or delete scope can delete
        assert await ResourceAuthorization.check_delete("resource-1", auth, db)
        
        # Admin can always delete
        auth = AuthContext({
            "roles": ["admin"]
        })
        assert await ResourceAuthorization.check_delete("resource-1", auth, db)
        
        # Regular user cannot delete
        auth = AuthContext({
            "roles": ["contributor"],
            "scp": "write"
        })
        assert not await ResourceAuthorization.check_delete("resource-1", auth, db)

class TestDecorators:
    """Test authentication decorators"""
    
    @pytest.mark.asyncio
    async def test_require_auth_decorator(self):
        """Test require_auth decorator"""
        @require_auth
        async def protected_function(auth: AuthContext):
            return f"Hello {auth.user_id}"
        
        # With authentication
        auth = AuthContext({
            "oid": TEST_USER_ID,
            "tid": TEST_TENANT_ID
        })
        
        result = await protected_function(auth=auth)
        assert result == f"Hello {TEST_USER_ID}"
        
        # Without authentication
        auth = AuthContext({})
        
        with pytest.raises(Exception) as exc_info:
            await protected_function(auth=auth)
        assert "Authentication required" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_require_roles_decorator(self):
        """Test require_roles decorator"""
        @require_roles("admin", "manager")
        async def admin_function(auth: AuthContext):
            return "Admin action"
        
        # With admin role
        auth = AuthContext({
            "roles": ["admin"]
        })
        
        result = await admin_function(auth=auth)
        assert result == "Admin action"
        
        # With manager role
        auth = AuthContext({
            "roles": ["manager"]
        })
        
        result = await admin_function(auth=auth)
        assert result == "Admin action"
        
        # Without required role
        auth = AuthContext({
            "roles": ["user"]
        })
        
        with pytest.raises(Exception) as exc_info:
            await admin_function(auth=auth)
        assert "Required role" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_require_admin_decorator(self):
        """Test require_admin decorator"""
        @require_admin
        async def admin_only_function(auth: AuthContext):
            return "Admin only"
        
        # Admin user
        auth = AuthContext({
            "roles": ["admin"]
        })
        
        result = await admin_only_function(auth=auth)
        assert result == "Admin only"
        
        # Non-admin user
        auth = AuthContext({
            "roles": ["user"]
        })
        
        with pytest.raises(Exception) as exc_info:
            await admin_only_function(auth=auth)
        assert "Admin privileges required" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])