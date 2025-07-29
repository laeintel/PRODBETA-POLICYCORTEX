"""
Unit tests for AuthManager functionality.
"""

import pytest
import jwt
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from backend.services.api_gateway.auth import AuthManager, AuthenticationError, AuthorizationError


class TestAuthManager:
    """Test AuthManager class."""

    def test_auth_manager_initialization(self):
        """Test auth manager initialization."""
        auth_manager = AuthManager()

        assert auth_manager.redis_client is not None
        assert hasattr(auth_manager, 'logger')

    @pytest.mark.asyncio
    async def test_verify_token_valid_jwt(self, mock_redis):
        """Test token verification with valid JWT."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Create a test JWT token
        payload = {
            "user_id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User",
            "roles": ["user"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }

        secret_key = "test-secret-key"
        token = jwt.encode(payload, secret_key, algorithm="HS256")

        # Mock Redis responses
        mock_redis.get.return_value = None  # Token not blacklisted

        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.return_value = payload

            user_info = await auth_manager.verify_token(token)

            assert user_info["user_id"] == "test-user-id"
            assert user_info["email"] == "test@example.com"
            assert user_info["name"] == "Test User"
            assert "user" in user_info["roles"]
            mock_decode.assert_called_once_with(token)

    @pytest.mark.asyncio
    async def test_verify_token_expired(self, mock_redis):
        """Test token verification with expired JWT."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Create an expired JWT token
        payload = {
            "user_id": "test-user-id",
            "email": "test@example.com",
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired
        }

        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

            with pytest.raises(AuthenticationError) as exc_info:
                await auth_manager.verify_token("expired-token")

            assert "Token expired" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_invalid_signature(self, mock_redis):
        """Test token verification with invalid signature."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.side_effect = jwt.InvalidSignatureError("Invalid signature")

            with pytest.raises(AuthenticationError) as exc_info:
                await auth_manager.verify_token("invalid-token")

            assert "Invalid signature" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_blacklisted(self, mock_redis):
        """Test token verification with blacklisted token."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response for blacklisted token
        mock_redis.get.return_value = "blacklisted"

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_manager.verify_token("blacklisted-token")

        assert "Token has been revoked" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_verify_token_malformed(self, mock_redis):
        """Test token verification with malformed token."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.side_effect = jwt.DecodeError("Invalid token format")

            with pytest.raises(AuthenticationError) as exc_info:
                await auth_manager.verify_token("malformed-token")

            assert "Invalid token format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_blacklist_token(self, mock_redis):
        """Test token blacklisting."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.setex.return_value = True

        # Mock JWT decoding to get expiry
        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.return_value = {
                "exp": datetime.utcnow() + timedelta(hours=1)
            }

            success = await auth_manager.blacklist_token("test-token")

            assert success is True
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_blacklist_token_without_expiry(self, mock_redis):
        """Test token blacklisting without expiry."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.set.return_value = True

        # Mock JWT decoding without expiry
        with patch.object(auth_manager, '_decode_jwt_token') as mock_decode:
            mock_decode.return_value = {"user_id": "test-user"}

            success = await auth_manager.blacklist_token("test-token")

            assert success is True
            mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_token_blacklisted(self, mock_redis):
        """Test checking if token is blacklisted."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = "blacklisted"

        is_blacklisted = await auth_manager.is_token_blacklisted("test-token")

        assert is_blacklisted is True
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_token_not_blacklisted(self, mock_redis):
        """Test checking if token is not blacklisted."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = None

        is_blacklisted = await auth_manager.is_token_blacklisted("test-token")

        assert is_blacklisted is False
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_permissions(self, mock_redis):
        """Test getting user permissions."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read", "write", "admin"]'

        permissions = await auth_manager.get_user_permissions("test-user-id")

        assert "read" in permissions
        assert "write" in permissions
        assert "admin" in permissions
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_permissions_not_found(self, mock_redis):
        """Test getting user permissions when not found."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = None

        permissions = await auth_manager.get_user_permissions("test-user-id")

        assert permissions == []

    @pytest.mark.asyncio
    async def test_check_permission_allowed(self, mock_redis):
        """Test permission checking when allowed."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read", "write"]'

        has_permission = await auth_manager.check_permission("test-user-id", "read")

        assert has_permission is True

    @pytest.mark.asyncio
    async def test_check_permission_denied(self, mock_redis):
        """Test permission checking when denied."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read"]'

        has_permission = await auth_manager.check_permission("test-user-id", "admin")

        assert has_permission is False

    @pytest.mark.asyncio
    async def test_require_permission_success(self, mock_redis):
        """Test requiring permission when user has it."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["admin"]'

        # Should not raise exception
        await auth_manager.require_permission("test-user-id", "admin")

    @pytest.mark.asyncio
    async def test_require_permission_failure(self, mock_redis):
        """Test requiring permission when user doesn't have it."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read"]'

        with pytest.raises(AuthorizationError) as exc_info:
            await auth_manager.require_permission("test-user-id", "admin")

        assert "Insufficient permissions" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_require_any_permission_success(self, mock_redis):
        """Test requiring any permission when user has one."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read", "write"]'

        # Should not raise exception
        await auth_manager.require_any_permission("test-user-id", ["admin", "write"])

    @pytest.mark.asyncio
    async def test_require_any_permission_failure(self, mock_redis):
        """Test requiring any permission when user has none."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read"]'

        with pytest.raises(AuthorizationError) as exc_info:
            await auth_manager.require_any_permission("test-user-id", ["admin", "write"])

        assert "Insufficient permissions" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_require_all_permissions_success(self, mock_redis):
        """Test requiring all permissions when user has them."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read", "write", "admin"]'

        # Should not raise exception
        await auth_manager.require_all_permissions("test-user-id", ["read", "write"])

    @pytest.mark.asyncio
    async def test_require_all_permissions_failure(self, mock_redis):
        """Test requiring all permissions when user is missing some."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.get.return_value = '["read"]'

        with pytest.raises(AuthorizationError) as exc_info:
            await auth_manager.require_all_permissions("test-user-id", ["read", "write"])

        assert "Insufficient permissions" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_user_info(self, mock_redis):
        """Test getting user information."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        user_data = {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User",
            "roles": ["user"]
        }
        mock_redis.get.return_value = '{"id": "test-user-id", "email": "test@example.com", "name": "Test User", "roles": ["user"]}'

        user_info = await auth_manager.get_user_info("test-user-id")

        assert user_info["id"] == "test-user-id"
        assert user_info["email"] == "test@example.com"
        assert user_info["name"] == "Test User"
        assert "user" in user_info["roles"]

    @pytest.mark.asyncio
    async def test_cache_user_info(self, mock_redis):
        """Test caching user information."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.setex.return_value = True

        user_data = {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User"
        }

        success = await auth_manager.cache_user_info("test-user-id", user_data, ttl=3600)

        assert success is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_user_cache(self, mock_redis):
        """Test invalidating user cache."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis response
        mock_redis.delete.return_value = 1

        success = await auth_manager.invalidate_user_cache("test-user-id")

        assert success is True
        mock_redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_token(self, mock_redis):
        """Test token refresh."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock the token refresh process
        old_token = "old-token"
        new_token = "new-token"

        with patch.object(auth_manager, '_generate_new_token') as mock_generate:
            mock_generate.return_value = new_token
            mock_redis.setex.return_value = True

            result = await auth_manager.refresh_token(old_token)

            assert result["access_token"] == new_token
            assert "expires_in" in result
            mock_generate.assert_called_once()

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Invalid credentials")

        assert str(error) == "Invalid credentials"
        assert error.args[0] == "Invalid credentials"

    def test_authorization_error(self):
        """Test AuthorizationError exception."""
        error = AuthorizationError("Access denied")

        assert str(error) == "Access denied"
        assert error.args[0] == "Access denied"

    @pytest.mark.asyncio
    async def test_redis_connection_error(self, mock_redis):
        """Test handling Redis connection errors."""
        auth_manager = AuthManager()
        auth_manager.redis_client = mock_redis

        # Mock Redis connection error
        mock_redis.get.side_effect = Exception("Redis connection error")

        with patch.object(auth_manager, 'logger') as mock_logger:
            # Should return empty permissions on Redis error
            permissions = await auth_manager.get_user_permissions("test-user-id")

            assert permissions == []
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_token_format(self):
        """Test token format validation."""
        auth_manager = AuthManager()

        # Test valid token format
        valid_token = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        assert auth_manager._validate_token_format(valid_token) is True

        # Test invalid token formats
        invalid_tokens = [
            "invalid-token",
            "bearer invalid-token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Only header
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",  # Missing signature
            "",
            None
        ]

        for token in invalid_tokens:
            assert auth_manager._validate_token_format(token) is False
