"""
Authentication manager for Conversation service.
Handles JWT token validation and user session management.
"""

import json
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import jwt
import redis.asyncio as redis
import structlog
from jose import JWTError
from jose import jwt as jose_jwt

try:
    from azure.identity.aio import DefaultAzureCredential
    from azure.keyvault.secrets.aio import SecretClient

    AZURE_KEYVAULT_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
    AZURE_KEYVAULT_AVAILABLE = False

from shared.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class AuthManager:
    """Authentication manager for validating tokens and managing user sessions."""

    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.azure_credential = None
        self.key_vault_client = None

    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for session management."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True,
            )
        return self.redis_client

    async def _get_key_vault_client(self):
        """Get Azure Key Vault client."""
        if not AZURE_KEYVAULT_AVAILABLE:
            raise ImportError("Azure KeyVault client not available")

        if self.key_vault_client is None:
            if self.azure_credential is None:
                self.azure_credential = DefaultAzureCredential()
            self.key_vault_client = SecretClient(
                vault_url=self.settings.azure.key_vault_url, credential=self.azure_credential
            )
        return self.key_vault_client

    async def _get_jwt_secret(self) -> str:
        """Get JWT secret from Key Vault or configuration."""
        try:
            if self.settings.is_production():
                # In production, get secret from Key Vault
                key_vault_client = await self._get_key_vault_client()
                secret = await key_vault_client.get_secret("jwt-secret-key")
                return secret.value
            else:
                # In development, use configuration
                return self.settings.security.jwt_secret_key
        except Exception as e:
            logger.warning("failed_to_get_jwt_secret_from_keyvault", error=str(e))
            return self.settings.security.jwt_secret_key

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user information."""
        try:
            # Get JWT secret
            jwt_secret = await self._get_jwt_secret()

            # Decode and verify token
            payload = jose_jwt.decode(
                token, jwt_secret, algorithms=[self.settings.security.jwt_algorithm]
            )

            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Token has expired")

            # Get user information from token
            user_info = {
                "id": payload.get("sub"),
                "email": payload.get("email"),
                "name": payload.get("name"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", []),
                "tenant_id": payload.get("tenant_id"),
                "subscription_ids": payload.get("subscription_ids", []),
            }

            # Validate session if session ID is present
            session_id = payload.get("session_id")
            if session_id:
                await self._validate_session(session_id, user_info["id"])

            logger.info("token_verified", user_id=user_info["id"], email=user_info["email"])

            return user_info

        except JWTError as e:
            logger.error("jwt_verification_failed", error=str(e))
            raise Exception(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error("token_verification_error", error=str(e))
            raise Exception(f"Token verification failed: {str(e)}")

    async def _validate_session(self, session_id: str, user_id: str) -> None:
        """Validate user session in Redis."""
        try:
            redis_client = await self._get_redis_client()
            session_key = f"auth_session:{session_id}"

            # Get session data
            session_data = await redis_client.get(session_key)
            if not session_data:
                raise Exception("Session not found")

            session_info = json.loads(session_data)

            # Validate session user
            if session_info.get("user_id") != user_id:
                raise Exception("Session user mismatch")

            # Check if session is revoked
            if session_info.get("revoked", False):
                raise Exception("Session has been revoked")

            # Update last activity
            session_info["last_activity"] = datetime.utcnow().isoformat()
            await redis_client.set(
                session_key,
                json.dumps(session_info),
                ex=self.settings.security.jwt_expiration_minutes * 60,
            )

        except Exception as e:
            logger.error("session_validation_failed", error=str(e))
            raise Exception(f"Session validation failed: {str(e)}")

    async def get_user_conversation_permissions(self, user_id: str) -> List[str]:
        """Get user conversation permissions."""
        try:
            redis_client = await self._get_redis_client()
            permissions_key = f"user_conversation_permissions:{user_id}"

            # Try to get from cache first
            cached_permissions = await redis_client.get(permissions_key)
            if cached_permissions:
                return json.loads(cached_permissions)

            # Default conversation permissions
            permissions = [
                "conversation:read",
                "conversation:write",
                "conversation:history",
                "conversation:export",
                "intent:classify",
                "entity:extract",
            ]

            # Cache permissions for 1 hour
            await redis_client.set(permissions_key, json.dumps(permissions), ex=3600)

            return permissions

        except Exception as e:
            logger.error("get_conversation_permissions_failed", error=str(e))
            return []

    async def check_conversation_permission(
        self, user_info: Dict[str, Any], required_permission: str
    ) -> bool:
        """Check if user has required conversation permission."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])

            # Check direct permission
            if required_permission in user_permissions:
                return True

            # Check role-based permissions
            admin_roles = ["admin", "global_admin", "conversation_admin"]
            if any(role in admin_roles for role in user_roles):
                return True

            # Check conversation-specific permissions
            conversation_permissions = await self.get_user_conversation_permissions(user_info["id"])
            if required_permission in conversation_permissions:
                return True

            return False

        except Exception as e:
            logger.error("conversation_permission_check_failed", error=str(e))
            return False

    async def track_conversation_access(self, user_id: str, session_id: str, action: str) -> None:
        """Track conversation access for auditing."""
        try:
            redis_client = await self._get_redis_client()
            access_key = f"conversation_access:{user_id}:{session_id}"

            access_data = {
                "user_id": user_id,
                "session_id": session_id,
                "action": action,
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": "unknown",  # Would be populated from request
            }

            # Store access log
            await redis_client.lpush(access_key, json.dumps(access_data))
            await redis_client.expire(access_key, 86400 * 30)  # Keep for 30 days

            logger.info(
                "conversation_access_tracked", user_id=user_id, session_id=session_id, action=action
            )

        except Exception as e:
            logger.error("conversation_access_tracking_failed", error=str(e))

    async def get_user_conversation_quota(self, user_id: str) -> Dict[str, Any]:
        """Get user conversation quota information."""
        try:
            redis_client = await self._get_redis_client()
            quota_key = f"conversation_quota:{user_id}"

            # Get current quota usage
            quota_data = await redis_client.get(quota_key)
            if quota_data:
                quota_info = json.loads(quota_data)
            else:
                quota_info = {
                    "daily_conversations": 0,
                    "monthly_conversations": 0,
                    "daily_messages": 0,
                    "monthly_messages": 0,
                    "last_reset": datetime.utcnow().isoformat(),
                }

            # Default limits
            limits = {
                "daily_conversation_limit": 50,
                "monthly_conversation_limit": 1000,
                "daily_message_limit": 500,
                "monthly_message_limit": 10000,
            }

            return {
                "usage": quota_info,
                "limits": limits,
                "remaining": {
                    "daily_conversations": max(
                        0, limits["daily_conversation_limit"] - quota_info["daily_conversations"]
                    ),
                    "monthly_conversations": max(
                        0,
                        limits["monthly_conversation_limit"] - quota_info["monthly_conversations"],
                    ),
                    "daily_messages": max(
                        0, limits["daily_message_limit"] - quota_info["daily_messages"]
                    ),
                    "monthly_messages": max(
                        0, limits["monthly_message_limit"] - quota_info["monthly_messages"]
                    ),
                },
            }

        except Exception as e:
            logger.error("get_conversation_quota_failed", error=str(e))
            return {"usage": {}, "limits": {}, "remaining": {}}

    async def increment_conversation_quota(self, user_id: str, increment_type: str) -> bool:
        """Increment user conversation quota and check limits."""
        try:
            redis_client = await self._get_redis_client()
            quota_key = f"conversation_quota:{user_id}"

            # Get current quota
            quota_info = await self.get_user_conversation_quota(user_id)
            usage = quota_info["usage"]
            limits = quota_info["limits"]

            # Check if increment would exceed limits
            if increment_type == "conversation":
                if usage.get("daily_conversations", 0) >= limits.get(
                    "daily_conversation_limit", 50
                ) or usage.get("monthly_conversations", 0) >= limits.get(
                    "monthly_conversation_limit", 1000
                ):
                    return False

                usage["daily_conversations"] = usage.get("daily_conversations", 0) + 1
                usage["monthly_conversations"] = usage.get("monthly_conversations", 0) + 1

            elif increment_type == "message":
                if usage.get("daily_messages", 0) >= limits.get(
                    "daily_message_limit", 500
                ) or usage.get("monthly_messages", 0) >= limits.get("monthly_message_limit", 10000):
                    return False

                usage["daily_messages"] = usage.get("daily_messages", 0) + 1
                usage["monthly_messages"] = usage.get("monthly_messages", 0) + 1

            # Update quota
            usage["last_updated"] = datetime.utcnow().isoformat()
            await redis_client.set(quota_key, json.dumps(usage), ex=86400 * 32)  # Keep for 32 days

            return True

        except Exception as e:
            logger.error("increment_conversation_quota_failed", error=str(e))
            return False

    async def reset_daily_quota(self, user_id: str) -> None:
        """Reset daily conversation quota (called by scheduled task)."""
        try:
            redis_client = await self._get_redis_client()
            quota_key = f"conversation_quota:{user_id}"

            quota_data = await redis_client.get(quota_key)
            if quota_data:
                quota_info = json.loads(quota_data)
                quota_info["daily_conversations"] = 0
                quota_info["daily_messages"] = 0
                quota_info["last_daily_reset"] = datetime.utcnow().isoformat()

                await redis_client.set(quota_key, json.dumps(quota_info), ex=86400 * 32)

                logger.info("daily_quota_reset", user_id=user_id)

        except Exception as e:
            logger.error("reset_daily_quota_failed", error=str(e))

    async def get_conversation_security_context(self, user_id: str) -> Dict[str, Any]:
        """Get conversation security context for a user."""
        try:
            redis_client = await self._get_redis_client()
            security_key = f"conversation_security:{user_id}"

            # Get security context
            security_data = await redis_client.get(security_key)
            if security_data:
                return json.loads(security_data)

            # Default security context
            security_context = {
                "data_classification": "internal",
                "retention_days": 90,
                "encryption_enabled": True,
                "audit_enabled": True,
                "pii_filtering": True,
                "content_filtering": True,
            }

            # Cache security context
            await redis_client.set(security_key, json.dumps(security_context), ex=3600)

            return security_context

        except Exception as e:
            logger.error("get_conversation_security_context_failed", error=str(e))
            return {}

    async def validate_conversation_access(self, user_id: str, session_id: str) -> bool:
        """Validate if user has access to a conversation session."""
        try:
            redis_client = await self._get_redis_client()

            # Check if session belongs to user
            session_key = f"conversation_session:{session_id}"
            session_data = await redis_client.get(session_key)

            if not session_data:
                return False

            session_info = json.loads(session_data)
            if session_info.get("user_id") != user_id:
                return False

            # Check if session is active
            if session_info.get("status") != "active":
                return False

            return True

        except Exception as e:
            logger.error("validate_conversation_access_failed", error=str(e))
            return False
