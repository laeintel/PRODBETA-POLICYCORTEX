"""
Authentication manager for Data Processing service.
Handles JWT token validation and user session management.
"""

import json
import uuid
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import jwt
import redis.asyncio as redis
import structlog
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient
from jose import JWTError
from jose import jwt as jose_jwt
from shared.config import get_settings

from .models import DataConnectorHealth

settings = get_settings()
logger = structlog.get_logger(__name__)


class AuthManager:
    """Authentication manager for validating tokens and managing user sessions."""

    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.azure_credential = None
        self.key_vault_client = None
        self._permissions_cache = {}

    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client for session management."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
        return self.redis_client

    async def _get_key_vault_client(self) -> SecretClient:
        """Get Azure Key Vault client."""
        if self.key_vault_client is None:
            if self.azure_credential is None:
                self.azure_credential = DefaultAzureCredential()
            self.key_vault_client = SecretClient(
                vault_url=self.settings.azure.key_vault_url,
                credential=self.azure_credential
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
            logger.warning(
                "failed_to_get_jwt_secret_from_keyvault",
                error=str(e)
            )
            return self.settings.security.jwt_secret_key

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return user information."""
        try:
            # Get JWT secret
            jwt_secret = await self._get_jwt_secret()

            # Decode and verify token
            payload = jose_jwt.decode(
                token,
                jwt_secret,
                algorithms=[self.settings.security.jwt_algorithm]
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
                "subscription_ids": payload.get("subscription_ids", [])
            }

            # Validate session if session ID is present
            session_id = payload.get("session_id")
            if session_id:
                await self._validate_session(session_id, user_info["id"])

            logger.info(
                "token_verified",
                user_id=user_info["id"],
                email=user_info["email"],
                service="data_processing"
            )

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
            session_key = f"session:{session_id}"

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
                ex=self.settings.security.jwt_expiration_minutes * 60
            )

        except Exception as e:
            logger.error("session_validation_failed", error=str(e))
            raise Exception(f"Session validation failed: {str(e)}")

    async def check_permission(self, user_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission for data processing operations."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])

            # Check direct permission
            if required_permission in user_permissions:
                return True

            # Check role-based permissions for data processing
            data_processing_permissions = {
                "data_read": ["data_reader", "data_analyst", "data_scientist", "data_engineer", "admin"],
                "data_write": ["data_writer", "data_engineer", "admin"],
                "pipeline_create": ["data_engineer", "pipeline_admin", "admin"],
                "pipeline_execute": ["data_engineer", "pipeline_admin", "admin"],
                "pipeline_delete": ["pipeline_admin", "admin"],
                "stream_process": ["stream_admin", "data_engineer", "admin"],
                "data_transform": ["data_engineer", "data_scientist", "admin"],
                "data_validate": ["data_engineer", "data_analyst", "admin"],
                "data_export": ["data_exporter", "data_engineer", "admin"],
                "lineage_view": ["data_analyst", "data_scientist", "data_engineer", "admin"],
                "quality_monitor": ["data_analyst", "data_engineer", "admin"],
                "admin_access": ["admin", "global_admin", "data_admin"]
            }

            allowed_roles = data_processing_permissions.get(required_permission, [])
            if any(role in allowed_roles for role in user_roles):
                return True

            # Check admin roles
            admin_roles = ["admin", "global_admin", "data_admin"]
            if any(role in admin_roles for role in user_roles):
                return True

            return False

        except Exception as e:
            logger.error("permission_check_failed", error=str(e))
            return False

    async def get_user_data_access_scope(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get user's data access scope and restrictions."""
        try:
            user_id = user_info.get("id")
            user_roles = user_info.get("roles", [])
            tenant_id = user_info.get("tenant_id")
            subscription_ids = user_info.get("subscription_ids", [])

            # Check cache first
            cache_key = f"user_data_scope:{user_id}"
            redis_client = await self._get_redis_client()
            cached_scope = await redis_client.get(cache_key)

            if cached_scope:
                return json.loads(cached_scope)

            # Determine data access scope based on roles
            scope = {
                "can_access_all_tenants": False,
                "can_access_all_subscriptions": False,
                "allowed_data_sources": [],
                "data_classification_access": [],
                "row_level_filters": {},
                "column_level_filters": {},
                "export_restrictions": {},
                "processing_restrictions": {}
            }

            # Admin roles get full access
            if any(role in ["admin", "global_admin"] for role in user_roles):
                scope.update({
                    "can_access_all_tenants": True,
                    "can_access_all_subscriptions": True,
                    "allowed_data_sources": ["*"],
                    "data_classification_access": ["public", "internal", "confidential", "restricted"]
                })

            # Data admin gets broad access within tenant
            elif "data_admin" in user_roles:
                scope.update({
                    "can_access_all_subscriptions": True,
                    "allowed_data_sources": ["*"],
                    "data_classification_access": ["public", "internal", "confidential"]
                })

            # Data engineer gets processing access
            elif "data_engineer" in user_roles:
                scope.update({
                    "allowed_data_sources": ["azure_sql", "cosmos_db", "blob_storage", "data_lake"],
                    "data_classification_access": ["public", "internal"]
                })

            # Data scientist gets analysis access
            elif "data_scientist" in user_roles:
                scope.update({
                    "allowed_data_sources": ["azure_sql", "cosmos_db", "blob_storage"],
                    "data_classification_access": ["public", "internal"],
                    "export_restrictions": {"max_records": 100000}
                })

            # Data analyst gets read access
            elif "data_analyst" in user_roles:
                scope.update({
                    "allowed_data_sources": ["azure_sql", "cosmos_db"],
                    "data_classification_access": ["public"],
                    "export_restrictions": {"max_records": 50000}
                })

            # Add tenant and subscription restrictions
            if not scope["can_access_all_tenants"]:
                scope["tenant_restrictions"] = [tenant_id] if tenant_id else []

            if not scope["can_access_all_subscriptions"]:
                scope["subscription_restrictions"] = subscription_ids

            # Cache the scope for 1 hour
            await redis_client.set(
                cache_key,
                json.dumps(scope),
                ex=3600
            )

            return scope

        except Exception as e:
            logger.error("get_user_data_access_scope_failed", error=str(e))
            return {
                "can_access_all_tenants": False,
                "can_access_all_subscriptions": False,
                "allowed_data_sources": [],
                "data_classification_access": ["public"],
                "row_level_filters": {},
                "column_level_filters": {},
                "export_restrictions": {"max_records": 1000},
                "processing_restrictions": {}
            }

    async def audit_data_access(self, user_info: Dict[str, Any], operation: str,
                               resource: str, details: Dict[str, Any]) -> None:
        """Audit data access and processing operations."""
        try:
            audit_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_info.get("id"),
                "user_email": user_info.get("email"),
                "tenant_id": user_info.get("tenant_id"),
                "operation": operation,
                "resource": resource,
                "details": details,
                "service": "data_processing"
            }

            # Store audit record in Redis and optionally send to audit service
            redis_client = await self._get_redis_client()
            audit_key = (
                f"audit:{datetime.utcnow().strftime('%Y-%m-%d')}:{user_info.get('id')}:{uuid.uuid4()}"
            )

            await redis_client.set(
                audit_key,
                json.dumps(audit_record),
                ex=86400 * 30  # Keep for 30 days
            )

            logger.info(
                "data_access_audited",
                user_id=user_info.get("id"),
                operation=operation,
                resource=resource
            )

        except Exception as e:
            logger.error("audit_data_access_failed", error=str(e))

    async def validate_data_processing_request(self, user_info: Dict[str, Any],
                                             request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data processing request against user permissions and data access scope."""
        try:
            # Get user's data access scope
            scope = await self.get_user_data_access_scope(user_info)

            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "applied_restrictions": {}
            }

            # Check data source access
            if "source_config" in request_data:
                source_type = request_data["source_config"].get("source_type")
                if source_type and "*" not in scope["allowed_data_sources"]:
                    if source_type not in scope["allowed_data_sources"]:
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(f"Access denied to data source: {source_type}")

            # Check export restrictions
            if "export" in request_data.get("operation", ""):
                export_restrictions = scope.get("export_restrictions", {})
                if export_restrictions.get("max_records"):
                    validation_result["applied_restrictions"]["max_records"] = (
                        export_restrictions["max_records"]
                    )
                    validation_result["warnings"].append(
                        f"Export limited to {export_restrictions['max_records']} records"
                    )

            # Check tenant restrictions
            if "tenant_restrictions" in scope:
                tenant_restrictions = scope["tenant_restrictions"]
                if tenant_restrictions and request_data.get("tenant_id") not in tenant_restrictions:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Access denied to specified tenant")

            # Check subscription restrictions
            if "subscription_restrictions" in scope:
                subscription_restrictions = scope["subscription_restrictions"]
                if subscription_restrictions and
                    request_data.get("subscription_id") not in subscription_restrictions:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append("Access denied to specified subscription")

            return validation_result

        except Exception as e:
            logger.error("validate_data_processing_request_failed", error=str(e))
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "applied_restrictions": {}
            }

    async def cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions and cached permissions."""
        try:
            redis_client = await self._get_redis_client()

            # Clean up expired session data
            # This would typically be run as a background task

            # Clear permissions cache
            self._permissions_cache.clear()

            logger.info("data_processing_session_cleanup_completed")

        except Exception as e:
            logger.error("data_processing_session_cleanup_failed", error=str(e))
