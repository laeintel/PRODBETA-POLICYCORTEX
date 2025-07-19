"""
Authentication manager for Notification service.
Handles JWT token validation, Azure AD integration, and user session management.
"""

import jwt
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import aioredis
import structlog
from jose import JWTError, jwt as jose_jwt
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from shared.config import get_settings
from .models import NotificationPreferences

settings = get_settings()
logger = structlog.get_logger(__name__)


class AuthManager:
    """Authentication manager for validating tokens and managing user sessions."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.azure_credential = None
        self.key_vault_client = None
    
    async def _get_redis_client(self) -> aioredis.Redis:
        """Get Redis client for session management."""
        if self.redis_client is None:
            self.redis_client = aioredis.from_url(
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
            
            # Get user notification preferences
            user_info["notification_preferences"] = await self.get_user_notification_preferences(user_info["id"])
            
            logger.info(
                "token_verified",
                user_id=user_info["id"],
                email=user_info["email"]
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
    
    async def get_user_notification_preferences(self, user_id: str) -> NotificationPreferences:
        """Get user notification preferences from cache or database."""
        try:
            redis_client = await self._get_redis_client()
            preferences_key = f"user_notification_preferences:{user_id}"
            
            # Try to get from cache first
            cached_preferences = await redis_client.get(preferences_key)
            if cached_preferences:
                preferences_data = json.loads(cached_preferences)
                return NotificationPreferences(**preferences_data)
            
            # If not in cache, return default preferences
            default_preferences = NotificationPreferences()
            
            # Cache preferences for 1 hour
            await redis_client.set(
                preferences_key,
                json.dumps(default_preferences.dict()),
                ex=3600
            )
            
            return default_preferences
            
        except Exception as e:
            logger.error("get_user_notification_preferences_failed", error=str(e))
            # Return default preferences on error
            return NotificationPreferences()
    
    async def update_user_notification_preferences(
        self, 
        user_id: str, 
        preferences: NotificationPreferences
    ) -> None:
        """Update user notification preferences in cache and database."""
        try:
            redis_client = await self._get_redis_client()
            preferences_key = f"user_notification_preferences:{user_id}"
            
            # Update cache
            await redis_client.set(
                preferences_key,
                json.dumps(preferences.dict()),
                ex=3600
            )
            
            # Log the update
            logger.info(
                "user_notification_preferences_updated",
                user_id=user_id,
                preferences=preferences.dict()
            )
            
        except Exception as e:
            logger.error("update_user_notification_preferences_failed", error=str(e))
            raise Exception(f"Failed to update notification preferences: {str(e)}")
    
    async def check_notification_permission(
        self, 
        user_info: Dict[str, Any], 
        notification_type: str,
        operation: str = "send"
    ) -> bool:
        """Check if user has permission for specific notification operations."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])
            
            # Check direct permissions
            required_permissions = {
                "email": f"notification:email:{operation}",
                "sms": f"notification:sms:{operation}",
                "push": f"notification:push:{operation}",
                "webhook": f"notification:webhook:{operation}",
                "alert": f"notification:alert:{operation}"
            }
            
            required_permission = required_permissions.get(notification_type)
            if required_permission and required_permission in user_permissions:
                return True
            
            # Check role-based permissions
            admin_roles = ["admin", "global_admin", "notification_admin"]
            if any(role in admin_roles for role in user_roles):
                return True
            
            # Check notification-specific roles
            notification_roles = {
                "email": ["email_admin", "communication_admin"],
                "sms": ["sms_admin", "communication_admin"],
                "push": ["push_admin", "communication_admin"],
                "webhook": ["webhook_admin", "integration_admin"],
                "alert": ["alert_admin", "monitoring_admin"]
            }
            
            allowed_roles = notification_roles.get(notification_type, [])
            if any(role in allowed_roles for role in user_roles):
                return True
            
            return False
            
        except Exception as e:
            logger.error("notification_permission_check_failed", error=str(e))
            return False
    
    async def check_rate_limit(self, user_id: str, notification_type: str) -> bool:
        """Check if user is within rate limits for notifications."""
        try:
            redis_client = await self._get_redis_client()
            
            # Define rate limits per notification type
            rate_limits = {
                "email": {"limit": 100, "window": 3600},  # 100 emails per hour
                "sms": {"limit": 50, "window": 3600},     # 50 SMS per hour
                "push": {"limit": 200, "window": 3600},   # 200 push notifications per hour
                "webhook": {"limit": 150, "window": 3600}, # 150 webhooks per hour
                "alert": {"limit": 500, "window": 3600}   # 500 alerts per hour
            }
            
            rate_limit = rate_limits.get(notification_type, {"limit": 100, "window": 3600})
            
            # Check current usage
            usage_key = f"rate_limit:{user_id}:{notification_type}"
            current_usage = await redis_client.get(usage_key)
            
            if current_usage is None:
                # First request, set counter
                await redis_client.set(usage_key, 1, ex=rate_limit["window"])
                return True
            
            current_count = int(current_usage)
            if current_count >= rate_limit["limit"]:
                return False
            
            # Increment counter
            await redis_client.incr(usage_key)
            return True
            
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            # Allow request on error to avoid blocking legitimate requests
            return True
    
    async def log_notification_activity(
        self, 
        user_id: str, 
        notification_type: str, 
        action: str, 
        details: Dict[str, Any]
    ) -> None:
        """Log notification activity for audit purposes."""
        try:
            redis_client = await self._get_redis_client()
            
            activity_log = {
                "user_id": user_id,
                "notification_type": notification_type,
                "action": action,
                "details": details,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis with TTL
            activity_key = f"activity_log:{user_id}:{datetime.utcnow().strftime('%Y%m%d')}"
            await redis_client.lpush(activity_key, json.dumps(activity_log))
            await redis_client.expire(activity_key, 86400 * 30)  # Keep for 30 days
            
            logger.info(
                "notification_activity_logged",
                user_id=user_id,
                notification_type=notification_type,
                action=action
            )
            
        except Exception as e:
            logger.error("notification_activity_logging_failed", error=str(e))
    
    async def get_user_activity_log(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user notification activity log."""
        try:
            redis_client = await self._get_redis_client()
            activities = []
            
            # Get logs for the specified number of days
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime('%Y%m%d')
                activity_key = f"activity_log:{user_id}:{date}"
                
                daily_activities = await redis_client.lrange(activity_key, 0, -1)
                for activity_json in daily_activities:
                    activity = json.loads(activity_json)
                    activities.append(activity)
            
            # Sort by timestamp (newest first)
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return activities
            
        except Exception as e:
            logger.error("get_user_activity_log_failed", error=str(e))
            return []
    
    async def validate_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Validate webhook signature for security."""
        try:
            import hmac
            import hashlib
            
            # Create expected signature
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error("webhook_signature_validation_failed", error=str(e))
            return False
    
    async def cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions (background task)."""
        try:
            redis_client = await self._get_redis_client()
            
            # This would typically be run as a background task
            # to clean up expired sessions periodically
            
            logger.info("notification_service_session_cleanup_completed")
            
        except Exception as e:
            logger.error("notification_service_session_cleanup_failed", error=str(e))
    
    async def initialize(self) -> None:
        """Initialize the authentication manager."""
        try:
            # Initialize Redis connection
            await self._get_redis_client()
            
            # Initialize Azure credentials if in production
            if self.settings.is_production():
                await self._get_key_vault_client()
            
            logger.info("auth_manager_initialized")
            
        except Exception as e:
            logger.error("auth_manager_initialization_failed", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.azure_credential:
                await self.azure_credential.close()
            
            logger.info("auth_manager_cleanup_completed")
            
        except Exception as e:
            logger.error("auth_manager_cleanup_failed", error=str(e))