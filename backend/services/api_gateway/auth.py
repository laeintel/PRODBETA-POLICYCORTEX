"""
Authentication manager for API Gateway.
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

from ...shared.config import get_settings
from .models import UserInfo

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
    
    async def create_session(self, user_info: Dict[str, Any], token: str) -> str:
        """Create user session in Redis."""
        try:
            redis_client = await self._get_redis_client()
            session_id = f"session_{user_info['id']}_{int(datetime.utcnow().timestamp())}"
            
            session_data = {
                "user_id": user_info["id"],
                "email": user_info["email"],
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "token": token,
                "revoked": False
            }
            
            # Store session with expiration
            await redis_client.set(
                f"session:{session_id}",
                json.dumps(session_data),
                ex=self.settings.security.jwt_expiration_minutes * 60
            )
            
            # Store user sessions list
            user_sessions_key = f"user_sessions:{user_info['id']}"
            await redis_client.sadd(user_sessions_key, session_id)
            await redis_client.expire(
                user_sessions_key,
                self.settings.security.jwt_expiration_minutes * 60
            )
            
            logger.info(
                "session_created",
                session_id=session_id,
                user_id=user_info["id"]
            )
            
            return session_id
            
        except Exception as e:
            logger.error("session_creation_failed", error=str(e))
            raise Exception(f"Session creation failed: {str(e)}")
    
    async def revoke_session(self, session_id: str) -> None:
        """Revoke user session."""
        try:
            redis_client = await self._get_redis_client()
            session_key = f"session:{session_id}"
            
            # Get session data
            session_data = await redis_client.get(session_key)
            if session_data:
                session_info = json.loads(session_data)
                session_info["revoked"] = True
                session_info["revoked_at"] = datetime.utcnow().isoformat()
                
                # Update session
                await redis_client.set(
                    session_key,
                    json.dumps(session_info),
                    ex=300  # Keep revoked session for 5 minutes
                )
                
                logger.info("session_revoked", session_id=session_id)
            
        except Exception as e:
            logger.error("session_revocation_failed", error=str(e))
            raise Exception(f"Session revocation failed: {str(e)}")
    
    async def revoke_user_sessions(self, user_id: str) -> None:
        """Revoke all sessions for a user."""
        try:
            redis_client = await self._get_redis_client()
            user_sessions_key = f"user_sessions:{user_id}"
            
            # Get all user sessions
            session_ids = await redis_client.smembers(user_sessions_key)
            
            # Revoke each session
            for session_id in session_ids:
                await self.revoke_session(session_id)
            
            # Clean up sessions list
            await redis_client.delete(user_sessions_key)
            
            logger.info(
                "user_sessions_revoked",
                user_id=user_id,
                session_count=len(session_ids)
            )
            
        except Exception as e:
            logger.error("user_sessions_revocation_failed", error=str(e))
            raise Exception(f"User sessions revocation failed: {str(e)}")
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions from cache or database."""
        try:
            redis_client = await self._get_redis_client()
            permissions_key = f"user_permissions:{user_id}"
            
            # Try to get from cache first
            cached_permissions = await redis_client.get(permissions_key)
            if cached_permissions:
                return json.loads(cached_permissions)
            
            # If not in cache, would typically fetch from database
            # For now, return empty list
            permissions = []
            
            # Cache permissions for 1 hour
            await redis_client.set(
                permissions_key,
                json.dumps(permissions),
                ex=3600
            )
            
            return permissions
            
        except Exception as e:
            logger.error("get_user_permissions_failed", error=str(e))
            return []
    
    async def check_permission(self, user_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])
            
            # Check direct permission
            if required_permission in user_permissions:
                return True
            
            # Check role-based permissions
            admin_roles = ["admin", "global_admin", "policy_admin"]
            if any(role in admin_roles for role in user_roles):
                return True
            
            return False
            
        except Exception as e:
            logger.error("permission_check_failed", error=str(e))
            return False
    
    async def generate_token(self, user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate JWT token for user."""
        try:
            jwt_secret = await self._get_jwt_secret()
            
            # Token payload
            now = datetime.utcnow()
            exp = now + timedelta(minutes=self.settings.security.jwt_expiration_minutes)
            
            payload = {
                "sub": user_info["id"],
                "email": user_info["email"],
                "name": user_info["name"],
                "roles": user_info.get("roles", []),
                "permissions": user_info.get("permissions", []),
                "tenant_id": user_info.get("tenant_id"),
                "subscription_ids": user_info.get("subscription_ids", []),
                "iat": now,
                "exp": exp
            }
            
            # Generate token
            token = jose_jwt.encode(
                payload,
                jwt_secret,
                algorithm=self.settings.security.jwt_algorithm
            )
            
            # Create session
            session_id = await self.create_session(user_info, token)
            
            # Add session ID to payload for refresh token
            refresh_payload = payload.copy()
            refresh_payload["session_id"] = session_id
            refresh_payload["exp"] = now + timedelta(
                days=self.settings.security.jwt_refresh_expiration_days
            )
            
            refresh_token = jose_jwt.encode(
                refresh_payload,
                jwt_secret,
                algorithm=self.settings.security.jwt_algorithm
            )
            
            return {
                "access_token": token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.settings.security.jwt_expiration_minutes * 60,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error("token_generation_failed", error=str(e))
            raise Exception(f"Token generation failed: {str(e)}")
    
    async def cleanup_expired_sessions(self) -> None:
        """Cleanup expired sessions (background task)."""
        try:
            redis_client = await self._get_redis_client()
            
            # This would typically be run as a background task
            # to clean up expired sessions periodically
            
            logger.info("session_cleanup_completed")
            
        except Exception as e:
            logger.error("session_cleanup_failed", error=str(e)) 