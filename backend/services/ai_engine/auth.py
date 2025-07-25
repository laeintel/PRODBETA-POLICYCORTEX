"""
Authentication manager for AI Engine.
Handles JWT token validation and user authorization for AI/ML operations.
"""

import jwt
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import redis.asyncio as redis
import structlog
from jose import JWTError, jwt as jose_jwt
from azure.identity.aio import DefaultAzureCredential
from azure.keyvault.secrets.aio import SecretClient

from shared.config import get_settings
from .models import ModelInfo

settings = get_settings()
logger = structlog.get_logger(__name__)


class AuthManager:
    """Authentication manager for AI Engine service."""
    
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
                service="ai_engine"
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
    
    async def check_ai_permission(self, user_info: Dict[str, Any], operation: str) -> bool:
        """Check if user has permission for AI/ML operations."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])
            
            # Define AI/ML operation permissions
            ai_permissions = {
                "policy_analysis": ["ai.policy.analyze", "ai.nlp.access"],
                "anomaly_detection": ["ai.anomaly.detect", "ai.monitoring.access"],
                "cost_optimization": ["ai.cost.optimize", "ai.finance.access"],
                "predictive_analytics": ["ai.predict.access", "ai.analytics.access"],
                "sentiment_analysis": ["ai.sentiment.analyze", "ai.nlp.access"],
                "model_management": ["ai.model.manage", "ai.admin.access"],
                "model_training": ["ai.model.train", "ai.admin.access"],
                "feature_engineering": ["ai.feature.engineer", "ai.data.access"],
                "batch_prediction": ["ai.batch.predict", "ai.inference.access"],
                "model_monitoring": ["ai.model.monitor", "ai.admin.access"]
            }
            
            required_permissions = ai_permissions.get(operation, [])
            
            # Check direct permissions
            if any(perm in user_permissions for perm in required_permissions):
                return True
            
            # Check role-based permissions
            admin_roles = ["admin", "global_admin", "ai_admin", "ml_engineer"]
            if any(role in admin_roles for role in user_roles):
                return True
            
            # Check specific AI roles
            ai_roles = {
                "policy_analysis": ["policy_analyst", "compliance_officer"],
                "anomaly_detection": ["security_analyst", "monitoring_engineer"],
                "cost_optimization": ["cost_analyst", "finance_manager"],
                "predictive_analytics": ["data_scientist", "business_analyst"],
                "sentiment_analysis": ["content_analyst", "quality_manager"],
                "model_management": ["ml_engineer", "data_scientist"],
                "model_training": ["ml_engineer", "data_scientist"],
                "feature_engineering": ["data_engineer", "ml_engineer"],
                "batch_prediction": ["data_scientist", "ml_engineer"],
                "model_monitoring": ["ml_engineer", "monitoring_engineer"]
            }
            
            allowed_roles = ai_roles.get(operation, [])
            if any(role in user_roles for role in allowed_roles):
                return True
            
            logger.warning(
                "ai_permission_denied",
                user_id=user_info["id"],
                operation=operation,
                user_roles=user_roles,
                user_permissions=user_permissions
            )
            
            return False
            
        except Exception as e:
            logger.error("ai_permission_check_failed", error=str(e))
            return False
    
    async def check_model_access(self, user_info: Dict[str, Any], model_name: str, operation: str) -> bool:
        """Check if user has access to specific model operations."""
        try:
            # Check general AI permission first
            if not await self.check_ai_permission(user_info, "model_management"):
                return False
            
            # Check model-specific permissions
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])
            
            # Model access patterns
            model_permission = f"ai.model.{model_name}.{operation}"
            if model_permission in user_permissions:
                return True
            
            # Check admin roles
            admin_roles = ["admin", "global_admin", "ai_admin", "ml_engineer"]
            if any(role in admin_roles for role in user_roles):
                return True
            
            # Check if user has general model access
            general_model_permissions = [
                f"ai.model.{operation}",
                f"ai.model.*.{operation}",
                "ai.model.access.all"
            ]
            
            if any(perm in user_permissions for perm in general_model_permissions):
                return True
            
            return False
            
        except Exception as e:
            logger.error("model_access_check_failed", error=str(e))
            return False
    
    async def check_data_access(self, user_info: Dict[str, Any], data_type: str, tenant_id: str = None) -> bool:
        """Check if user has access to specific data types."""
        try:
            user_permissions = user_info.get("permissions", [])
            user_roles = user_info.get("roles", [])
            user_tenant_id = user_info.get("tenant_id")
            
            # Check tenant isolation
            if tenant_id and user_tenant_id != tenant_id:
                # Check if user has cross-tenant access
                if "data.access.cross_tenant" not in user_permissions:
                    return False
            
            # Data access permissions
            data_permissions = {
                "azure_resources": ["data.azure.resources", "azure.read"],
                "cost_data": ["data.cost.access", "finance.read"],
                "security_data": ["data.security.access", "security.read"],
                "compliance_data": ["data.compliance.access", "compliance.read"],
                "performance_data": ["data.performance.access", "monitoring.read"],
                "user_data": ["data.user.access", "user.read"],
                "audit_data": ["data.audit.access", "audit.read"]
            }
            
            required_permissions = data_permissions.get(data_type, [])
            
            # Check direct permissions
            if any(perm in user_permissions for perm in required_permissions):
                return True
            
            # Check role-based access
            admin_roles = ["admin", "global_admin", "data_admin"]
            if any(role in admin_roles for role in user_roles):
                return True
            
            # Check specific data roles
            data_roles = {
                "azure_resources": ["azure_admin", "infrastructure_admin"],
                "cost_data": ["finance_manager", "cost_analyst"],
                "security_data": ["security_admin", "security_analyst"],
                "compliance_data": ["compliance_officer", "audit_manager"],
                "performance_data": ["monitoring_engineer", "performance_analyst"],
                "user_data": ["user_admin", "hr_manager"],
                "audit_data": ["audit_manager", "compliance_officer"]
            }
            
            allowed_roles = data_roles.get(data_type, [])
            if any(role in user_roles for role in allowed_roles):
                return True
            
            return False
            
        except Exception as e:
            logger.error("data_access_check_failed", error=str(e))
            return False
    
    async def log_ai_operation(self, user_info: Dict[str, Any], operation: str, 
                              details: Dict[str, Any] = None) -> None:
        """Log AI/ML operation for audit purposes."""
        try:
            redis_client = await self._get_redis_client()
            
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_info["id"],
                "user_email": user_info["email"],
                "operation": operation,
                "service": "ai_engine",
                "details": details or {},
                "tenant_id": user_info.get("tenant_id"),
                "session_id": user_info.get("session_id")
            }
            
            # Store audit log
            audit_key = f"audit:ai:{datetime.utcnow().strftime('%Y%m%d')}:{user_info['id']}"
            await redis_client.lpush(audit_key, json.dumps(audit_entry))
            await redis_client.expire(audit_key, 86400 * 90)  # Keep for 90 days
            
            logger.info(
                "ai_operation_logged",
                user_id=user_info["id"],
                operation=operation,
                details=details
            )
            
        except Exception as e:
            logger.error("ai_operation_logging_failed", error=str(e))
    
    async def get_user_ai_usage(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user AI/ML usage statistics."""
        try:
            redis_client = await self._get_redis_client()
            
            # Get usage data for the specified period
            usage_stats = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "operations": {},
                "models_used": {},
                "period_days": days
            }
            
            # This would typically aggregate from audit logs
            # For now, return basic structure
            
            return usage_stats
            
        except Exception as e:
            logger.error("get_user_ai_usage_failed", error=str(e))
            return {"error": "Failed to retrieve usage statistics"}
    
    async def check_rate_limit(self, user_info: Dict[str, Any], operation: str) -> bool:
        """Check if user has exceeded rate limits for AI operations."""
        try:
            redis_client = await self._get_redis_client()
            
            # Define rate limits for different operations
            rate_limits = {
                "policy_analysis": {"requests": 100, "window": 3600},  # 100 per hour
                "anomaly_detection": {"requests": 50, "window": 3600},  # 50 per hour
                "cost_optimization": {"requests": 20, "window": 3600},  # 20 per hour
                "predictive_analytics": {"requests": 30, "window": 3600},  # 30 per hour
                "sentiment_analysis": {"requests": 200, "window": 3600},  # 200 per hour
                "model_training": {"requests": 5, "window": 86400},  # 5 per day
                "batch_prediction": {"requests": 10, "window": 3600}  # 10 per hour
            }
            
            limit_config = rate_limits.get(operation, {"requests": 100, "window": 3600})
            
            # Check current usage
            rate_limit_key = f"rate_limit:{operation}:{user_info['id']}"
            current_count = await redis_client.get(rate_limit_key)
            
            if current_count is None:
                # First request in window
                await redis_client.set(rate_limit_key, 1, ex=limit_config["window"])
                return True
            elif int(current_count) < limit_config["requests"]:
                # Within limit
                await redis_client.incr(rate_limit_key)
                return True
            else:
                # Exceeded limit
                logger.warning(
                    "ai_rate_limit_exceeded",
                    user_id=user_info["id"],
                    operation=operation,
                    current_count=current_count,
                    limit=limit_config["requests"]
                )
                return False
                
        except Exception as e:
            logger.error("rate_limit_check_failed", error=str(e))
            return True  # Allow on error
    
    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            if self.key_vault_client:
                await self.key_vault_client.close()
                
            logger.info("auth_manager_cleanup_completed")
            
        except Exception as e:
            logger.error("auth_manager_cleanup_failed", error=str(e))