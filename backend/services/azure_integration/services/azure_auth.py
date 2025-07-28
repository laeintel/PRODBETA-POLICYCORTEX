"""
Azure authentication service for managing Azure AD authentication and credentials.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import structlog
from azure.identity.aio import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.subscription.aio import SubscriptionClient
from azure.core.exceptions import AzureError
    import jwt
from jose import JWTError, jwt as jose_jwt

from shared.config import get_settings
from ..models import AzureAuthResponse

settings = get_settings()
logger = structlog.get_logger(__name__)


class AzureAuthService:
    """Service for Azure authentication and credential management."""

    def __init__(self):
        self.settings = settings
        self.credentials_cache = {}
        self.subscription_cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def authenticate(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        subscription_ids: Optional[List[str]] = None
    ) -> AzureAuthResponse:
        """Authenticate with Azure AD and return tokens."""
        try:
            # Create credential
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )

            # Get access token
            token = await credential.get_token("https://management.azure.com/.default")

            # Get available subscriptions
            if not subscription_ids:
                subscription_ids = await self._get_subscriptions(credential)

            # Generate JWT tokens
            user_info = {
                "id": client_id,
                "tenant_id": tenant_id,
                "subscription_ids": subscription_ids,
                "type": "service_principal"
            }

            access_token = self._generate_jwt(user_info, expires_minutes=30)
            refresh_token = self._generate_jwt(user_info, expires_minutes=10080)  # 7 days

            # Cache credentials
            cache_key = f"{tenant_id}:{client_id}"
            self.credentials_cache[cache_key] = {
                "credential": credential,
                "timestamp": time.time(),
                "subscription_ids": subscription_ids
            }

            logger.info(
                "azure_authentication_successful",
                tenant_id=tenant_id,
                client_id=client_id,
                subscription_count=len(subscription_ids)
            )

            return AzureAuthResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="Bearer",
                expires_in=1800,  # 30 minutes
                tenant_id=tenant_id,
                subscription_ids=subscription_ids,
                user_info=user_info
            )

        except Exception as e:
            logger.error(
                "azure_authentication_failed",
                error=str(e),
                tenant_id=tenant_id,
                client_id=client_id
            )
            raise Exception(f"Azure authentication failed: {str(e)}")

    async def refresh_token(self, refresh_token: str) -> AzureAuthResponse:
        """Refresh authentication token."""
        try:
            # Decode refresh token
            payload = jose_jwt.decode(
                refresh_token,
                settings.security.jwt_secret_key,
                algorithms=[settings.security.jwt_algorithm]
            )

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise JWTError("Refresh token has expired")

            # Extract user info
            user_info = {
                "id": payload.get("id"),
                "tenant_id": payload.get("tenant_id"),
                "subscription_ids": payload.get("subscription_ids", []),
                "type": payload.get("type", "service_principal")
            }

            # Generate new access token
            access_token = self._generate_jwt(user_info, expires_minutes=30)

            logger.info(
                "token_refresh_successful",
                tenant_id=user_info["tenant_id"],
                user_id=user_info["id"]
            )

            return AzureAuthResponse(
                access_token=access_token,
                refresh_token=refresh_token,  # Return same refresh token
                token_type="Bearer",
                expires_in=1800,  # 30 minutes
                tenant_id=user_info["tenant_id"],
                subscription_ids=user_info["subscription_ids"],
                user_info=user_info
            )

        except JWTError as e:
            logger.error("token_refresh_failed", error=str(e))
            raise Exception(f"Invalid refresh token: {str(e)}")
        except Exception as e:
            logger.error("token_refresh_error", error=str(e))
            raise Exception(f"Token refresh failed: {str(e)}")

    async def get_credential(self, tenant_id: str, client_id: Optional[str] = None):
        """Get Azure credential from cache or create new one."""
        try:
            # Check cache
            if client_id:
                cache_key = f"{tenant_id}:{client_id}"
                cached = self.credentials_cache.get(cache_key)
                if cached and (time.time() - cached["timestamp"]) < self.cache_ttl:
                    return cached["credential"]

            # Use default credential if no client_id provided
            if not client_id:
                return DefaultAzureCredential()

            # Create new credential using settings
            credential = ClientSecretCredential(
                tenant_id=tenant_id or settings.azure.tenant_id,
                client_id=client_id or settings.azure.client_id,
                client_secret=settings.azure.client_secret
            )

            return credential

        except Exception as e:
            logger.error("get_credential_failed", error=str(e))
            raise Exception(f"Failed to get Azure credential: {str(e)}")

    async def verify_azure_connection(self) -> bool:
        """Verify Azure connection is working."""
        try:
            credential = await self.get_credential(settings.azure.tenant_id)

            # Try to get a token
            token = await credential.get_token("https://management.azure.com/.default")

            return token is not None

        except Exception as e:
            logger.error("azure_connection_verification_failed", error=str(e))
            return False

    async def _get_subscriptions(self, credential) -> List[str]:
        """Get list of accessible subscriptions."""
        try:
            # Check cache
            cache_key = "subscriptions"
            cached = self.subscription_cache.get(cache_key)
            if cached and (time.time() - cached["timestamp"]) < self.cache_ttl:
                return cached["subscription_ids"]

            # Get subscriptions from Azure
            async with SubscriptionClient(credential) as client:
                subscriptions = []
                async for sub in client.subscriptions.list():
                    subscriptions.append(sub.subscription_id)

            # Cache results
            self.subscription_cache[cache_key] = {
                "subscription_ids": subscriptions,
                "timestamp": time.time()
            }

            return subscriptions

        except Exception as e:
            logger.error("get_subscriptions_failed", error=str(e))
            # Return configured subscription as fallback
            return [settings.azure.subscription_id]

    def _generate_jwt(self, user_info: Dict[str, Any], expires_minutes: int) -> str:
        """Generate JWT token."""
        try:
            now = datetime.utcnow()
            exp = now + timedelta(minutes=expires_minutes)

            payload = {
                "sub": user_info["id"],
                "id": user_info["id"],
                "tenant_id": user_info["tenant_id"],
                "subscription_ids": user_info["subscription_ids"],
                "type": user_info.get("type", "service_principal"),
                "iat": now,
                "exp": exp
            }

            token = jose_jwt.encode(
                payload,
                settings.security.jwt_secret_key,
                algorithm=settings.security.jwt_algorithm
            )

            return token

        except Exception as e:
            logger.error("jwt_generation_failed", error=str(e))
            raise Exception(f"Failed to generate JWT: {str(e)}")

    async def validate_subscription_access(
        self,
        credential,
        subscription_id: str
    ) -> bool:
        """Validate if credential has access to subscription."""
        try:
            async with SubscriptionClient(credential) as client:
                subscription = await client.subscriptions.get(subscription_id)
                return subscription is not None

        except AzureError:
            return False
        except Exception as e:
            logger.error(
                "subscription_validation_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            return False
