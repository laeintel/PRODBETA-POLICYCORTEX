"""
Subscription manager for handling user notification preferences and subscriptions.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis.asyncio as redis
import structlog
from shared.config import get_settings

    SubscriptionRequest,
    Subscription,
    NotificationPreferences,
    SubscriptionStatus,
    NotificationType
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class SubscriptionManager:
    """Service for managing user notification subscriptions and preferences."""

    def __init__(self):
        self.settings = settings
        self.redis_client = None

    async def initialize(self) -> None:
        """Initialize the subscription manager."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )

            logger.info("subscription_manager_initialized")

        except Exception as e:
            logger.error("subscription_manager_initialization_failed", error=str(e))
            raise

    async def create_subscription(self, request: SubscriptionRequest) -> str:
        """Create new subscription."""
        try:
            subscription_id = request.id or str(uuid.uuid4())

            subscription_data = {
                "id": subscription_id,
                "user_id": request.user_id,
                "channel": request.channel.value,
                "topic": request.topic,
                "filters": request.filters or {},
                "preferences": request.preferences.dict() if request.preferences else {},
                "status": request.status.value,
                "metadata": request.metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "expires_at": request.expires_at.isoformat() if request.expires_at else None
            }

            # Store subscription
            await self.redis_client.set(
                f"subscription:{subscription_id}",
                json.dumps(subscription_data),
                ex=86400 * 365  # 1 year
            )

            # Add to user's subscriptions
            await self.redis_client.sadd(f"user_subscriptions:{request.user_id}", subscription_id)

            # Add to topic subscriptions
            await self.redis_client.sadd(f"topic_subscriptions:{request.topic}", subscription_id)

            logger.info(
                "subscription_created",
                subscription_id=subscription_id,
                user_id=request.user_id
            )

            return subscription_id

        except Exception as e:
            logger.error("subscription_creation_failed", error=str(e))
            raise

    async def get_user_subscriptions(self, user_id: str) -> List[Subscription]:
        """Get all subscriptions for a user."""
        try:
            subscription_ids = await self.redis_client.smembers(f"user_subscriptions:{user_id}")
            subscriptions = []

            for subscription_id in subscription_ids:
                subscription_data = await self.redis_client.get(f"subscription:{subscription_id}")
                if subscription_data:
                    subscription = json.loads(subscription_data)
                    subscriptions.append(Subscription(**subscription))

            return subscriptions

        except Exception as e:
            logger.error("user_subscriptions_retrieval_failed", error=str(e))
            return []

    async def update_preferences(self, user_id: str, preferences: NotificationPreferences) -> None:
        """Update user notification preferences."""
        try:
            preferences_data = preferences.dict()

            await self.redis_client.set(
                f"user_preferences:{user_id}",
                json.dumps(preferences_data),
                ex=86400 * 365  # 1 year
            )

            logger.info("user_preferences_updated", user_id=user_id)

        except Exception as e:
            logger.error("user_preferences_update_failed", error=str(e))
            raise

    async def health_check(self) -> bool:
        """Check subscription manager health."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error("subscription_manager_health_check_failed", error=str(e))
            return False

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            logger.info("subscription_manager_cleanup_completed")
        except Exception as e:
            logger.error("subscription_manager_cleanup_failed", error=str(e))
