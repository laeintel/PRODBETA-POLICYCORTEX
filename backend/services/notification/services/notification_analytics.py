"""
Notification analytics service for tracking delivery metrics and performance.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import structlog

from shared.config import get_settings
from ..models import (
    NotificationStats,
    DeliveryStatus,
    DeliveryStatusEnum,
    AnalyticsEvent
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class NotificationAnalytics:
    """Service for tracking notification analytics and metrics."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        
    async def initialize(self) -> None:
        """Initialize the notification analytics service."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            logger.info("notification_analytics_initialized")
            
        except Exception as e:
            logger.error("notification_analytics_initialization_failed", error=str(e))
            raise
    
    async def track_notification(
        self,
        notification_id: str,
        notification_type: str,
        recipient_count: int,
        delivery_time: float
    ) -> None:
        """Track notification metrics."""
        try:
            # Update daily stats
            today = datetime.utcnow().strftime("%Y-%m-%d")
            stats_key = f"notification_stats:{today}"
            
            # Increment counters
            await self.redis_client.hincrby(stats_key, "total_sent", 1)
            await self.redis_client.hincrby(stats_key, f"sent_{notification_type}", 1)
            await self.redis_client.hincrby(stats_key, "total_recipients", recipient_count)
            
            # Track delivery time
            await self.redis_client.lpush(
                f"delivery_times:{notification_type}:{today}",
                delivery_time
            )
            await self.redis_client.expire(f"delivery_times:{notification_type}:{today}", 86400 * 7)
            
            # Set expiration for stats
            await self.redis_client.expire(stats_key, 86400 * 30)
            
        except Exception as e:
            logger.error("notification_tracking_failed", error=str(e))
    
    async def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> NotificationStats:
        """Get notification statistics."""
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=7)
            if not end_date:
                end_date = datetime.utcnow()
            
            stats = NotificationStats(
                total_sent=0,
                total_delivered=0,
                total_failed=0,
                delivery_rate=0.0,
                avg_delivery_time=0.0,
                stats_by_type={},
                stats_by_priority={}
            )
            
            # Aggregate stats for date range
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                stats_key = f"notification_stats:{date_str}"
                
                daily_stats = await self.redis_client.hgetall(stats_key)
                if daily_stats:
                    stats.total_sent += int(daily_stats.get("total_sent", 0))
                    stats.total_delivered += int(daily_stats.get("total_delivered", 0))
                    stats.total_failed += int(daily_stats.get("total_failed", 0))
                
                current_date += timedelta(days=1)
            
            # Calculate delivery rate
            if stats.total_sent > 0:
                stats.delivery_rate = stats.total_delivered / stats.total_sent
            
            return stats
            
        except Exception as e:
            logger.error("notification_stats_retrieval_failed", error=str(e))
            return NotificationStats(
                total_sent=0,
                total_delivered=0,
                total_failed=0,
                delivery_rate=0.0,
                avg_delivery_time=0.0,
                stats_by_type={},
                stats_by_priority={}
            )
    
    async def get_delivery_status(self, notification_id: str) -> Optional[DeliveryStatus]:
        """Get delivery status for a notification."""
        try:
            # Check different delivery type keys
            for delivery_type in ["email", "sms", "push", "webhook"]:
                delivery_key = f"{delivery_type}_delivery:{notification_id}"
                delivery_data = await self.redis_client.get(delivery_key)
                
                if delivery_data:
                    delivery = json.loads(delivery_data)
                    return DeliveryStatus(
                        notification_id=notification_id,
                        status=DeliveryStatusEnum.DELIVERED if delivery.get("delivered_count", 0) > 0 else DeliveryStatusEnum.FAILED,
                        sent_at=datetime.fromisoformat(delivery["sent_at"]),
                        delivered_at=datetime.fromisoformat(delivery["sent_at"]) if delivery.get("delivered_count", 0) > 0 else None,
                        failed_at=datetime.fromisoformat(delivery["sent_at"]) if delivery.get("failed_count", 0) > 0 else None,
                        retry_count=0,
                        tracking_events=delivery.get("delivery_details", [])
                    )
            
            return None
            
        except Exception as e:
            logger.error("delivery_status_retrieval_failed", error=str(e))
            return None
    
    async def run_analytics(self) -> None:
        """Run analytics processing."""
        try:
            while True:
                await self._process_analytics()
                await asyncio.sleep(300)  # Process every 5 minutes
                
        except Exception as e:
            logger.error("analytics_run_failed", error=str(e))
    
    async def _process_analytics(self) -> None:
        """Process analytics data."""
        try:
            # This would contain more complex analytics processing
            # For now, just log that analytics are running
            logger.info("analytics_processing_completed")
            
        except Exception as e:
            logger.error("analytics_processing_failed", error=str(e))
    
    async def health_check(self) -> bool:
        """Check analytics service health."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error("notification_analytics_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            logger.info("notification_analytics_cleanup_completed")
        except Exception as e:
            logger.error("notification_analytics_cleanup_failed", error=str(e))