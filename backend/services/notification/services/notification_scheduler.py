"""
Notification scheduler for handling scheduled and recurring notifications.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
import structlog
import uuid
from croniter import croniter

from shared.config import get_settings
from ..models import (
    ScheduledNotificationRequest,
    NotificationRequest,
    NotificationResponse
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class NotificationScheduler:
    """Service for scheduling and managing notification delivery."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.running = False
        self.notification_services = {}
        
    async def initialize(self) -> None:
        """Initialize the notification scheduler."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            logger.info("notification_scheduler_initialized")
            
        except Exception as e:
            logger.error("notification_scheduler_initialization_failed", error=str(e))
            raise
    
    def set_notification_services(self, services: Dict[str, Any]) -> None:
        """Set notification services."""
        self.notification_services = services
    
    async def schedule_notification(self, request: ScheduledNotificationRequest) -> str:
        """Schedule a notification for future delivery."""
        try:
            scheduled_id = str(uuid.uuid4())
            
            scheduled_data = {
                "id": scheduled_id,
                "notification": request.dict(),
                "scheduled_time": request.scheduled_time.isoformat(),
                "recurrence": request.recurrence,
                "end_date": request.end_date.isoformat() if request.end_date else None,
                "max_occurrences": request.max_occurrences,
                "current_occurrences": 0,
                "created_at": datetime.utcnow().isoformat(),
                "status": "scheduled"
            }
            
            # Store scheduled notification
            await self.redis_client.set(
                f"scheduled_notification:{scheduled_id}",
                json.dumps(scheduled_data),
                ex=86400 * 365  # 1 year
            )
            
            # Add to scheduler queue
            await self.redis_client.zadd(
                "notification_schedule",
                {scheduled_id: request.scheduled_time.timestamp()}
            )
            
            logger.info("notification_scheduled", scheduled_id=scheduled_id)
            
            return scheduled_id
            
        except Exception as e:
            logger.error("notification_scheduling_failed", error=str(e))
            raise
    
    async def run_scheduler(self) -> None:
        """Run the notification scheduler."""
        self.running = True
        
        while self.running:
            try:
                await self._process_scheduled_notifications()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error("scheduler_run_failed", error=str(e))
                await asyncio.sleep(60)
    
    async def _process_scheduled_notifications(self) -> None:
        """Process scheduled notifications."""
        try:
            now = datetime.utcnow().timestamp()
            
            # Get notifications ready to send
            ready_notifications = await self.redis_client.zrangebyscore(
                "notification_schedule",
                0,
                now,
                withscores=True
            )
            
            for scheduled_id, score in ready_notifications:
                try:
                    await self._send_scheduled_notification(scheduled_id)
                    
                    # Remove from schedule
                    await self.redis_client.zrem("notification_schedule", scheduled_id)
                    
                except Exception as e:
                    logger.error("scheduled_notification_processing_failed", 
                               error=str(e), scheduled_id=scheduled_id)
                    continue
                    
        except Exception as e:
            logger.error("scheduled_notifications_processing_failed", error=str(e))
    
    async def _send_scheduled_notification(self, scheduled_id: str) -> None:
        """Send a scheduled notification."""
        try:
            # Get scheduled notification data
            scheduled_data = await self.redis_client.get(f"scheduled_notification:{scheduled_id}")
            if not scheduled_data:
                logger.warning("scheduled_notification_not_found", scheduled_id=scheduled_id)
                return
            
            scheduled = json.loads(scheduled_data)
            notification_data = scheduled["notification"]
            
            # Create notification request
            notification_request = NotificationRequest(**notification_data)
            
            # Send notification via appropriate service
            service = self.notification_services.get(notification_request.type)
            if service:
                await service.send_notification(notification_request)
            
            # Handle recurrence
            if scheduled.get("recurrence"):
                await self._handle_recurrence(scheduled_id, scheduled)
            
            # Update occurrence count
            scheduled["current_occurrences"] += 1
            await self.redis_client.set(
                f"scheduled_notification:{scheduled_id}",
                json.dumps(scheduled),
                ex=86400 * 365
            )
            
            logger.info("scheduled_notification_sent", scheduled_id=scheduled_id)
            
        except Exception as e:
            logger.error("scheduled_notification_send_failed", error=str(e))
    
    async def _handle_recurrence(self, scheduled_id: str, scheduled: Dict[str, Any]) -> None:
        """Handle recurring notifications."""
        try:
            recurrence = scheduled["recurrence"]
            current_time = datetime.utcnow()
            
            # Check if we should continue recurring
            if scheduled.get("end_date"):
                end_date = datetime.fromisoformat(scheduled["end_date"])
                if current_time >= end_date:
                    return
            
            if scheduled.get("max_occurrences"):
                if scheduled["current_occurrences"] >= scheduled["max_occurrences"]:
                    return
            
            # Calculate next occurrence
            cron = croniter(recurrence, current_time)
            next_time = cron.get_next(datetime)
            
            # Schedule next occurrence
            await self.redis_client.zadd(
                "notification_schedule",
                {scheduled_id: next_time.timestamp()}
            )
            
        except Exception as e:
            logger.error("recurrence_handling_failed", error=str(e))
    
    async def health_check(self) -> bool:
        """Check scheduler health."""
        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error("notification_scheduler_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.running = False
            if self.redis_client:
                await self.redis_client.close()
            logger.info("notification_scheduler_cleanup_completed")
        except Exception as e:
            logger.error("notification_scheduler_cleanup_failed", error=str(e))