"""
Alert manager for handling alerts with escalation rules and automatic resolution.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aioredis
import structlog
import uuid
from enum import Enum

from ....shared.config import get_settings
from ..models import (
    AlertRequest,
    AlertStatus,
    AlertSeverity,
    EscalationRule,
    NotificationRequest,
    NotificationResponse,
    NotificationType,
    NotificationPriority,
    NotificationContent,
    NotificationRecipient
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class AlertState(str, Enum):
    """Alert state enumeration."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"


class AlertManager:
    """Service for managing alerts with escalation rules."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.email_service = None
        self.sms_service = None
        self.push_service = None
        self.webhook_service = None
        self.escalation_tasks = {}
        
    async def initialize(self) -> None:
        """Initialize the alert manager."""
        try:
            # Initialize Redis client
            self.redis_client = aioredis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            # Start background tasks
            asyncio.create_task(self._process_escalations())
            asyncio.create_task(self._process_auto_resolutions())
            
            logger.info("alert_manager_initialized")
            
        except Exception as e:
            logger.error("alert_manager_initialization_failed", error=str(e))
            raise
    
    def set_notification_services(self, email_service, sms_service, push_service, webhook_service):
        """Set notification services for sending alerts."""
        self.email_service = email_service
        self.sms_service = sms_service
        self.push_service = push_service
        self.webhook_service = webhook_service
    
    async def create_alert(self, request: AlertRequest) -> str:
        """Create new alert."""
        try:
            alert_id = request.id or str(uuid.uuid4())
            
            # Create alert data
            alert_data = {
                "id": alert_id,
                "title": request.title,
                "description": request.description,
                "severity": request.severity.value,
                "source": request.source,
                "category": request.category,
                "initial_recipients": [r.dict() for r in request.initial_recipients],
                "escalation_rules": [r.dict() for r in request.escalation_rules] if request.escalation_rules else [],
                "auto_resolve": request.auto_resolve,
                "resolve_conditions": request.resolve_conditions or {},
                "metadata": request.metadata or {},
                "tags": request.tags or [],
                "state": AlertState.ACTIVE.value,
                "created_at": datetime.utcnow().isoformat(),
                "last_escalated": None,
                "resolved_at": None,
                "acknowledged_at": None,
                "current_level": 0,
                "notification_count": 0,
                "escalation_history": []
            }
            
            # Store alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert_data),
                ex=86400 * 30  # 30 days
            )
            
            # Add to active alerts
            await self.redis_client.sadd("active_alerts", alert_id)
            await self.redis_client.sadd(f"alerts_by_severity:{request.severity.value}", alert_id)
            
            # Send initial notifications
            await self._send_initial_notifications(alert_data)
            
            # Schedule escalation if rules exist
            if request.escalation_rules:
                await self._schedule_escalation(alert_id, request.escalation_rules[0])
            
            logger.info("alert_created", alert_id=alert_id, severity=request.severity.value)
            
            return alert_id
            
        except Exception as e:
            logger.error("alert_creation_failed", error=str(e))
            raise
    
    async def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get alert by ID."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if alert_data:
                return json.loads(alert_data)
            
            return None
            
        except Exception as e:
            logger.error("alert_retrieval_failed", error=str(e))
            return None
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> None:
        """Update alert."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if not alert_data:
                raise Exception(f"Alert {alert_id} not found")
            
            alert = json.loads(alert_data)
            
            # Update fields
            for key, value in updates.items():
                if key in alert:
                    alert[key] = value
            
            alert["updated_at"] = datetime.utcnow().isoformat()
            
            # Store updated alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert),
                ex=86400 * 30
            )
            
            logger.info("alert_updated", alert_id=alert_id, updates=list(updates.keys()))
            
        except Exception as e:
            logger.error("alert_update_failed", error=str(e))
            raise
    
    async def acknowledge_alert(self, alert_id: str, user_id: str) -> None:
        """Acknowledge alert."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if not alert_data:
                raise Exception(f"Alert {alert_id} not found")
            
            alert = json.loads(alert_data)
            
            # Update alert state
            alert["state"] = AlertState.ACKNOWLEDGED.value
            alert["acknowledged_at"] = datetime.utcnow().isoformat()
            alert["acknowledged_by"] = user_id
            
            # Store updated alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert),
                ex=86400 * 30
            )
            
            # Cancel escalation
            await self._cancel_escalation(alert_id)
            
            logger.info("alert_acknowledged", alert_id=alert_id, user_id=user_id)
            
        except Exception as e:
            logger.error("alert_acknowledgment_failed", error=str(e))
            raise
    
    async def resolve_alert(self, alert_id: str, user_id: Optional[str] = None, auto_resolved: bool = False) -> None:
        """Resolve alert."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if not alert_data:
                raise Exception(f"Alert {alert_id} not found")
            
            alert = json.loads(alert_data)
            
            # Update alert state
            alert["state"] = AlertState.RESOLVED.value
            alert["resolved_at"] = datetime.utcnow().isoformat()
            alert["auto_resolved"] = auto_resolved
            
            if user_id:
                alert["resolved_by"] = user_id
            
            # Store updated alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert),
                ex=86400 * 30
            )
            
            # Remove from active alerts
            await self.redis_client.srem("active_alerts", alert_id)
            await self.redis_client.srem(f"alerts_by_severity:{alert['severity']}", alert_id)
            
            # Cancel escalation
            await self._cancel_escalation(alert_id)
            
            # Send resolution notification
            await self._send_resolution_notification(alert)
            
            logger.info("alert_resolved", alert_id=alert_id, user_id=user_id, auto_resolved=auto_resolved)
            
        except Exception as e:
            logger.error("alert_resolution_failed", error=str(e))
            raise
    
    async def suppress_alert(self, alert_id: str, duration_minutes: int, user_id: str) -> None:
        """Suppress alert for specified duration."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if not alert_data:
                raise Exception(f"Alert {alert_id} not found")
            
            alert = json.loads(alert_data)
            
            # Update alert state
            alert["state"] = AlertState.SUPPRESSED.value
            alert["suppressed_at"] = datetime.utcnow().isoformat()
            alert["suppressed_by"] = user_id
            alert["suppressed_until"] = (datetime.utcnow() + timedelta(minutes=duration_minutes)).isoformat()
            
            # Store updated alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert),
                ex=86400 * 30
            )
            
            # Cancel escalation
            await self._cancel_escalation(alert_id)
            
            # Schedule unsuppression
            await self.redis_client.set(
                f"suppressed_alert:{alert_id}",
                json.dumps({"alert_id": alert_id, "unsuppress_at": alert["suppressed_until"]}),
                ex=duration_minutes * 60
            )
            
            logger.info("alert_suppressed", alert_id=alert_id, duration_minutes=duration_minutes, user_id=user_id)
            
        except Exception as e:
            logger.error("alert_suppression_failed", error=str(e))
            raise
    
    async def escalate_alert(self, alert_id: str, level: int) -> None:
        """Escalate alert to next level."""
        try:
            alert_data = await self.redis_client.get(f"alert:{alert_id}")
            
            if not alert_data:
                raise Exception(f"Alert {alert_id} not found")
            
            alert = json.loads(alert_data)
            
            # Check if alert is still active
            if alert["state"] not in [AlertState.ACTIVE.value, AlertState.ESCALATED.value]:
                logger.info("alert_escalation_skipped", alert_id=alert_id, state=alert["state"])
                return
            
            # Find escalation rule for this level
            escalation_rules = alert.get("escalation_rules", [])
            escalation_rule = None
            
            for rule in escalation_rules:
                if rule["level"] == level:
                    escalation_rule = rule
                    break
            
            if not escalation_rule:
                logger.warning("escalation_rule_not_found", alert_id=alert_id, level=level)
                return
            
            # Update alert state
            alert["state"] = AlertState.ESCALATED.value
            alert["current_level"] = level
            alert["last_escalated"] = datetime.utcnow().isoformat()
            
            # Add to escalation history
            alert["escalation_history"].append({
                "level": level,
                "escalated_at": datetime.utcnow().isoformat(),
                "recipients": escalation_rule["recipients"]
            })
            
            # Store updated alert
            await self.redis_client.set(
                f"alert:{alert_id}",
                json.dumps(alert),
                ex=86400 * 30
            )
            
            # Send escalation notifications
            await self._send_escalation_notifications(alert, escalation_rule)
            
            # Schedule next escalation if available
            next_level = level + 1
            next_rule = None
            for rule in escalation_rules:
                if rule["level"] == next_level:
                    next_rule = rule
                    break
            
            if next_rule:
                await self._schedule_escalation(alert_id, next_rule)
            
            logger.info("alert_escalated", alert_id=alert_id, level=level)
            
        except Exception as e:
            logger.error("alert_escalation_failed", error=str(e))
    
    async def _send_initial_notifications(self, alert_data: Dict[str, Any]) -> None:
        """Send initial alert notifications."""
        try:
            recipients = [NotificationRecipient(**r) for r in alert_data["initial_recipients"]]
            
            # Create notification content
            content = NotificationContent(
                title=f"Alert: {alert_data['title']}",
                body=f"Severity: {alert_data['severity']}\n\n{alert_data['description']}",
                subject=f"[{alert_data['severity'].upper()}] {alert_data['title']}"
            )
            
            # Send notifications via different channels
            await self._send_alert_notifications(recipients, content, alert_data)
            
            # Update notification count
            alert_data["notification_count"] += len(recipients)
            
        except Exception as e:
            logger.error("initial_notifications_send_failed", error=str(e))
    
    async def _send_escalation_notifications(self, alert_data: Dict[str, Any], escalation_rule: Dict[str, Any]) -> None:
        """Send escalation notifications."""
        try:
            recipients = [NotificationRecipient(**r) for r in escalation_rule["recipients"]]
            
            # Create escalation content
            content = NotificationContent(
                title=f"ESCALATED Alert: {alert_data['title']}",
                body=f"Severity: {alert_data['severity']}\nEscalation Level: {escalation_rule['level']}\n\n{alert_data['description']}",
                subject=f"[ESCALATED-{escalation_rule['level']}] {alert_data['title']}"
            )
            
            # Send notifications via specified channels
            notification_types = escalation_rule.get("notification_types", [NotificationType.EMAIL])
            await self._send_alert_notifications(recipients, content, alert_data, notification_types)
            
            # Update notification count
            alert_data["notification_count"] += len(recipients)
            
        except Exception as e:
            logger.error("escalation_notifications_send_failed", error=str(e))
    
    async def _send_resolution_notification(self, alert_data: Dict[str, Any]) -> None:
        """Send alert resolution notification."""
        try:
            recipients = [NotificationRecipient(**r) for r in alert_data["initial_recipients"]]
            
            # Create resolution content
            content = NotificationContent(
                title=f"RESOLVED: {alert_data['title']}",
                body=f"Alert has been resolved.\n\nOriginal alert:\nSeverity: {alert_data['severity']}\n{alert_data['description']}",
                subject=f"[RESOLVED] {alert_data['title']}"
            )
            
            # Send notifications
            await self._send_alert_notifications(recipients, content, alert_data)
            
        except Exception as e:
            logger.error("resolution_notification_send_failed", error=str(e))
    
    async def _send_alert_notifications(
        self, 
        recipients: List[NotificationRecipient], 
        content: NotificationContent, 
        alert_data: Dict[str, Any],
        notification_types: Optional[List[NotificationType]] = None
    ) -> None:
        """Send alert notifications via multiple channels."""
        try:
            if not notification_types:
                notification_types = [NotificationType.EMAIL, NotificationType.SMS, NotificationType.PUSH]
            
            # Group recipients by notification type
            for notification_type in notification_types:
                try:
                    # Create notification request
                    notification_request = NotificationRequest(
                        type=notification_type,
                        priority=self._get_priority_from_severity(alert_data["severity"]),
                        recipients=recipients,
                        content=content,
                        metadata={
                            "alert_id": alert_data["id"],
                            "alert_severity": alert_data["severity"],
                            "alert_source": alert_data["source"]
                        },
                        tags=["alert"] + alert_data.get("tags", [])
                    )
                    
                    # Send via appropriate service
                    if notification_type == NotificationType.EMAIL and self.email_service:
                        await self.email_service.send_email(notification_request)
                    elif notification_type == NotificationType.SMS and self.sms_service:
                        await self.sms_service.send_sms(notification_request)
                    elif notification_type == NotificationType.PUSH and self.push_service:
                        await self.push_service.send_push_notification(notification_request)
                    elif notification_type == NotificationType.WEBHOOK and self.webhook_service:
                        await self.webhook_service.send_webhook(notification_request)
                    
                except Exception as e:
                    logger.error("alert_notification_send_failed", error=str(e), type=notification_type)
                    continue
            
        except Exception as e:
            logger.error("alert_notifications_send_failed", error=str(e))
    
    def _get_priority_from_severity(self, severity: str) -> NotificationPriority:
        """Get notification priority from alert severity."""
        severity_priority_map = {
            AlertSeverity.CRITICAL: NotificationPriority.URGENT,
            AlertSeverity.ERROR: NotificationPriority.HIGH,
            AlertSeverity.WARNING: NotificationPriority.MEDIUM,
            AlertSeverity.INFO: NotificationPriority.LOW
        }
        return severity_priority_map.get(severity, NotificationPriority.MEDIUM)
    
    async def _schedule_escalation(self, alert_id: str, escalation_rule: Dict[str, Any]) -> None:
        """Schedule alert escalation."""
        try:
            escalation_time = datetime.utcnow() + timedelta(minutes=escalation_rule["delay_minutes"])
            
            escalation_data = {
                "alert_id": alert_id,
                "level": escalation_rule["level"],
                "escalate_at": escalation_time.isoformat()
            }
            
            # Store escalation schedule
            await self.redis_client.set(
                f"escalation:{alert_id}:{escalation_rule['level']}",
                json.dumps(escalation_data),
                ex=escalation_rule["delay_minutes"] * 60 + 3600  # Extra hour buffer
            )
            
            logger.info("escalation_scheduled", alert_id=alert_id, level=escalation_rule["level"], delay_minutes=escalation_rule["delay_minutes"])
            
        except Exception as e:
            logger.error("escalation_scheduling_failed", error=str(e))
    
    async def _cancel_escalation(self, alert_id: str) -> None:
        """Cancel all escalations for alert."""
        try:
            # Find all escalation keys for this alert
            escalation_keys = await self.redis_client.keys(f"escalation:{alert_id}:*")
            
            if escalation_keys:
                await self.redis_client.delete(*escalation_keys)
                logger.info("escalations_canceled", alert_id=alert_id, count=len(escalation_keys))
            
        except Exception as e:
            logger.error("escalation_cancellation_failed", error=str(e))
    
    async def _process_escalations(self) -> None:
        """Background task to process escalations."""
        try:
            while True:
                try:
                    # Find escalations ready to process
                    escalation_keys = await self.redis_client.keys("escalation:*")
                    
                    for key in escalation_keys:
                        try:
                            escalation_data = await self.redis_client.get(key)
                            if not escalation_data:
                                continue
                            
                            escalation = json.loads(escalation_data)
                            escalate_at = datetime.fromisoformat(escalation["escalate_at"])
                            
                            if datetime.utcnow() >= escalate_at:
                                # Process escalation
                                await self.escalate_alert(escalation["alert_id"], escalation["level"])
                                
                                # Remove processed escalation
                                await self.redis_client.delete(key)
                                
                        except Exception as e:
                            logger.error("escalation_processing_failed", error=str(e), key=key)
                            continue
                    
                    # Wait before next check
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error("escalation_processing_loop_failed", error=str(e))
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.error("escalation_processing_task_failed", error=str(e))
    
    async def _process_auto_resolutions(self) -> None:
        """Background task to process auto-resolutions."""
        try:
            while True:
                try:
                    # Find active alerts with auto-resolve enabled
                    active_alerts = await self.redis_client.smembers("active_alerts")
                    
                    for alert_id in active_alerts:
                        try:
                            alert_data = await self.redis_client.get(f"alert:{alert_id}")
                            if not alert_data:
                                continue
                            
                            alert = json.loads(alert_data)
                            
                            if alert.get("auto_resolve", False) and alert.get("resolve_conditions"):
                                # Check resolution conditions
                                should_resolve = await self._check_resolution_conditions(alert)
                                
                                if should_resolve:
                                    await self.resolve_alert(alert_id, auto_resolved=True)
                                    
                        except Exception as e:
                            logger.error("auto_resolution_processing_failed", error=str(e), alert_id=alert_id)
                            continue
                    
                    # Wait before next check
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                except Exception as e:
                    logger.error("auto_resolution_processing_loop_failed", error=str(e))
                    await asyncio.sleep(300)
                    
        except Exception as e:
            logger.error("auto_resolution_processing_task_failed", error=str(e))
    
    async def _check_resolution_conditions(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be auto-resolved."""
        try:
            conditions = alert.get("resolve_conditions", {})
            
            # Simple condition checking (can be extended)
            if "timeout_minutes" in conditions:
                created_at = datetime.fromisoformat(alert["created_at"])
                timeout_minutes = conditions["timeout_minutes"]
                
                if datetime.utcnow() >= created_at + timedelta(minutes=timeout_minutes):
                    return True
            
            # Add more condition types as needed
            
            return False
            
        except Exception as e:
            logger.error("resolution_conditions_check_failed", error=str(e))
            return False
    
    async def get_active_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active alerts."""
        try:
            if severity:
                alert_ids = await self.redis_client.smembers(f"alerts_by_severity:{severity}")
            else:
                alert_ids = await self.redis_client.smembers("active_alerts")
            
            alerts = []
            for alert_id in alert_ids:
                alert_data = await self.redis_client.get(f"alert:{alert_id}")
                if alert_data:
                    alerts.append(json.loads(alert_data))
            
            # Sort by severity and creation time
            severity_order = {"critical": 0, "error": 1, "warning": 2, "info": 3}
            alerts.sort(key=lambda x: (severity_order.get(x["severity"], 999), x["created_at"]))
            
            return alerts
            
        except Exception as e:
            logger.error("active_alerts_retrieval_failed", error=str(e))
            return []
    
    async def get_alert_stats(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get alert statistics."""
        try:
            # Get all alert keys
            alert_keys = await self.redis_client.keys("alert:*")
            
            stats = {
                "total_alerts": 0,
                "active_alerts": 0,
                "resolved_alerts": 0,
                "acknowledged_alerts": 0,
                "escalated_alerts": 0,
                "alerts_by_severity": {},
                "alerts_by_source": {},
                "avg_resolution_time": 0.0,
                "escalation_rate": 0.0
            }
            
            total_resolution_time = 0.0
            resolved_count = 0
            escalated_count = 0
            
            for key in alert_keys:
                try:
                    alert_data = await self.redis_client.get(key)
                    if not alert_data:
                        continue
                    
                    alert = json.loads(alert_data)
                    created_at = datetime.fromisoformat(alert["created_at"])
                    
                    # Filter by date range
                    if start_date and created_at < start_date:
                        continue
                    if end_date and created_at > end_date:
                        continue
                    
                    stats["total_alerts"] += 1
                    
                    # Count by state
                    state = alert["state"]
                    if state == AlertState.ACTIVE.value:
                        stats["active_alerts"] += 1
                    elif state == AlertState.RESOLVED.value:
                        stats["resolved_alerts"] += 1
                        resolved_count += 1
                        
                        # Calculate resolution time
                        if alert.get("resolved_at"):
                            resolved_at = datetime.fromisoformat(alert["resolved_at"])
                            resolution_time = (resolved_at - created_at).total_seconds() / 3600  # hours
                            total_resolution_time += resolution_time
                    elif state == AlertState.ACKNOWLEDGED.value:
                        stats["acknowledged_alerts"] += 1
                    elif state == AlertState.ESCALATED.value:
                        stats["escalated_alerts"] += 1
                        escalated_count += 1
                    
                    # Count by severity
                    severity = alert["severity"]
                    stats["alerts_by_severity"][severity] = stats["alerts_by_severity"].get(severity, 0) + 1
                    
                    # Count by source
                    source = alert["source"]
                    stats["alerts_by_source"][source] = stats["alerts_by_source"].get(source, 0) + 1
                    
                except Exception as e:
                    logger.error("alert_stats_processing_failed", error=str(e), key=key)
                    continue
            
            # Calculate averages
            if resolved_count > 0:
                stats["avg_resolution_time"] = total_resolution_time / resolved_count
            
            if stats["total_alerts"] > 0:
                stats["escalation_rate"] = escalated_count / stats["total_alerts"]
            
            return stats
            
        except Exception as e:
            logger.error("alert_stats_retrieval_failed", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check alert manager health."""
        try:
            # Check Redis connection
            await self.redis_client.ping()
            
            return True
            
        except Exception as e:
            logger.error("alert_manager_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Cancel all escalation tasks
            for task in self.escalation_tasks.values():
                task.cancel()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("alert_manager_cleanup_completed")
            
        except Exception as e:
            logger.error("alert_manager_cleanup_failed", error=str(e))