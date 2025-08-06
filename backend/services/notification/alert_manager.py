"""
Enhanced Alert Manager Module
Advanced alert management with escalation and correlation
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


class AlertSource(str, Enum):
    COMPLIANCE = "compliance"
    SECURITY = "security"
    COST = "cost"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    GOVERNANCE = "governance"
    POLICY = "policy"
    AUTOMATION = "automation"


class EscalationLevel(str, Enum):
    LEVEL_0 = "level_0"  # Initial notification
    LEVEL_1 = "level_1"  # Team lead
    LEVEL_2 = "level_2"  # Manager
    LEVEL_3 = "level_3"  # Director
    LEVEL_4 = "level_4"  # Executive


@dataclass
class Alert:
    """Represents an alert"""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: AlertSource
    status: AlertStatus
    tenant_id: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    assigned_to: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    child_alert_ids: List[str] = field(default_factory=list)


@dataclass
class EscalationRule:
    """Escalation rule configuration"""

    rule_id: str
    alert_conditions: Dict[str, Any]
    escalation_levels: List[Dict[str, Any]]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class NotificationChannel:
    """Notification channel configuration"""

    channel_id: str
    channel_type: str  # email, sms, slack, teams, webhook
    config: Dict[str, Any]
    enabled: bool = True


class AlertManager:
    """
    Enhanced Alert Manager with escalation, correlation, and advanced routing
    """

    def __init__(self):
        self.alerts = {}
        self.escalation_rules = {}
        self.notification_channels = {}
        self.alert_correlations = {}
        self.suppression_rules = {}
        self.escalation_tasks = {}

    async def initialize(self) -> None:
        """Initialize the alert manager"""
        # Load default escalation rules
        await self._load_default_escalation_rules()

        # Load default notification channels
        await self._load_default_channels()

        # Start background tasks
        asyncio.create_task(self._run_escalation_processor())
        asyncio.create_task(self._run_correlation_engine())

    async def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        source: AlertSource,
        tenant_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new alert

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity level
            source: Alert source system
            tenant_id: Tenant identifier
            metadata: Additional metadata
            tags: Alert tags

        Returns:
            Alert ID
        """

        alert_id = str(uuid.uuid4())

        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            status=AlertStatus.OPEN,
            tenant_id=tenant_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=tags or [],
            metadata=metadata or {},
        )

        # Check for suppression
        if await self._should_suppress_alert(alert):
            alert.status = AlertStatus.SUPPRESSED
            logger.info(f"Alert {alert_id} suppressed due to suppression rules")

        # Check for correlation
        correlation_id = await self._correlate_alert(alert)
        if correlation_id:
            alert.correlation_id = correlation_id

        self.alerts[alert_id] = alert

        # Trigger initial notifications if not suppressed
        if alert.status != AlertStatus.SUPPRESSED:
            await self._trigger_notifications(alert)

        # Start escalation timer
        await self._schedule_escalation(alert)

        logger.info(
            f"Created alert {alert_id}",
            alert_id=alert_id,
            severity=severity.value,
            source=source.value,
            tenant_id=tenant_id,
        )

        return alert_id

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None
    ) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: User acknowledging the alert
            notes: Optional acknowledgment notes

        Returns:
            Success status
        """

        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]

        if alert.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
            logger.warning(f"Cannot acknowledge resolved/closed alert {alert_id}")
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = acknowledged_by
        alert.updated_at = datetime.utcnow()

        if notes:
            alert.metadata["acknowledgment_notes"] = notes

        # Cancel escalation
        await self._cancel_escalation(alert_id)

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

        return True

    async def assign_alert(self, alert_id: str, assigned_to: str, assigned_by: str) -> bool:
        """
        Assign an alert to a user

        Args:
            alert_id: Alert to assign
            assigned_to: User to assign to
            assigned_by: User making the assignment

        Returns:
            Success status
        """

        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.assigned_to = assigned_to
        alert.updated_at = datetime.utcnow()
        alert.metadata["assigned_by"] = assigned_by

        # Notify assigned user
        await self._notify_assignment(alert, assigned_to, assigned_by)

        logger.info(f"Alert {alert_id} assigned to {assigned_to} by {assigned_by}")

        return True

    async def resolve_alert(
        self, alert_id: str, resolved_by: str, resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert to resolve
            resolved_by: User resolving the alert
            resolution_notes: Resolution notes

        Returns:
            Success status
        """

        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.updated_at = datetime.utcnow()
        alert.metadata["resolved_by"] = resolved_by

        if resolution_notes:
            alert.metadata["resolution_notes"] = resolution_notes

        # Cancel escalation
        await self._cancel_escalation(alert_id)

        # Resolve correlated alerts
        if alert.correlation_id:
            await self._resolve_correlated_alerts(alert.correlation_id, resolved_by)

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")

        return True

    async def close_alert(
        self, alert_id: str, closed_by: str, close_reason: Optional[str] = None
    ) -> bool:
        """
        Close an alert

        Args:
            alert_id: Alert to close
            closed_by: User closing the alert
            close_reason: Reason for closing

        Returns:
            Success status
        """

        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]

        alert.status = AlertStatus.CLOSED
        alert.updated_at = datetime.utcnow()
        alert.metadata["closed_by"] = closed_by

        if close_reason:
            alert.metadata["close_reason"] = close_reason

        # Cancel escalation
        await self._cancel_escalation(alert_id)

        logger.info(f"Alert {alert_id} closed by {closed_by}")

        return True

    async def get_alerts(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        source: Optional[AlertSource] = None,
        assigned_to: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alerts with filtering"""

        alerts = list(self.alerts.values())

        # Apply filters
        if tenant_id:
            alerts = [a for a in alerts if a.tenant_id == tenant_id]
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        if assigned_to:
            alerts = [a for a in alerts if a.assigned_to == assigned_to]

        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x.created_at, reverse=True)

        return alerts[:limit]

    async def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert"""
        return self.alerts.get(alert_id)

    async def create_escalation_rule(
        self,
        rule_name: str,
        alert_conditions: Dict[str, Any],
        escalation_levels: List[Dict[str, Any]],
    ) -> str:
        """
        Create an escalation rule

        Args:
            rule_name: Name of the rule
            alert_conditions: Conditions for applying the rule
            escalation_levels: List of escalation level configurations

        Returns:
            Rule ID
        """

        rule_id = str(uuid.uuid4())

        rule = EscalationRule(
            rule_id=rule_id, alert_conditions=alert_conditions, escalation_levels=escalation_levels
        )

        self.escalation_rules[rule_id] = rule

        logger.info(f"Created escalation rule {rule_name} ({rule_id})")

        return rule_id

    async def create_notification_channel(
        self, channel_name: str, channel_type: str, config: Dict[str, Any]
    ) -> str:
        """
        Create a notification channel

        Args:
            channel_name: Channel name
            channel_type: Type of channel (email, sms, slack, etc.)
            config: Channel configuration

        Returns:
            Channel ID
        """

        channel_id = str(uuid.uuid4())

        channel = NotificationChannel(
            channel_id=channel_id, channel_type=channel_type, config=config
        )

        self.notification_channels[channel_id] = channel

        logger.info(f"Created notification channel {channel_name} ({channel_id})")

        return channel_id

    async def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""

        for rule in self.suppression_rules.values():
            if await self._matches_suppression_rule(alert, rule):
                return True

        # Check for duplicate alerts
        duplicate_threshold = timedelta(minutes=5)
        for existing_alert in self.alerts.values():
            if (
                existing_alert.title == alert.title
                and existing_alert.tenant_id == alert.tenant_id
                and existing_alert.status in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]
                and (alert.created_at - existing_alert.created_at) < duplicate_threshold
            ):
                return True

        return False

    async def _matches_suppression_rule(self, alert: Alert, rule: Dict[str, Any]) -> bool:
        """Check if alert matches suppression rule"""

        conditions = rule.get("conditions", {})

        # Check severity
        if "severity" in conditions:
            if alert.severity not in conditions["severity"]:
                return False

        # Check source
        if "source" in conditions:
            if alert.source not in conditions["source"]:
                return False

        # Check tags
        if "tags" in conditions:
            required_tags = set(conditions["tags"])
            alert_tags = set(alert.tags)
            if not required_tags.issubset(alert_tags):
                return False

        return True

    async def _correlate_alert(self, alert: Alert) -> Optional[str]:
        """Correlate alert with existing alerts"""

        # Simple correlation based on title similarity and time window
        correlation_window = timedelta(minutes=10)

        for existing_alert in self.alerts.values():
            if (
                existing_alert.tenant_id == alert.tenant_id
                and existing_alert.source == alert.source
                and existing_alert.status in [AlertStatus.OPEN, AlertStatus.ACKNOWLEDGED]
                and (alert.created_at - existing_alert.created_at) < correlation_window
            ):

                # Calculate title similarity (simplified)
                title_similarity = self._calculate_similarity(alert.title, existing_alert.title)

                if title_similarity > 0.8:  # 80% similarity threshold
                    correlation_id = existing_alert.correlation_id or existing_alert.alert_id

                    # Update correlation mapping
                    if correlation_id not in self.alert_correlations:
                        self.alert_correlations[correlation_id] = []
                    self.alert_correlations[correlation_id].append(alert.alert_id)

                    return correlation_id

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified implementation)"""

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    async def _trigger_notifications(self, alert: Alert) -> None:
        """Trigger initial notifications for an alert"""

        # Find applicable escalation rules
        applicable_rules = []
        for rule in self.escalation_rules.values():
            if await self._matches_escalation_conditions(alert, rule.alert_conditions):
                applicable_rules.append(rule)

        if not applicable_rules:
            # Use default notification
            await self._send_default_notification(alert)
            return

        # Use first matching rule for initial notification
        rule = applicable_rules[0]
        if rule.escalation_levels:
            level_0 = rule.escalation_levels[0]
            await self._send_escalation_notification(alert, level_0, EscalationLevel.LEVEL_0)

    async def _matches_escalation_conditions(
        self, alert: Alert, conditions: Dict[str, Any]
    ) -> bool:
        """Check if alert matches escalation conditions"""

        # Check severity
        if "severity" in conditions:
            if alert.severity.value not in conditions["severity"]:
                return False

        # Check source
        if "source" in conditions:
            if alert.source.value not in conditions["source"]:
                return False

        # Check tags
        if "tags" in conditions:
            required_tags = set(conditions["tags"])
            alert_tags = set(alert.tags)
            if not required_tags.issubset(alert_tags):
                return False

        return True

    async def _send_default_notification(self, alert: Alert) -> None:
        """Send default notification for an alert"""

        # Find default channels based on severity
        channels = []

        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            # Send to all configured channels
            channels = list(self.notification_channels.values())
        else:
            # Send to email only for lower severity
            channels = [c for c in self.notification_channels.values() if c.channel_type == "email"]

        for channel in channels:
            if channel.enabled:
                await self._send_channel_notification(alert, channel)

    async def _send_escalation_notification(
        self, alert: Alert, level_config: Dict[str, Any], level: EscalationLevel
    ) -> None:
        """Send escalation notification"""

        channel_ids = level_config.get("channels", [])

        for channel_id in channel_ids:
            if channel_id in self.notification_channels:
                channel = self.notification_channels[channel_id]
                await self._send_channel_notification(alert, channel, level)

    async def _send_channel_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send notification through specific channel"""

        try:
            if channel.channel_type == "email":
                await self._send_email_notification(alert, channel, level)
            elif channel.channel_type == "sms":
                await self._send_sms_notification(alert, channel, level)
            elif channel.channel_type == "slack":
                await self._send_slack_notification(alert, channel, level)
            elif channel.channel_type == "teams":
                await self._send_teams_notification(alert, channel, level)
            elif channel.channel_type == "webhook":
                await self._send_webhook_notification(alert, channel, level)

        except Exception as e:
            logger.error(f"Failed to send notification through {channel.channel_type}: {e}")

    async def _send_email_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send email notification"""

        subject = f"[{alert.severity.upper()}] {alert.title}"
        if level:
            subject = f"[ESCALATION {level.value.upper()}] {subject}"

        body = f"""
        Alert: {alert.title}
        Severity: {alert.severity.value}
        Source: {alert.source.value}
        Status: {alert.status.value}
        Created: {alert.created_at.isoformat()}
        
        Description:
        {alert.description}
        
        Alert ID: {alert.alert_id}
        Tenant: {alert.tenant_id}
        """

        recipients = channel.config.get("recipients", [])

        # Simulate email sending
        logger.info(f"Sending email notification for alert {alert.alert_id} to {recipients}")

    async def _send_sms_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send SMS notification"""

        message = f"Alert: {alert.title} ({alert.severity.value})"
        if level:
            message = f"ESCALATION: {message}"

        recipients = channel.config.get("recipients", [])

        # Simulate SMS sending
        logger.info(f"Sending SMS notification for alert {alert.alert_id} to {recipients}")

    async def _send_slack_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send Slack notification"""

        webhook_url = channel.config.get("webhook_url")
        if not webhook_url:
            return

        color_map = {
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.HIGH: "#ff8000",
            AlertSeverity.MEDIUM: "#ffff00",
            AlertSeverity.LOW: "#00ff00",
            AlertSeverity.INFO: "#0000ff",
        }

        message = {
            "text": f"Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source.value, "short": True},
                        {"title": "Status", "value": alert.status.value, "short": True},
                        {"title": "Created", "value": alert.created_at.isoformat(), "short": True},
                    ],
                    "text": alert.description,
                }
            ],
        }

        if level:
            message["text"] = f"[ESCALATION {level.value.upper()}] {message['text']}"

        # Simulate Slack webhook call
        logger.info(f"Sending Slack notification for alert {alert.alert_id}")

    async def _send_teams_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send Microsoft Teams notification"""

        webhook_url = channel.config.get("webhook_url")
        if not webhook_url:
            return

        # Simulate Teams webhook call
        logger.info(f"Sending Teams notification for alert {alert.alert_id}")

    async def _send_webhook_notification(
        self, alert: Alert, channel: NotificationChannel, level: Optional[EscalationLevel] = None
    ) -> None:
        """Send webhook notification"""

        webhook_url = channel.config.get("url")
        if not webhook_url:
            return

        payload = {
            "alert_id": alert.alert_id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "source": alert.source.value,
            "status": alert.status.value,
            "tenant_id": alert.tenant_id,
            "created_at": alert.created_at.isoformat(),
            "metadata": alert.metadata,
            "escalation_level": level.value if level else None,
        }

        # Simulate webhook call
        logger.info(f"Sending webhook notification for alert {alert.alert_id}")

    async def _schedule_escalation(self, alert: Alert) -> None:
        """Schedule escalation for an alert"""

        # Find applicable escalation rules
        for rule in self.escalation_rules.values():
            if await self._matches_escalation_conditions(alert, rule.alert_conditions):
                task = asyncio.create_task(self._run_escalation(alert, rule))
                self.escalation_tasks[alert.alert_id] = task
                break

    async def _run_escalation(self, alert: Alert, rule: EscalationRule) -> None:
        """Run escalation process for an alert"""

        try:
            for i, level_config in enumerate(
                rule.escalation_levels[1:], 1
            ):  # Skip level 0 (already sent)
                delay_minutes = level_config.get("delay_minutes", 30)

                # Wait for delay
                await asyncio.sleep(delay_minutes * 60)

                # Check if alert is still open
                current_alert = self.alerts.get(alert.alert_id)
                if not current_alert or current_alert.status not in [AlertStatus.OPEN]:
                    logger.info(f"Escalation stopped for alert {alert.alert_id} - status changed")
                    break

                # Send escalation notification
                level = EscalationLevel(f"level_{i}")
                await self._send_escalation_notification(alert, level_config, level)

                logger.info(f"Escalated alert {alert.alert_id} to {level.value}")

        except asyncio.CancelledError:
            logger.info(f"Escalation cancelled for alert {alert.alert_id}")
        except Exception as e:
            logger.error(f"Escalation failed for alert {alert.alert_id}: {e}")

    async def _cancel_escalation(self, alert_id: str) -> None:
        """Cancel escalation for an alert"""

        if alert_id in self.escalation_tasks:
            task = self.escalation_tasks[alert_id]
            task.cancel()
            del self.escalation_tasks[alert_id]
            logger.info(f"Cancelled escalation for alert {alert_id}")

    async def _resolve_correlated_alerts(self, correlation_id: str, resolved_by: str) -> None:
        """Resolve all alerts in a correlation group"""

        if correlation_id not in self.alert_correlations:
            return

        correlated_alert_ids = self.alert_correlations[correlation_id]

        for alert_id in correlated_alert_ids:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                if alert.status == AlertStatus.OPEN:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = datetime.utcnow()
                    alert.metadata["auto_resolved"] = True
                    alert.metadata["resolved_by"] = f"correlation:{resolved_by}"
                    await self._cancel_escalation(alert_id)

    async def _notify_assignment(self, alert: Alert, assigned_to: str, assigned_by: str) -> None:
        """Send notification about alert assignment"""

        # Find user's preferred notification channel
        # For now, just log
        logger.info(f"Alert {alert.alert_id} assigned to {assigned_to} by {assigned_by}")

    async def _run_escalation_processor(self) -> None:
        """Background task to process escalations"""

        while True:
            try:
                # Clean up completed escalation tasks
                completed_tasks = [
                    alert_id for alert_id, task in self.escalation_tasks.items() if task.done()
                ]

                for alert_id in completed_tasks:
                    del self.escalation_tasks[alert_id]

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Escalation processor error: {e}")
                await asyncio.sleep(60)

    async def _run_correlation_engine(self) -> None:
        """Background task to process alert correlations"""

        while True:
            try:
                # Perform advanced correlation analysis
                await self._advanced_correlation_analysis()

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Correlation engine error: {e}")
                await asyncio.sleep(300)

    async def _advanced_correlation_analysis(self) -> None:
        """Perform advanced correlation analysis"""

        # Get recent open alerts
        recent_alerts = [
            alert
            for alert in self.alerts.values()
            if (
                alert.status == AlertStatus.OPEN
                and (datetime.utcnow() - alert.created_at) < timedelta(hours=1)
            )
        ]

        # Group by tenant and analyze patterns
        tenant_alerts = {}
        for alert in recent_alerts:
            if alert.tenant_id not in tenant_alerts:
                tenant_alerts[alert.tenant_id] = []
            tenant_alerts[alert.tenant_id].append(alert)

        for tenant_id, alerts in tenant_alerts.items():
            if len(alerts) >= 3:  # Potential alert storm
                await self._handle_alert_storm(tenant_id, alerts)

    async def _handle_alert_storm(self, tenant_id: str, alerts: List[Alert]) -> None:
        """Handle alert storm scenario"""

        logger.warning(f"Alert storm detected for tenant {tenant_id}: {len(alerts)} alerts")

        # Create a summary alert
        storm_alert_id = await self.create_alert(
            title=f"Alert Storm Detected - {len(alerts)} alerts",
            description=f"Multiple alerts detected in short timeframe for tenant {tenant_id}",
            severity=AlertSeverity.HIGH,
            source=AlertSource.GOVERNANCE,
            tenant_id=tenant_id,
            metadata={
                "storm_alert_count": len(alerts),
                "storm_alert_ids": [a.alert_id for a in alerts],
                "storm_sources": list(set(a.source.value for a in alerts)),
            },
            tags=["alert_storm", "auto_generated"],
        )

        # Suppress individual alerts
        for alert in alerts:
            if alert.status == AlertStatus.OPEN:
                alert.status = AlertStatus.SUPPRESSED
                alert.metadata["suppressed_by"] = f"storm:{storm_alert_id}"
                await self._cancel_escalation(alert.alert_id)

    async def _load_default_escalation_rules(self) -> None:
        """Load default escalation rules"""

        # Critical alerts escalation
        await self.create_escalation_rule(
            "Critical Alerts",
            {"severity": ["critical"]},
            [
                {"delay_minutes": 0, "channels": ["email_primary", "sms_oncall"]},
                {"delay_minutes": 15, "channels": ["email_manager", "sms_manager"]},
                {"delay_minutes": 30, "channels": ["email_director", "slack_executives"]},
                {"delay_minutes": 60, "channels": ["email_executives", "sms_executives"]},
            ],
        )

        # High severity alerts escalation
        await self.create_escalation_rule(
            "High Severity Alerts",
            {"severity": ["high"]},
            [
                {"delay_minutes": 0, "channels": ["email_primary"]},
                {"delay_minutes": 30, "channels": ["email_manager", "slack_team"]},
                {"delay_minutes": 120, "channels": ["email_director"]},
            ],
        )

    async def _load_default_channels(self) -> None:
        """Load default notification channels"""

        # Email channels
        await self.create_notification_channel(
            "Primary Email", "email", {"recipients": ["oncall@policycortex.com"]}
        )

        await self.create_notification_channel(
            "Manager Email", "email", {"recipients": ["manager@policycortex.com"]}
        )

        # SMS channels
        await self.create_notification_channel(
            "On-call SMS", "sms", {"recipients": ["+1234567890"]}
        )

    async def health_check(self) -> bool:
        """Health check for the alert manager"""
        try:
            # Check if we can process alerts
            return len(self.escalation_rules) > 0
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Cancel all escalation tasks
        for task in self.escalation_tasks.values():
            task.cancel()
        self.escalation_tasks.clear()

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""

        total_alerts = len(self.alerts)

        if total_alerts == 0:
            return {"total_alerts": 0}

        # Count by status
        status_counts = {}
        for status in AlertStatus:
            status_counts[status.value] = sum(1 for a in self.alerts.values() if a.status == status)

        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = sum(
                1 for a in self.alerts.values() if a.severity == severity
            )

        # Count by source
        source_counts = {}
        for source in AlertSource:
            source_counts[source.value] = sum(1 for a in self.alerts.values() if a.source == source)

        # Average resolution time
        resolved_alerts = [a for a in self.alerts.values() if a.resolved_at]
        avg_resolution_time = None
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.created_at).total_seconds() / 60 for a in resolved_alerts
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)

        return {
            "total_alerts": total_alerts,
            "status_distribution": status_counts,
            "severity_distribution": severity_counts,
            "source_distribution": source_counts,
            "average_resolution_time_minutes": avg_resolution_time,
            "active_escalations": len(self.escalation_tasks),
            "correlation_groups": len(self.alert_correlations),
        }
