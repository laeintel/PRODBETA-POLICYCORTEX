"""
Comprehensive Audit Logging System for PolicyCortex
Implements enterprise-grade audit trails with Azure Monitor integration
"""

import asyncio
import hashlib
import json
from collections import deque
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import redis.asyncio as redis
import structlog
from azure.core.exceptions import HttpResponseError
from azure.identity.aio import DefaultAzureCredential
from azure.monitor.ingestion import LogsIngestionClient
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import metrics
from opentelemetry import trace
from opentelemetry.metrics import get_meter
from shared.config import get_settings
from shared.database import AuditLog
from shared.database import DatabaseUtils
from shared.database import async_db_transaction

settings = get_settings()
logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)
meter = get_meter(__name__)


class AuditEventType(Enum):
    """Types of audit events for compliance tracking"""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    MFA_CHALLENGE = "mfa_challenge"
    PASSWORD_CHANGE = "password_change"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_ASSIGNMENT = "role_assignment"

    # Data events
    DATA_CREATE = "data_create"
    DATA_READ = "data_read"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"

    # Policy events
    POLICY_CREATE = "policy_create"
    POLICY_UPDATE = "policy_update"
    POLICY_DELETE = "policy_delete"
    POLICY_VIOLATION = "policy_violation"
    POLICY_EVALUATION = "policy_evaluation"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    COMPLIANCE_REMEDIATION = "compliance_remediation"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    ERROR = "error"
    WARNING = "warning"

    # Security events
    SECURITY_ALERT = "security_alert"
    THREAT_DETECTED = "threat_detected"
    VULNERABILITY_FOUND = "vulnerability_found"

    # Administrative events
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"
    TENANT_CREATE = "tenant_create"
    TENANT_UPDATE = "tenant_update"
    TENANT_DELETE = "tenant_delete"


class AuditSeverity(Enum):
    """Severity levels for audit events"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ComprehensiveAuditLogger:
    """
    Enterprise-grade audit logging system with compliance tracking
    Ensures all actions are logged for regulatory compliance
    """

    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.azure_credential = None
        self.logs_client = None
        self._audit_queue = deque(maxlen=1000)
        self._batch_size = 100
        self._flush_interval = 5  # seconds
        self._last_flush = datetime.utcnow()
        self._metrics_counter = {}

        # Initialize Azure Monitor if in production
        if settings.is_production():
            self._initialize_azure_monitor()

        # Start background tasks
        asyncio.create_task(self._batch_processor())

    def _initialize_azure_monitor(self):
        """Initialize Azure Monitor integration"""
        try:
            configure_azure_monitor(
                connection_string=settings.azure.application_insights_connection_string,
                logger_name="PolicyCortex.Audit",
            )

            # Initialize metrics
            self.event_counter = meter.create_counter(
                "audit_events_total", description="Total number of audit events", unit="1"
            )

            self.event_latency = meter.create_histogram(
                "audit_event_latency", description="Latency of audit event processing", unit="ms"
            )

            logger.info("azure_monitor_initialized")

        except Exception as e:
            logger.error("azure_monitor_initialization_failed", error=str(e))

    async def log_event(
        self,
        event_type: AuditEventType,
        tenant_id: str,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Log an audit event with full context
        Returns audit event ID for tracking
        """
        with tracer.start_as_current_span("log_audit_event") as span:
            span.set_attribute("event_type", event_type.value)
            span.set_attribute("tenant_id", tenant_id)

            # Generate audit event ID
            event_id = self._generate_event_id(event_type, tenant_id, user_id)

            # Create audit event
            audit_event = {
                "event_id": event_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type.value,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action or event_type.value,
                "result": result,
                "severity": severity.value,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "session_id": session_id,
                "correlation_id": correlation_id or event_id,
                "details": details or {},
                "compliance_frameworks": await self._get_applicable_frameworks(event_type),
                "retention_required": self._calculate_retention_period(event_type, severity),
            }

            # Add to queue for batch processing
            self._audit_queue.append(audit_event)

            # Log immediately if critical
            if severity == AuditSeverity.CRITICAL:
                await self._flush_events([audit_event])

            # Update metrics
            if settings.is_production():
                self.event_counter.add(
                    1,
                    {
                        "event_type": event_type.value,
                        "severity": severity.value,
                        "tenant_id": tenant_id,
                    },
                )

            # Store in database for persistence
            await self._persist_to_database(audit_event)

            # Send to Azure Monitor if configured
            if settings.is_production():
                await self._send_to_azure_monitor(audit_event)

            # Check for security alerts
            if await self._is_security_event(event_type, severity, details):
                await self._trigger_security_alert(audit_event)

            logger.info(
                "audit_event_logged",
                event_id=event_id,
                event_type=event_type.value,
                severity=severity.value,
            )

            return event_id

    def _generate_event_id(
        self, event_type: AuditEventType, tenant_id: str, user_id: Optional[str]
    ) -> str:
        """Generate unique audit event ID"""
        components = [
            event_type.value,
            tenant_id,
            user_id or "system",
            str(datetime.utcnow().timestamp()),
        ]

        return hashlib.sha256("_".join(components).encode()).hexdigest()[:32]

    async def _get_applicable_frameworks(self, event_type: AuditEventType) -> List[str]:
        """Get compliance frameworks that require this event type"""
        framework_requirements = {
            "SOC2": [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.DATA_DELETE,
                AuditEventType.CONFIG_CHANGE,
                AuditEventType.SECURITY_ALERT,
            ],
            "GDPR": [
                AuditEventType.DATA_CREATE,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_UPDATE,
                AuditEventType.DATA_DELETE,
                AuditEventType.DATA_EXPORT,
                AuditEventType.USER_DELETE,
            ],
            "HIPAA": [
                AuditEventType.DATA_READ,
                AuditEventType.DATA_UPDATE,
                AuditEventType.DATA_DELETE,
                AuditEventType.ACCESS_GRANTED,
                AuditEventType.ACCESS_DENIED,
            ],
            "PCI-DSS": [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.ACCESS_GRANTED,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.DATA_READ,
            ],
            "ISO27001": [
                AuditEventType.SECURITY_ALERT,
                AuditEventType.CONFIG_CHANGE,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.POLICY_VIOLATION,
            ],
        }

        applicable = []
        for framework, required_events in framework_requirements.items():
            if event_type in required_events:
                applicable.append(framework)

        return applicable

    def _calculate_retention_period(
        self, event_type: AuditEventType, severity: AuditSeverity
    ) -> int:
        """Calculate retention period in days based on compliance requirements"""
        # Critical events kept for 7 years
        if severity == AuditSeverity.CRITICAL:
            return 2555

        # Security and compliance events kept for 3 years
        security_events = [
            AuditEventType.SECURITY_ALERT,
            AuditEventType.THREAT_DETECTED,
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.COMPLIANCE_VIOLATION,
        ]
        if event_type in security_events:
            return 1095

        # Data modification events kept for 1 year
        data_events = [
            AuditEventType.DATA_CREATE,
            AuditEventType.DATA_UPDATE,
            AuditEventType.DATA_DELETE,
            AuditEventType.DATA_EXPORT,
        ]
        if event_type in data_events:
            return 365

        # Authentication events kept for 6 months
        auth_events = [
            AuditEventType.LOGIN_SUCCESS,
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.LOGOUT,
        ]
        if event_type in auth_events:
            return 180

        # Default retention: 90 days
        return 90

    async def _persist_to_database(self, audit_event: Dict[str, Any]) -> None:
        """Persist audit event to database"""
        try:
            async with async_db_transaction() as session:
                audit_log = AuditLog(
                    entity_type=audit_event.get("entity_type", "system"),
                    entity_id=audit_event.get("entity_id", audit_event["event_id"]),
                    action=audit_event["action"],
                    user_id=audit_event.get("user_id"),
                    session_id=audit_event.get("session_id"),
                    ip_address=audit_event.get("ip_address"),
                    user_agent=audit_event.get("user_agent"),
                    details=json.dumps(audit_event),
                )
                session.add(audit_log)
                await session.flush()

        except Exception as e:
            logger.error("audit_persistence_failed", event_id=audit_event["event_id"], error=str(e))

    async def _send_to_azure_monitor(self, audit_event: Dict[str, Any]) -> None:
        """Send audit event to Azure Monitor"""
        try:
            if not self.logs_client:
                self.azure_credential = DefaultAzureCredential()
                self.logs_client = LogsIngestionClient(
                    endpoint=settings.azure.log_analytics_endpoint, credential=self.azure_credential
                )

            # Format for Azure Monitor
            log_entry = {
                "TimeGenerated": audit_event["timestamp"],
                "EventId": audit_event["event_id"],
                "EventType": audit_event["event_type"],
                "TenantId": audit_event["tenant_id"],
                "UserId": audit_event.get("user_id", ""),
                "Severity": audit_event["severity"],
                "Result": audit_event["result"],
                "Details": json.dumps(audit_event["details"]),
                "ComplianceFrameworks": ",".join(audit_event["compliance_frameworks"]),
            }

            # Send to Log Analytics
            await self.logs_client.upload(
                rule_id=settings.azure.log_analytics_rule_id,
                stream_name="Custom-PolicyCortexAudit",
                logs=[log_entry],
            )

        except Exception as e:
            logger.error(
                "azure_monitor_send_failed", event_id=audit_event["event_id"], error=str(e)
            )

    async def _is_security_event(
        self, event_type: AuditEventType, severity: AuditSeverity, details: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if event requires security alert"""
        # Critical severity always triggers alert
        if severity == AuditSeverity.CRITICAL:
            return True

        # Security-related event types
        security_types = [
            AuditEventType.SECURITY_ALERT,
            AuditEventType.THREAT_DETECTED,
            AuditEventType.VULNERABILITY_FOUND,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.POLICY_VIOLATION,
        ]

        if event_type in security_types:
            return True

        # Multiple failed login attempts
        if event_type == AuditEventType.LOGIN_FAILURE:
            if details and details.get("attempt_count", 0) > 3:
                return True

        return False

    async def _trigger_security_alert(self, audit_event: Dict[str, Any]) -> None:
        """Trigger security alert for critical events"""
        logger.warning(
            "security_alert_triggered",
            event_id=audit_event["event_id"],
            event_type=audit_event["event_type"],
            severity=audit_event["severity"],
        )

        # This would integrate with notification service
        # For now, just log the alert
        alert = {
            "alert_id": hashlib.sha256(f"alert_{audit_event['event_id']}".encode()).hexdigest()[
                :16
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "audit_event_id": audit_event["event_id"],
            "tenant_id": audit_event["tenant_id"],
            "severity": audit_event["severity"],
            "description": f"Security event: {audit_event['event_type']}",
            "details": audit_event["details"],
        }

        # Store alert
        redis_client = await self._get_redis_client()
        alert_key = f"security_alert:{audit_event['tenant_id']}:{alert['alert_id']}"
        await redis_client.set(alert_key, json.dumps(alert), ex=86400)  # Keep for 24 hours

    async def _batch_processor(self) -> None:
        """Background task to process audit events in batches"""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)

                # Check if we should flush
                if (
                    len(self._audit_queue) >= self._batch_size
                    or (datetime.utcnow() - self._last_flush).seconds >= self._flush_interval
                ):

                    # Get events to flush
                    events_to_flush = []
                    while self._audit_queue and len(events_to_flush) < self._batch_size:
                        events_to_flush.append(self._audit_queue.popleft())

                    if events_to_flush:
                        await self._flush_events(events_to_flush)
                        self._last_flush = datetime.utcnow()

            except Exception as e:
                logger.error("batch_processor_error", error=str(e))

    async def _flush_events(self, events: List[Dict[str, Any]]) -> None:
        """Flush batch of events to storage"""
        if not events:
            return

        try:
            # Group by tenant for efficient storage
            events_by_tenant = {}
            for event in events:
                tenant_id = event["tenant_id"]
                if tenant_id not in events_by_tenant:
                    events_by_tenant[tenant_id] = []
                events_by_tenant[tenant_id].append(event)

            # Store in Redis for real-time access
            redis_client = await self._get_redis_client()

            for tenant_id, tenant_events in events_by_tenant.items():
                # Create daily audit log key
                date_key = datetime.utcnow().strftime("%Y%m%d")
                audit_key = f"audit_log:{tenant_id}:{date_key}"

                # Add events to sorted set with timestamp as score
                for event in tenant_events:
                    score = datetime.fromisoformat(event["timestamp"]).timestamp()
                    await redis_client.zadd(audit_key, {json.dumps(event): score})

                # Set expiration based on max retention
                await redis_client.expire(audit_key, 86400 * 2555)  # 7 years max

            logger.info("audit_events_flushed", count=len(events), tenants=len(events_by_tenant))

        except Exception as e:
            logger.error("flush_events_failed", error=str(e))

    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                settings.database.redis_url,
                password=settings.database.redis_password,
                ssl=settings.database.redis_ssl,
                decode_responses=True,
            )
        return self.redis_client

    async def query_audit_logs(
        self,
        tenant_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Query audit logs with filtering and pagination
        Returns (events, total_count)
        """
        with tracer.start_as_current_span("query_audit_logs") as span:
            span.set_attribute("tenant_id", tenant_id)

            # Default date range: last 30 days
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = end_date - timedelta(days=30)

            redis_client = await self._get_redis_client()
            all_events = []

            # Iterate through daily logs
            current_date = start_date
            while current_date <= end_date:
                date_key = current_date.strftime("%Y%m%d")
                audit_key = f"audit_log:{tenant_id}:{date_key}"

                # Get events for this day
                start_score = current_date.timestamp()
                end_score = min(
                    (current_date + timedelta(days=1)).timestamp(), end_date.timestamp()
                )

                events_json = await redis_client.zrangebyscore(audit_key, start_score, end_score)

                for event_json in events_json:
                    event = json.loads(event_json)

                    # Apply filters
                    if event_types and AuditEventType(event["event_type"]) not in event_types:
                        continue
                    if user_id and event.get("user_id") != user_id:
                        continue
                    if entity_id and event.get("entity_id") != entity_id:
                        continue
                    if severity and event["severity"] != severity.value:
                        continue

                    all_events.append(event)

                current_date += timedelta(days=1)

            # Sort by timestamp descending
            all_events.sort(key=lambda x: x["timestamp"], reverse=True)

            # Apply pagination
            total_count = len(all_events)
            paginated_events = all_events[offset : offset + limit]

            logger.info(
                "audit_logs_queried",
                tenant_id=tenant_id,
                total_count=total_count,
                returned_count=len(paginated_events),
            )

            return paginated_events, total_count

    async def generate_compliance_report(
        self, tenant_id: str, framework: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        logger.info("generating_compliance_report", tenant_id=tenant_id, framework=framework)

        # Get framework requirements
        required_events = self._get_framework_requirements(framework)

        # Query relevant audit logs
        events, total_count = await self.query_audit_logs(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            event_types=required_events,
        )

        # Analyze compliance
        report = {
            "tenant_id": tenant_id,
            "framework": framework,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_events": total_count,
            "event_breakdown": {},
            "compliance_score": 0,
            "findings": [],
            "recommendations": [],
        }

        # Count events by type
        for event in events:
            event_type = event["event_type"]
            if event_type not in report["event_breakdown"]:
                report["event_breakdown"][event_type] = 0
            report["event_breakdown"][event_type] += 1

        # Calculate compliance score
        expected_events = len(required_events)
        actual_events = len(report["event_breakdown"])
        report["compliance_score"] = (
            (actual_events / expected_events * 100) if expected_events > 0 else 0
        )

        # Generate findings
        for required_event in required_events:
            if required_event.value not in report["event_breakdown"]:
                report["findings"].append(
                    {
                        "severity": "high",
                        "issue": f"Missing required audit events: {required_event.value}",
                        "impact": f"Non-compliance with {framework} requirements",
                    }
                )

        # Generate recommendations
        if report["compliance_score"] < 100:
            report["recommendations"].append("Ensure all required event types are being logged")

        return report

    def _get_framework_requirements(self, framework: str) -> List[AuditEventType]:
        """Get required audit events for compliance framework"""
        requirements = {
            "SOC2": [
                AuditEventType.LOGIN_SUCCESS,
                AuditEventType.LOGIN_FAILURE,
                AuditEventType.ACCESS_DENIED,
                AuditEventType.DATA_DELETE,
                AuditEventType.CONFIG_CHANGE,
            ],
            "GDPR": [
                AuditEventType.DATA_CREATE,
                AuditEventType.DATA_READ,
                AuditEventType.DATA_UPDATE,
                AuditEventType.DATA_DELETE,
                AuditEventType.DATA_EXPORT,
            ],
            "HIPAA": [
                AuditEventType.DATA_READ,
                AuditEventType.DATA_UPDATE,
                AuditEventType.ACCESS_GRANTED,
                AuditEventType.ACCESS_DENIED,
            ],
        }

        return requirements.get(framework, [])

    async def export_audit_logs(
        self, tenant_id: str, start_date: datetime, end_date: datetime, format: str = "json"
    ) -> str:
        """Export audit logs for compliance or analysis"""
        events, total_count = await self.query_audit_logs(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            limit=10000,  # Export limit
        )

        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_events": total_count,
            "events": events,
        }

        if format == "json":
            return json.dumps(export_data, indent=2)
        else:
            # Could support CSV, XML, etc.
            return json.dumps(export_data)


# Global audit logger instance
audit_logger = ComprehensiveAuditLogger()
