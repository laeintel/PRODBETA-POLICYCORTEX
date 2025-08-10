"""
Comprehensive Monitoring and Alerting Service for PolicyCortex
Provides real-time monitoring, alerting, and observability
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import statistics

# Prometheus metrics
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
import aiohttp
from asyncio import Queue

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    severity: AlertSeverity
    condition: str
    message: str
    resource_id: Optional[str]
    metric_name: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class Metric:
    """Metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class MonitoringRule:
    """Monitoring rule definition"""
    id: str
    name: str
    metric_name: str
    condition: str  # e.g., "value > 80", "rate > 100"
    threshold: float
    severity: AlertSeverity
    evaluation_period: int  # seconds
    cooldown_period: int  # seconds
    enabled: bool = True
    actions: List[str] = field(default_factory=list)  # webhook, email, etc.

class MonitoringService:
    """Comprehensive monitoring and alerting service"""
    
    def __init__(self):
        """Initialize monitoring service"""
        self.metrics_buffer: Dict[str, deque] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, MonitoringRule] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.metric_processors: Dict[str, Callable] = {}
        self.alert_channels: List[Callable] = []
        
        # Prometheus metrics
        self.request_counter = Counter('policycortex_requests_total', 'Total requests', ['method', 'endpoint'])
        self.error_counter = Counter('policycortex_errors_total', 'Total errors', ['type', 'service'])
        self.compliance_gauge = Gauge('policycortex_compliance_score', 'Current compliance score')
        self.cost_gauge = Gauge('policycortex_monthly_cost', 'Current monthly cost')
        self.resource_gauge = Gauge('policycortex_resources_total', 'Total resources', ['type', 'provider'])
        self.latency_histogram = Histogram('policycortex_request_latency_seconds', 'Request latency', ['endpoint'])
        self.alert_summary = Summary('policycortex_alerts_summary', 'Alert summary', ['severity'])
        
        # Initialize default rules
        self._initialize_default_rules()
        
        # Start monitoring loop
        self.monitoring_task = None
        
    def _initialize_default_rules(self):
        """Initialize default monitoring rules"""
        default_rules = [
            MonitoringRule(
                id="rule-cpu-high",
                name="High CPU Usage",
                metric_name="cpu_utilization",
                condition="value > threshold",
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                evaluation_period=300,
                cooldown_period=600,
                actions=["notify", "auto_scale"]
            ),
            MonitoringRule(
                id="rule-memory-critical",
                name="Critical Memory Usage",
                metric_name="memory_utilization",
                condition="value > threshold",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                evaluation_period=180,
                cooldown_period=300,
                actions=["notify", "remediate"]
            ),
            MonitoringRule(
                id="rule-compliance-low",
                name="Low Compliance Score",
                metric_name="compliance_score",
                condition="value < threshold",
                threshold=70.0,
                severity=AlertSeverity.MEDIUM,
                evaluation_period=3600,
                cooldown_period=7200,
                actions=["notify", "report"]
            ),
            MonitoringRule(
                id="rule-cost-spike",
                name="Cost Spike Detected",
                metric_name="daily_cost",
                condition="rate > threshold",
                threshold=1.5,  # 50% increase
                severity=AlertSeverity.HIGH,
                evaluation_period=86400,
                cooldown_period=86400,
                actions=["notify", "analyze"]
            ),
            MonitoringRule(
                id="rule-security-breach",
                name="Security Breach Detected",
                metric_name="security_events",
                condition="value > threshold",
                threshold=0,
                severity=AlertSeverity.CRITICAL,
                evaluation_period=60,
                cooldown_period=1800,
                actions=["notify", "lockdown", "investigate"]
            ),
            MonitoringRule(
                id="rule-api-latency",
                name="High API Latency",
                metric_name="api_latency_ms",
                condition="value > threshold",
                threshold=1000,  # 1 second
                severity=AlertSeverity.MEDIUM,
                evaluation_period=300,
                cooldown_period=600,
                actions=["notify", "scale"]
            ),
            MonitoringRule(
                id="rule-error-rate",
                name="High Error Rate",
                metric_name="error_rate",
                condition="value > threshold",
                threshold=5.0,  # 5% error rate
                severity=AlertSeverity.HIGH,
                evaluation_period=300,
                cooldown_period=900,
                actions=["notify", "investigate"]
            ),
            MonitoringRule(
                id="rule-backup-failure",
                name="Backup Failure",
                metric_name="backup_success",
                condition="value < threshold",
                threshold=1.0,
                severity=AlertSeverity.HIGH,
                evaluation_period=86400,
                cooldown_period=3600,
                actions=["notify", "retry", "escalate"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    async def start_monitoring(self):
        """Start the monitoring loop"""
        if not self.monitoring_task:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Monitoring service started")
    
    async def stop_monitoring(self):
        """Stop the monitoring loop"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            await asyncio.gather(self.monitoring_task, return_exceptions=True)
            self.monitoring_task = None
            logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Evaluate all rules
                for rule_id, rule in self.rules.items():
                    if rule.enabled:
                        await self._evaluate_rule(rule)
                
                # Process metric aggregations
                await self._process_aggregations()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next evaluation
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def record_metric(self, metric: Metric):
        """Record a metric data point"""
        # Store in buffer
        if metric.name not in self.metrics_buffer:
            self.metrics_buffer[metric.name] = deque(maxlen=10000)
        
        self.metrics_buffer[metric.name].append({
            'value': metric.value,
            'timestamp': metric.timestamp,
            'labels': metric.labels
        })
        
        # Update Prometheus metrics
        if metric.metric_type == MetricType.COUNTER:
            if metric.name == "requests":
                self.request_counter.labels(**metric.labels).inc(metric.value)
            elif metric.name == "errors":
                self.error_counter.labels(**metric.labels).inc(metric.value)
        elif metric.metric_type == MetricType.GAUGE:
            if metric.name == "compliance_score":
                self.compliance_gauge.set(metric.value)
            elif metric.name == "monthly_cost":
                self.cost_gauge.set(metric.value)
            elif metric.name == "resources":
                self.resource_gauge.labels(**metric.labels).set(metric.value)
        elif metric.metric_type == MetricType.HISTOGRAM:
            if metric.name == "latency":
                self.latency_histogram.labels(**metric.labels).observe(metric.value)
        
        # Run custom processors
        if metric.name in self.metric_processors:
            await self.metric_processors[metric.name](metric)
    
    async def _evaluate_rule(self, rule: MonitoringRule):
        """Evaluate a monitoring rule"""
        if rule.metric_name not in self.metrics_buffer:
            return
        
        # Get recent metrics
        metrics = self.metrics_buffer[rule.metric_name]
        if not metrics:
            return
        
        # Get metrics within evaluation period
        cutoff_time = datetime.utcnow() - timedelta(seconds=rule.evaluation_period)
        recent_metrics = [m for m in metrics if m['timestamp'] > cutoff_time]
        
        if not recent_metrics:
            return
        
        # Calculate current value based on condition
        current_value = self._calculate_metric_value(recent_metrics, rule.condition)
        
        # Check if threshold is breached
        threshold_breached = self._check_threshold(current_value, rule.threshold, rule.condition)
        
        alert_id = f"{rule.id}-{rule.metric_name}"
        
        if threshold_breached:
            # Check if alert already exists
            if alert_id not in self.active_alerts:
                # Check cooldown
                if self._check_cooldown(alert_id, rule.cooldown_period):
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        name=rule.name,
                        severity=rule.severity,
                        condition=rule.condition,
                        message=f"{rule.name}: {rule.metric_name} is {current_value:.2f} (threshold: {rule.threshold})",
                        resource_id=None,
                        metric_name=rule.metric_name,
                        threshold_value=rule.threshold,
                        current_value=current_value,
                        triggered_at=datetime.utcnow(),
                        metadata={'rule_id': rule.id}
                    )
                    
                    await self._trigger_alert(alert, rule.actions)
        else:
            # Resolve alert if it exists
            if alert_id in self.active_alerts:
                await self._resolve_alert(alert_id)
    
    def _calculate_metric_value(self, metrics: List[Dict], condition: str) -> float:
        """Calculate metric value based on condition"""
        values = [m['value'] for m in metrics]
        
        if not values:
            return 0.0
        
        if 'rate' in condition:
            # Calculate rate of change
            if len(values) >= 2:
                return (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
            return 0.0
        elif 'avg' in condition:
            return statistics.mean(values)
        elif 'max' in condition:
            return max(values)
        elif 'min' in condition:
            return min(values)
        else:
            # Default to latest value
            return values[-1]
    
    def _check_threshold(self, value: float, threshold: float, condition: str) -> bool:
        """Check if threshold is breached"""
        if '>' in condition:
            return value > threshold
        elif '<' in condition:
            return value < threshold
        elif '>=' in condition:
            return value >= threshold
        elif '<=' in condition:
            return value <= threshold
        elif '==' in condition:
            return abs(value - threshold) < 0.001
        return False
    
    def _check_cooldown(self, alert_id: str, cooldown_period: int) -> bool:
        """Check if alert is in cooldown period"""
        for alert in self.alert_history:
            if alert.id == alert_id:
                time_since_resolved = datetime.utcnow() - (alert.resolved_at or alert.triggered_at)
                if time_since_resolved.total_seconds() < cooldown_period:
                    return False
        return True
    
    async def _trigger_alert(self, alert: Alert, actions: List[str]):
        """Trigger an alert"""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Update Prometheus
        self.alert_summary.labels(severity=alert.severity.value).observe(1)
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        # Execute actions
        for action in actions:
            try:
                await self._execute_action(action, alert)
                alert.actions_taken.append(action)
            except Exception as e:
                logger.error(f"Failed to execute action {action}: {e}")
        
        # Send to alert channels
        for channel in self.alert_channels:
            try:
                await channel(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to channel: {e}")
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.name}")
            
            # Notify channels
            for channel in self.alert_channels:
                try:
                    await channel(alert)
                except Exception as e:
                    logger.error(f"Failed to notify channel of resolution: {e}")
    
    async def _execute_action(self, action: str, alert: Alert):
        """Execute an alert action"""
        if action == "notify":
            # Send notification (webhook, email, etc.)
            await self._send_notification(alert)
        elif action == "remediate":
            # Trigger automated remediation
            await self._trigger_remediation(alert)
        elif action == "scale":
            # Trigger auto-scaling
            await self._trigger_scaling(alert)
        elif action == "lockdown":
            # Security lockdown
            await self._trigger_lockdown(alert)
        elif action == "investigate":
            # Create investigation ticket
            await self._create_investigation(alert)
        elif action == "report":
            # Generate report
            await self._generate_report(alert)
        elif action == "escalate":
            # Escalate to higher level
            await self._escalate_alert(alert)
        elif action == "retry":
            # Retry failed operation
            await self._retry_operation(alert)
    
    async def _send_notification(self, alert: Alert):
        """Send alert notification"""
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "alert_id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "metric": alert.metric_name,
                    "value": alert.current_value,
                    "threshold": alert.threshold_value,
                    "timestamp": alert.triggered_at.isoformat()
                }
                
                try:
                    async with session.post(webhook_url, json=payload) as resp:
                        if resp.status != 200:
                            logger.error(f"Failed to send webhook: {resp.status}")
                except Exception as e:
                    logger.error(f"Webhook error: {e}")
    
    async def _trigger_remediation(self, alert: Alert):
        """Trigger automated remediation"""
        logger.info(f"Triggering remediation for {alert.name}")
        # Implementation would connect to remediation service
    
    async def _trigger_scaling(self, alert: Alert):
        """Trigger auto-scaling"""
        logger.info(f"Triggering auto-scaling for {alert.name}")
        # Implementation would connect to scaling service
    
    async def _trigger_lockdown(self, alert: Alert):
        """Trigger security lockdown"""
        logger.critical(f"SECURITY LOCKDOWN triggered for {alert.name}")
        # Implementation would connect to security service
    
    async def _create_investigation(self, alert: Alert):
        """Create investigation ticket"""
        logger.info(f"Creating investigation ticket for {alert.name}")
        # Implementation would connect to ticketing system
    
    async def _generate_report(self, alert: Alert):
        """Generate alert report"""
        logger.info(f"Generating report for {alert.name}")
        # Implementation would generate detailed report
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate alert to higher level"""
        logger.warning(f"Escalating alert {alert.name}")
        # Implementation would escalate to management
    
    async def _retry_operation(self, alert: Alert):
        """Retry failed operation"""
        logger.info(f"Retrying operation for {alert.name}")
        # Implementation would retry the failed operation
    
    async def _process_aggregations(self):
        """Process metric aggregations"""
        # Calculate aggregated metrics
        for metric_name, values in self.metrics_buffer.items():
            if values and len(values) > 10:
                recent_values = [v['value'] for v in list(values)[-100:]]
                
                # Calculate statistics
                avg_value = statistics.mean(recent_values)
                p95_value = statistics.quantiles(recent_values, n=20)[18] if len(recent_values) >= 20 else max(recent_values)
                
                # Store aggregations
                await self.record_metric(Metric(
                    name=f"{metric_name}_avg",
                    value=avg_value,
                    timestamp=datetime.utcnow(),
                    metric_type=MetricType.GAUGE
                ))
                
                await self.record_metric(Metric(
                    name=f"{metric_name}_p95",
                    value=p95_value,
                    timestamp=datetime.utcnow(),
                    metric_type=MetricType.GAUGE
                ))
    
    async def _cleanup_old_data(self):
        """Clean up old metric data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for metric_name in self.metrics_buffer:
            buffer = self.metrics_buffer[metric_name]
            # Remove old entries
            while buffer and buffer[0]['timestamp'] < cutoff_time:
                buffer.popleft()
    
    def add_rule(self, rule: MonitoringRule):
        """Add a monitoring rule"""
        self.rules[rule.id] = rule
        logger.info(f"Added monitoring rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove a monitoring rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed monitoring rule: {rule_id}")
    
    def add_alert_channel(self, channel: Callable):
        """Add an alert notification channel"""
        self.alert_channels.append(channel)
    
    def add_metric_processor(self, metric_name: str, processor: Callable):
        """Add a custom metric processor"""
        self.metric_processors[metric_name] = processor
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history"""
        return list(self.alert_history)[-limit:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        
        for metric_name, values in self.metrics_buffer.items():
            if values:
                recent_values = [v['value'] for v in list(values)[-100:]]
                summary[metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'average': statistics.mean(recent_values) if recent_values else 0,
                    'min': min(recent_values) if recent_values else 0,
                    'max': max(recent_values) if recent_values else 0,
                    'count': len(values)
                }
        
        return summary
    
    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus metrics in text format"""
        return generate_latest()

# Singleton instance
monitoring_service = MonitoringService()