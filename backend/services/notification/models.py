"""
Pydantic models for Notification service.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr, Field, validator


class NotificationType(str, Enum):
    """Notification type enumeration."""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    ALERT = "alert"


class NotificationPriority(str, Enum):
    """Notification priority enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class DeliveryStatusEnum(str, Enum):
    """Delivery status enumeration."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    """Alert severity enumeration."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class WebhookMethod(str, Enum):
    """Webhook HTTP method enumeration."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class TemplateType(str, Enum):
    """Template type enumeration."""

    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


# Base models
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class APIResponse(BaseModel):
    """Generic API response model."""

    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    message: Optional[str] = Field(None, description="Response message")
    request_id: Optional[str] = Field(None, description="Request identifier")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Notification models
class NotificationRecipient(BaseModel):
    """Notification recipient model."""

    id: Optional[str] = Field(None, description="Recipient identifier")
    email: Optional[EmailStr] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    device_token: Optional[str] = Field(None, description="Device token for push notifications")
    user_id: Optional[str] = Field(None, description="User identifier")
    name: Optional[str] = Field(None, description="Recipient name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional recipient metadata")


class NotificationContent(BaseModel):
    """Notification content model."""

    title: Optional[str] = Field(None, description="Notification title")
    body: str = Field(..., description="Notification body/message")
    html_body: Optional[str] = Field(None, description="HTML version of body")
    subject: Optional[str] = Field(None, description="Email subject")
    template_id: Optional[str] = Field(None, description="Template identifier")
    template_variables: Optional[Dict[str, Any]] = Field(None, description="Template variables")


class NotificationRequest(BaseModel):
    """Base notification request model."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Notification ID"
    )
    type: NotificationType = Field(..., description="Notification type")
    priority: NotificationPriority = Field(
        NotificationPriority.MEDIUM, description="Notification priority"
    )
    recipients: List[NotificationRecipient] = Field(..., description="Recipients list")
    content: NotificationContent = Field(..., description="Notification content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled delivery time")
    expires_at: Optional[datetime] = Field(None, description="Notification expiration time")
    retry_count: Optional[int] = Field(3, description="Number of retry attempts")
    callback_url: Optional[str] = Field(None, description="Callback URL for delivery status")
    tags: Optional[List[str]] = Field(None, description="Notification tags")

    @validator("recipients")
    def validate_recipients(cls, v):
        """Validate recipients list."""
        if not v:
            raise ValueError("At least one recipient is required")
        return v


class NotificationResponse(BaseModel):
    """Notification response model."""

    notification_id: str = Field(..., description="Notification identifier")
    status: DeliveryStatusEnum = Field(..., description="Delivery status")
    message: str = Field(..., description="Response message")
    sent_at: datetime = Field(..., description="Timestamp when sent")
    delivered_count: int = Field(0, description="Number of successful deliveries")
    failed_count: int = Field(0, description="Number of failed deliveries")
    delivery_details: Optional[List[Dict[str, Any]]] = Field(
        None, description="Detailed delivery information"
    )
    tracking_id: Optional[str] = Field(None, description="External tracking identifier")


# Email-specific models
class EmailRequest(NotificationRequest):
    """Email notification request model."""

    type: NotificationType = Field(NotificationType.EMAIL, description="Notification type")
    from_email: Optional[EmailStr] = Field(None, description="Sender email address")
    from_name: Optional[str] = Field(None, description="Sender name")
    reply_to: Optional[EmailStr] = Field(None, description="Reply-to email address")
    cc: Optional[List[EmailStr]] = Field(None, description="CC recipients")
    bcc: Optional[List[EmailStr]] = Field(None, description="BCC recipients")
    attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Email attachments")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom email headers")
    track_opens: bool = Field(True, description="Track email opens")
    track_clicks: bool = Field(True, description="Track link clicks")


class EmailTemplate(BaseModel):
    """Email template model."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Template ID")
    name: str = Field(..., description="Template name")
    subject: str = Field(..., description="Email subject template")
    html_body: str = Field(..., description="HTML body template")
    text_body: Optional[str] = Field(None, description="Plain text body template")
    variables: Optional[List[str]] = Field(None, description="Template variables")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Template metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )


# SMS-specific models
class SMSRequest(NotificationRequest):
    """SMS notification request model."""

    type: NotificationType = Field(NotificationType.SMS, description="Notification type")
    from_number: Optional[str] = Field(None, description="Sender phone number")
    provider: Optional[str] = Field(None, description="SMS provider to use")
    message_type: Optional[str] = Field("transactional", description="Message type")
    delivery_receipt: bool = Field(True, description="Request delivery receipt")


# Push notification models
class PushNotificationRequest(NotificationRequest):
    """Push notification request model."""

    type: NotificationType = Field(NotificationType.PUSH, description="Notification type")
    platform: Optional[str] = Field(None, description="Target platform (ios, android, web)")
    badge: Optional[int] = Field(None, description="Badge number")
    sound: Optional[str] = Field(None, description="Notification sound")
    icon: Optional[str] = Field(None, description="Notification icon")
    image: Optional[str] = Field(None, description="Notification image")
    action_buttons: Optional[List[Dict[str, str]]] = Field(None, description="Action buttons")
    deep_link: Optional[str] = Field(None, description="Deep link URL")
    custom_data: Optional[Dict[str, Any]] = Field(None, description="Custom push data")


# Webhook models
class WebhookRequest(NotificationRequest):
    """Webhook notification request model."""

    type: NotificationType = Field(NotificationType.WEBHOOK, description="Notification type")
    url: str = Field(..., description="Webhook URL")
    method: WebhookMethod = Field(WebhookMethod.POST, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")
    auth_token: Optional[str] = Field(None, description="Authentication token")
    timeout: Optional[int] = Field(30, description="Request timeout in seconds")
    verify_ssl: bool = Field(True, description="Verify SSL certificate")


# Alert models
class EscalationRule(BaseModel):
    """Alert escalation rule model."""

    level: int = Field(..., description="Escalation level")
    delay_minutes: int = Field(..., description="Delay before escalation")
    recipients: List[NotificationRecipient] = Field(..., description="Escalation recipients")
    notification_types: List[NotificationType] = Field(..., description="Notification types to use")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Escalation conditions")


class AlertRequest(BaseModel):
    """Alert request model."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Alert ID")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    severity: AlertSeverity = Field(..., description="Alert severity")
    source: str = Field(..., description="Alert source")
    category: Optional[str] = Field(None, description="Alert category")
    initial_recipients: List[NotificationRecipient] = Field(..., description="Initial recipients")
    escalation_rules: Optional[List[EscalationRule]] = Field(None, description="Escalation rules")
    auto_resolve: bool = Field(False, description="Auto-resolve when conditions are met")
    resolve_conditions: Optional[Dict[str, Any]] = Field(
        None, description="Auto-resolve conditions"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Alert metadata")
    tags: Optional[List[str]] = Field(None, description="Alert tags")


class AlertStatus(BaseModel):
    """Alert status model."""

    alert_id: str = Field(..., description="Alert identifier")
    status: str = Field(..., description="Alert status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_escalated: Optional[datetime] = Field(None, description="Last escalation timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    current_level: int = Field(0, description="Current escalation level")
    notification_count: int = Field(0, description="Number of notifications sent")


# Subscription models
class NotificationPreferences(BaseModel):
    """User notification preferences model."""

    email_enabled: bool = Field(True, description="Enable email notifications")
    sms_enabled: bool = Field(True, description="Enable SMS notifications")
    push_enabled: bool = Field(True, description="Enable push notifications")
    webhook_enabled: bool = Field(True, description="Enable webhook notifications")
    quiet_hours_start: Optional[str] = Field(None, description="Quiet hours start time (HH:MM)")
    quiet_hours_end: Optional[str] = Field(None, description="Quiet hours end time (HH:MM)")
    timezone: str = Field("UTC", description="User timezone")
    digest_frequency: Optional[str] = Field(None, description="Digest frequency")
    categories: Optional[Dict[str, bool]] = Field(None, description="Category preferences")


class SubscriptionRequest(BaseModel):
    """Subscription request model."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Subscription ID"
    )
    user_id: str = Field(..., description="User identifier")
    channel: NotificationType = Field(..., description="Notification channel")
    topic: str = Field(..., description="Subscription topic")
    filters: Optional[Dict[str, Any]] = Field(None, description="Subscription filters")
    preferences: Optional[NotificationPreferences] = Field(None, description="User preferences")
    status: SubscriptionStatus = Field(SubscriptionStatus.ACTIVE, description="Subscription status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Subscription metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Subscription expiration")


class Subscription(BaseModel):
    """Subscription model."""

    id: str = Field(..., description="Subscription identifier")
    user_id: str = Field(..., description="User identifier")
    channel: NotificationType = Field(..., description="Notification channel")
    topic: str = Field(..., description="Subscription topic")
    filters: Optional[Dict[str, Any]] = Field(None, description="Subscription filters")
    preferences: Optional[NotificationPreferences] = Field(None, description="User preferences")
    status: SubscriptionStatus = Field(..., description="Subscription status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Subscription metadata")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    expires_at: Optional[datetime] = Field(None, description="Subscription expiration")


# Template models
class NotificationTemplate(BaseModel):
    """Notification template model."""

    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Template ID")
    name: str = Field(..., description="Template name")
    type: TemplateType = Field(..., description="Template type")
    subject: Optional[str] = Field(None, description="Subject template")
    title: Optional[str] = Field(None, description="Title template")
    body: str = Field(..., description="Body template")
    html_body: Optional[str] = Field(None, description="HTML body template")
    variables: Optional[List[str]] = Field(None, description="Template variables")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Template metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    version: int = Field(1, description="Template version")
    is_active: bool = Field(True, description="Template active status")


# Bulk and scheduled notification models
class BulkNotificationRequest(BaseModel):
    """Bulk notification request model."""

    notifications: List[NotificationRequest] = Field(..., description="List of notifications")
    batch_size: Optional[int] = Field(100, description="Batch processing size")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    callback_url: Optional[str] = Field(None, description="Batch completion callback URL")


class ScheduledNotificationRequest(NotificationRequest):
    """Scheduled notification request model."""

    scheduled_time: datetime = Field(..., description="Scheduled delivery time")
    recurrence: Optional[str] = Field(None, description="Recurrence pattern")
    end_date: Optional[datetime] = Field(None, description="Recurrence end date")
    max_occurrences: Optional[int] = Field(None, description="Maximum recurrence count")


# Analytics and tracking models
class DeliveryStatus(BaseModel):
    """Delivery status model."""

    notification_id: str = Field(..., description="Notification identifier")
    status: DeliveryStatusEnum = Field(..., description="Delivery status")
    sent_at: datetime = Field(..., description="Sent timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Delivered timestamp")
    failed_at: Optional[datetime] = Field(None, description="Failed timestamp")
    error_message: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(0, description="Retry count")
    tracking_events: Optional[List[Dict[str, Any]]] = Field(None, description="Tracking events")


class NotificationStats(BaseModel):
    """Notification statistics model."""

    total_sent: int = Field(..., description="Total notifications sent")
    total_delivered: int = Field(..., description="Total notifications delivered")
    total_failed: int = Field(..., description="Total notifications failed")
    delivery_rate: float = Field(..., description="Delivery rate percentage")
    avg_delivery_time: float = Field(..., description="Average delivery time in seconds")
    stats_by_type: Dict[str, Dict[str, int]] = Field(
        ..., description="Statistics by notification type"
    )
    stats_by_priority: Dict[str, Dict[str, int]] = Field(..., description="Statistics by priority")
    time_series: Optional[List[Dict[str, Any]]] = Field(None, description="Time series data")
    top_failures: Optional[List[Dict[str, Any]]] = Field(None, description="Top failure reasons")


class AnalyticsEvent(BaseModel):
    """Analytics event model."""

    event_id: str = Field(..., description="Event identifier")
    notification_id: str = Field(..., description="Notification identifier")
    event_type: str = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    recipient_id: Optional[str] = Field(None, description="Recipient identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Event metadata")


# Provider configuration models
class SMSProviderConfig(BaseModel):
    """SMS provider configuration model."""

    provider: str = Field(..., description="Provider name")
    api_key: str = Field(..., description="API key")
    api_secret: Optional[str] = Field(None, description="API secret")
    endpoint: Optional[str] = Field(None, description="Provider endpoint")
    from_number: Optional[str] = Field(None, description="Default from number")
    settings: Optional[Dict[str, Any]] = Field(None, description="Provider-specific settings")


class EmailProviderConfig(BaseModel):
    """Email provider configuration model."""

    provider: str = Field(..., description="Provider name")
    smtp_host: Optional[str] = Field(None, description="SMTP host")
    smtp_port: Optional[int] = Field(None, description="SMTP port")
    smtp_username: Optional[str] = Field(None, description="SMTP username")
    smtp_password: Optional[str] = Field(None, description="SMTP password")
    api_key: Optional[str] = Field(None, description="API key")
    from_email: str = Field(..., description="Default from email")
    from_name: Optional[str] = Field(None, description="Default from name")
    settings: Optional[Dict[str, Any]] = Field(None, description="Provider-specific settings")


class PushProviderConfig(BaseModel):
    """Push notification provider configuration model."""

    provider: str = Field(..., description="Provider name")
    api_key: str = Field(..., description="API key")
    app_id: Optional[str] = Field(None, description="Application ID")
    certificate_path: Optional[str] = Field(None, description="Certificate file path")
    certificate_password: Optional[str] = Field(None, description="Certificate password")
    sandbox: bool = Field(False, description="Use sandbox environment")
    settings: Optional[Dict[str, Any]] = Field(None, description="Provider-specific settings")


# Azure Communication Services models
class AzureCommunicationConfig(BaseModel):
    """Azure Communication Services configuration model."""

    connection_string: str = Field(..., description="ACS connection string")
    email_domain: Optional[str] = Field(None, description="Email domain")
    sms_number: Optional[str] = Field(None, description="SMS number")
    resource_endpoint: Optional[str] = Field(None, description="Resource endpoint")
    settings: Optional[Dict[str, Any]] = Field(None, description="Additional settings")


class AzureEmailRequest(BaseModel):
    """Azure Communication Services email request model."""

    sender: str = Field(..., description="Sender email address")
    recipients: List[str] = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    html_content: Optional[str] = Field(None, description="HTML content")
    plain_text_content: Optional[str] = Field(None, description="Plain text content")
    attachments: Optional[List[Dict[str, Any]]] = Field(None, description="Email attachments")
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers")


class AzureSMSRequest(BaseModel):
    """Azure Communication Services SMS request model."""

    from_number: str = Field(..., description="Sender phone number")
    to_numbers: List[str] = Field(..., description="Recipient phone numbers")
    message: str = Field(..., description="SMS message")
    enable_delivery_receipt: bool = Field(True, description="Enable delivery receipt")
    tag: Optional[str] = Field(None, description="Message tag")
