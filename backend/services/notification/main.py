"""
Notification Service for PolicyCortex.
Handles email, SMS, push notifications, webhooks, alerts, and subscription management.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from shared.config import get_settings
from shared.database import get_async_db, DatabaseUtils
from services.notification.auth import AuthManager
    HealthResponse,
    APIResponse,
    ErrorResponse,
    NotificationRequest,
    NotificationResponse,
    EmailRequest,
    SMSRequest,
    PushNotificationRequest,
    WebhookRequest,
    AlertRequest,
    SubscriptionRequest,
    NotificationStats,
    DeliveryStatus,
    NotificationTemplate,
    BulkNotificationRequest,
    ScheduledNotificationRequest,
    NotificationPreferences
)
    EmailService,
    SMSService,
    PushNotificationService,
    WebhookService,
    AlertManager,
    SubscriptionManager,
    NotificationScheduler,
    NotificationAnalytics,
    AzureCommunicationService
)

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'notification_requests_total',
    'Total notification requests',
    ['method',
    'endpoint',
    'status']
)
REQUEST_DURATION = Histogram('notification_request_duration_seconds', 'Request duration')
NOTIFICATION_COUNT = Counter(
    'notifications_sent_total',
    'Total notifications sent',
    ['type',
    'status']
)
NOTIFICATION_DELIVERY_TIME = Histogram(
    'notification_delivery_seconds',
    'Notification delivery time',
    ['type']
)

# Security
security = HTTPBearer(auto_error=False)

# Global services
auth_manager = AuthManager()
email_service = EmailService()
sms_service = SMSService()
push_service = PushNotificationService()
webhook_service = WebhookService()
alert_manager = AlertManager()
subscription_manager = SubscriptionManager()
notification_scheduler = NotificationScheduler()
notification_analytics = NotificationAnalytics()
azure_communication_service = AzureCommunicationService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("notification_service_starting")

    # Initialize services
    await email_service.initialize()
    await sms_service.initialize()
    await push_service.initialize()
    await webhook_service.initialize()
    await alert_manager.initialize()
    await subscription_manager.initialize()
    await notification_scheduler.initialize()
    await notification_analytics.initialize()
    await azure_communication_service.initialize()

    # Start background tasks
    scheduler_task = asyncio.create_task(notification_scheduler.run_scheduler())
    analytics_task = asyncio.create_task(notification_analytics.run_analytics())

    logger.info("notification_service_started")

    yield

    # Cleanup
    logger.info("notification_service_stopping")

    # Cancel background tasks
    scheduler_task.cancel()
    analytics_task.cancel()

    # Cleanup services
    await email_service.cleanup()
    await sms_service.cleanup()
    await push_service.cleanup()
    await webhook_service.cleanup()
    await alert_manager.cleanup()
    await subscription_manager.cleanup()
    await notification_scheduler.cleanup()
    await notification_analytics.cleanup()
    await azure_communication_service.cleanup()

    logger.info("notification_service_stopped")


# FastAPI app
app = FastAPI(
    title="PolicyCortex Notification Service",
    description="Comprehensive notification service for PolicyCortex platform",
    version=settings.service.service_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=settings.security.cors_methods,
    allow_headers=settings.security.cors_headers,
        )

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Add request ID to headers
        request.state.request_id = request_id

        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            REQUEST_DURATION.observe(duration)

            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Log error
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )

            # Update error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()

            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)


async def verify_authentication(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Verify authentication for protected endpoints."""

    # Skip authentication for health checks and public endpoints
    if request.url.path in ["/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return None

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        user_info = await auth_manager.verify_token(credentials.credentials)
        request.state.user = user_info
        return user_info
    except Exception as e:
        logger.error("authentication_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="notification-service",
        version=settings.service.service_version
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    # Check service dependencies
    checks = {
        "email_service": await email_service.health_check(),
        "sms_service": await sms_service.health_check(),
        "push_service": await push_service.health_check(),
        "webhook_service": await webhook_service.health_check(),
        "alert_manager": await alert_manager.health_check(),
        "subscription_manager": await subscription_manager.health_check(),
        "notification_scheduler": await notification_scheduler.health_check(),
        "azure_communication": await azure_communication_service.health_check()
    }

    failed_checks = [name for name, status in checks.items() if not status]

    if failed_checks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service dependencies unhealthy: {', '.join(failed_checks)}"
        )

    return HealthResponse(
        status="ready",
        timestamp=datetime.utcnow(),
        service="notification-service",
        version=settings.service.service_version,
        details={"service_checks": checks}
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# Notification endpoints
@app.post("/api/v1/notifications/send", response_model=NotificationResponse)
async def send_notification(
    request: NotificationRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send a notification through appropriate channel."""
    try:
        start_time = time.time()

        # Validate request
        if not request.recipients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one recipient is required"
            )

        # Process notification based on type
        result = None
        if request.type == "email":
            result = await email_service.send_email(request)
        elif request.type == "sms":
            result = await sms_service.send_sms(request)
        elif request.type == "push":
            result = await push_service.send_push_notification(request)
        elif request.type == "webhook":
            result = await webhook_service.send_webhook(request)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported notification type: {request.type}"
            )

        # Update metrics
        delivery_time = time.time() - start_time
        NOTIFICATION_COUNT.labels(type=request.type, status="success").inc()
        NOTIFICATION_DELIVERY_TIME.labels(type=request.type).observe(delivery_time)

        # Track analytics in background
        background_tasks.add_task(
            notification_analytics.track_notification,
            result.notification_id,
            request.type,
            len(request.recipients),
            delivery_time
        )

        logger.info(
            "notification_sent",
            notification_id=result.notification_id,
            type=request.type,
            recipient_count=len(request.recipients),
            delivery_time_ms=round(delivery_time * 1000, 2)
        )

        return result

    except Exception as e:
        NOTIFICATION_COUNT.labels(type=request.type, status="error").inc()
        logger.error("notification_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send notification: {str(e)}"
        )


@app.post("/api/v1/notifications/bulk", response_model=List[NotificationResponse])
async def send_bulk_notifications(
    request: BulkNotificationRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send multiple notifications in bulk."""
    try:
        results = []

        for notification in request.notifications:
            try:
                result = await send_notification(notification, background_tasks, user)
                results.append(result)
            except Exception as e:
                logger.error(
                    "bulk_notification_failed",
                    notification_id=notification.id,
                    error=str(e)
                )
                # Continue with other notifications
                continue

        return results

    except Exception as e:
        logger.error("bulk_notification_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send bulk notifications: {str(e)}"
        )


@app.post("/api/v1/notifications/schedule", response_model=APIResponse)
async def schedule_notification(
    request: ScheduledNotificationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Schedule a notification for future delivery."""
    try:
        scheduled_id = await notification_scheduler.schedule_notification(request)

        return APIResponse(
            success=True,
            data={"scheduled_id": scheduled_id},
            message="Notification scheduled successfully"
        )

    except Exception as e:
        logger.error("notification_schedule_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule notification: {str(e)}"
        )


# Email endpoints
@app.post("/api/v1/notifications/email/send", response_model=NotificationResponse)
async def send_email(
    request: EmailRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send email notification."""
    try:
        result = await email_service.send_email(request)

        # Track analytics
        background_tasks.add_task(
            notification_analytics.track_email_delivery,
            result.notification_id,
            request.recipients,
            result.status
        )

        return result

    except Exception as e:
        logger.error("email_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send email: {str(e)}"
        )


@app.post("/api/v1/notifications/email/template", response_model=APIResponse)
async def create_email_template(
    request: NotificationTemplate,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Create email template."""
    try:
        template_id = await email_service.create_template(request)

        return APIResponse(
            success=True,
            data={"template_id": template_id},
            message="Email template created successfully"
        )

    except Exception as e:
        logger.error("email_template_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create email template: {str(e)}"
        )


# SMS endpoints
@app.post("/api/v1/notifications/sms/send", response_model=NotificationResponse)
async def send_sms(
    request: SMSRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send SMS notification."""
    try:
        result = await sms_service.send_sms(request)

        # Track analytics
        background_tasks.add_task(
            notification_analytics.track_sms_delivery,
            result.notification_id,
            request.recipients,
            result.status
        )

        return result

    except Exception as e:
        logger.error("sms_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send SMS: {str(e)}"
        )


# Push notification endpoints
@app.post("/api/v1/notifications/push/send", response_model=NotificationResponse)
async def send_push_notification(
    request: PushNotificationRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send push notification."""
    try:
        result = await push_service.send_push_notification(request)

        # Track analytics
        background_tasks.add_task(
            notification_analytics.track_push_delivery,
            result.notification_id,
            request.recipients,
            result.status
        )

        return result

    except Exception as e:
        logger.error("push_notification_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send push notification: {str(e)}"
        )


# Webhook endpoints
@app.post("/api/v1/notifications/webhook/send", response_model=NotificationResponse)
async def send_webhook(
    request: WebhookRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Send webhook notification."""
    try:
        result = await webhook_service.send_webhook(request)

        # Track analytics
        background_tasks.add_task(
            notification_analytics.track_webhook_delivery,
            result.notification_id,
            request.url,
            result.status
        )

        return result

    except Exception as e:
        logger.error("webhook_send_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send webhook: {str(e)}"
        )


# Alert endpoints
@app.post("/api/v1/notifications/alerts/create", response_model=APIResponse)
async def create_alert(
    request: AlertRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Create alert with escalation rules."""
    try:
        alert_id = await alert_manager.create_alert(request)

        return APIResponse(
            success=True,
            data={"alert_id": alert_id},
            message="Alert created successfully"
        )

    except Exception as e:
        logger.error("alert_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create alert: {str(e)}"
        )


@app.get("/api/v1/notifications/alerts/{alert_id}", response_model=APIResponse)
async def get_alert(
    alert_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get alert details."""
    try:
        alert = await alert_manager.get_alert(alert_id)

        return APIResponse(
            success=True,
            data={"alert": alert},
            message="Alert retrieved successfully"
        )

    except Exception as e:
        logger.error("alert_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alert: {str(e)}"
        )


# Subscription endpoints
@app.post("/api/v1/notifications/subscriptions", response_model=APIResponse)
async def create_subscription(
    request: SubscriptionRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Create notification subscription."""
    try:
        subscription_id = await subscription_manager.create_subscription(request)

        return APIResponse(
            success=True,
            data={"subscription_id": subscription_id},
            message="Subscription created successfully"
        )

    except Exception as e:
        logger.error("subscription_creation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create subscription: {str(e)}"
        )


@app.get("/api/v1/notifications/subscriptions/{user_id}", response_model=APIResponse)
async def get_user_subscriptions(
    user_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get user notification subscriptions."""
    try:
        subscriptions = await subscription_manager.get_user_subscriptions(user_id)

        return APIResponse(
            success=True,
            data={"subscriptions": subscriptions},
            message="Subscriptions retrieved successfully"
        )

    except Exception as e:
        logger.error("subscriptions_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve subscriptions: {str(e)}"
        )


@app.put("/api/v1/notifications/preferences/{user_id}", response_model=APIResponse)
async def update_notification_preferences(
    user_id: str,
    request: NotificationPreferences,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Update user notification preferences."""
    try:
        await subscription_manager.update_preferences(user_id, request)

        return APIResponse(
            success=True,
            message="Notification preferences updated successfully"
        )

    except Exception as e:
        logger.error("preferences_update_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update preferences: {str(e)}"
        )


# Analytics endpoints
@app.get("/api/v1/notifications/analytics/stats", response_model=NotificationStats)
async def get_notification_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get notification analytics and statistics."""
    try:
        stats = await notification_analytics.get_stats(start_date, end_date)
        return stats

    except Exception as e:
        logger.error("analytics_stats_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


@app.get("/api/v1/notifications/{notification_id}/status", response_model=DeliveryStatus)
async def get_notification_status(
    notification_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get notification delivery status."""
    try:
        status_info = await notification_analytics.get_delivery_status(notification_id)
        return status_info

    except Exception as e:
        logger.error("notification_status_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve notification status: {str(e)}"
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        request_id=getattr(request.state, "request_id", "unknown")
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message=str(exc) if settings.debug else "An unexpected error occurred",
            request_id=getattr(request.state, "request_id", "unknown")
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.service.service_host,
        port=settings.service.service_port,
        workers=1,  # Single worker for development
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )
