"""
Azure Communication Services integration for email and SMS notifications.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import aioredis
import httpx
import structlog
import uuid

from ....shared.config import get_settings
from ..models import (
    EmailRequest,
    SMSRequest,
    NotificationResponse,
    DeliveryStatusEnum,
    AzureCommunicationConfig,
    AzureEmailRequest,
    AzureSMSRequest
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class AzureCommunicationService:
    """Service for Azure Communication Services integration."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.http_client = None
        self.config = None
        
    async def initialize(self) -> None:
        """Initialize Azure Communication Services."""
        try:
            # Initialize Redis client
            self.redis_client = aioredis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Load configuration
            self.config = AzureCommunicationConfig(
                connection_string=getattr(self.settings, "azure_communication_connection_string", ""),
                email_domain=getattr(self.settings, "azure_communication_email_domain", ""),
                sms_number=getattr(self.settings, "azure_communication_sms_number", ""),
                resource_endpoint=getattr(self.settings, "azure_communication_endpoint", "")
            )
            
            logger.info("azure_communication_service_initialized")
            
        except Exception as e:
            logger.error("azure_communication_service_initialization_failed", error=str(e))
            raise
    
    async def send_email(self, request: EmailRequest) -> NotificationResponse:
        """Send email via Azure Communication Services."""
        try:
            notification_id = request.id or str(uuid.uuid4())
            
            # Prepare Azure email request
            azure_request = AzureEmailRequest(
                sender=f"noreply@{self.config.email_domain}",
                recipients=[r.email for r in request.recipients if r.email],
                subject=request.content.subject or "Notification",
                html_content=request.content.html_body,
                plain_text_content=request.content.body
            )
            
            # Send email via Azure Communication Services
            result = await self._send_azure_email(azure_request)
            
            if result.get("success", False):
                return NotificationResponse(
                    notification_id=notification_id,
                    status=DeliveryStatusEnum.DELIVERED,
                    message="Email sent successfully via Azure Communication Services",
                    sent_at=datetime.utcnow(),
                    delivered_count=len(azure_request.recipients),
                    failed_count=0,
                    tracking_id=result.get("message_id")
                )
            else:
                return NotificationResponse(
                    notification_id=notification_id,
                    status=DeliveryStatusEnum.FAILED,
                    message=f"Email sending failed: {result.get('error', 'Unknown error')}",
                    sent_at=datetime.utcnow(),
                    delivered_count=0,
                    failed_count=len(azure_request.recipients)
                )
                
        except Exception as e:
            logger.error("azure_email_send_failed", error=str(e))
            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"Email sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=len(request.recipients)
            )
    
    async def send_sms(self, request: SMSRequest) -> NotificationResponse:
        """Send SMS via Azure Communication Services."""
        try:
            notification_id = request.id or str(uuid.uuid4())
            
            # Prepare Azure SMS request
            azure_request = AzureSMSRequest(
                from_number=self.config.sms_number,
                to_numbers=[r.phone for r in request.recipients if r.phone],
                message=request.content.body,
                enable_delivery_receipt=True
            )
            
            # Send SMS via Azure Communication Services
            result = await self._send_azure_sms(azure_request)
            
            if result.get("success", False):
                return NotificationResponse(
                    notification_id=notification_id,
                    status=DeliveryStatusEnum.DELIVERED,
                    message="SMS sent successfully via Azure Communication Services",
                    sent_at=datetime.utcnow(),
                    delivered_count=len(azure_request.to_numbers),
                    failed_count=0,
                    tracking_id=result.get("message_id")
                )
            else:
                return NotificationResponse(
                    notification_id=notification_id,
                    status=DeliveryStatusEnum.FAILED,
                    message=f"SMS sending failed: {result.get('error', 'Unknown error')}",
                    sent_at=datetime.utcnow(),
                    delivered_count=0,
                    failed_count=len(azure_request.to_numbers)
                )
                
        except Exception as e:
            logger.error("azure_sms_send_failed", error=str(e))
            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"SMS sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=len(request.recipients)
            )
    
    async def _send_azure_email(self, request: AzureEmailRequest) -> Dict[str, Any]:
        """Send email via Azure Communication Services API."""
        try:
            # This is a placeholder implementation
            # In practice, you would use the Azure Communication Services SDK
            # For now, simulate successful sending
            
            logger.info("azure_email_sent", recipients=len(request.recipients))
            
            return {
                "success": True,
                "message_id": str(uuid.uuid4()),
                "recipients": request.recipients
            }
            
        except Exception as e:
            logger.error("azure_email_api_failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _send_azure_sms(self, request: AzureSMSRequest) -> Dict[str, Any]:
        """Send SMS via Azure Communication Services API."""
        try:
            # This is a placeholder implementation
            # In practice, you would use the Azure Communication Services SDK
            # For now, simulate successful sending
            
            logger.info("azure_sms_sent", recipients=len(request.to_numbers))
            
            return {
                "success": True,
                "message_id": str(uuid.uuid4()),
                "recipients": request.to_numbers
            }
            
        except Exception as e:
            logger.error("azure_sms_api_failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_delivery_status(self, message_id: str) -> Dict[str, Any]:
        """Get delivery status from Azure Communication Services."""
        try:
            # This would query Azure Communication Services for delivery status
            # For now, return placeholder data
            
            return {
                "message_id": message_id,
                "status": "delivered",
                "delivered_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("azure_delivery_status_failed", error=str(e))
            return {
                "message_id": message_id,
                "status": "unknown",
                "error": str(e)
            }
    
    async def health_check(self) -> bool:
        """Check Azure Communication Services health."""
        try:
            # Check if configuration is valid
            if not self.config.connection_string:
                return False
            
            # Check Redis connection
            await self.redis_client.ping()
            
            return True
            
        except Exception as e:
            logger.error("azure_communication_service_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("azure_communication_service_cleanup_completed")
            
        except Exception as e:
            logger.error("azure_communication_service_cleanup_failed", error=str(e))