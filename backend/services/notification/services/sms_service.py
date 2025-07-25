"""
SMS service for sending SMS notifications with multiple provider support.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import redis.asyncio as redis
import httpx
import structlog
import uuid
from urllib.parse import quote_plus

from shared.config import get_settings
from ..models import (
    SMSRequest,
    NotificationResponse,
    DeliveryStatusEnum,
    SMSProviderConfig
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class SMSService:
    """Service for sending SMS notifications with multiple provider support."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.providers = {}
        self.provider_configs = {}
        
    async def initialize(self) -> None:
        """Initialize the SMS service."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            # Load provider configurations
            await self._load_provider_configs()
            
            # Initialize providers
            await self._initialize_providers()
            
            logger.info("sms_service_initialized")
            
        except Exception as e:
            logger.error("sms_service_initialization_failed", error=str(e))
            raise
    
    async def _load_provider_configs(self) -> None:
        """Load SMS provider configurations."""
        try:
            # Twilio configuration
            twilio_config = SMSProviderConfig(
                provider="twilio",
                api_key=getattr(self.settings, "twilio_account_sid", ""),
                api_secret=getattr(self.settings, "twilio_auth_token", ""),
                from_number=getattr(self.settings, "twilio_from_number", ""),
                settings={
                    "api_url": "https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
                }
            )
            
            # AWS SNS configuration
            aws_sns_config = SMSProviderConfig(
                provider="aws_sns",
                api_key=getattr(self.settings, "aws_access_key_id", ""),
                api_secret=getattr(self.settings, "aws_secret_access_key", ""),
                settings={
                    "region": getattr(self.settings, "aws_region", "us-east-1"),
                    "endpoint_url": f"https://sns.{getattr(self.settings, 'aws_region', 'us-east-1')}.amazonaws.com"
                }
            )
            
            # MessageBird configuration
            messagebird_config = SMSProviderConfig(
                provider="messagebird",
                api_key=getattr(self.settings, "messagebird_api_key", ""),
                from_number=getattr(self.settings, "messagebird_from_number", ""),
                settings={
                    "api_url": "https://rest.messagebird.com/messages"
                }
            )
            
            # Nexmo/Vonage configuration
            nexmo_config = SMSProviderConfig(
                provider="nexmo",
                api_key=getattr(self.settings, "nexmo_api_key", ""),
                api_secret=getattr(self.settings, "nexmo_api_secret", ""),
                from_number=getattr(self.settings, "nexmo_from_number", ""),
                settings={
                    "api_url": "https://rest.nexmo.com/sms/json"
                }
            )
            
            # Store configurations
            self.provider_configs = {
                "twilio": twilio_config,
                "aws_sns": aws_sns_config,
                "messagebird": messagebird_config,
                "nexmo": nexmo_config
            }
            
            logger.info("sms_provider_configs_loaded", count=len(self.provider_configs))
            
        except Exception as e:
            logger.error("sms_provider_configs_loading_failed", error=str(e))
            raise
    
    async def _initialize_providers(self) -> None:
        """Initialize SMS providers."""
        try:
            # Initialize provider instances
            for provider_name, config in self.provider_configs.items():
                if config.api_key:  # Only initialize if API key is available
                    if provider_name == "twilio":
                        self.providers[provider_name] = TwilioProvider(config)
                    elif provider_name == "aws_sns":
                        self.providers[provider_name] = AWSSNSProvider(config)
                    elif provider_name == "messagebird":
                        self.providers[provider_name] = MessageBirdProvider(config)
                    elif provider_name == "nexmo":
                        self.providers[provider_name] = NexmoProvider(config)
            
            logger.info("sms_providers_initialized", providers=list(self.providers.keys()))
            
        except Exception as e:
            logger.error("sms_providers_initialization_failed", error=str(e))
            raise
    
    async def send_sms(self, request: SMSRequest) -> NotificationResponse:
        """Send SMS notification."""
        try:
            notification_id = request.id or str(uuid.uuid4())
            
            # Select provider
            provider_name = request.provider or await self._select_provider()
            provider = self.providers.get(provider_name)
            
            if not provider:
                raise Exception(f"SMS provider '{provider_name}' not available")
            
            # Send to all recipients
            delivery_details = []
            delivered_count = 0
            failed_count = 0
            
            for recipient in request.recipients:
                try:
                    if not recipient.phone:
                        logger.warning("recipient_missing_phone", recipient_id=recipient.id)
                        failed_count += 1
                        continue
                    
                    # Send SMS
                    result = await provider.send_sms(
                        to_number=recipient.phone,
                        message=request.content.body,
                        from_number=request.from_number
                    )
                    
                    if result.get("success", False):
                        delivered_count += 1
                        delivery_details.append({
                            "recipient": recipient.phone,
                            "status": "delivered",
                            "provider": provider_name,
                            "message_id": result.get("message_id"),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    else:
                        failed_count += 1
                        delivery_details.append({
                            "recipient": recipient.phone,
                            "status": "failed",
                            "provider": provider_name,
                            "error": result.get("error"),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    logger.info(
                        "sms_sent",
                        notification_id=notification_id,
                        recipient=recipient.phone,
                        provider=provider_name,
                        success=result.get("success", False)
                    )
                    
                except Exception as e:
                    failed_count += 1
                    delivery_details.append({
                        "recipient": recipient.phone,
                        "status": "failed",
                        "provider": provider_name,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    logger.error(
                        "sms_send_failed",
                        notification_id=notification_id,
                        recipient=recipient.phone,
                        provider=provider_name,
                        error=str(e)
                    )
            
            # Store delivery status
            await self._store_delivery_status(
                notification_id,
                request,
                delivery_details,
                delivered_count,
                failed_count
            )
            
            status = DeliveryStatusEnum.DELIVERED if delivered_count > 0 else DeliveryStatusEnum.FAILED
            
            return NotificationResponse(
                notification_id=notification_id,
                status=status,
                message=f"SMS sent to {delivered_count} recipients, {failed_count} failed",
                sent_at=datetime.utcnow(),
                delivered_count=delivered_count,
                failed_count=failed_count,
                delivery_details=delivery_details
            )
            
        except Exception as e:
            logger.error("sms_service_send_failed", error=str(e))
            
            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"SMS sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=len(request.recipients),
                delivery_details=[]
            )
    
    async def _select_provider(self) -> str:
        """Select the best available SMS provider."""
        try:
            # Get provider health scores
            provider_scores = {}
            
            for provider_name in self.providers.keys():
                score = await self._get_provider_health_score(provider_name)
                provider_scores[provider_name] = score
            
            # Select provider with highest score
            if provider_scores:
                best_provider = max(provider_scores, key=provider_scores.get)
                return best_provider
            
            # Fallback to first available provider
            return next(iter(self.providers.keys()))
            
        except Exception as e:
            logger.error("sms_provider_selection_failed", error=str(e))
            return next(iter(self.providers.keys())) if self.providers else "twilio"
    
    async def _get_provider_health_score(self, provider_name: str) -> float:
        """Get provider health score based on recent performance."""
        try:
            # Get recent delivery stats from Redis
            stats_key = f"sms_provider_stats:{provider_name}"
            stats_data = await self.redis_client.get(stats_key)
            
            if not stats_data:
                return 0.5  # Default score for new providers
            
            stats = json.loads(stats_data)
            
            # Calculate score based on delivery rate and response time
            delivery_rate = stats.get("delivery_rate", 0.5)
            avg_response_time = stats.get("avg_response_time", 5.0)
            
            # Score formula: delivery_rate * (1 - normalized_response_time)
            normalized_response_time = min(avg_response_time / 10.0, 1.0)
            score = delivery_rate * (1 - normalized_response_time)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error("provider_health_score_calculation_failed", error=str(e))
            return 0.5
    
    async def _store_delivery_status(
        self,
        notification_id: str,
        request: SMSRequest,
        delivery_details: List[Dict[str, Any]],
        delivered_count: int,
        failed_count: int
    ) -> None:
        """Store delivery status in cache."""
        try:
            status_data = {
                "notification_id": notification_id,
                "type": "sms",
                "sent_at": datetime.utcnow().isoformat(),
                "delivered_count": delivered_count,
                "failed_count": failed_count,
                "total_count": len(request.recipients),
                "delivery_details": delivery_details,
                "request_data": request.dict()
            }
            
            # Store in Redis with TTL (30 days)
            await self.redis_client.set(
                f"sms_delivery:{notification_id}",
                json.dumps(status_data),
                ex=86400 * 30
            )
            
        except Exception as e:
            logger.error("sms_delivery_status_storage_failed", error=str(e))
    
    async def update_provider_stats(
        self,
        provider_name: str,
        success: bool,
        response_time: float
    ) -> None:
        """Update provider statistics."""
        try:
            stats_key = f"sms_provider_stats:{provider_name}"
            
            # Get current stats
            stats_data = await self.redis_client.get(stats_key)
            if stats_data:
                stats = json.loads(stats_data)
            else:
                stats = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "total_response_time": 0.0,
                    "delivery_rate": 0.0,
                    "avg_response_time": 0.0
                }
            
            # Update stats
            stats["total_requests"] += 1
            if success:
                stats["successful_requests"] += 1
            stats["total_response_time"] += response_time
            
            # Recalculate rates
            stats["delivery_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["avg_response_time"] = stats["total_response_time"] / stats["total_requests"]
            
            # Store updated stats
            await self.redis_client.set(
                stats_key,
                json.dumps(stats),
                ex=86400 * 7  # Keep for 7 days
            )
            
        except Exception as e:
            logger.error("sms_provider_stats_update_failed", error=str(e))
    
    async def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all SMS providers."""
        try:
            status = {}
            
            for provider_name in self.providers.keys():
                provider_status = await self.providers[provider_name].health_check()
                health_score = await self._get_provider_health_score(provider_name)
                
                status[provider_name] = {
                    "healthy": provider_status,
                    "health_score": health_score,
                    "config": {
                        "from_number": self.provider_configs[provider_name].from_number,
                        "has_api_key": bool(self.provider_configs[provider_name].api_key)
                    }
                }
            
            return status
            
        except Exception as e:
            logger.error("sms_provider_status_retrieval_failed", error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check SMS service health."""
        try:
            # Check Redis connection
            await self.redis_client.ping()
            
            # Check if at least one provider is available
            if not self.providers:
                return False
            
            # Check provider health
            healthy_providers = 0
            for provider in self.providers.values():
                if await provider.health_check():
                    healthy_providers += 1
            
            return healthy_providers > 0
            
        except Exception as e:
            logger.error("sms_service_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            # Cleanup providers
            for provider in self.providers.values():
                await provider.cleanup()
            
            logger.info("sms_service_cleanup_completed")
            
        except Exception as e:
            logger.error("sms_service_cleanup_failed", error=str(e))


class BaseSMSProvider:
    """Base class for SMS providers."""
    
    def __init__(self, config: SMSProviderConfig):
        self.config = config
        self.http_client = None
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def send_sms(self, to_number: str, message: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send SMS message."""
        raise NotImplementedError("Subclasses must implement send_sms method")
    
    async def health_check(self) -> bool:
        """Check provider health."""
        return self.http_client is not None and bool(self.config.api_key)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()


class TwilioProvider(BaseSMSProvider):
    """Twilio SMS provider."""
    
    async def send_sms(self, to_number: str, message: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send SMS via Twilio."""
        try:
            if not self.http_client:
                await self.initialize()
            
            from_number = from_number or self.config.from_number
            if not from_number:
                raise Exception("From number not configured")
            
            # Prepare request
            url = self.config.settings["api_url"].format(account_sid=self.config.api_key)
            auth = (self.config.api_key, self.config.api_secret)
            
            data = {
                "From": from_number,
                "To": to_number,
                "Body": message
            }
            
            # Send request
            response = await self.http_client.post(url, auth=auth, data=data)
            
            if response.status_code == 201:
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("sid"),
                    "status": result.get("status")
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("twilio_sms_send_failed", error=str(e))
            return {"success": False, "error": str(e)}


class AWSSNSProvider(BaseSMSProvider):
    """AWS SNS SMS provider."""
    
    async def send_sms(self, to_number: str, message: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send SMS via AWS SNS."""
        try:
            # This would require boto3 implementation
            # For now, return not implemented
            return {"success": False, "error": "AWS SNS provider not implemented"}
            
        except Exception as e:
            logger.error("aws_sns_sms_send_failed", error=str(e))
            return {"success": False, "error": str(e)}


class MessageBirdProvider(BaseSMSProvider):
    """MessageBird SMS provider."""
    
    async def send_sms(self, to_number: str, message: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send SMS via MessageBird."""
        try:
            if not self.http_client:
                await self.initialize()
            
            from_number = from_number or self.config.from_number
            if not from_number:
                raise Exception("From number not configured")
            
            # Prepare request
            url = self.config.settings["api_url"]
            headers = {
                "Authorization": f"AccessKey {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "originator": from_number,
                "recipients": [to_number],
                "body": message
            }
            
            # Send request
            response = await self.http_client.post(url, headers=headers, json=data)
            
            if response.status_code == 201:
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("id"),
                    "status": "sent"
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("messagebird_sms_send_failed", error=str(e))
            return {"success": False, "error": str(e)}


class NexmoProvider(BaseSMSProvider):
    """Nexmo/Vonage SMS provider."""
    
    async def send_sms(self, to_number: str, message: str, from_number: Optional[str] = None) -> Dict[str, Any]:
        """Send SMS via Nexmo."""
        try:
            if not self.http_client:
                await self.initialize()
            
            from_number = from_number or self.config.from_number
            if not from_number:
                raise Exception("From number not configured")
            
            # Prepare request
            url = self.config.settings["api_url"]
            
            data = {
                "api_key": self.config.api_key,
                "api_secret": self.config.api_secret,
                "from": from_number,
                "to": to_number,
                "text": message
            }
            
            # Send request
            response = await self.http_client.post(url, data=data)
            
            if response.status_code == 200:
                result = response.json()
                messages = result.get("messages", [])
                if messages and messages[0].get("status") == "0":
                    return {
                        "success": True,
                        "message_id": messages[0].get("message-id"),
                        "status": "sent"
                    }
                else:
                    return {
                        "success": False,
                        "error": messages[0].get("error-text", "Unknown error") if messages else "No messages"
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("nexmo_sms_send_failed", error=str(e))
            return {"success": False, "error": str(e)}