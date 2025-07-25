"""
Webhook service for sending HTTP webhook notifications.
"""

import asyncio
import json
import hashlib
import hmac
from typing import Dict, Any, Optional, List
from datetime import datetime
import redis.asyncio as redis
import httpx
import structlog
import uuid
from urllib.parse import urlencode

from shared.config import get_settings
from ..models import (
    WebhookRequest,
    NotificationResponse,
    DeliveryStatusEnum,
    WebhookMethod
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class WebhookService:
    """Service for sending webhook notifications."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.http_client = None
        self.retry_delays = [1, 5, 15, 30, 60]  # Retry delays in seconds
        
    async def initialize(self) -> None:
        """Initialize the webhook service."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                self.settings.database.redis_url,
                password=self.settings.database.redis_password,
                ssl=self.settings.database.redis_ssl,
                decode_responses=True
            )
            
            # Initialize HTTP client with retry configuration
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=100, max_connections=200)
            )
            
            logger.info("webhook_service_initialized")
            
        except Exception as e:
            logger.error("webhook_service_initialization_failed", error=str(e))
            raise
    
    async def send_webhook(self, request: WebhookRequest) -> NotificationResponse:
        """Send webhook notification."""
        try:
            notification_id = request.id or str(uuid.uuid4())
            
            # Validate webhook URL
            if not self._is_valid_url(request.url):
                raise Exception(f"Invalid webhook URL: {request.url}")
            
            # Prepare webhook payload
            payload = self._prepare_webhook_payload(request)
            
            # Send webhook with retry logic
            result = await self._send_webhook_with_retry(
                request.url,
                request.method,
                payload,
                request.headers,
                request.auth_token,
                request.timeout or 30,
                request.verify_ssl
            )
            
            # Store delivery status
            delivery_details = [{
                "url": request.url,
                "method": request.method.value,
                "status": "delivered" if result["success"] else "failed",
                "status_code": result.get("status_code"),
                "response_time": result.get("response_time"),
                "error": result.get("error"),
                "timestamp": datetime.utcnow().isoformat()
            }]
            
            await self._store_delivery_status(
                notification_id,
                request,
                delivery_details,
                1 if result["success"] else 0,
                0 if result["success"] else 1
            )
            
            status = DeliveryStatusEnum.DELIVERED if result["success"] else DeliveryStatusEnum.FAILED
            message = result.get("message", "Webhook sent successfully" if result["success"] else "Webhook failed")
            
            logger.info(
                "webhook_sent",
                notification_id=notification_id,
                url=request.url,
                method=request.method.value,
                success=result["success"],
                status_code=result.get("status_code"),
                response_time=result.get("response_time")
            )
            
            return NotificationResponse(
                notification_id=notification_id,
                status=status,
                message=message,
                sent_at=datetime.utcnow(),
                delivered_count=1 if result["success"] else 0,
                failed_count=0 if result["success"] else 1,
                delivery_details=delivery_details,
                tracking_id=result.get("tracking_id")
            )
            
        except Exception as e:
            logger.error("webhook_service_send_failed", error=str(e))
            
            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"Webhook sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=1,
                delivery_details=[]
            )
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate webhook URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.scheme in ["http", "https"]
        except Exception:
            return False
    
    def _prepare_webhook_payload(self, request: WebhookRequest) -> Dict[str, Any]:
        """Prepare webhook payload."""
        return {
            "id": request.id,
            "type": request.type,
            "timestamp": datetime.utcnow().isoformat(),
            "priority": request.priority,
            "content": request.content.dict(),
            "metadata": request.metadata or {},
            "tags": request.tags or []
        }
    
    async def _send_webhook_with_retry(
        self,
        url: str,
        method: WebhookMethod,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]],
        auth_token: Optional[str],
        timeout: int,
        verify_ssl: bool
    ) -> Dict[str, Any]:
        """Send webhook with retry logic."""
        start_time = datetime.utcnow()
        
        for attempt in range(len(self.retry_delays) + 1):
            try:
                # Prepare headers
                request_headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "PolicyCortex-Webhook/1.0"
                }
                
                if headers:
                    request_headers.update(headers)
                
                if auth_token:
                    request_headers["Authorization"] = f"Bearer {auth_token}"
                
                # Add webhook signature
                signature = self._generate_webhook_signature(payload, auth_token or "")
                if signature:
                    request_headers["X-Webhook-Signature"] = signature
                
                # Send request
                response = await self.http_client.request(
                    method=method.value,
                    url=url,
                    json=payload,
                    headers=request_headers,
                    timeout=timeout,
                    verify=verify_ssl
                )
                
                response_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Check if request was successful
                if 200 <= response.status_code < 300:
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "response_time": response_time,
                        "response_body": response.text,
                        "tracking_id": response.headers.get("X-Tracking-ID")
                    }
                else:
                    # If not successful but not a server error, don't retry
                    if response.status_code < 500:
                        return {
                            "success": False,
                            "status_code": response.status_code,
                            "response_time": response_time,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "message": "Webhook request failed with client error"
                        }
                    
                    # Server error, will retry
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except httpx.TimeoutException:
                last_error = "Request timeout"
            except httpx.ConnectError:
                last_error = "Connection error"
            except Exception as e:
                last_error = str(e)
            
            # Wait before retry (except for last attempt)
            if attempt < len(self.retry_delays):
                await asyncio.sleep(self.retry_delays[attempt])
                logger.warning(
                    "webhook_retry_attempt",
                    url=url,
                    attempt=attempt + 1,
                    error=last_error
                )
        
        # All retries failed
        response_time = (datetime.utcnow() - start_time).total_seconds()
        return {
            "success": False,
            "response_time": response_time,
            "error": last_error,
            "message": f"Webhook failed after {len(self.retry_delays) + 1} attempts"
        }
    
    def _generate_webhook_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate webhook signature for security."""
        try:
            if not secret:
                return ""
            
            payload_str = json.dumps(payload, sort_keys=True)
            signature = hmac.new(
                secret.encode('utf-8'),
                payload_str.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return f"sha256={signature}"
            
        except Exception as e:
            logger.error("webhook_signature_generation_failed", error=str(e))
            return ""
    
    async def _store_delivery_status(
        self,
        notification_id: str,
        request: WebhookRequest,
        delivery_details: List[Dict[str, Any]],
        delivered_count: int,
        failed_count: int
    ) -> None:
        """Store delivery status in cache."""
        try:
            status_data = {
                "notification_id": notification_id,
                "type": "webhook",
                "sent_at": datetime.utcnow().isoformat(),
                "delivered_count": delivered_count,
                "failed_count": failed_count,
                "total_count": 1,
                "delivery_details": delivery_details,
                "request_data": request.dict()
            }
            
            # Store in Redis with TTL (30 days)
            await self.redis_client.set(
                f"webhook_delivery:{notification_id}",
                json.dumps(status_data),
                ex=86400 * 30
            )
            
        except Exception as e:
            logger.error("webhook_delivery_status_storage_failed", error=str(e))
    
    async def validate_webhook_endpoint(self, url: str, method: WebhookMethod = WebhookMethod.POST) -> Dict[str, Any]:
        """Validate webhook endpoint."""
        try:
            # Send test payload
            test_payload = {
                "event": "test",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"message": "This is a test webhook"}
            }
            
            start_time = datetime.utcnow()
            
            response = await self.http_client.request(
                method=method.value,
                url=url,
                json=test_payload,
                timeout=10
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "valid": 200 <= response.status_code < 300,
                "status_code": response.status_code,
                "response_time": response_time,
                "response_body": response.text[:1000],  # Limit response body
                "headers": dict(response.headers)
            }
            
        except Exception as e:
            logger.error("webhook_endpoint_validation_failed", error=str(e), url=url)
            return {
                "valid": False,
                "error": str(e),
                "response_time": None
            }
    
    async def get_webhook_history(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook delivery history."""
        try:
            delivery_data = await self.redis_client.get(f"webhook_delivery:{notification_id}")
            
            if delivery_data:
                return json.loads(delivery_data)
            
            return None
            
        except Exception as e:
            logger.error("webhook_history_retrieval_failed", error=str(e))
            return None
    
    async def get_webhook_stats(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get webhook statistics."""
        try:
            # Get all webhook delivery records
            webhook_keys = await self.redis_client.keys("webhook_delivery:*")
            
            stats = {
                "total_webhooks": 0,
                "successful_webhooks": 0,
                "failed_webhooks": 0,
                "avg_response_time": 0.0,
                "status_codes": {},
                "failure_reasons": {}
            }
            
            total_response_time = 0.0
            
            for key in webhook_keys:
                try:
                    delivery_data = await self.redis_client.get(key)
                    if delivery_data:
                        delivery = json.loads(delivery_data)
                        sent_at = datetime.fromisoformat(delivery["sent_at"])
                        
                        # Filter by date range if specified
                        if start_date and sent_at < start_date:
                            continue
                        if end_date and sent_at > end_date:
                            continue
                        
                        stats["total_webhooks"] += 1
                        
                        if delivery["delivered_count"] > 0:
                            stats["successful_webhooks"] += 1
                        else:
                            stats["failed_webhooks"] += 1
                        
                        # Process delivery details
                        for detail in delivery.get("delivery_details", []):
                            if detail.get("status_code"):
                                status_code = str(detail["status_code"])
                                stats["status_codes"][status_code] = stats["status_codes"].get(status_code, 0) + 1
                            
                            if detail.get("response_time"):
                                total_response_time += detail["response_time"]
                            
                            if detail.get("error"):
                                error = detail["error"]
                                stats["failure_reasons"][error] = stats["failure_reasons"].get(error, 0) + 1
                                
                except Exception as e:
                    logger.error("webhook_stats_processing_failed", error=str(e), key=key)
                    continue
            
            # Calculate averages
            if stats["total_webhooks"] > 0:
                stats["success_rate"] = stats["successful_webhooks"] / stats["total_webhooks"]
                stats["avg_response_time"] = total_response_time / stats["total_webhooks"]
            else:
                stats["success_rate"] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error("webhook_stats_retrieval_failed", error=str(e))
            return {}
    
    async def create_webhook_subscription(
        self,
        url: str,
        events: List[str],
        headers: Optional[Dict[str, str]] = None,
        auth_token: Optional[str] = None
    ) -> str:
        """Create webhook subscription."""
        try:
            subscription_id = str(uuid.uuid4())
            
            subscription_data = {
                "id": subscription_id,
                "url": url,
                "events": events,
                "headers": headers or {},
                "auth_token": auth_token,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
            
            # Store subscription
            await self.redis_client.set(
                f"webhook_subscription:{subscription_id}",
                json.dumps(subscription_data),
                ex=86400 * 365  # 1 year
            )
            
            # Add to active subscriptions list
            await self.redis_client.sadd("active_webhook_subscriptions", subscription_id)
            
            logger.info("webhook_subscription_created", subscription_id=subscription_id, url=url)
            
            return subscription_id
            
        except Exception as e:
            logger.error("webhook_subscription_creation_failed", error=str(e))
            raise
    
    async def delete_webhook_subscription(self, subscription_id: str) -> None:
        """Delete webhook subscription."""
        try:
            # Remove subscription
            await self.redis_client.delete(f"webhook_subscription:{subscription_id}")
            
            # Remove from active subscriptions list
            await self.redis_client.srem("active_webhook_subscriptions", subscription_id)
            
            logger.info("webhook_subscription_deleted", subscription_id=subscription_id)
            
        except Exception as e:
            logger.error("webhook_subscription_deletion_failed", error=str(e))
            raise
    
    async def get_webhook_subscriptions(self) -> List[Dict[str, Any]]:
        """Get all webhook subscriptions."""
        try:
            subscription_ids = await self.redis_client.smembers("active_webhook_subscriptions")
            subscriptions = []
            
            for subscription_id in subscription_ids:
                subscription_data = await self.redis_client.get(f"webhook_subscription:{subscription_id}")
                if subscription_data:
                    subscription = json.loads(subscription_data)
                    # Remove sensitive data
                    subscription.pop("auth_token", None)
                    subscriptions.append(subscription)
            
            return subscriptions
            
        except Exception as e:
            logger.error("webhook_subscriptions_retrieval_failed", error=str(e))
            return []
    
    async def trigger_webhook_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger webhook event to all subscribed endpoints."""
        try:
            # Get all active subscriptions
            subscription_ids = await self.redis_client.smembers("active_webhook_subscriptions")
            
            for subscription_id in subscription_ids:
                try:
                    subscription_data = await self.redis_client.get(f"webhook_subscription:{subscription_id}")
                    if not subscription_data:
                        continue
                    
                    subscription = json.loads(subscription_data)
                    
                    # Check if subscription is interested in this event
                    if event_type not in subscription.get("events", []):
                        continue
                    
                    # Prepare webhook payload
                    webhook_payload = {
                        "event": event_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": event_data,
                        "subscription_id": subscription_id
                    }
                    
                    # Send webhook (fire and forget)
                    asyncio.create_task(self._send_subscription_webhook(subscription, webhook_payload))
                    
                except Exception as e:
                    logger.error("webhook_event_trigger_failed", error=str(e), subscription_id=subscription_id)
                    continue
            
        except Exception as e:
            logger.error("webhook_event_triggering_failed", error=str(e))
    
    async def _send_subscription_webhook(self, subscription: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Send webhook for subscription."""
        try:
            headers = subscription.get("headers", {})
            auth_token = subscription.get("auth_token")
            
            await self._send_webhook_with_retry(
                subscription["url"],
                WebhookMethod.POST,
                payload,
                headers,
                auth_token,
                30,
                True
            )
            
        except Exception as e:
            logger.error("subscription_webhook_send_failed", error=str(e), subscription_id=subscription["id"])
    
    async def health_check(self) -> bool:
        """Check webhook service health."""
        try:
            # Check Redis connection
            await self.redis_client.ping()
            
            # Check HTTP client
            return self.http_client is not None
            
        except Exception as e:
            logger.error("webhook_service_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("webhook_service_cleanup_completed")
            
        except Exception as e:
            logger.error("webhook_service_cleanup_failed", error=str(e))