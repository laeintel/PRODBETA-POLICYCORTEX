"""
Push notification service for sending push notifications to mobile and web devices.
"""

import asyncio
import json
import ssl
from typing import Dict, Any, Optional, List
from datetime import datetime
import redis.asyncio as redis
import httpx
import structlog
import uuid
from pathlib import Path
import jwt
import time

from shared.config import get_settings
from ..models import (
    PushNotificationRequest,
    NotificationResponse,
    DeliveryStatusEnum,
    PushProviderConfig
)

settings = get_settings()
logger = structlog.get_logger(__name__)


class PushNotificationService:
    """Service for sending push notifications with multiple provider support."""
    
    def __init__(self):
        self.settings = settings
        self.redis_client = None
        self.providers = {}
        self.provider_configs = {}
        
    async def initialize(self) -> None:
        """Initialize the push notification service."""
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
            
            logger.info("push_notification_service_initialized")
            
        except Exception as e:
            logger.error("push_notification_service_initialization_failed", error=str(e))
            raise
    
    async def _load_provider_configs(self) -> None:
        """Load push notification provider configurations."""
        try:
            # Firebase Cloud Messaging (FCM) configuration
            fcm_config = PushProviderConfig(
                provider="fcm",
                api_key=getattr(self.settings, "fcm_server_key", ""),
                settings={
                    "project_id": getattr(self.settings, "fcm_project_id", ""),
                    "api_url": "https://fcm.googleapis.com/fcm/send",
                    "v1_api_url": "https://fcm.googleapis.com/v1/projects/{project_id}/messages:send",
                    "service_account_path": getattr(self.settings, "fcm_service_account_path", "")
                }
            )
            
            # Apple Push Notification Service (APNS) configuration
            apns_config = PushProviderConfig(
                provider="apns",
                api_key=getattr(self.settings, "apns_key_id", ""),
                certificate_path=getattr(self.settings, "apns_certificate_path", ""),
                certificate_password=getattr(self.settings, "apns_certificate_password", ""),
                sandbox=getattr(self.settings, "apns_sandbox", True),
                settings={
                    "team_id": getattr(self.settings, "apns_team_id", ""),
                    "bundle_id": getattr(self.settings, "apns_bundle_id", ""),
                    "auth_key_path": getattr(self.settings, "apns_auth_key_path", ""),
                    "production_url": "https://api.push.apple.com/3/device/",
                    "sandbox_url": "https://api.sandbox.push.apple.com/3/device/"
                }
            )
            
            # OneSignal configuration
            onesignal_config = PushProviderConfig(
                provider="onesignal",
                api_key=getattr(self.settings, "onesignal_rest_api_key", ""),
                app_id=getattr(self.settings, "onesignal_app_id", ""),
                settings={
                    "api_url": "https://onesignal.com/api/v1/notifications"
                }
            )
            
            # Pusher configuration
            pusher_config = PushProviderConfig(
                provider="pusher",
                api_key=getattr(self.settings, "pusher_app_key", ""),
                settings={
                    "app_id": getattr(self.settings, "pusher_app_id", ""),
                    "secret": getattr(self.settings, "pusher_secret", ""),
                    "cluster": getattr(self.settings, "pusher_cluster", "mt1"),
                    "api_url": "https://api.pusherapp.com/apps/{app_id}/events"
                }
            )
            
            # Store configurations
            self.provider_configs = {
                "fcm": fcm_config,
                "apns": apns_config,
                "onesignal": onesignal_config,
                "pusher": pusher_config
            }
            
            logger.info("push_provider_configs_loaded", count=len(self.provider_configs))
            
        except Exception as e:
            logger.error("push_provider_configs_loading_failed", error=str(e))
            raise
    
    async def _initialize_providers(self) -> None:
        """Initialize push notification providers."""
        try:
            # Initialize provider instances
            for provider_name, config in self.provider_configs.items():
                if config.api_key or config.certificate_path:  # Only initialize if credentials are available
                    if provider_name == "fcm":
                        self.providers[provider_name] = FCMProvider(config)
                    elif provider_name == "apns":
                        self.providers[provider_name] = APNSProvider(config)
                    elif provider_name == "onesignal":
                        self.providers[provider_name] = OneSignalProvider(config)
                    elif provider_name == "pusher":
                        self.providers[provider_name] = PusherProvider(config)
                    
                    await self.providers[provider_name].initialize()
            
            logger.info("push_providers_initialized", providers=list(self.providers.keys()))
            
        except Exception as e:
            logger.error("push_providers_initialization_failed", error=str(e))
            raise
    
    async def send_push_notification(self, request: PushNotificationRequest) -> NotificationResponse:
        """Send push notification."""
        try:
            notification_id = request.id or str(uuid.uuid4())
            
            # Group recipients by platform
            platform_recipients = self._group_recipients_by_platform(request.recipients)
            
            # Send notifications for each platform
            delivery_details = []
            delivered_count = 0
            failed_count = 0
            
            for platform, recipients in platform_recipients.items():
                if not recipients:
                    continue
                
                # Select appropriate provider for platform
                provider = self._select_provider_for_platform(platform)
                if not provider:
                    logger.warning(f"No provider available for platform: {platform}")
                    failed_count += len(recipients)
                    continue
                
                # Send notifications
                for recipient in recipients:
                    try:
                        result = await provider.send_push_notification(
                            device_token=recipient.device_token,
                            title=request.content.title,
                            body=request.content.body,
                            data=request.custom_data or {},
                            badge=request.badge,
                            sound=request.sound,
                            icon=request.icon,
                            image=request.image,
                            action_buttons=request.action_buttons,
                            deep_link=request.deep_link
                        )
                        
                        if result.get("success", False):
                            delivered_count += 1
                            delivery_details.append({
                                "recipient": recipient.device_token,
                                "platform": platform,
                                "status": "delivered",
                                "provider": provider.config.provider,
                                "message_id": result.get("message_id"),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            failed_count += 1
                            delivery_details.append({
                                "recipient": recipient.device_token,
                                "platform": platform,
                                "status": "failed",
                                "provider": provider.config.provider,
                                "error": result.get("error"),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        
                        logger.info(
                            "push_notification_sent",
                            notification_id=notification_id,
                            platform=platform,
                            provider=provider.config.provider,
                            success=result.get("success", False)
                        )
                        
                    except Exception as e:
                        failed_count += 1
                        delivery_details.append({
                            "recipient": recipient.device_token,
                            "platform": platform,
                            "status": "failed",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        logger.error(
                            "push_notification_send_failed",
                            notification_id=notification_id,
                            platform=platform,
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
                message=f"Push notification sent to {delivered_count} devices, {failed_count} failed",
                sent_at=datetime.utcnow(),
                delivered_count=delivered_count,
                failed_count=failed_count,
                delivery_details=delivery_details
            )
            
        except Exception as e:
            logger.error("push_notification_service_send_failed", error=str(e))
            
            return NotificationResponse(
                notification_id=notification_id,
                status=DeliveryStatusEnum.FAILED,
                message=f"Push notification sending failed: {str(e)}",
                sent_at=datetime.utcnow(),
                delivered_count=0,
                failed_count=len(request.recipients),
                delivery_details=[]
            )
    
    def _group_recipients_by_platform(self, recipients: List[Any]) -> Dict[str, List[Any]]:
        """Group recipients by platform."""
        platform_groups = {"ios": [], "android": [], "web": []}
        
        for recipient in recipients:
            if not recipient.device_token:
                continue
            
            # Simple platform detection based on device token format
            # In production, this would be more sophisticated
            platform = getattr(recipient, "platform", None)
            if not platform:
                # Detect platform from token format
                if len(recipient.device_token) == 64:  # APNS token
                    platform = "ios"
                elif recipient.device_token.startswith("f") or recipient.device_token.startswith("c"):  # FCM token
                    platform = "android"
                else:
                    platform = "web"
            
            platform_groups[platform].append(recipient)
        
        return platform_groups
    
    def _select_provider_for_platform(self, platform: str) -> Optional[Any]:
        """Select appropriate provider for platform."""
        if platform == "ios" and "apns" in self.providers:
            return self.providers["apns"]
        elif platform in ["android", "web"] and "fcm" in self.providers:
            return self.providers["fcm"]
        elif "onesignal" in self.providers:
            return self.providers["onesignal"]
        
        return None
    
    async def _store_delivery_status(
        self,
        notification_id: str,
        request: PushNotificationRequest,
        delivery_details: List[Dict[str, Any]],
        delivered_count: int,
        failed_count: int
    ) -> None:
        """Store delivery status in cache."""
        try:
            status_data = {
                "notification_id": notification_id,
                "type": "push",
                "sent_at": datetime.utcnow().isoformat(),
                "delivered_count": delivered_count,
                "failed_count": failed_count,
                "total_count": len(request.recipients),
                "delivery_details": delivery_details,
                "request_data": request.dict()
            }
            
            # Store in Redis with TTL (30 days)
            await self.redis_client.set(
                f"push_delivery:{notification_id}",
                json.dumps(status_data),
                ex=86400 * 30
            )
            
        except Exception as e:
            logger.error("push_delivery_status_storage_failed", error=str(e))
    
    async def register_device(self, user_id: str, device_token: str, platform: str) -> None:
        """Register device for push notifications."""
        try:
            device_data = {
                "user_id": user_id,
                "device_token": device_token,
                "platform": platform,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            # Store device registration
            await self.redis_client.set(
                f"push_device:{user_id}:{platform}",
                json.dumps(device_data),
                ex=86400 * 365  # 1 year
            )
            
            # Add to user's device list
            await self.redis_client.sadd(f"user_devices:{user_id}", f"{platform}:{device_token}")
            
            logger.info("push_device_registered", user_id=user_id, platform=platform)
            
        except Exception as e:
            logger.error("push_device_registration_failed", error=str(e))
            raise
    
    async def unregister_device(self, user_id: str, device_token: str, platform: str) -> None:
        """Unregister device from push notifications."""
        try:
            # Remove device registration
            await self.redis_client.delete(f"push_device:{user_id}:{platform}")
            
            # Remove from user's device list
            await self.redis_client.srem(f"user_devices:{user_id}", f"{platform}:{device_token}")
            
            logger.info("push_device_unregistered", user_id=user_id, platform=platform)
            
        except Exception as e:
            logger.error("push_device_unregistration_failed", error=str(e))
            raise
    
    async def get_user_devices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get registered devices for user."""
        try:
            device_keys = await self.redis_client.smembers(f"user_devices:{user_id}")
            devices = []
            
            for device_key in device_keys:
                platform, device_token = device_key.split(":", 1)
                device_data = await self.redis_client.get(f"push_device:{user_id}:{platform}")
                
                if device_data:
                    device_info = json.loads(device_data)
                    devices.append(device_info)
            
            return devices
            
        except Exception as e:
            logger.error("get_user_devices_failed", error=str(e))
            return []
    
    async def health_check(self) -> bool:
        """Check push notification service health."""
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
            logger.error("push_notification_service_health_check_failed", error=str(e))
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            # Cleanup providers
            for provider in self.providers.values():
                await provider.cleanup()
            
            logger.info("push_notification_service_cleanup_completed")
            
        except Exception as e:
            logger.error("push_notification_service_cleanup_failed", error=str(e))


class BasePushProvider:
    """Base class for push notification providers."""
    
    def __init__(self, config: PushProviderConfig):
        self.config = config
        self.http_client = None
    
    async def initialize(self) -> None:
        """Initialize the provider."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send push notification."""
        raise NotImplementedError("Subclasses must implement send_push_notification method")
    
    async def health_check(self) -> bool:
        """Check provider health."""
        return self.http_client is not None and bool(self.config.api_key or self.config.certificate_path)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()


class FCMProvider(BasePushProvider):
    """Firebase Cloud Messaging provider."""
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send push notification via FCM."""
        try:
            if not self.http_client:
                await self.initialize()
            
            # Prepare notification payload
            payload = {
                "to": device_token,
                "notification": {
                    "title": title,
                    "body": body
                },
                "data": data
            }
            
            # Add optional parameters
            if kwargs.get("badge"):
                payload["notification"]["badge"] = kwargs["badge"]
            if kwargs.get("sound"):
                payload["notification"]["sound"] = kwargs["sound"]
            if kwargs.get("icon"):
                payload["notification"]["icon"] = kwargs["icon"]
            if kwargs.get("image"):
                payload["notification"]["image"] = kwargs["image"]
            
            # Send request
            headers = {
                "Authorization": f"key={self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.http_client.post(
                self.config.settings["api_url"],
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", 0) > 0:
                    return {
                        "success": True,
                        "message_id": result.get("results", [{}])[0].get("message_id"),
                        "response": result
                    }
                else:
                    return {
                        "success": False,
                        "error": result.get("results", [{}])[0].get("error", "Unknown error")
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("fcm_push_notification_send_failed", error=str(e))
            return {"success": False, "error": str(e)}


class APNSProvider(BasePushProvider):
    """Apple Push Notification Service provider."""
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send push notification via APNS."""
        try:
            if not self.http_client:
                await self.initialize()
            
            # Prepare notification payload
            payload = {
                "aps": {
                    "alert": {
                        "title": title,
                        "body": body
                    },
                    "sound": kwargs.get("sound", "default")
                }
            }
            
            # Add badge if specified
            if kwargs.get("badge"):
                payload["aps"]["badge"] = kwargs["badge"]
            
            # Add custom data
            if data:
                payload.update(data)
            
            # Prepare headers
            headers = {
                "apns-topic": self.config.settings["bundle_id"],
                "apns-priority": "10",
                "content-type": "application/json"
            }
            
            # Use JWT authentication if available
            if self.config.settings.get("auth_key_path"):
                jwt_token = self._generate_jwt_token()
                headers["authorization"] = f"bearer {jwt_token}"
            
            # Select URL based on sandbox setting
            base_url = (self.config.settings["sandbox_url"] 
                       if self.config.sandbox 
                       else self.config.settings["production_url"])
            url = f"{base_url}{device_token}"
            
            # Send request
            response = await self.http_client.post(
                url,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message_id": response.headers.get("apns-id"),
                    "response": response.headers
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("apns_push_notification_send_failed", error=str(e))
            return {"success": False, "error": str(e)}
    
    def _generate_jwt_token(self) -> str:
        """Generate JWT token for APNS authentication."""
        try:
            # This would require reading the private key file
            # For now, return empty token
            return ""
        except Exception as e:
            logger.error("apns_jwt_generation_failed", error=str(e))
            return ""


class OneSignalProvider(BasePushProvider):
    """OneSignal push notification provider."""
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send push notification via OneSignal."""
        try:
            if not self.http_client:
                await self.initialize()
            
            # Prepare notification payload
            payload = {
                "app_id": self.config.app_id,
                "include_player_ids": [device_token],
                "headings": {"en": title},
                "contents": {"en": body},
                "data": data
            }
            
            # Add optional parameters
            if kwargs.get("badge"):
                payload["ios_badgeType"] = "Increase"
                payload["ios_badgeCount"] = kwargs["badge"]
            
            if kwargs.get("sound"):
                payload["ios_sound"] = kwargs["sound"]
                payload["android_sound"] = kwargs["sound"]
            
            if kwargs.get("icon"):
                payload["chrome_icon"] = kwargs["icon"]
                payload["firefox_icon"] = kwargs["icon"]
            
            if kwargs.get("image"):
                payload["big_picture"] = kwargs["image"]
                payload["ios_attachments"] = {"image": kwargs["image"]}
            
            # Send request
            headers = {
                "Authorization": f"Basic {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            response = await self.http_client.post(
                self.config.settings["api_url"],
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "message_id": result.get("id"),
                    "recipients": result.get("recipients", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error("onesignal_push_notification_send_failed", error=str(e))
            return {"success": False, "error": str(e)}


class PusherProvider(BasePushProvider):
    """Pusher push notification provider."""
    
    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send push notification via Pusher."""
        try:
            # Pusher is typically used for real-time events, not push notifications
            # This is a placeholder implementation
            return {"success": False, "error": "Pusher push notifications not implemented"}
            
        except Exception as e:
            logger.error("pusher_push_notification_send_failed", error=str(e))
            return {"success": False, "error": str(e)}