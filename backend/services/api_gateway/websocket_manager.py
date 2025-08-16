"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
WebSocket manager for real-time updates and bidirectional communication
"""

import asyncio
import json
import uuid
from typing import Dict, Set, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Subscriptions
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    
    # Real-time updates
    RESOURCE_UPDATE = "resource.update"
    POLICY_UPDATE = "policy.update"
    COMPLIANCE_UPDATE = "compliance.update"
    COST_UPDATE = "cost.update"
    SECURITY_ALERT = "security.alert"
    
    # Notifications
    NOTIFICATION = "notification"
    APPROVAL_REQUEST = "approval.request"
    APPROVAL_UPDATE = "approval.update"
    
    # Dashboard updates
    DASHBOARD_UPDATE = "dashboard.update"
    METRICS_UPDATE = "metrics.update"
    
    # Chat
    CHAT_MESSAGE = "chat.message"
    CHAT_RESPONSE = "chat.response"
    
    # System
    SYSTEM_MESSAGE = "system.message"
    ERROR = "error"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    id: str
    type: str
    channel: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON"""
        return json.dumps({
            "id": self.id,
            "type": self.type,
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON"""
        data = json.loads(json_str)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow()
        return cls(**data)


class ConnectionInfo:
    """WebSocket connection information"""
    
    def __init__(
        self,
        connection_id: str,
        websocket: WebSocket,
        user_id: str,
        tenant_id: str,
        roles: List[str] = None
    ):
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.roles = roles or []
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.utcnow()
        self.last_ping = datetime.utcnow()
        self.metadata: Dict[str, Any] = {}
        
    def is_alive(self) -> bool:
        """Check if connection is still alive"""
        return (datetime.utcnow() - self.last_ping).seconds < 60
    
    def can_access_channel(self, channel: str) -> bool:
        """Check if user can access channel"""
        # Tenant-specific channels
        if channel.startswith(f"tenant:{self.tenant_id}"):
            return True
            
        # User-specific channels
        if channel.startswith(f"user:{self.user_id}"):
            return True
            
        # Role-based channels
        for role in self.roles:
            if channel.startswith(f"role:{role}"):
                return True
                
        # Public channels
        if channel.startswith("public:"):
            return True
            
        # Admin channels
        if channel.startswith("admin:") and "admin" in self.roles:
            return True
            
        return False


class WebSocketManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.channels: Dict[str, Set[str]] = {}  # channel -> connection_ids
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.handlers: Dict[str, List[callable]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start WebSocket manager background tasks"""
        if self._running:
            return
            
        self._running = True
        
        # Start message processor
        self._tasks.append(asyncio.create_task(self._process_messages()))
        
        # Start heartbeat monitor
        self._tasks.append(asyncio.create_task(self._monitor_connections()))
        
        logger.info("WebSocket manager started")
        
    async def stop(self):
        """Stop WebSocket manager"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close all connections
        for conn_info in list(self.connections.values()):
            await self.disconnect(conn_info.connection_id)
            
        logger.info("WebSocket manager stopped")
        
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        tenant_id: str,
        roles: List[str] = None
    ) -> str:
        """Handle new WebSocket connection"""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        conn_info = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            user_id=user_id,
            tenant_id=tenant_id,
            roles=roles or []
        )
        
        self.connections[connection_id] = conn_info
        
        # Auto-subscribe to user and tenant channels
        await self.subscribe(connection_id, f"user:{user_id}")
        await self.subscribe(connection_id, f"tenant:{tenant_id}")
        
        # Send connection confirmation
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.CONNECT.value,
                channel=None,
                data={
                    "connection_id": connection_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "message": "Connected successfully"
                },
                timestamp=datetime.utcnow()
            )
        )
        
        logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        
        return connection_id
        
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id not in self.connections:
            return
            
        conn_info = self.connections[connection_id]
        
        # Unsubscribe from all channels
        for channel in list(conn_info.subscriptions):
            await self.unsubscribe(connection_id, channel)
            
        # Close WebSocket if still open
        if conn_info.websocket.application_state == WebSocketState.CONNECTED:
            await conn_info.websocket.close()
            
        # Remove connection
        del self.connections[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}")
        
    async def subscribe(self, connection_id: str, channel: str) -> bool:
        """Subscribe connection to channel"""
        if connection_id not in self.connections:
            return False
            
        conn_info = self.connections[connection_id]
        
        # Check access permissions
        if not conn_info.can_access_channel(channel):
            logger.warning(f"Access denied to channel {channel} for connection {connection_id}")
            return False
            
        # Add to subscriptions
        conn_info.subscriptions.add(channel)
        
        # Add to channel mapping
        if channel not in self.channels:
            self.channels[channel] = set()
        self.channels[channel].add(connection_id)
        
        logger.debug(f"Connection {connection_id} subscribed to {channel}")
        
        return True
        
    async def unsubscribe(self, connection_id: str, channel: str):
        """Unsubscribe connection from channel"""
        if connection_id not in self.connections:
            return
            
        conn_info = self.connections[connection_id]
        
        # Remove from subscriptions
        conn_info.subscriptions.discard(channel)
        
        # Remove from channel mapping
        if channel in self.channels:
            self.channels[channel].discard(connection_id)
            
            # Clean up empty channels
            if not self.channels[channel]:
                del self.channels[channel]
                
        logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
        
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.connections:
            return False
            
        conn_info = self.connections[connection_id]
        
        try:
            if conn_info.websocket.application_state == WebSocketState.CONNECTED:
                await conn_info.websocket.send_text(message.to_json())
                return True
        except Exception as e:
            logger.error(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect(connection_id)
            
        return False
        
    async def send_to_channel(self, channel: str, message: WebSocketMessage):
        """Send message to all connections in channel"""
        message.channel = channel
        
        if channel not in self.channels:
            return
            
        # Send to all subscribed connections
        disconnected = []
        
        for connection_id in self.channels[channel]:
            if not await self.send_to_connection(connection_id, message):
                disconnected.append(connection_id)
                
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
            
    async def send_to_user(self, user_id: str, message: WebSocketMessage):
        """Send message to specific user"""
        await self.send_to_channel(f"user:{user_id}", message)
        
    async def send_to_tenant(self, tenant_id: str, message: WebSocketMessage):
        """Send message to all users in tenant"""
        await self.send_to_channel(f"tenant:{tenant_id}", message)
        
    async def broadcast(self, message: WebSocketMessage):
        """Broadcast message to all connections"""
        disconnected = []
        
        for connection_id in list(self.connections.keys()):
            if not await self.send_to_connection(connection_id, message):
                disconnected.append(connection_id)
                
        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)
            
    async def handle_message(self, connection_id: str, raw_message: str):
        """Handle incoming WebSocket message"""
        try:
            message = WebSocketMessage.from_json(raw_message)
            
            # Update last ping time
            if connection_id in self.connections:
                self.connections[connection_id].last_ping = datetime.utcnow()
                
            # Handle different message types
            if message.type == MessageType.PING.value:
                await self._handle_ping(connection_id, message)
                
            elif message.type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(connection_id, message)
                
            elif message.type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(connection_id, message)
                
            else:
                # Queue message for processing
                await self.message_queue.put((connection_id, message))
                
                # Call registered handlers
                if message.type in self.handlers:
                    for handler in self.handlers[message.type]:
                        await handler(connection_id, message)
                        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from connection {connection_id}: {e}")
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.ERROR.value,
                    channel=None,
                    data={"error": "Invalid JSON format"},
                    timestamp=datetime.utcnow()
                )
            )
            
        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            
    async def _handle_ping(self, connection_id: str, message: WebSocketMessage):
        """Handle ping message"""
        await self.send_to_connection(
            connection_id,
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.PONG.value,
                channel=None,
                data={"timestamp": datetime.utcnow().isoformat()},
                timestamp=datetime.utcnow(),
                correlation_id=message.id
            )
        )
        
    async def _handle_subscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle subscribe request"""
        channel = message.data.get("channel")
        if channel:
            success = await self.subscribe(connection_id, channel)
            
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.SUBSCRIBE.value,
                    channel=channel,
                    data={
                        "channel": channel,
                        "subscribed": success,
                        "message": "Subscribed successfully" if success else "Access denied"
                    },
                    timestamp=datetime.utcnow(),
                    correlation_id=message.id
                )
            )
            
    async def _handle_unsubscribe(self, connection_id: str, message: WebSocketMessage):
        """Handle unsubscribe request"""
        channel = message.data.get("channel")
        if channel:
            await self.unsubscribe(connection_id, channel)
            
            await self.send_to_connection(
                connection_id,
                WebSocketMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.UNSUBSCRIBE.value,
                    channel=channel,
                    data={
                        "channel": channel,
                        "unsubscribed": True
                    },
                    timestamp=datetime.utcnow(),
                    correlation_id=message.id
                )
            )
            
    async def _process_messages(self):
        """Process queued messages"""
        while self._running:
            try:
                # Get message from queue with timeout
                connection_id, message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process message (implement your business logic here)
                logger.debug(f"Processing message from {connection_id}: {message.type}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    async def _monitor_connections(self):
        """Monitor connection health and clean up stale connections"""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                disconnected = []
                
                for connection_id, conn_info in list(self.connections.items()):
                    # Check if connection is alive
                    if not conn_info.is_alive():
                        logger.warning(f"Connection {connection_id} is stale, disconnecting")
                        disconnected.append(connection_id)
                        continue
                        
                    # Send ping to keep connection alive
                    await self.send_to_connection(
                        connection_id,
                        WebSocketMessage(
                            id=str(uuid.uuid4()),
                            type=MessageType.PING.value,
                            channel=None,
                            data={"timestamp": datetime.utcnow().isoformat()},
                            timestamp=datetime.utcnow()
                        )
                    )
                    
                # Clean up stale connections
                for connection_id in disconnected:
                    await self.disconnect(connection_id)
                    
            except Exception as e:
                logger.error(f"Error monitoring connections: {e}")
                
    def register_handler(self, message_type: str, handler: callable):
        """Register handler for message type"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
            
        self.handlers[message_type].append(handler)
        
    def get_connection_info(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection information"""
        return self.connections.get(connection_id)
        
    def get_channel_connections(self, channel: str) -> List[str]:
        """Get all connections in a channel"""
        return list(self.channels.get(channel, set()))
        
    def get_user_connections(self, user_id: str) -> List[str]:
        """Get all connections for a user"""
        return [
            conn_id
            for conn_id, conn_info in self.connections.items()
            if conn_info.user_id == user_id
        ]
        
    def get_tenant_connections(self, tenant_id: str) -> List[str]:
        """Get all connections for a tenant"""
        return [
            conn_id
            for conn_id, conn_info in self.connections.items()
            if conn_info.tenant_id == tenant_id
        ]
        
    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        return {
            "total_connections": len(self.connections),
            "total_channels": len(self.channels),
            "connections_by_tenant": {},  # Could aggregate by tenant
            "messages_in_queue": self.message_queue.qsize(),
            "uptime": datetime.utcnow().isoformat()
        }


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


class WebSocketNotificationService:
    """Service for sending notifications via WebSocket"""
    
    def __init__(self, manager: WebSocketManager):
        self.manager = manager
        
    async def notify_user(
        self,
        user_id: str,
        title: str,
        message: str,
        level: str = "info",
        data: Dict[str, Any] = None
    ):
        """Send notification to user"""
        await self.manager.send_to_user(
            user_id,
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.NOTIFICATION.value,
                channel=f"user:{user_id}",
                data={
                    "title": title,
                    "message": message,
                    "level": level,
                    "data": data or {}
                },
                timestamp=datetime.utcnow()
            )
        )
        
    async def notify_tenant(
        self,
        tenant_id: str,
        title: str,
        message: str,
        level: str = "info",
        data: Dict[str, Any] = None
    ):
        """Send notification to all users in tenant"""
        await self.manager.send_to_tenant(
            tenant_id,
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.NOTIFICATION.value,
                channel=f"tenant:{tenant_id}",
                data={
                    "title": title,
                    "message": message,
                    "level": level,
                    "data": data or {}
                },
                timestamp=datetime.utcnow()
            )
        )
        
    async def send_approval_request(
        self,
        approver_id: str,
        approval_request: Dict[str, Any]
    ):
        """Send approval request to approver"""
        await self.manager.send_to_user(
            approver_id,
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.APPROVAL_REQUEST.value,
                channel=f"user:{approver_id}",
                data=approval_request,
                timestamp=datetime.utcnow()
            )
        )
        
    async def broadcast_system_message(
        self,
        message: str,
        level: str = "info"
    ):
        """Broadcast system message to all users"""
        await self.manager.broadcast(
            WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.SYSTEM_MESSAGE.value,
                channel="public:system",
                data={
                    "message": message,
                    "level": level
                },
                timestamp=datetime.utcnow()
            )
        )


# Export key components
__all__ = [
    "WebSocketManager",
    "WebSocketMessage",
    "MessageType",
    "ConnectionInfo",
    "WebSocketNotificationService",
    "websocket_manager"
]