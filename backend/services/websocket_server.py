"""
Patent #4: WebSocket Server for Real-time Predictions
Streaming predictions and model updates to clients
Author: PolicyCortex Engineering Team
Date: January 2025
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Set, Any, Optional
from datetime import datetime
import aioredis
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass, asdict
import numpy as np
import torch

logger = logging.getLogger(__name__)

@dataclass
class PredictionUpdate:
    """Real-time prediction update message"""
    prediction_id: str
    resource_id: str
    tenant_id: str
    violation_probability: float
    risk_level: str
    time_to_violation: Optional[float]
    confidence_score: float
    timestamp: str
    update_type: str  # 'prediction', 'drift_alert', 'remediation', 'model_update'
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class WebSocketManager:
    """
    Manage WebSocket connections for real-time ML updates
    Patent Requirement: Real-time prediction streaming
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}  # tenant_id -> connections
        self.connection_metadata: Dict[WebSocketServerProtocol, Dict] = {}
        self.redis_url = redis_url
        self.redis_client = None
        self.pubsub = None
        self.prediction_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize Redis connection for pub/sub"""
        self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        
        # Subscribe to ML prediction channels
        await self.pubsub.subscribe(
            'ml:predictions',
            'ml:drift_alerts',
            'ml:model_updates',
            'ml:remediation'
        )
        
        logger.info("WebSocket manager initialized with Redis pub/sub")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)
            
            if not self._authenticate(auth_data):
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Authentication failed'
                }))
                return
            
            tenant_id = auth_data.get('tenant_id')
            user_role = auth_data.get('user_role', 'viewer')
            
            # Register connection
            await self._register_connection(websocket, tenant_id, user_role)
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to ML prediction stream',
                'tenant_id': tenant_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except asyncio.TimeoutError:
            logger.warning("Connection timeout - no authentication received")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await self._unregister_connection(websocket)
    
    def _authenticate(self, auth_data: Dict) -> bool:
        """Authenticate WebSocket connection"""
        # Simplified authentication - in production, validate JWT token
        return 'tenant_id' in auth_data and 'auth_token' in auth_data
    
    async def _register_connection(self, websocket: WebSocketServerProtocol, 
                                  tenant_id: str, user_role: str):
        """Register new connection"""
        if tenant_id not in self.connections:
            self.connections[tenant_id] = set()
        
        self.connections[tenant_id].add(websocket)
        self.connection_metadata[websocket] = {
            'tenant_id': tenant_id,
            'user_role': user_role,
            'connected_at': datetime.now(),
            'subscriptions': set()
        }
        
        logger.info(f"Registered WebSocket for tenant {tenant_id}")
    
    async def _unregister_connection(self, websocket: WebSocketServerProtocol):
        """Remove connection from registry"""
        if websocket in self.connection_metadata:
            metadata = self.connection_metadata[websocket]
            tenant_id = metadata['tenant_id']
            
            if tenant_id in self.connections:
                self.connections[tenant_id].discard(websocket)
                if not self.connections[tenant_id]:
                    del self.connections[tenant_id]
            
            del self.connection_metadata[websocket]
            logger.info(f"Unregistered WebSocket for tenant {tenant_id}")
    
    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'subscribe':
                await self._handle_subscription(websocket, data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscription(websocket, data)
            elif message_type == 'request_prediction':
                await self._handle_prediction_request(websocket, data)
            elif message_type == 'ping':
                await websocket.send(json.dumps({'type': 'pong'}))
            else:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON message'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_subscription(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle subscription request"""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return
        
        resource_ids = data.get('resource_ids', [])
        prediction_types = data.get('prediction_types', ['all'])
        
        metadata['subscriptions'].update(resource_ids)
        
        await websocket.send(json.dumps({
            'type': 'subscribed',
            'resource_ids': resource_ids,
            'prediction_types': prediction_types
        }))
    
    async def _handle_unsubscription(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle unsubscription request"""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return
        
        resource_ids = data.get('resource_ids', [])
        for resource_id in resource_ids:
            metadata['subscriptions'].discard(resource_id)
        
        await websocket.send(json.dumps({
            'type': 'unsubscribed',
            'resource_ids': resource_ids
        }))
    
    async def _handle_prediction_request(self, websocket: WebSocketServerProtocol, data: Dict):
        """Handle on-demand prediction request"""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return
        
        resource_id = data.get('resource_id')
        if not resource_id:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'resource_id required'
            }))
            return
        
        # Queue prediction request
        await self.prediction_queue.put({
            'tenant_id': metadata['tenant_id'],
            'resource_id': resource_id,
            'websocket': websocket,
            'priority': data.get('priority', 1)
        })
        
        await websocket.send(json.dumps({
            'type': 'prediction_queued',
            'resource_id': resource_id
        }))
    
    async def broadcast_prediction(self, update: PredictionUpdate):
        """Broadcast prediction update to relevant clients"""
        tenant_connections = self.connections.get(update.tenant_id, set())
        
        if not tenant_connections:
            return
        
        message = update.to_json()
        disconnected = set()
        
        for websocket in tenant_connections:
            try:
                metadata = self.connection_metadata.get(websocket)
                if metadata:
                    # Check if client subscribed to this resource
                    if 'all' in metadata['subscriptions'] or \
                       update.resource_id in metadata['subscriptions']:
                        await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            await self._unregister_connection(ws)
    
    async def process_redis_messages(self):
        """Process messages from Redis pub/sub"""
        while True:
            try:
                # Get message from Redis
                message = await self.pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    channel = message['channel'].decode('utf-8')
                    data = json.loads(message['data'].decode('utf-8'))
                    
                    # Create prediction update
                    update = PredictionUpdate(
                        prediction_id=data.get('prediction_id', str(uuid.uuid4())),
                        resource_id=data['resource_id'],
                        tenant_id=data['tenant_id'],
                        violation_probability=data.get('violation_probability', 0),
                        risk_level=data.get('risk_level', 'unknown'),
                        time_to_violation=data.get('time_to_violation'),
                        confidence_score=data.get('confidence_score', 0),
                        timestamp=datetime.now().isoformat(),
                        update_type=channel.split(':')[1]  # Extract type from channel
                    )
                    
                    # Broadcast to connected clients
                    await self.broadcast_prediction(update)
                    
            except Exception as e:
                logger.error(f"Error processing Redis message: {e}")
                await asyncio.sleep(1)
    
    async def process_prediction_queue(self):
        """Process queued prediction requests"""
        while True:
            try:
                # Get request from queue
                request = await self.prediction_queue.get()
                
                # Make prediction (simplified - would call actual ML service)
                prediction = await self._make_prediction(
                    request['tenant_id'],
                    request['resource_id']
                )
                
                # Send prediction to requesting client
                websocket = request['websocket']
                if websocket in self.connection_metadata:
                    await websocket.send(json.dumps({
                        'type': 'prediction_result',
                        'resource_id': request['resource_id'],
                        'prediction': prediction
                    }))
                    
            except Exception as e:
                logger.error(f"Error processing prediction queue: {e}")
    
    async def _make_prediction(self, tenant_id: str, resource_id: str) -> Dict:
        """Make prediction for resource (placeholder)"""
        # In production, this would call the actual ML prediction service
        return {
            'violation_probability': np.random.random(),
            'risk_level': np.random.choice(['critical', 'high', 'medium', 'low']),
            'confidence_score': 0.85 + np.random.random() * 0.14,
            'time_to_violation': np.random.randint(1, 72)
        }
    
    async def cleanup(self):
        """Clean up resources"""
        # Close all WebSocket connections
        for tenant_connections in self.connections.values():
            for websocket in tenant_connections:
                await websocket.close()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()


class MLWebSocketServer:
    """
    Main WebSocket server for ML predictions
    Patent Requirement: Real-time streaming with <100ms latency
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.manager = WebSocketManager()
        self.server = None
        
    async def start(self):
        """Start WebSocket server"""
        await self.manager.initialize()
        
        # Start background tasks
        asyncio.create_task(self.manager.process_redis_messages())
        asyncio.create_task(self.manager.process_prediction_queue())
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.manager.handle_connection,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(f"ML WebSocket server started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        await self.manager.cleanup()
        logger.info("ML WebSocket server stopped")
    
    async def simulate_predictions(self):
        """Simulate prediction updates for testing"""
        while True:
            await asyncio.sleep(5)  # Generate prediction every 5 seconds
            
            # Create simulated prediction
            update = PredictionUpdate(
                prediction_id=str(uuid.uuid4()),
                resource_id=f"resource-{np.random.randint(1, 100)}",
                tenant_id="org-1",  # Use test tenant
                violation_probability=np.random.random(),
                risk_level=np.random.choice(['critical', 'high', 'medium', 'low']),
                time_to_violation=np.random.randint(1, 72),
                confidence_score=0.85 + np.random.random() * 0.14,
                timestamp=datetime.now().isoformat(),
                update_type='prediction'
            )
            
            # Broadcast to connected clients
            await self.manager.broadcast_prediction(update)


async def main():
    """Main entry point"""
    server = MLWebSocketServer()
    await server.start()
    
    # Start simulation for testing
    if __name__ == "__main__":
        asyncio.create_task(server.simulate_predictions())
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())