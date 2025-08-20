#!/usr/bin/env python3
"""
Simplified WebSocket Server for Real-time Predictions
Patent #4 Implementation - Testing Version
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Set, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedWebSocketServer:
    """Simplified WebSocket server for testing ML predictions"""
    
    def __init__(self):
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.connection_metadata: Dict[WebSocketServerProtocol, Dict] = {}
        
    async def handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection"""
        try:
            logger.info(f"New connection from {websocket.remote_address}")
            
            # Wait for authentication message
            try:
                auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                auth_data = json.loads(auth_message)
                logger.info(f"Received auth: {auth_data}")
            except asyncio.TimeoutError:
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': 'Authentication timeout'
                }))
                return
            
            tenant_id = auth_data.get('tenant_id', 'default')
            
            # Register connection
            if tenant_id not in self.connections:
                self.connections[tenant_id] = set()
            self.connections[tenant_id].add(websocket)
            self.connection_metadata[websocket] = {
                'tenant_id': tenant_id,
                'connected_at': datetime.now().isoformat()
            }
            
            # Send welcome message
            await websocket.send(json.dumps({
                'type': 'connected',
                'message': 'Connected to ML prediction stream',
                'tenant_id': tenant_id,
                'timestamp': datetime.now().isoformat()
            }))
            
            logger.info(f"Client authenticated: tenant_id={tenant_id}")
            
            # Start sending periodic predictions
            prediction_task = asyncio.create_task(
                self.send_periodic_predictions(websocket, tenant_id)
            )
            
            # Handle incoming messages
            try:
                async for message in websocket:
                    await self.handle_message(websocket, message, tenant_id)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection closed for tenant {tenant_id}")
            finally:
                prediction_task.cancel()
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            # Clean up connection
            if websocket in self.connection_metadata:
                tenant_id = self.connection_metadata[websocket]['tenant_id']
                if tenant_id in self.connections:
                    self.connections[tenant_id].discard(websocket)
                del self.connection_metadata[websocket]
                logger.info(f"Cleaned up connection for tenant {tenant_id}")
    
    async def handle_message(self, websocket: WebSocketServerProtocol, message: str, tenant_id: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            logger.info(f"Received action: {action} from tenant {tenant_id}")
            
            if action == 'ping':
                await websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': datetime.now().isoformat()
                }))
                
            elif action == 'subscribe':
                channels = data.get('channels', [])
                await websocket.send(json.dumps({
                    'type': 'subscribed',
                    'channels': channels,
                    'message': f'Subscribed to {len(channels)} channels'
                }))
                
            elif action == 'predict':
                # Generate a mock prediction
                resource_id = data.get('resource_id', f'resource-{uuid.uuid4().hex[:8]}')
                prediction = self.generate_mock_prediction(resource_id, tenant_id)
                await websocket.send(json.dumps(prediction))
                
            else:
                await websocket.send(json.dumps({
                    'type': 'unknown_action',
                    'message': f'Unknown action: {action}'
                }))
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def send_periodic_predictions(self, websocket: WebSocketServerProtocol, tenant_id: str):
        """Send periodic prediction updates to simulate real-time stream"""
        try:
            while True:
                await asyncio.sleep(5)  # Send update every 5 seconds
                
                if websocket.closed:
                    break
                    
                # Generate random prediction update
                prediction = self.generate_mock_prediction(
                    f'auto-resource-{uuid.uuid4().hex[:8]}',
                    tenant_id
                )
                
                await websocket.send(json.dumps(prediction))
                logger.info(f"Sent periodic prediction to tenant {tenant_id}")
                
        except asyncio.CancelledError:
            logger.info(f"Stopping periodic predictions for tenant {tenant_id}")
        except Exception as e:
            logger.error(f"Error sending periodic prediction: {e}")
    
    def generate_mock_prediction(self, resource_id: str, tenant_id: str) -> Dict:
        """Generate a mock prediction for testing"""
        violation_prob = random.random()
        risk_level = 'critical' if violation_prob > 0.8 else 'high' if violation_prob > 0.6 else 'medium' if violation_prob > 0.3 else 'low'
        
        return {
            'type': 'prediction',
            'prediction_id': str(uuid.uuid4()),
            'resource_id': resource_id,
            'tenant_id': tenant_id,
            'violation_probability': round(violation_prob, 3),
            'risk_level': risk_level,
            'time_to_violation': random.randint(1, 72) if violation_prob > 0.5 else None,
            'confidence_score': round(0.85 + random.random() * 0.14, 3),  # 85-99% confidence
            'timestamp': datetime.now().isoformat(),
            'update_type': 'prediction',
            'recommendations': [
                'Enable encryption for data at rest',
                'Review network access controls',
                'Update compliance tags'
            ] if violation_prob > 0.5 else []
        }
    
    async def broadcast_to_tenant(self, tenant_id: str, message: Dict):
        """Broadcast message to all connections for a tenant"""
        if tenant_id in self.connections:
            disconnected = set()
            for websocket in self.connections[tenant_id]:
                try:
                    await websocket.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
            
            # Clean up disconnected clients
            for ws in disconnected:
                self.connections[tenant_id].discard(ws)
                if ws in self.connection_metadata:
                    del self.connection_metadata[ws]


async def main():
    """Start the WebSocket server"""
    server = SimplifiedWebSocketServer()
    
    host = "localhost"
    port = 8765
    
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    logger.info("Patent #4: Real-time Prediction Streaming")
    logger.info("Press Ctrl+C to stop")
    
    async with websockets.serve(server.handle_connection, host, port):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")