#!/usr/bin/env python3
"""
Test script for WebSocket server functionality
Tests real-time prediction streaming for Patent #4
"""

import asyncio
import json
import websockets
import sys
from datetime import datetime
import uuid

async def test_websocket_client():
    """Test WebSocket connection and message handling"""
    uri = "ws://localhost:8765"
    
    try:
        print(f"[{datetime.now().isoformat()}] Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            # Send authentication message
            auth_msg = {
                "tenant_id": "test-tenant-1",
                "auth_token": "test-token-123",
                "user_role": "admin"
            }
            
            print(f"[{datetime.now().isoformat()}] Sending authentication...")
            await websocket.send(json.dumps(auth_msg))
            
            # Receive welcome message
            response = await websocket.recv()
            response_data = json.loads(response)
            print(f"[{datetime.now().isoformat()}] Received: {response_data}")
            
            if response_data.get('type') == 'error':
                print(f"Authentication failed: {response_data.get('message')}")
                return
            
            # Subscribe to prediction updates
            subscribe_msg = {
                "action": "subscribe",
                "channels": ["predictions", "drift_alerts", "model_updates"]
            }
            
            print(f"[{datetime.now().isoformat()}] Subscribing to channels...")
            await websocket.send(json.dumps(subscribe_msg))
            
            # Request a prediction
            prediction_request = {
                "action": "predict",
                "resource_id": f"vm-{uuid.uuid4().hex[:8]}",
                "configuration": {
                    "encryption_enabled": False,
                    "public_access": True,
                    "network_isolation": False
                }
            }
            
            print(f"[{datetime.now().isoformat()}] Requesting prediction...")
            await websocket.send(json.dumps(prediction_request))
            
            # Listen for messages for 30 seconds
            print(f"[{datetime.now().isoformat()}] Listening for messages (30 seconds)...")
            
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    msg_data = json.loads(message)
                    print(f"[{datetime.now().isoformat()}] Message received:")
                    print(f"  Type: {msg_data.get('type', 'unknown')}")
                    
                    if 'prediction_id' in msg_data:
                        print(f"  Prediction ID: {msg_data['prediction_id']}")
                        print(f"  Risk Level: {msg_data.get('risk_level', 'N/A')}")
                        print(f"  Violation Probability: {msg_data.get('violation_probability', 'N/A')}")
                        print(f"  Confidence: {msg_data.get('confidence_score', 'N/A')}")
                    elif 'message' in msg_data:
                        print(f"  Message: {msg_data['message']}")
                    
                    print("-" * 50)
                    
            except asyncio.TimeoutError:
                print(f"[{datetime.now().isoformat()}] No more messages received (timeout)")
            
            # Send ping to check connection
            ping_msg = {"action": "ping"}
            await websocket.send(json.dumps(ping_msg))
            
            # Close connection gracefully
            print(f"[{datetime.now().isoformat()}] Closing connection...")
            
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused. Is the WebSocket server running on {uri}?")
        print("Start the server with: python backend/services/websocket_server.py")
        return False
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"[{datetime.now().isoformat()}] Test completed successfully!")
    return True


async def test_multiple_clients():
    """Test multiple concurrent WebSocket connections"""
    print("\n" + "="*60)
    print("Testing multiple concurrent connections...")
    print("="*60)
    
    tasks = []
    for i in range(3):
        tenant_id = f"test-tenant-{i+1}"
        print(f"Creating client for {tenant_id}")
        task = asyncio.create_task(test_client_instance(tenant_id, i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = sum(1 for r in results if r is True)
    print(f"\n[SUMMARY] {success_count}/{len(tasks)} clients connected successfully")
    

async def test_client_instance(tenant_id: str, client_num: int):
    """Test a single client instance"""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            # Authenticate
            auth_msg = {
                "tenant_id": tenant_id,
                "auth_token": f"token-{client_num}",
                "user_role": "viewer"
            }
            await websocket.send(json.dumps(auth_msg))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            response_data = json.loads(response)
            
            if response_data.get('type') == 'connected':
                print(f"  ✓ Client {client_num} ({tenant_id}) connected")
                return True
            else:
                print(f"  ✗ Client {client_num} ({tenant_id}) failed: {response_data}")
                return False
                
    except Exception as e:
        print(f"  ✗ Client {client_num} ({tenant_id}) error: {e}")
        return False


def main():
    """Main test runner"""
    print("="*60)
    print("PolicyCortex WebSocket Server Test")
    print("Patent #4: Real-time Prediction Streaming")
    print("="*60)
    
    # Run single client test
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(test_websocket_client())
    
    if success:
        # Run multi-client test
        loop.run_until_complete(test_multiple_clients())
    
    print("\n" + "="*60)
    print("Test suite completed")
    print("="*60)


if __name__ == "__main__":
    main()