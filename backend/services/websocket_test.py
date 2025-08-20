"""
Simple WebSocket server for testing
"""
import asyncio
import websockets
import json
from datetime import datetime

async def handle_connection(websocket, path):
    """Handle incoming WebSocket connections"""
    print(f"Client connected from {websocket.remote_address}")
    
    try:
        # Send welcome message
        await websocket.send(json.dumps({
            "type": "welcome",
            "message": "Connected to ML WebSocket Server",
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        # Echo messages back
        async for message in websocket:
            print(f"Received: {message}")
            response = {
                "type": "echo",
                "original": message,
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Start the WebSocket server"""
    print("Starting WebSocket server on port 8765...")
    async with websockets.serve(handle_connection, "0.0.0.0", 8765):
        print("WebSocket server is running on ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())