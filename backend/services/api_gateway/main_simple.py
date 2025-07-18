"""
Simplified API Gateway Service for testing.
"""

import time
from typing import Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException

# Initialize FastAPI app
app = FastAPI(
    title="PolicyCortex API Gateway",
    description="Central entry point for all microservices",
    version="1.0.0"
)

# Simple health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)