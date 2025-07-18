"""
Simplified Azure Integration Service for testing.
"""

import time
from fastapi import FastAPI
from prometheus_client import generate_latest

# Initialize FastAPI app
app = FastAPI(
    title="PolicyCortex Azure Integration",
    description="Azure services integration service",
    version="1.0.0"
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {"status": "ready", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)