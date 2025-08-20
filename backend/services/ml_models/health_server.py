"""
Simple health check server for ML services testing
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
from datetime import datetime

app = FastAPI(title="ML Health Check Server")

class HealthStatus(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

@app.get("/")
async def root():
    return {"message": "ML Service Health Check Server"}

@app.get("/health", response_model=HealthStatus)
async def health():
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        service="ml-prediction-server",
        version="1.0.0-test"
    )

@app.get("/api/v1/predictions")
async def predictions():
    return {
        "predictions": [],
        "message": "ML predictions endpoint (test mode)"
    }

@app.get("/metrics")
async def metrics():
    return """
# HELP ml_requests_total Total number of ML requests
# TYPE ml_requests_total counter
ml_requests_total 0

# HELP ml_model_loaded Model loaded status
# TYPE ml_model_loaded gauge
ml_model_loaded 1
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)