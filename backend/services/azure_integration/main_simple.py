"""
Simplified Azure Integration Service for Container Apps deployment.
Basic health checks and Azure service endpoints without heavy dependencies.
"""

import os
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "azure-integration")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI app
app = FastAPI(
    title="PolicyCortex Azure Integration Service",
    description="Azure Integration and Resource Management Service",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": SERVICE_NAME,
        "environment": ENVIRONMENT,
        "version": "1.0.0",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "service": SERVICE_NAME,
        "environment": ENVIRONMENT,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PolicyCortex Azure Integration Service",
        "status": "running",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "operational",
        "service": "azure_integration",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/azure/status")
async def azure_status():
    """Azure integration status endpoint."""
    return {
        "azure_integration": "operational",
        "services": [
            "resource_management",
            "policy_management",
            "rbac_management",
            "cost_management",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_simple:app", host="0.0.0.0", port=SERVICE_PORT, log_level=LOG_LEVEL.lower())
