"""
Simplified Notification Service for Container Apps deployment.
Basic health checks and notification endpoints without heavy dependencies.
"""

import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "notification")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8005"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI app
app = FastAPI(
    title="PolicyCortex Notification Service",
    description="Alert and Notification Management Service",
    version="1.0.0"
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
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "service": SERVICE_NAME,
        "environment": ENVIRONMENT
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PolicyCortex Notification Service",
        "status": "running",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/status")
async def api_status():
    """API status endpoint."""
    return {
        "api_version": "v1",
        "status": "operational",
        "service": "notification",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/notifications/status")
async def notifications_status():
    """Notification system status endpoint."""
    return {
        "notification_services": {
            "email_service": "operational",
            "sms_service": "operational",
            "push_service": "operational",
            "webhook_service": "operational"
        },
        "alert_manager": "running",
        "subscription_manager": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level=LOG_LEVEL.lower()
    )