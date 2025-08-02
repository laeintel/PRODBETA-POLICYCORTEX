"""
Simplified API Gateway for Container Apps deployment.
Basic health checks and service routing without heavy dependencies.
"""

import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "api-gateway")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8001"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI app
app = FastAPI(
    title="PolicyCortex API Gateway",
    description="Central API Gateway for PolicyCortex microservices",
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
        "message": "PolicyCortex API Gateway",
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
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/conversation/governance")
async def conversation_governance(request: Request):
    """Mock conversation endpoint for development."""
    data = await request.json()
    user_input = data.get("user_input", "")
    session_id = data.get("session_id", "")
    
    # Mock AI response for development
    mock_response = {
        "response": f"I understand you're asking about: '{user_input}'. This is a mock response from the development API. In production, this would connect to the AI Engine service to provide intelligent Azure governance insights.",
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "mock-api-gateway",
        "suggestions": [
            "Review your current Azure Policy assignments",
            "Check compliance status of your resources",
            "Optimize cost management policies"
        ]
    }
    
    return mock_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_simple:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        log_level=LOG_LEVEL.lower()
    )
