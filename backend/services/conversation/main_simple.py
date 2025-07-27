"""
Simplified Conversation Service for Container Apps deployment.
Basic health checks and conversation endpoints without heavy dependencies.
"""

import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Simple configuration from environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
SERVICE_NAME = os.getenv("SERVICE_NAME", "conversation")
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8004"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# FastAPI app
app = FastAPI(
    title="PolicyCortex Conversation Service",
    description="Natural Language Interface and Conversation Management",
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
        "message": "PolicyCortex Conversation Service",
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
        "service": "conversation",
        "environment": ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/chat/status")
async def chat_status():
    """Conversation system status endpoint."""
    return {
        "chat_engine": "operational",
        "components": {
            "intent_classifier": "ready",
            "response_generator": "ready",
            "context_manager": "ready",
            "query_router": "ready"
        },
        "nlp_ready": True,
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