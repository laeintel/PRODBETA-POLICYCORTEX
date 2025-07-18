"""
Conversation Service for PolicyCortex.
Natural language interface for Azure governance with multi-turn conversation management.
"""

import time
import uuid
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocketState
import asyncio

from ...shared.config import get_settings
from ...shared.database import get_async_db, DatabaseUtils
from .auth import AuthManager
from .models import (
    HealthResponse,
    ConversationRequest,
    ConversationResponse,
    ConversationHistoryResponse,
    ConversationSessionResponse,
    IntentClassificationResponse,
    EntityExtractionResponse,
    WebSocketMessage,
    ConversationAnalytics,
    ErrorResponse
)
from .services.conversation_manager import ConversationManager
from .services.intent_classifier import IntentClassifier
from .services.context_manager import ContextManager
from .services.response_generator import ResponseGenerator
from .services.query_router import QueryRouter
from .services.analytics_service import AnalyticsService

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter('conversation_requests_total', 'Total conversation requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('conversation_request_duration_seconds', 'Request duration')
CONVERSATION_COUNT = Counter('conversations_total', 'Total conversations', ['intent', 'status'])
WEBSOCKET_CONNECTIONS = Counter('websocket_connections_total', 'Total WebSocket connections')
WEBSOCKET_MESSAGES = Counter('websocket_messages_total', 'Total WebSocket messages', ['type'])

# FastAPI app
app = FastAPI(
    title="PolicyCortex Conversation Service",
    description="Natural language interface for Azure governance with AI-powered conversation management",
    version=settings.service.service_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Security
security = HTTPBearer(auto_error=False)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=settings.security.cors_methods,
    allow_headers=settings.security.cors_headers,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Global components
auth_manager = AuthManager()
conversation_manager = ConversationManager()
intent_classifier = IntentClassifier()
context_manager = ContextManager()
response_generator = ResponseGenerator()
query_router = QueryRouter()
analytics_service = AnalyticsService()

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time conversations."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """Connect a new WebSocket."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.user_sessions[user_id] = session_id
        WEBSOCKET_CONNECTIONS.inc()
        logger.info(
            "websocket_connected",
            session_id=session_id,
            user_id=user_id
        )
    
    def disconnect(self, session_id: str, user_id: str):
        """Disconnect a WebSocket."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        logger.info(
            "websocket_disconnected",
            session_id=session_id,
            user_id=user_id
        )
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
                WEBSOCKET_MESSAGES.labels(type="outbound").inc()
            except Exception as e:
                logger.error(
                    "websocket_send_failed",
                    session_id=session_id,
                    error=str(e)
                )
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        """Broadcast message to all user sessions."""
        session_id = self.user_sessions.get(user_id)
        if session_id:
            await self.send_personal_message(message, session_id)

connection_manager = ConnectionManager()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and metrics."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to headers
        request.state.request_id = request_id
        
        # Log request
        logger.info(
            "request_started",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else None
        )
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            REQUEST_DURATION.observe(duration)
            
            # Log response
            logger.info(
                "request_completed",
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2)
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "request_failed",
                request_id=request_id,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            
            # Update error metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            raise


# Add middleware
app.add_middleware(RequestLoggingMiddleware)


async def verify_authentication(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Verify authentication for protected endpoints."""
    
    # Skip authentication for health checks and public endpoints
    if request.url.path in ["/health", "/ready", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        user_info = await auth_manager.verify_token(credentials.credentials)
        request.state.user = user_info
        return user_info
    except Exception as e:
        logger.error("authentication_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="conversation",
        version=settings.service.service_version
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    # Check AI engine availability
    try:
        await response_generator.health_check()
        return HealthResponse(
            status="ready",
            timestamp=datetime.utcnow(),
            service="conversation",
            version=settings.service.service_version,
            details={"ai_engine": "healthy"}
        )
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI engine is not available"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# Conversation endpoints
@app.post("/api/v1/conversations", response_model=ConversationResponse)
async def start_conversation(
    request: ConversationRequest,
    user: Dict[str, Any] = Depends(verify_authentication)
):
    """Start a new conversation or continue existing one."""
    try:
        # Create or get conversation session
        session = await conversation_manager.create_or_get_session(
            user_id=user["id"],
            session_id=request.session_id
        )
        
        # Classify intent
        intent_result = await intent_classifier.classify_intent(
            text=request.message,
            context=session.context
        )
        
        # Extract entities
        entities = await intent_classifier.extract_entities(
            text=request.message,
            intent=intent_result.intent
        )
        
        # Update conversation context
        await context_manager.update_context(
            session_id=session.session_id,
            message=request.message,
            intent=intent_result.intent,
            entities=entities
        )
        
        # Route query to appropriate service
        service_response = await query_router.route_query(
            intent=intent_result.intent,
            entities=entities,
            message=request.message,
            user_info=user
        )
        
        # Generate response
        response = await response_generator.generate_response(
            message=request.message,
            intent=intent_result.intent,
            entities=entities,
            context=session.context,
            service_data=service_response
        )
        
        # Save conversation turn
        await conversation_manager.add_message(
            session_id=session.session_id,
            user_message=request.message,
            assistant_message=response.message,
            intent=intent_result.intent,
            entities=entities
        )
        
        # Update metrics
        CONVERSATION_COUNT.labels(
            intent=intent_result.intent,
            status="success"
        ).inc()
        
        # Send WebSocket update if connected
        if session.session_id in connection_manager.active_connections:
            await connection_manager.send_personal_message(
                {
                    "type": "conversation_update",
                    "session_id": session.session_id,
                    "message": response.message,
                    "intent": intent_result.intent,
                    "entities": entities
                },
                session.session_id
            )
        
        return ConversationResponse(
            session_id=session.session_id,
            message=response.message,
            intent=intent_result.intent,
            entities=entities,
            confidence=intent_result.confidence,
            suggestions=response.suggestions,
            follow_up_questions=response.follow_up_questions,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(
            "conversation_failed",
            error=str(e),
            user_id=user["id"],
            message=request.message
        )
        
        CONVERSATION_COUNT.labels(
            intent="unknown",
            status="error"
        ).inc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation processing failed: {str(e)}"
        )


@app.get("/api/v1/conversations/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    session_id: str,
    limit: int = 50,
    offset: int = 0,
    user: Dict[str, Any] = Depends(verify_authentication)
):
    """Get conversation history for a session."""
    try:
        history = await conversation_manager.get_conversation_history(
            session_id=session_id,
            user_id=user["id"],
            limit=limit,
            offset=offset
        )
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=history.messages,
            total_count=history.total_count,
            has_more=history.has_more
        )
        
    except Exception as e:
        logger.error(
            "get_conversation_history_failed",
            error=str(e),
            session_id=session_id,
            user_id=user["id"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@app.get("/api/v1/conversations", response_model=List[ConversationSessionResponse])
async def get_user_conversations(
    user: Dict[str, Any] = Depends(verify_authentication),
    limit: int = 20,
    offset: int = 0
):
    """Get all conversation sessions for a user."""
    try:
        sessions = await conversation_manager.get_user_sessions(
            user_id=user["id"],
            limit=limit,
            offset=offset
        )
        
        return [
            ConversationSessionResponse(
                session_id=session.session_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=session.message_count,
                last_message_preview=session.last_message_preview
            )
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(
            "get_user_conversations_failed",
            error=str(e),
            user_id=user["id"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user conversations: {str(e)}"
        )


@app.delete("/api/v1/conversations/{session_id}")
async def delete_conversation(
    session_id: str,
    user: Dict[str, Any] = Depends(verify_authentication)
):
    """Delete a conversation session."""
    try:
        await conversation_manager.delete_session(
            session_id=session_id,
            user_id=user["id"]
        )
        
        return {"message": "Conversation deleted successfully"}
        
    except Exception as e:
        logger.error(
            "delete_conversation_failed",
            error=str(e),
            session_id=session_id,
            user_id=user["id"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


# Intent classification endpoints
@app.post("/api/v1/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(
    request: ConversationRequest,
    user: Dict[str, Any] = Depends(verify_authentication)
):
    """Classify intent of a message."""
    try:
        result = await intent_classifier.classify_intent(
            text=request.message,
            context={}
        )
        
        return IntentClassificationResponse(
            intent=result.intent,
            confidence=result.confidence,
            entities=result.entities,
            sub_intents=result.sub_intents
        )
        
    except Exception as e:
        logger.error(
            "intent_classification_failed",
            error=str(e),
            message=request.message
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intent classification failed: {str(e)}"
        )


@app.post("/api/v1/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(
    request: ConversationRequest,
    user: Dict[str, Any] = Depends(verify_authentication)
):
    """Extract entities from a message."""
    try:
        entities = await intent_classifier.extract_entities(
            text=request.message,
            intent=request.intent
        )
        
        return EntityExtractionResponse(
            entities=entities,
            message=request.message
        )
        
    except Exception as e:
        logger.error(
            "entity_extraction_failed",
            error=str(e),
            message=request.message
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {str(e)}"
        )


# Analytics endpoints
@app.get("/api/v1/analytics", response_model=ConversationAnalytics)
async def get_conversation_analytics(
    user: Dict[str, Any] = Depends(verify_authentication),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get conversation analytics."""
    try:
        analytics = await analytics_service.get_conversation_analytics(
            user_id=user["id"],
            start_date=start_date,
            end_date=end_date
        )
        
        return analytics
        
    except Exception as e:
        logger.error(
            "get_analytics_failed",
            error=str(e),
            user_id=user["id"]
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


# WebSocket endpoint
@app.websocket("/api/v1/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: str
):
    """WebSocket endpoint for real-time conversations."""
    try:
        # Verify authentication
        user_info = await auth_manager.verify_token(token)
        
        # Connect WebSocket
        await connection_manager.connect(websocket, session_id, user_info["id"])
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                WEBSOCKET_MESSAGES.labels(type="inbound").inc()
                
                message = WebSocketMessage(**data)
                
                if message.type == "conversation":
                    # Process conversation message
                    conversation_request = ConversationRequest(
                        message=message.content,
                        session_id=session_id
                    )
                    
                    # Process through conversation pipeline
                    response = await start_conversation(
                        conversation_request,
                        user_info
                    )
                    
                    # Send response back
                    await connection_manager.send_personal_message(
                        {
                            "type": "conversation_response",
                            "session_id": session_id,
                            "message": response.message,
                            "intent": response.intent,
                            "entities": response.entities,
                            "confidence": response.confidence,
                            "suggestions": response.suggestions,
                            "follow_up_questions": response.follow_up_questions,
                            "timestamp": response.timestamp.isoformat()
                        },
                        session_id
                    )
                
                elif message.type == "typing":
                    # Handle typing indicator
                    await connection_manager.send_personal_message(
                        {
                            "type": "typing_indicator",
                            "session_id": session_id,
                            "is_typing": message.content.get("is_typing", False)
                        },
                        session_id
                    )
                
        except WebSocketDisconnect:
            connection_manager.disconnect(session_id, user_info["id"])
        except Exception as e:
            logger.error(
                "websocket_error",
                error=str(e),
                session_id=session_id,
                user_id=user_info["id"]
            )
            connection_manager.disconnect(session_id, user_info["id"])
            
    except Exception as e:
        logger.error(
            "websocket_connection_failed",
            error=str(e),
            session_id=session_id
        )
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close(code=1000)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "unhandled_exception",
        error=str(exc),
        request_id=getattr(request.state, "request_id", "unknown")
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            message=str(exc) if settings.debug else "An unexpected error occurred",
            request_id=getattr(request.state, "request_id", "unknown")
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.service.service_host,
        port=settings.service.service_port,
        workers=1,  # Single worker for WebSocket support
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )