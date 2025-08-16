"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Error Learning Middleware for Continuous Learning System
Automatically captures application errors and sends them for AI learning
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import logging
import traceback
from datetime import datetime
import asyncio
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

class ErrorLearningMiddleware(BaseHTTPMiddleware):
    """
    Middleware that captures errors and sends them to the continuous learning system
    """
    
    def __init__(self, app, continuous_learner=None, batch_size: int = 10, 
                 batch_interval: float = 30.0):
        super().__init__(app)
        self.continuous_learner = continuous_learner
        self.error_buffer = deque(maxlen=1000)
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.last_batch_time = datetime.utcnow()
        self._batch_task = None
        
        # Start batch processing task if learner is available
        if self.continuous_learner:
            self._start_batch_processor()
    
    def _start_batch_processor(self):
        """Start the background task for batch processing errors"""
        self._batch_task = asyncio.create_task(self._batch_processor())
    
    async def _batch_processor(self):
        """Background task that processes error batches"""
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                await self._process_error_batch()
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    async def _process_error_batch(self):
        """Process accumulated errors in batch"""
        if not self.error_buffer or not self.continuous_learner:
            return
        
        # Extract errors for processing
        errors_to_process = []
        while self.error_buffer and len(errors_to_process) < self.batch_size:
            errors_to_process.append(self.error_buffer.popleft())
        
        if errors_to_process:
            try:
                # Send to continuous learning system
                error_events = await self.continuous_learner.collect_errors_from_application(
                    errors_to_process
                )
                await self.continuous_learner.learn_from_errors(error_events)
                
                logger.info(f"Sent {len(errors_to_process)} errors to continuous learning")
            except Exception as e:
                logger.error(f"Failed to send errors to learning system: {e}")
                # Put errors back in buffer for retry
                for error in reversed(errors_to_process):
                    self.error_buffer.appendleft(error)
    
    def _classify_error(self, error: Exception, request: Request) -> str:
        """Classify error severity based on type and context"""
        if isinstance(error, HTTPException):
            if error.status_code >= 500:
                return "critical"
            elif error.status_code >= 400:
                return "high"
            else:
                return "medium"
        elif isinstance(error, (ValueError, KeyError, AttributeError)):
            return "medium"
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return "high"
        else:
            return "high"  # Unknown errors are treated as high severity
    
    def _detect_domain(self, request: Request, error: Exception) -> str:
        """Detect error domain based on request path and error type"""
        path = request.url.path.lower()
        error_str = str(error).lower()
        
        # Check path-based domains
        if any(word in path for word in ['/policies', '/compliance', '/rbac']):
            return 'security'
        elif any(word in path for word in ['/costs', '/finops', '/budget']):
            return 'cloud'
        elif any(word in path for word in ['/network', '/vpc', '/firewall']):
            return 'network'
        
        # Check error message-based domains
        if any(word in error_str for word in ['auth', 'token', 'permission', 'forbidden']):
            return 'security'
        elif any(word in error_str for word in ['connection', 'timeout', 'socket']):
            return 'network'
        elif any(word in error_str for word in ['azure', 'aws', 'gcp', 'resource']):
            return 'cloud'
        
        return 'other'
    
    def _create_error_hash(self, error: Dict[str, Any]) -> str:
        """Create a hash for error deduplication"""
        key_parts = [
            error.get('error_type', ''),
            error.get('error_message', '')[:100],  # First 100 chars
            error.get('path', '')
        ]
        key = '|'.join(key_parts)
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def dispatch(self, request: Request, call_next):
        """Intercept requests and capture errors for learning"""
        try:
            # Process the request
            response = await call_next(request)
            
            # Check for error status codes
            if response.status_code >= 400:
                # Capture HTTP errors
                error_data = {
                    "error_type": f"HTTP_{response.status_code}",
                    "error_message": f"HTTP {response.status_code} error on {request.url.path}",
                    "stack_trace": None,
                    "context": {
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "client": request.client.host if request.client else None
                    },
                    "severity": "high" if response.status_code >= 500 else "medium",
                    "domain": self._detect_domain(request, None),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Add to buffer for batch processing
                if self.continuous_learner:
                    self.error_buffer.append(error_data)
                    
                    # Process immediately if buffer is full
                    if len(self.error_buffer) >= self.batch_size:
                        await self._process_error_batch()
            
            return response
            
        except Exception as error:
            # Capture unhandled exceptions
            error_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "stack_trace": traceback.format_exc(),
                "context": {
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "client": request.client.host if request.client else None
                },
                "severity": self._classify_error(error, request),
                "domain": self._detect_domain(request, error),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add error hash for deduplication
            error_data["error_hash"] = self._create_error_hash(error_data)
            
            # Log the error
            logger.error(f"Captured error for learning: {error_data['error_type']} - {error_data['error_message'][:100]}")
            
            # Add to buffer for batch processing
            if self.continuous_learner:
                self.error_buffer.append(error_data)
                
                # Trigger immediate learning for critical errors
                if error_data["severity"] == "critical":
                    await self._process_error_batch()
            
            # Re-raise the error or return error response
            if isinstance(error, HTTPException):
                return JSONResponse(
                    status_code=error.status_code,
                    content={"detail": error.detail}
                )
            else:
                # Internal server error
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "error_id": error_data.get("error_hash"),
                        "learning_enabled": bool(self.continuous_learner)
                    }
                )

class ErrorPredictionHelper:
    """
    Helper class to get AI predictions for errors in real-time
    """
    
    def __init__(self, continuous_learner):
        self.continuous_learner = continuous_learner
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def get_solution_suggestion(self, error_message: str, 
                                     domain: str = "other") -> Optional[Dict[str, Any]]:
        """Get AI-predicted solution for an error"""
        if not self.continuous_learner:
            return None
        
        # Check cache
        cache_key = f"{domain}:{hashlib.md5(error_message.encode()).hexdigest()}"
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if (datetime.utcnow() - cached["timestamp"]).total_seconds() < self.cache_ttl:
                return cached["prediction"]
        
        try:
            # Get prediction from AI model
            prediction = self.continuous_learner.predict_solution(error_message, domain)
            
            # Cache the prediction
            self.prediction_cache[cache_key] = {
                "prediction": prediction,
                "timestamp": datetime.utcnow()
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to get error prediction: {e}")
            return None
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.prediction_cache),
            "ttl_seconds": self.cache_ttl,
            "oldest_entry": min(
                (v["timestamp"] for v in self.prediction_cache.values()),
                default=None
            )
        }

def setup_error_learning(app, continuous_learner=None):
    """
    Setup error learning middleware and helper
    
    Args:
        app: FastAPI application instance
        continuous_learner: ContinuousLearningSystem instance
    
    Returns:
        ErrorPredictionHelper instance
    """
    # Add middleware
    app.add_middleware(
        ErrorLearningMiddleware,
        continuous_learner=continuous_learner
    )
    
    # Create and return prediction helper
    return ErrorPredictionHelper(continuous_learner)

# Export main components
__all__ = [
    'ErrorLearningMiddleware',
    'ErrorPredictionHelper',
    'setup_error_learning'
]