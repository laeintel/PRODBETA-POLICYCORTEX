"""
AI Engine Service for PolicyCortex.
Provides AI/ML capabilities for policy analysis, anomaly detection, and predictive analytics.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import json

import structlog
from fastapi import FastAPI, Request, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from backend.shared.config import get_settings
from backend.shared.database import get_async_db, DatabaseUtils
from .auth import AuthManager
from .models import (
    HealthResponse,
    APIResponse,
    ErrorResponse,
    PolicyAnalysisRequest,
    PolicyAnalysisResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    CostOptimizationRequest,
    CostOptimizationResponse,
    PredictiveAnalyticsRequest,
    PredictiveAnalyticsResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    ModelInfo,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelMetrics
)
from .services.model_manager import ModelManager
from .services.nlp_service import NLPService
from .services.anomaly_detector import AnomalyDetector
from .services.cost_optimizer import CostOptimizer
from .services.predictive_analytics import PredictiveAnalyticsService
from .services.sentiment_analyzer import SentimentAnalyzer
from .services.feature_engineer import FeatureEngineer
from .services.model_monitor import ModelMonitor
from .ml_models.compliance_predictor import CompliancePredictor
from .ml_models.correlation_engine import CrossDomainCorrelationEngine
from .services.automation_orchestrator import WorkflowEngine
from .services.automation_engine import AutomationTrigger, AutomationStatus

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter('ai_engine_requests_total', 'Total AI Engine requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('ai_engine_request_duration_seconds', 'Request duration')
MODEL_INFERENCE_COUNT = Counter('ai_engine_model_inference_total', 'Model inference requests', ['model_name', 'status'])
MODEL_INFERENCE_DURATION = Histogram('ai_engine_model_inference_duration_seconds', 'Model inference duration', ['model_name'])
ACTIVE_MODELS = Gauge('ai_engine_active_models', 'Number of active models')
MODEL_ACCURACY = Gauge('ai_engine_model_accuracy', 'Model accuracy metrics', ['model_name', 'metric_type'])

# FastAPI app
app = FastAPI(
    title="PolicyCortex AI Engine",
    description="AI/ML service for policy analysis, anomaly detection, and predictive analytics",
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
model_manager = ModelManager()
nlp_service = NLPService()
anomaly_detector = AnomalyDetector()
cost_optimizer = CostOptimizer()
predictive_analytics = PredictiveAnalyticsService()
sentiment_analyzer = SentimentAnalyzer()
feature_engineer = FeatureEngineer()
model_monitor = ModelMonitor()
compliance_predictor = CompliancePredictor()
correlation_engine = CrossDomainCorrelationEngine()
workflow_engine = WorkflowEngine()


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


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting AI Engine service")
    
    try:
        # Initialize model manager
        await model_manager.initialize()
        
        # Load pre-trained models
        await model_manager.load_default_models()
        
        # Initialize services
        await nlp_service.initialize()
        await anomaly_detector.initialize()
        await cost_optimizer.initialize()
        await predictive_analytics.initialize()
        await sentiment_analyzer.initialize()
        await feature_engineer.initialize()
        await model_monitor.initialize()
        await compliance_predictor.initialize()
        await correlation_engine.initialize()
        await workflow_engine.initialize()
        
        # Update metrics
        ACTIVE_MODELS.set(len(model_manager.active_models))
        
        logger.info("AI Engine service started successfully")
        
    except Exception as e:
        logger.error("Failed to start AI Engine service", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down AI Engine service")
    
    try:
        # Cleanup services
        await model_manager.cleanup()
        await nlp_service.cleanup()
        await anomaly_detector.cleanup()
        await cost_optimizer.cleanup()
        await predictive_analytics.cleanup()
        await sentiment_analyzer.cleanup()
        await feature_engineer.cleanup()
        await model_monitor.cleanup()
        await workflow_engine.cleanup()
        
        logger.info("AI Engine service shutdown complete")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Health endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="ai-engine",
        version=settings.service.service_version
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check if core services are ready
        services_ready = {
            "model_manager": model_manager.is_ready(),
            "nlp_service": nlp_service.is_ready(),
            "anomaly_detector": anomaly_detector.is_ready(),
            "cost_optimizer": cost_optimizer.is_ready(),
            "predictive_analytics": predictive_analytics.is_ready(),
            "sentiment_analyzer": sentiment_analyzer.is_ready()
        }
        
        if all(services_ready.values()):
            return HealthResponse(
                status="ready",
                timestamp=datetime.utcnow(),
                service="ai-engine",
                version=settings.service.service_version,
                details={"services": services_ready}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Some services are not ready: {services_ready}"
            )
            
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# Model Management endpoints
@app.get("/api/v1/models", response_model=List[ModelInfo])
async def list_models(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """List available models."""
    try:
        models = await model_manager.list_models()
        return models
    except Exception as e:
        logger.error("list_models_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.get("/api/v1/models/{model_name}", response_model=ModelInfo)
async def get_model_info(
    model_name: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get model information."""
    try:
        model_info = await model_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        return model_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_model_info_failed", model_name=model_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/api/v1/models/{model_name}/train", response_model=ModelTrainingResponse)
async def train_model(
    model_name: str,
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Train or retrain a model."""
    try:
        # Start training in background
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            model_manager.train_model,
            model_name,
            request.training_data,
            request.parameters,
            task_id
        )
        
        return ModelTrainingResponse(
            task_id=task_id,
            model_name=model_name,
            status="training_started",
            message="Model training started in background"
        )
        
    except Exception as e:
        logger.error("train_model_failed", model_name=model_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start model training: {str(e)}"
        )


@app.get("/api/v1/models/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_name: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get model performance metrics."""
    try:
        metrics = await model_monitor.get_model_metrics(model_name)
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Metrics for model '{model_name}' not found"
            )
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_model_metrics_failed", model_name=model_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )


# AI/ML Capability endpoints
@app.post("/api/v1/policy-analysis", response_model=PolicyAnalysisResponse)
async def analyze_policy(
    request: PolicyAnalysisRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze policy documents using NLP."""
    start_time = time.time()
    
    try:
        logger.info("policy_analysis_started", request_id=request.request_id)
        
        result = await nlp_service.analyze_policy(
            policy_text=request.policy_text,
            analysis_type=request.analysis_type,
            options=request.options
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="policy_analyzer", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="policy_analyzer").observe(duration)
        
        logger.info("policy_analysis_completed", 
                   request_id=request.request_id,
                   duration_ms=round(duration * 1000, 2))
        
        return PolicyAnalysisResponse(
            request_id=request.request_id,
            analysis_results=result,
            confidence_score=result.get("confidence", 0.0),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="policy_analyzer", status="error").inc()
        
        logger.error("policy_analysis_failed", 
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy analysis failed: {str(e)}"
        )


@app.post("/api/v1/anomaly-detection", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Detect anomalies in Azure resources."""
    start_time = time.time()
    
    try:
        logger.info("anomaly_detection_started", request_id=request.request_id)
        
        result = await anomaly_detector.detect_anomalies(
            resource_data=request.resource_data,
            detection_type=request.detection_type,
            threshold=request.threshold
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="anomaly_detector", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="anomaly_detector").observe(duration)
        
        logger.info("anomaly_detection_completed",
                   request_id=request.request_id,
                   anomalies_found=len(result.get("anomalies", [])),
                   duration_ms=round(duration * 1000, 2))
        
        return AnomalyDetectionResponse(
            request_id=request.request_id,
            anomalies=result.get("anomalies", []),
            analysis_summary=result.get("summary", {}),
            confidence_score=result.get("confidence", 0.0),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="anomaly_detector", status="error").inc()
        
        logger.error("anomaly_detection_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@app.post("/api/v1/cost-optimization", response_model=CostOptimizationResponse)
async def optimize_costs(
    request: CostOptimizationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Generate cost optimization recommendations."""
    start_time = time.time()
    
    try:
        logger.info("cost_optimization_started", request_id=request.request_id)
        
        result = await cost_optimizer.optimize_costs(
            resource_data=request.resource_data,
            optimization_goals=request.optimization_goals,
            constraints=request.constraints
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="cost_optimizer", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="cost_optimizer").observe(duration)
        
        logger.info("cost_optimization_completed",
                   request_id=request.request_id,
                   recommendations_count=len(result.get("recommendations", [])),
                   duration_ms=round(duration * 1000, 2))
        
        return CostOptimizationResponse(
            request_id=request.request_id,
            recommendations=result.get("recommendations", []),
            projected_savings=result.get("projected_savings", {}),
            implementation_plan=result.get("implementation_plan", {}),
            confidence_score=result.get("confidence", 0.0),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="cost_optimizer", status="error").inc()
        
        logger.error("cost_optimization_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cost optimization failed: {str(e)}"
        )


@app.post("/api/v1/predictive-analytics", response_model=PredictiveAnalyticsResponse)
async def predict_resource_usage(
    request: PredictiveAnalyticsRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Predict resource usage patterns."""
    start_time = time.time()
    
    try:
        logger.info("predictive_analytics_started", request_id=request.request_id)
        
        result = await predictive_analytics.predict_usage(
            historical_data=request.historical_data,
            prediction_horizon=request.prediction_horizon,
            metrics=request.metrics
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="predictive_analytics", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="predictive_analytics").observe(duration)
        
        logger.info("predictive_analytics_completed",
                   request_id=request.request_id,
                   duration_ms=round(duration * 1000, 2))
        
        return PredictiveAnalyticsResponse(
            request_id=request.request_id,
            predictions=result.get("predictions", []),
            trends=result.get("trends", {}),
            forecast_accuracy=result.get("forecast_accuracy", {}),
            confidence_intervals=result.get("confidence_intervals", {}),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="predictive_analytics", status="error").inc()
        
        logger.error("predictive_analytics_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Predictive analytics failed: {str(e)}"
        )


@app.post("/api/v1/sentiment-analysis", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze sentiment of compliance reports."""
    start_time = time.time()
    
    try:
        logger.info("sentiment_analysis_started", request_id=request.request_id)
        
        result = await sentiment_analyzer.analyze_sentiment(
            text=request.text,
            analysis_type=request.analysis_type,
            options=request.options
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="sentiment_analyzer", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="sentiment_analyzer").observe(duration)
        
        logger.info("sentiment_analysis_completed",
                   request_id=request.request_id,
                   sentiment=result.get("sentiment", "unknown"),
                   duration_ms=round(duration * 1000, 2))
        
        return SentimentAnalysisResponse(
            request_id=request.request_id,
            sentiment=result.get("sentiment", "neutral"),
            confidence_score=result.get("confidence", 0.0),
            emotions=result.get("emotions", {}),
            key_phrases=result.get("key_phrases", []),
            processing_time_ms=round(duration * 1000, 2)
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="sentiment_analyzer", status="error").inc()
        
        logger.error("sentiment_analysis_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sentiment analysis failed: {str(e)}"
        )


# Enhanced AI/ML Capability endpoints for Patent Implementation

@app.post("/api/v1/compliance-prediction", response_model=APIResponse)
async def predict_compliance(
    request: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Predict compliance violations using advanced temporal analysis and drift detection."""
    start_time = time.time()
    request_id = request.get('request_id', str(uuid.uuid4()))
    
    try:
        logger.info("compliance_prediction_started", request_id=request_id)
        
        # Extract parameters
        resource_data = request.get('resource_data', {})
        horizon_hours = request.get('horizon_hours', 24)
        
        # Perform compliance prediction
        prediction_results = await compliance_predictor.predict_compliance(
            resource_data=resource_data,
            horizon_hours=horizon_hours
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="compliance_predictor", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="compliance_predictor").observe(duration)
        
        logger.info("compliance_prediction_completed",
                   request_id=request_id,
                   duration_ms=round(duration * 1000, 2))
        
        return APIResponse(
            success=True,
            data=prediction_results,
            message="Compliance prediction completed successfully"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="compliance_predictor", status="error").inc()
        
        logger.error("compliance_prediction_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance prediction failed: {str(e)}"
        )


@app.post("/api/v1/cross-domain-correlation", response_model=APIResponse)
async def analyze_cross_domain_correlations(
    request: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze cross-domain correlations with graph neural networks and impact prediction."""
    start_time = time.time()
    request_id = request.get('request_id', str(uuid.uuid4()))
    
    try:
        logger.info("correlation_analysis_started", request_id=request_id)
        
        # Extract governance data
        governance_data = request.get('governance_data', {})
        
        # Perform correlation analysis
        correlation_results = await correlation_engine.analyze_governance_correlations(
            governance_data=governance_data
        )
        
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="correlation_engine", status="success").inc()
        MODEL_INFERENCE_DURATION.labels(model_name="correlation_engine").observe(duration)
        
        logger.info("correlation_analysis_completed",
                   request_id=request_id,
                   duration_ms=round(duration * 1000, 2))
        
        return APIResponse(
            success=True,
            data=correlation_results,
            message="Cross-domain correlation analysis completed successfully"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="correlation_engine", status="error").inc()
        
        logger.error("correlation_analysis_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-domain correlation analysis failed: {str(e)}"
        )


@app.post("/api/v1/train-compliance-model", response_model=APIResponse)
async def train_compliance_model(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Train the compliance prediction model with historical data."""
    try:
        logger.info("compliance_model_training_started")
        
        # Extract training data
        historical_data = request.get('historical_data', {})
        
        # Start training in background
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            compliance_predictor.train,
            historical_data
        )
        
        return APIResponse(
            success=True,
            data={
                'task_id': task_id,
                'status': 'training_started',
                'message': 'Compliance model training started'
            },
            message="Training initiated successfully"
        )
        
    except Exception as e:
        logger.error("compliance_model_training_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start compliance model training: {str(e)}"
        )


@app.post("/api/v1/drift-detection", response_model=APIResponse)
async def detect_configuration_drift(
    request: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Detect configuration drift using VAE-based drift detection."""
    start_time = time.time()
    request_id = request.get('request_id', str(uuid.uuid4()))
    
    try:
        logger.info("drift_detection_started", request_id=request_id)
        
        # Extract current configuration
        current_config = request.get('current_config', {})
        
        # Perform drift detection
        drift_results = await compliance_predictor.drift_detector.detect_drift(current_config)
        
        duration = time.time() - start_time
        
        logger.info("drift_detection_completed",
                   request_id=request_id,
                   drift_class=drift_results.get('drift_class'),
                   duration_ms=round(duration * 1000, 2))
        
        return APIResponse(
            success=True,
            data=drift_results,
            message="Configuration drift detection completed"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error("drift_detection_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}"
        )


@app.post("/api/v1/temporal-analysis", response_model=APIResponse)
async def analyze_temporal_patterns(
    request: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze temporal patterns in compliance data."""
    start_time = time.time()
    request_id = request.get('request_id', str(uuid.uuid4()))
    
    try:
        logger.info("temporal_analysis_started", request_id=request_id)
        
        # Extract time series data
        time_series_data = request.get('time_series', [])
        
        if not time_series_data:
            raise ValueError("Time series data is required")
        
        # Convert to pandas Series
        import pandas as pd
        ts_data = pd.Series(time_series_data)
        
        # Perform temporal analysis
        decomposition = await compliance_predictor.pattern_analyzer.decompose_time_series(ts_data)
        motifs = await compliance_predictor.pattern_analyzer.discover_motifs(ts_data.values)
        regime_changes = await compliance_predictor.pattern_analyzer.detect_regime_changes(ts_data.values)
        
        duration = time.time() - start_time
        
        results = {
            'decomposition': decomposition,
            'motifs': motifs[:5],  # Return top 5 motifs
            'regime_changes': regime_changes,
            'statistics': {
                'data_points': len(time_series_data),
                'motifs_found': len(motifs),
                'regime_changes_found': len(regime_changes)
            }
        }
        
        logger.info("temporal_analysis_completed",
                   request_id=request_id,
                   motifs_found=len(motifs),
                   regime_changes=len(regime_changes),
                   duration_ms=round(duration * 1000, 2))
        
        return APIResponse(
            success=True,
            data=results,
            message="Temporal pattern analysis completed"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error("temporal_analysis_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal analysis failed: {str(e)}"
        )


@app.get("/api/v1/governance-insights", response_model=APIResponse)
async def get_governance_insights(
    domains: Optional[str] = None,
    time_range: Optional[str] = "7d",
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get comprehensive governance insights across domains."""
    try:
        logger.info("governance_insights_requested",
                   domains=domains,
                   time_range=time_range)
        
        # Parse domains
        selected_domains = domains.split(',') if domains else ['policy', 'rbac', 'network', 'cost']
        
        # Generate mock insights (in production, this would query real data)
        insights = {
            'summary': {
                'total_resources': 1247,
                'compliance_score': 87.3,
                'risk_score': 6.2,
                'cost_trend': '+3.2%',
                'time_range': time_range
            },
            'domain_insights': {},
            'correlations': {
                'strong_correlations': [
                    {'domains': ['policy', 'cost'], 'strength': 0.82},
                    {'domains': ['rbac', 'network'], 'strength': 0.76}
                ],
                'risks': [
                    {'type': 'cascade_risk', 'score': 7.1, 'affected_domains': ['policy', 'rbac']},
                    {'type': 'volatility_risk', 'score': 5.8, 'affected_domains': ['cost']}
                ]
            },
            'recommendations': [
                {
                    'type': 'policy_optimization',
                    'priority': 'high',
                    'description': 'Consolidate overlapping policies to reduce complexity',
                    'expected_benefit': '15% reduction in compliance overhead'
                },
                {
                    'type': 'cost_optimization',
                    'priority': 'medium',
                    'description': 'Right-size underutilized resources',
                    'expected_benefit': '$12K monthly savings'
                }
            ]
        }
        
        # Add domain-specific insights
        for domain in selected_domains:
            insights['domain_insights'][domain] = {
                'status': 'healthy',
                'score': np.random.uniform(75, 95),
                'recent_changes': np.random.randint(5, 25),
                'alerts': np.random.randint(0, 5)
            }
        
        return APIResponse(
            success=True,
            data=insights,
            message="Governance insights retrieved successfully"
        )
        
    except Exception as e:
        logger.error("governance_insights_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve governance insights: {str(e)}"
        )


# Intelligent Automation Framework endpoints
@app.post("/api/v1/automation/trigger", response_model=APIResponse)
async def trigger_automation_workflow(
    trigger_event: Dict[str, Any],
    workflow_id: Optional[str] = None,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Trigger an automation workflow based on an event."""
    try:
        logger.info("automation_workflow_trigger_requested",
                   trigger_type=trigger_event.get('trigger_type'),
                   workflow_id=workflow_id)
        
        # Add user context to trigger event
        if user:
            trigger_event['triggered_by'] = user.get('id')
            trigger_event['user_permissions'] = user.get('permissions', [])
        
        # Trigger workflow
        execution_id = await workflow_engine.trigger_workflow(trigger_event, workflow_id)
        
        if execution_id:
            return APIResponse(
                success=True,
                data={
                    'execution_id': execution_id,
                    'status': 'triggered',
                    'workflow_id': workflow_id
                },
                message="Automation workflow triggered successfully"
            )
        else:
            return APIResponse(
                success=False,
                data={},
                message="No matching workflow found or conditions not met"
            )
        
    except Exception as e:
        logger.error("automation_trigger_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger automation workflow: {str(e)}"
        )


@app.get("/api/v1/automation/execution/{execution_id}", response_model=APIResponse)
async def get_automation_execution_status(
    execution_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get the status of an automation workflow execution."""
    try:
        execution_status = await workflow_engine.get_execution_status(execution_id)
        
        if execution_status:
            return APIResponse(
                success=True,
                data=execution_status,
                message="Execution status retrieved successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution not found: {execution_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_execution_status_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {str(e)}"
        )


@app.post("/api/v1/automation/execution/{execution_id}/cancel", response_model=APIResponse)
async def cancel_automation_execution(
    execution_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Cancel a running automation workflow execution."""
    try:
        success = await workflow_engine.cancel_execution(execution_id)
        
        if success:
            return APIResponse(
                success=True,
                data={'execution_id': execution_id, 'status': 'cancelled'},
                message="Execution cancelled successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Execution not found or cannot be cancelled: {execution_id}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("cancel_execution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )


@app.get("/api/v1/automation/workflows", response_model=APIResponse)
async def list_automation_workflows(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """List all available automation workflows."""
    try:
        workflows = []
        
        for workflow_id, workflow in workflow_engine.workflow_registry.items():
            workflows.append({
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'description': workflow.description,
                'trigger': workflow.trigger.value,
                'action_count': len(workflow.actions),
                'approval_required': workflow.approval_required,
                'auto_rollback': workflow.auto_rollback,
                'created_at': workflow.created_at.isoformat(),
                'created_by': workflow.created_by
            })
        
        return APIResponse(
            success=True,
            data={'workflows': workflows, 'total_count': len(workflows)},
            message="Workflows retrieved successfully"
        )
        
    except Exception as e:
        logger.error("list_workflows_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@app.get("/api/v1/automation/analytics", response_model=APIResponse)
async def get_automation_analytics(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get automation analytics and metrics."""
    try:
        analytics = await workflow_engine.get_workflow_analytics()
        
        return APIResponse(
            success=True,
            data=analytics,
            message="Automation analytics retrieved successfully"
        )
        
    except Exception as e:
        logger.error("get_automation_analytics_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automation analytics: {str(e)}"
        )


@app.post("/api/v1/automation/simulate", response_model=APIResponse)
async def simulate_automation_workflow(
    trigger_event: Dict[str, Any],
    workflow_id: Optional[str] = None,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Simulate an automation workflow without executing actions."""
    try:
        logger.info("automation_simulation_requested",
                   trigger_type=trigger_event.get('trigger_type'),
                   workflow_id=workflow_id)
        
        # Find appropriate workflow
        if workflow_id:
            workflow = workflow_engine.workflow_registry.get(workflow_id)
            if not workflow:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workflow not found: {workflow_id}"
                )
        else:
            trigger_type = AutomationTrigger(trigger_event.get('trigger_type', 'manual'))
            workflow = None
            for w in workflow_engine.workflow_registry.values():
                if w.trigger == trigger_type:
                    workflow = w
                    break
            
            if not workflow:
                return APIResponse(
                    success=False,
                    data={},
                    message="No matching workflow found for trigger"
                )
        
        # Check conditions
        conditions_met = await workflow_engine._evaluate_conditions(workflow.conditions, trigger_event)
        
        # Simulate execution plan
        simulation_result = {
            'workflow_id': workflow.workflow_id,
            'workflow_name': workflow.name,
            'conditions_met': conditions_met,
            'estimated_duration': f"{len(workflow.actions) * 30} seconds",
            'action_plan': [
                {
                    'action_id': action.action_id,
                    'name': action.name,
                    'action_type': action.action_type,
                    'target_resource': action.target_resource,
                    'priority': action.priority.value,
                    'dependencies': action.dependencies,
                    'approval_required': getattr(action, 'approval_required', False)
                }
                for action in workflow.actions
            ],
            'risk_assessment': {
                'rollback_available': workflow.auto_rollback,
                'approval_required': workflow.approval_required,
                'high_risk_actions': len([a for a in workflow.actions if a.priority.value in ['critical', 'high']])
            }
        }
        
        return APIResponse(
            success=True,
            data=simulation_result,
            message="Workflow simulation completed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("automation_simulation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to simulate workflow: {str(e)}"
        )


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
        workers=1,  # Single worker for development
        reload=settings.debug,
        log_level=settings.monitoring.log_level.lower()
    )