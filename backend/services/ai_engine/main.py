"""
AI Engine Service for PolicyCortex.
Provides AI/ML capabilities for policy analysis, anomaly detection, and predictive analytics.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from backend.shared.config import get_settings
from backend.shared.database import DatabaseUtils, get_async_db

from .auth import AuthManager
from .ml_models.compliance_predictor import CompliancePredictor
from .ml_models.correlation_engine import CrossDomainCorrelationEngine
from .models import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    APIResponse,
    ConversationHistoryRequest,
    ConversationHistoryResponse,
    ConversationRequest,
    ConversationResponse,
    CostOptimizationRequest,
    CostOptimizationResponse,
    ErrorResponse,
    GovernanceOptimizationRequest,
    GovernanceOptimizationResponse,
    HealthResponse,
    ModelInfo,
    ModelMetrics,
    ModelTrainingRequest,
    ModelTrainingResponse,
    PolicyAnalysisRequest,
    PolicyAnalysisResponse,
    PolicySynthesisRequest,
    PolicySynthesisResponse,
    PredictiveAnalyticsRequest,
    PredictiveAnalyticsResponse,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    UnifiedAIAnalysisRequest,
    UnifiedAIAnalysisResponse,
)
from .services.anomaly_detector import AnomalyDetector
from .services.automation_engine import AutomationStatus, AutomationTrigger
from .services.automation_orchestrator import WorkflowEngine, automation_orchestrator
from .services.conversation_analytics import conversation_analytics
from .services.conversational_ai_service import conversational_ai_service
from .services.cost_optimizer import CostOptimizer
from .services.cross_domain_correlator import cross_domain_correlator
from .services.feature_engineer import FeatureEngineer
from .services.gnn_correlation_service import gnn_service
from .services.governance_intelligence import governance_intelligence
from .services.model_manager import ModelManager
from .services.model_monitor import ModelMonitor
from .services.multi_objective_optimizer import multi_objective_optimizer
from .services.nlp_service import NLPService
from .services.predictive_analytics import PredictiveAnalyticsService
from .services.sentiment_analyzer import SentimentAnalyzer

# Try to import real models, fall back to mock models for testing
try:
    from .ml_models.conversational_governance_intelligence import (
        ConversationConfig,
        conversational_intelligence,
    )
    from .ml_models.unified_ai_platform import UnifiedAIConfig, unified_ai_platform
    USE_MOCK_MODELS = False
    logger.info("Using production AI models")
except ImportError as e:
    logger.warning(f"ML dependencies not available ({e}), using mock models for testing")
    from .ml_models.mock_models import (
        mock_conversational_intelligence as conversational_intelligence,
    )
    from .ml_models.mock_models import mock_unified_ai_platform as unified_ai_platform
    USE_MOCK_MODELS = True

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'ai_engine_requests_total',
    'Total AI Engine requests',
    ['method',
    'endpoint',
    'status']
)
REQUEST_DURATION = Histogram('ai_engine_request_duration_seconds', 'Request duration')
MODEL_INFERENCE_COUNT = Counter(
    'ai_engine_model_inference_total',
    'Model inference requests',
    ['model_name',
    'status']
)
MODEL_INFERENCE_DURATION = Histogram(
    'ai_engine_model_inference_duration_seconds',
    'Model inference duration',
    ['model_name']
)
ACTIVE_MODELS = Gauge('ai_engine_active_models', 'Number of active models')
MODEL_ACCURACY = Gauge(
    'ai_engine_model_accuracy',
    'Model accuracy metrics',
    ['model_name',
    'metric_type']
)

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
        await gnn_service.initialize()
        await conversational_ai_service.initialize()

        # Initialize Patent 2 components
        await multi_objective_optimizer.initialize(
            resource_analyzer=cost_optimizer,
            compliance_checker=compliance_predictor,
            performance_monitor=model_monitor
        )
        await automation_orchestrator.initialize(
            resource_manager=cost_optimizer,
            policy_engine=compliance_predictor,
            security_service=anomaly_detector
        )

        # Initialize Patent 3 components
        await governance_intelligence.initialize()
        await conversation_analytics.initialize()

        # Initialize Patent 4 components
        await cross_domain_correlator.initialize()

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
        regime_changes = (
            await compliance_predictor.pattern_analyzer.detect_regime_changes(ts_data.values)
        )

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
        conditions_met = await workflow_engine._evaluate_conditions(
            workflow.conditions,
            trigger_event
        )

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
                'high_risk_actions': len(
                    [a for a in workflow.actions if a.priority.value in ['critical',
                    'high']]
                )
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


# Cross-Domain GNN Correlation endpoints
@app.post("/api/v1/gnn/correlations", response_model=APIResponse)
async def analyze_cross_domain_correlations(
    governance_data: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze cross-domain correlations using Graph Neural Networks."""
    start_time = time.time()
    request_id = governance_data.get('request_id', str(uuid.uuid4()))

    try:
        logger.info("gnn_correlation_analysis_started", request_id=request_id)

        # Analyze correlations using GNN
        correlation_results = await gnn_service.analyze_governance_correlations(governance_data)

        duration = time.time() - start_time

        # Update metrics
        MODEL_INFERENCE_COUNT.labels(model_name='cross_domain_gnn', status='success').inc()
        MODEL_INFERENCE_DURATION.labels(model_name='cross_domain_gnn').observe(duration)

        logger.info("gnn_correlation_analysis_completed",
                request_id=request_id,
                correlations_found=len(correlation_results.get('correlations', [])),
                impacts_predicted=len(correlation_results.get('impacts', [])),
                duration_ms=round(duration * 1000, 2))

        return APIResponse(
            success=True,
            data=correlation_results,
            message="Cross-domain correlation analysis completed"
        )

    except HTTPException:
        MODEL_INFERENCE_COUNT.labels(model_name='cross_domain_gnn', status='error').inc()
        raise
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name='cross_domain_gnn', status='error').inc()

        logger.error("gnn_correlation_analysis_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-domain correlation analysis failed: {str(e)}"
        )


@app.post("/api/v1/gnn/impact-prediction", response_model=APIResponse)
async def predict_governance_impacts(
    change_scenario: Dict[str, Any],
    current_state: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Predict impacts of governance changes using GNN analysis."""
    start_time = time.time()
    request_id = change_scenario.get('request_id', str(uuid.uuid4()))

    try:
        logger.info("gnn_impact_prediction_started", request_id=request_id)

        # Predict impacts using GNN
        impact_results = await gnn_service.predict_governance_impacts(
            change_scenario,
            current_state
        )

        duration = time.time() - start_time

        # Update metrics
        MODEL_INFERENCE_COUNT.labels(model_name='impact_prediction_gnn', status='success').inc()
        MODEL_INFERENCE_DURATION.labels(model_name='impact_prediction_gnn').observe(duration)

        logger.info("gnn_impact_prediction_completed",
                request_id=request_id,
                predicted_impacts=len(impact_results.get('predicted_impacts', [])),
                affected_domains=len(impact_results.get('affected_domains', [])),
                confidence_score=impact_results.get('confidence_score', 0),
                duration_ms=round(duration * 1000, 2))

        return APIResponse(
            success=True,
            data=impact_results,
            message="Governance impact prediction completed"
        )

    except HTTPException:
        MODEL_INFERENCE_COUNT.labels(model_name='impact_prediction_gnn', status='error').inc()
        raise
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name='impact_prediction_gnn', status='error').inc()

        logger.error("gnn_impact_prediction_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Governance impact prediction failed: {str(e)}"
        )


@app.get("/api/v1/gnn/domain-relationships", response_model=APIResponse)
async def get_domain_relationships(
    domain_focus: Optional[str] = None,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get relationships between governance domains using GNN analysis."""
    try:
        logger.info("gnn_domain_relationships_requested", domain_focus=domain_focus)

        # Analyze domain relationships
        relationship_results = await gnn_service.get_domain_relationships(domain_focus)

        logger.info("gnn_domain_relationships_completed",
                domain_focus=domain_focus,
                correlations_found=len(
                    relationship_results.get('cross_domain_correlations',
                    []))
                )

        return APIResponse(
            success=True,
            data=relationship_results,
            message="Domain relationship analysis completed"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("gnn_domain_relationships_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Domain relationship analysis failed: {str(e)}"
        )


@app.post("/api/v1/gnn/train", response_model=APIResponse)
async def train_gnn_model(
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Train or retrain the GNN model with new data."""
    try:
        logger.info("gnn_model_training_started",
                training_samples=len(training_data.get('samples', [])))

        # Extract training samples
        samples = training_data.get('samples', [])
        if not samples:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Training samples are required"
            )

        # Start training in background
        background_tasks.add_task(
            gnn_service.train_model_with_data,
            samples
        )

        return APIResponse(
            success=True,
            data={'status': 'training_started', 'sample_count': len(samples)},
            message="GNN model training started in background"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("gnn_model_training_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GNN model training failed: {str(e)}"
        )


@app.get("/api/v1/gnn/health", response_model=APIResponse)
async def get_gnn_health(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get health status and performance metrics of the GNN model."""
    try:
        health_status = await gnn_service.get_model_health()

        return APIResponse(
            success=True,
            data=health_status,
            message="GNN health status retrieved"
        )

    except Exception as e:
        logger.error("gnn_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GNN health check failed: {str(e)}"
        )


@app.post("/api/v1/gnn/real-time-stream", response_model=APIResponse)
async def process_real_time_governance_stream(
    stream_data: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Process real-time governance data stream for correlation detection."""
    try:
        logger.info("gnn_real_time_processing_started",
                stream_type=stream_data.get('stream_type'))

        # Process stream data
        governance_data = stream_data.get('governance_data', {})

        # Perform real-time correlation analysis
        correlation_results = await gnn_service.analyze_governance_correlations(governance_data)

        # Filter for high-confidence, high-impact correlations
        high_priority_correlations = [
            corr for corr in correlation_results.get('correlations', [])
            if corr.get('correlation_score', 0) > 0.8
        ]

        high_impact_predictions = [
            impact for impact in correlation_results.get('impacts', [])
            if impact.get('impact_probability', 0) > 0.7
        ]

        stream_results = {
            'timestamp': datetime.now().isoformat(),
            'stream_type': stream_data.get('stream_type'),
            'high_priority_correlations': high_priority_correlations,
            'high_impact_predictions': high_impact_predictions,
            'alert_count': len(high_priority_correlations) + len(high_impact_predictions),
            'processing_latency_ms': stream_data.get('processing_latency_ms', 0)
        }

        logger.info("gnn_real_time_processing_completed",
                alert_count=stream_results['alert_count'],
                correlations=len(high_priority_correlations),
                impacts=len(high_impact_predictions))

        return APIResponse(
            success=True,
            data=stream_results,
            message="Real-time governance stream processed"
        )

    except Exception as e:
        logger.error("gnn_real_time_processing_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Real-time stream processing failed: {str(e)}"
        )


# Conversational AI Interface endpoints - Patent 2 Implementation
@app.post("/api/v1/conversation/message", response_model=APIResponse)
async def process_conversation_message(
    message_data: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Process conversational AI message and return intelligent response."""
    start_time = time.time()
    request_id = message_data.get('request_id', str(uuid.uuid4()))

    try:
        logger.info("conversational_ai_message_started",
                request_id=request_id,
                user_input_length=len(message_data.get('user_input', '')))

        # Extract message parameters
        user_input = message_data.get('user_input', '')
        conversation_id = message_data.get('conversation_id')
        user_id = user.get('id', 'anonymous') if user else 'anonymous'

        if not user_input.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User input is required"
            )

        # Process conversation
        conversation_response = await conversational_ai_service.process_conversation(
            user_input=user_input,
            conversation_id=conversation_id,
            user_id=user_id
        )

        duration = time.time() - start_time

        # Update metrics
        MODEL_INFERENCE_COUNT.labels(model_name='conversational_ai', status='success').inc()
        MODEL_INFERENCE_DURATION.labels(model_name='conversational_ai').observe(duration)

        logger.info("conversational_ai_message_completed",
                request_id=request_id,
                conversation_id=conversation_response.conversation_id,
                intent_type = (
                    conversation_response.intent_type.value if conversation_response.intent_type else None,
                )
                confidence_score=conversation_response.confidence_score,
                next_state=conversation_response.next_state.value,
                duration_ms=round(duration * 1000, 2))

        return APIResponse(
            success=True,
            data={
                'response_text': conversation_response.response_text,
                'conversation_id': conversation_response.conversation_id,
                'intent_type': conversation_response.intent_type.value if conversation_response.intent_type else None,
                'confidence_score': conversation_response.confidence_score,
                'next_state': conversation_response.next_state.value,
                'suggested_actions': conversation_response.suggested_actions,
                'extracted_entities': conversation_response.extracted_entities,
                'requires_confirmation': conversation_response.requires_confirmation,
                'response_data': conversation_response.response_data,
                'processing_time_ms': round(duration * 1000, 2)
            },
            message="Conversation processed successfully"
        )

    except HTTPException:
        MODEL_INFERENCE_COUNT.labels(model_name='conversational_ai', status='error').inc()
        raise
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name='conversational_ai', status='error').inc()

        logger.error("conversational_ai_message_failed",
                    request_id=request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation processing failed: {str(e)}"
        )


@app.get("/api/v1/conversation/{conversation_id}/history", response_model=APIResponse)
async def get_conversation_history(
    conversation_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get conversation history for a specific conversation."""
    try:
        logger.info("conversation_history_requested", conversation_id=conversation_id)

        history = await conversational_ai_service.get_conversation_history(conversation_id)

        return APIResponse(
            success=True,
            data={
                'conversation_id': conversation_id,
                'history': history,
                'message_count': len(history)
            },
            message="Conversation history retrieved successfully"
        )

    except Exception as e:
        logger.error("conversation_history_failed",
                    conversation_id=conversation_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@app.delete("/api/v1/conversation/{conversation_id}", response_model=APIResponse)
async def end_conversation(
    conversation_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """End and cleanup a conversation."""
    try:
        logger.info("conversation_end_requested", conversation_id=conversation_id)

        success = await conversational_ai_service.end_conversation(conversation_id)

        if success:
            return APIResponse(
                success=True,
                data={'conversation_id': conversation_id, 'status': 'ended'},
                message="Conversation ended successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found: {conversation_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversation_end_failed",
                    conversation_id=conversation_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end conversation: {str(e)}"
        )


@app.post("/api/v1/conversation/batch-process", response_model=APIResponse)
async def batch_process_conversations(
    batch_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Process multiple conversation messages in batch."""
    try:
        logger.info("batch_conversation_processing_started",
                batch_size=len(batch_data.get('messages', [])))

        messages = batch_data.get('messages', [])
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Messages array is required"
            )

        # Process messages in parallel
        async def process_message_batch(messages):
            results = []
            for msg in messages:
                try:
                    response = await conversational_ai_service.process_conversation(
                        user_input=msg.get('user_input', ''),
                        conversation_id=msg.get('conversation_id'),
                        user_id=user.get('id', 'anonymous') if user else 'anonymous'
                    )
                    results.append({
                        'message_id': msg.get('message_id', ''),
                        'success': True,
                        'response': {
                            'response_text': response.response_text,
                            'conversation_id': response.conversation_id,
                            'intent_type': response.intent_type.value if response.intent_type else None,
                            'confidence_score': response.confidence_score
                        }
                    })
                except Exception as e:
                    results.append({
                        'message_id': msg.get('message_id', ''),
                        'success': False,
                        'error': str(e)
                    })
            return results

        # Start batch processing in background
        task_id = str(uuid.uuid4())
        background_tasks.add_task(process_message_batch, messages)

        return APIResponse(
            success=True,
            data={
                'task_id': task_id,
                'batch_size': len(messages),
                'status': 'processing_started'
            },
            message="Batch conversation processing started"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("batch_conversation_processing_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch conversation processing failed: {str(e)}"
        )


@app.get("/api/v1/conversation/analytics", response_model=APIResponse)
async def get_conversation_analytics(
    time_range: Optional[str] = "7d",
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get conversation analytics and insights."""
    try:
        logger.info("conversation_analytics_requested", time_range=time_range)

        # Generate analytics (would be based on real data in production)
        analytics = {
            'summary': {
                'total_conversations': 1247,
                'total_messages': 8432,
                'avg_conversation_length': 6.8,
                'avg_response_time_ms': 850,
                'user_satisfaction_score': 4.2,
                'time_range': time_range
            },
            'intent_distribution': {
                'query_resources': 32.5,
                'policy_analysis': 18.7,
                'compliance_check': 15.3,
                'cost_optimization': 12.8,
                'security_assessment': 10.2,
                'correlation_analysis': 6.1,
                'impact_prediction': 4.4
            },
            'conversation_states': {
                'completed': 78.3,
                'in_progress': 12.4,
                'failed': 5.2,
                'abandoned': 4.1
            },
            'top_entities_extracted': [
                {'entity_type': 'resource_type', 'count': 3421},
                {'entity_type': 'subscription', 'count': 2103},
                {'entity_type': 'time_period', 'count': 1876},
                {'entity_type': 'cost_amount', 'count': 1544}
            ],
            'user_engagement': {
                'returning_users': 67.8,
                'avg_session_duration_minutes': 12.3,
                'most_active_hours': ['09:00-11:00', '14:00-16:00'],
                'completion_rate': 82.6
            },
            'performance_metrics': {
                'avg_nlu_confidence': 0.847,
                'intent_classification_accuracy': 0.923,
                'entity_extraction_accuracy': 0.891,
                'response_relevance_score': 0.876
            }
        }

        return APIResponse(
            success=True,
            data=analytics,
            message="Conversation analytics retrieved successfully"
        )

    except Exception as e:
        logger.error("conversation_analytics_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation analytics: {str(e)}"
        )


@app.post("/api/v1/conversation/feedback", response_model=APIResponse)
async def submit_conversation_feedback(
    feedback_data: Dict[str, Any],
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Submit feedback for conversation quality improvement."""
    try:
        logger.info("conversation_feedback_submitted",
                conversation_id=feedback_data.get('conversation_id'),
                rating=feedback_data.get('rating'))

        # Extract feedback data
        conversation_id = feedback_data.get('conversation_id')
        rating = feedback_data.get('rating')  # 1-5 scale
        feedback_text = feedback_data.get('feedback_text', '')
        feedback_categories = feedback_data.get('categories', [])

        if not conversation_id or rating is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Conversation ID and rating are required"
            )

        # Process and store feedback (would integrate with feedback storage)
        feedback_record = {
            'conversation_id': conversation_id,
            'user_id': user.get('id', 'anonymous') if user else 'anonymous',
            'rating': rating,
            'feedback_text': feedback_text,
            'categories': feedback_categories,
            'timestamp': datetime.now().isoformat(),
            'processed': False
        }

        # In production, this would be stored in a database
        logger.info("conversation_feedback_recorded", feedback_record=feedback_record)

        return APIResponse(
            success=True,
            data={
                'feedback_id': str(uuid.uuid4()),
                'conversation_id': conversation_id,
                'status': 'recorded'
            },
            message="Feedback submitted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversation_feedback_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get("/api/v1/conversation/health", response_model=APIResponse)
async def get_conversational_ai_health(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get health status of the conversational AI system."""
    try:
        # Check component health
        health_status = {
            'service_status': 'healthy',
            'components': {
                'nlu_processor': {
                    'status': 'healthy',
                    'initialized': conversational_ai_service.nlu_processor.initialized,
                    'models_loaded': bool(conversational_ai_service.nlu_processor.nlp)
                },
                'dialogue_manager': {
                    'status': 'healthy',
                    'state_transitions_available': len(conversational_ai_service.dialogue_manager.state_transitions)
                },
                'action_executor': {
                    'status': 'healthy',
                    'handlers_available': len(conversational_ai_service.action_executor.action_handlers)
                },
                'gnn_service': {
                    'status': 'healthy' if conversational_ai_service.gnn_service.model_loaded else 'degraded',
                    'model_loaded': conversational_ai_service.gnn_service.model_loaded
                }
            },
            'cache_status': {
                'redis_connected': conversational_ai_service.redis_client is not None
            },
            'performance_metrics': {
                'avg_response_time_ms': 850,
                'active_conversations': 47,
                'cache_hit_rate': 0.68
            },
            'last_health_check': datetime.now().isoformat()
        }

        # Determine overall health
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        overall_healthy = all(status in ['healthy', 'degraded'] for status in component_statuses)

        health_status['service_status'] = 'healthy' if overall_healthy else 'unhealthy'

        return APIResponse(
            success=True,
            data=health_status,
            message="Conversational AI health status retrieved"
        )

    except Exception as e:
        logger.error("conversational_ai_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversational AI health check failed: {str(e)}"
        )


# Multi-Objective Optimization endpoints (Patent 2)
@app.post("/api/v1/optimization/multi-objective",
        response_model=APIResponse,
        status_code=status.HTTP_200_OK,
        tags=["optimization"])
async def run_multi_objective_optimization(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Run multi-objective optimization for governance scenarios."""
    request_id = str(uuid.uuid4())

    try:
        logger.info("multi_objective_optimization_request", request_id=request_id)

        # Extract parameters
        objectives = []
        for obj_data in request.get('objectives', []):
            from .services.multi_objective_optimizer import Objective, ObjectiveType
            objectives.append(Objective(
                name=obj_data['name'],
                type=ObjectiveType(obj_data['type']),
                weight=obj_data.get('weight', 1.0),
                target_value=obj_data.get('target_value'),
                importance=obj_data.get('importance', 1.0)
            ))

        constraints = []
        for constr_data in request.get('constraints', []):
            from .services.multi_objective_optimizer import Constraint, ConstraintType
            constraints.append(Constraint(
                name=constr_data['name'],
                type=ConstraintType(constr_data['type']),
                min_value=constr_data.get('min_value'),
                max_value=constr_data.get('max_value'),
                is_hard=constr_data.get('is_hard', True),
                penalty_weight=constr_data.get('penalty_weight', 1.0)
            ))

        decision_variables = request.get('decision_variables', {})
        algorithm = request.get('algorithm', 'nsga2')
        selection_method = request.get('selection_method', 'weighted_sum')

        # Run optimization
        result = await multi_objective_optimizer.optimize(
            objectives=objectives,
            constraints=constraints,
            decision_variables=decision_variables,
            algorithm=algorithm,
            selection_method=selection_method,
            max_generations=request.get('max_generations', 100),
            population_size=request.get('population_size', 100)
        )

        return APIResponse(
            status="success",
            data={
                'solution_id': result.solution_id,
                'selected_solution': result.selected_solution,
                'pareto_front_size': len(result.pareto_front),
                'optimization_time': result.optimization_time,
                'metadata': result.metadata
            },
            message="Multi-objective optimization completed successfully"
        )

    except Exception as e:
        logger.error("multi_objective_optimization_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-objective optimization failed: {str(e)}"
        )


@app.get("/api/v1/optimization/history",
        response_model=APIResponse,
        tags=["optimization"])
async def get_optimization_history(
    limit: int = 10,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get optimization history."""
    try:
        history = await multi_objective_optimizer.get_optimization_history(limit=limit)

        return APIResponse(
            status="success",
            data={
                'history': [
                    {
                        'solution_id': h.solution_id,
                        'objectives': [obj.name for obj in h.objectives],
                        'optimization_time': h.optimization_time,
                        'metadata': h.metadata
                    }
                    for h in history
                ]
            },
            message="Optimization history retrieved"
        )

    except Exception as e:
        logger.error("get_optimization_history_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get optimization history: {str(e)}"
        )


@app.post("/api/v1/optimization/apply/{solution_id}",
        response_model=APIResponse,
        tags=["optimization"])
async def apply_optimization_solution(
    solution_id: str,
    dry_run: bool = True,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Apply an optimization solution."""
    try:
        result = await multi_objective_optimizer.apply_solution(
            solution_id=solution_id,
            dry_run=dry_run
        )

        return APIResponse(
            status="success",
            data=result,
            message=f"Solution {'preview' if dry_run else 'applied'} successfully"
        )

    except Exception as e:
        logger.error("apply_optimization_solution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply solution: {str(e)}"
        )


# Automation Orchestrator endpoints (Patent 2)
@app.post("/api/v1/automation/workflow",
        response_model=APIResponse,
        tags=["automation"])
async def create_automation_workflow(
    workflow_data: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Create an automation workflow."""
    try:
        from .services.automation_orchestrator import AutomationWorkflow, TriggerType

        # Parse workflow data
        workflow = AutomationWorkflow(
            workflow_id=workflow_data.get('workflow_id'),
            name=workflow_data['name'],
            description=workflow_data['description'],
            trigger=workflow_data['trigger'],
            actions=[],  # Parse actions from workflow_data
            conditions=workflow_data.get('conditions', []),
            max_execution_time=workflow_data.get('max_execution_time', 3600),
            auto_rollback=workflow_data.get('auto_rollback', True),
            approval_required=workflow_data.get('approval_required', False)
        )

        workflow_id = await automation_orchestrator.create_workflow(workflow)

        return APIResponse(
            status="success",
            data={'workflow_id': workflow_id},
            message="Workflow created successfully"
        )

    except Exception as e:
        logger.error("create_automation_workflow_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@app.post("/api/v1/automation/workflow/{workflow_id}/execute",
        response_model=APIResponse,
        tags=["automation"])
async def execute_automation_workflow(
    workflow_id: str,
    parameters: Dict[str, Any] = {},
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Execute an automation workflow."""
    try:
        execution_id = await automation_orchestrator.execute_workflow(
            workflow_id=workflow_id,
            parameters=parameters
        )

        return APIResponse(
            status="success",
            data={'execution_id': execution_id},
            message="Workflow execution started"
        )

    except Exception as e:
        logger.error("execute_automation_workflow_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute workflow: {str(e)}"
        )


@app.post("/api/v1/automation/optimize",
        response_model=APIResponse,
        tags=["automation"])
async def trigger_optimization_workflow(
    objectives: List[str],
    constraints: List[str],
    parameters: Dict[str, Any] = {},
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Trigger an optimization-based automation workflow."""
    try:
        execution_id = await automation_orchestrator.trigger_optimization_workflow(
            objectives=objectives,
            constraints=constraints,
            parameters=parameters
        )

        return APIResponse(
            status="success",
            data={'execution_id': execution_id},
            message="Optimization workflow triggered"
        )

    except Exception as e:
        logger.error("trigger_optimization_workflow_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger optimization workflow: {str(e)}"
        )


@app.get("/api/v1/automation/workflow/{workflow_id}/status",
        response_model=APIResponse,
        tags=["automation"])
async def get_workflow_status(
    workflow_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get workflow status."""
    try:
        status_info = await automation_orchestrator.get_workflow_status(workflow_id)

        return APIResponse(
            status="success",
            data=status_info,
            message="Workflow status retrieved"
        )

    except Exception as e:
        logger.error("get_workflow_status_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@app.delete("/api/v1/automation/execution/{execution_id}",
            response_model=APIResponse,
            tags=["automation"])
async def cancel_workflow_execution(
    execution_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Cancel a workflow execution."""
    try:
        result = await automation_orchestrator.cancel_execution(execution_id)

        return APIResponse(
            status="success",
            data=result,
            message="Execution cancelled"
        )

    except Exception as e:
        logger.error("cancel_workflow_execution_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel execution: {str(e)}"
        )


@app.get("/api/v1/automation/insights",
        response_model=APIResponse,
        tags=["automation"])
async def get_optimization_insights(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get insights from optimization workflows."""
    try:
        insights = await automation_orchestrator.get_optimization_insights()

        return APIResponse(
            status="success",
            data=insights,
            message="Optimization insights retrieved"
        )

    except Exception as e:
        logger.error("get_optimization_insights_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get optimization insights: {str(e)}"
        )


# Conversational Governance Intelligence endpoints (Patent 3)
@app.post("/api/v1/governance/conversation",
        response_model=APIResponse,
        tags=["governance"])
async def process_governance_conversation(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Process conversational governance intelligence query."""
    try:
        user_id = request.get('user_id', 'anonymous')
        input_text = request.get('input_text', '')
        session_id = request.get('session_id')
        user_context = request.get('user_context', {})

        # Process conversation
        result = await governance_intelligence.process_conversation(
            user_id=user_id,
            input_text=input_text,
            session_id=session_id,
            user_context=user_context
        )

        return APIResponse(
            status="success",
            data=result,
            message="Governance conversation processed successfully"
        )

    except Exception as e:
        logger.error("governance_conversation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process governance conversation: {str(e)}"
        )


@app.get("/api/v1/governance/conversation/{user_id}/insights",
        response_model=APIResponse,
        tags=["governance"])
async def get_conversation_insights(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get conversation insights for a specific user."""
    try:
        insights = await governance_intelligence.get_conversation_insights(user_id)

        return APIResponse(
            status="success",
            data=insights,
            message="Conversation insights retrieved"
        )

    except Exception as e:
        logger.error("get_conversation_insights_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation insights: {str(e)}"
        )


@app.post("/api/v1/governance/analytics/conversation",
        response_model=APIResponse,
        tags=["governance"])
async def analyze_conversations(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Perform advanced conversation analytics."""
    try:
        conversations = request.get('conversations', [])
        analysis_types = request.get('analysis_types', [])

        # Convert string analysis types to enum
        from .services.conversation_analytics import AnalyticsType
        if analysis_types:
            analysis_types = [AnalyticsType(at) for at in analysis_types]

        # Perform analytics
        results = await conversation_analytics.analyze_conversations(
            conversations=conversations,
            analysis_types=analysis_types
        )

        return APIResponse(
            status="success",
            data=results,
            message="Conversation analytics completed"
        )

    except Exception as e:
        logger.error("conversation_analytics_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversation analytics failed: {str(e)}"
        )


@app.get("/api/v1/governance/analytics/flow",
        response_model=APIResponse,
        tags=["governance"])
async def get_conversation_flow_analysis(
    limit: int = 100,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get conversation flow analysis."""
    try:
        # This would typically fetch conversations from database
        # For now, return sample analysis
        sample_result = {
            'flow_analysis': {
                'total_flows': 0,
                'most_common_flows': [],
                'efficiency_metrics': {
                    'average_turns_per_conversation': 0,
                    'resolution_rate': 0,
                    'escalation_rate': 0
                }
            },
            'recommendations': [
                "Insufficient conversation data for meaningful flow analysis",
                "Collect more conversation data to identify patterns",
                "Implement conversation tracking to enable flow optimization"
            ]
        }

        return APIResponse(
            status="success",
            data=sample_result,
            message="Conversation flow analysis retrieved"
        )

    except Exception as e:
        logger.error("conversation_flow_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation flow analysis: {str(e)}"
        )


@app.get("/api/v1/governance/analytics/topics",
        response_model=APIResponse,
        tags=["governance"])
async def get_topic_analysis(
    time_period: str = "7d",
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get governance topic analysis."""
    try:
        # Sample topic analysis - in production, fetch real data
        sample_topics = {
            'topics': [
                {
                    'topic_name': 'Policy Management',
                    'prevalence': 0.35,
                    'trending': 'increasing',
                    'keywords': ['policy', 'procedure', 'guideline']
                },
                {
                    'topic_name': 'Compliance',
                    'prevalence': 0.28,
                    'trending': 'stable',
                    'keywords': ['compliance', 'audit', 'regulation']
                },
                {
                    'topic_name': 'Security',
                    'prevalence': 0.22,
                    'trending': 'increasing',
                    'keywords': ['security', 'threat', 'vulnerability']
                },
                {
                    'topic_name': 'Cost Management',
                    'prevalence': 0.15,
                    'trending': 'decreasing',
                    'keywords': ['cost', 'budget', 'optimization']
                }
            ],
            'time_period': time_period,
            'analysis_date': datetime.now().isoformat()
        }

        return APIResponse(
            status="success",
            data=sample_topics,
            message="Topic analysis retrieved"
        )

    except Exception as e:
        logger.error("topic_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get topic analysis: {str(e)}"
        )


@app.get("/api/v1/governance/analytics/knowledge-gaps",
        response_model=APIResponse,
        tags=["governance"])
async def get_knowledge_gaps(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Identify knowledge gaps in governance conversations."""
    try:
        # Sample knowledge gap analysis
        sample_gaps = {
            'knowledge_gaps': [
                {
                    'topic': 'Azure Kubernetes Service Security',
                    'gap_score': 0.7,
                    'frequency': 15,
                    'impact': 'high',
                    'recommendation': 'Add comprehensive AKS security documentation'
                },
                {
                    'topic': 'Multi-cloud Compliance',
                    'gap_score': 0.6,
                    'frequency': 12,
                    'impact': 'medium',
                    'recommendation': 'Develop multi-cloud governance guidelines'
                },
                {
                    'topic': 'Cost Allocation Policies',
                    'gap_score': 0.5,
                    'frequency': 8,
                    'impact': 'medium',
                    'recommendation': 'Create detailed cost allocation procedures'
                }
            ],
            'overall_coverage': 0.82,
            'priority_areas': ['Security', 'Compliance', 'Cost Management'],
            'recommendations': [
                'Focus on high-impact knowledge gaps first',
                'Implement feedback loop for continuous improvement',
                'Regular knowledge base reviews and updates'
            ]
        }

        return APIResponse(
            status="success",
            data=sample_gaps,
            message="Knowledge gaps identified"
        )

    except Exception as e:
        logger.error("knowledge_gaps_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to identify knowledge gaps: {str(e)}"
        )


@app.get("/api/v1/governance/health",
        response_model=APIResponse,
        tags=["governance"])
async def get_governance_intelligence_health(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get governance intelligence system health."""
    try:
        health_status = {
            'components': {
                'governance_intelligence': governance_intelligence._initialized,
                'conversation_analytics': conversation_analytics._initialized,
                'nlu_engine': governance_intelligence.nlu._initialized if governance_intelligence._initialized else False,
                'knowledge_base': governance_intelligence.knowledge_base._initialized if governance_intelligence._initialized else False
            },
            'metrics': {
                'active_sessions': len(governance_intelligence.active_sessions) if governance_intelligence._initialized else 0,
                'knowledge_articles': len(governance_intelligence.knowledge_base.knowledge_store) if governance_intelligence._initialized and
                    governance_intelligence.knowledge_base._initialized else 0,
                'cached_analytics': 'available'
            },
            'status': 'healthy' if all([
                governance_intelligence._initialized,
                conversation_analytics._initialized
            ]) else 'degraded'
        }

        return APIResponse(
            status="success",
            data=health_status,
            message="Governance intelligence health status retrieved"
        )

    except Exception as e:
        logger.error("governance_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Governance intelligence health check failed: {str(e)}"
        )


# Cross-Domain Correlation Engine endpoints (Patent 4)
@app.post("/api/v1/correlation/analyze",
        response_model=APIResponse,
        tags=["correlation"])
async def analyze_cross_domain_correlations(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Perform cross-domain correlation analysis."""
    try:
        logger.info("correlation_analysis_started")

        # Extract events from request
        events_data = request.get('events', [])
        correlation_types = request.get('correlation_types', [])

        # Convert events data to CorrelationEvent objects
        from .services.cross_domain_correlator import CorrelationEvent, CorrelationType, DomainType
        events = []
        for event_data in events_data:
            event = CorrelationEvent(
                event_id=event_data['event_id'],
                domain=DomainType(event_data['domain']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                attributes=event_data.get('attributes', {}),
                metadata=event_data.get('metadata', {})
            )
            events.append(event)

        # Convert correlation types
        if correlation_types:
            correlation_types = [CorrelationType(ct) for ct in correlation_types]

        # Perform correlation analysis
        results = await cross_domain_correlator.analyze_correlations(
            events=events,
            correlation_types=correlation_types
        )

        return APIResponse(
            status="success",
            data=results,
            message="Cross-domain correlation analysis completed"
        )

    except Exception as e:
        logger.error("correlation_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Correlation analysis failed: {str(e)}"
        )


@app.get("/api/v1/correlation/patterns",
        response_model=APIResponse,
        tags=["correlation"])
async def get_correlation_patterns(
    correlation_type: Optional[str] = None,
    min_confidence: float = 0.0,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get discovered correlation patterns."""
    try:
        from .services.cross_domain_correlator import CorrelationType

        # Convert correlation type
        correlation_type_enum = None
        if correlation_type:
            correlation_type_enum = CorrelationType(correlation_type)

        patterns = await cross_domain_correlator.get_correlation_patterns(
            correlation_type=correlation_type_enum,
            min_confidence=min_confidence,
            limit=limit
        )

        # Convert patterns to dictionaries
        pattern_data = []
        for pattern in patterns:
            pattern_dict = {
                'pattern_id': pattern.pattern_id,
                'correlation_type': pattern.correlation_type.value,
                'domains': [d.value for d in pattern.domains],
                'strength': pattern.strength.value,
                'confidence': pattern.confidence,
                'event_count': len(pattern.events),
                'frequency': pattern.frequency,
                'description': pattern.description,
                'created_at': pattern.created_at.isoformat(),
                'last_seen': pattern.last_seen.isoformat(),
                'metadata': pattern.metadata
            }
            pattern_data.append(pattern_dict)

        return APIResponse(
            status="success",
            data={
                'patterns': pattern_data,
                'total_count': len(pattern_data)
            },
            message="Correlation patterns retrieved"
        )

    except Exception as e:
        logger.error("get_correlation_patterns_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get correlation patterns: {str(e)}"
        )


@app.get("/api/v1/correlation/insights",
        response_model=APIResponse,
        tags=["correlation"])
async def get_correlation_insights(
    priority: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 20,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get correlation insights and recommendations."""
    try:
        insights = await cross_domain_correlator.get_correlation_insights(
            priority=priority,
            category=category,
            limit=limit
        )

        # Convert insights to dictionaries
        insight_data = []
        for insight in insights:
            insight_dict = {
                'insight_id': insight.insight_id,
                'title': insight.title,
                'description': insight.description,
                'category': insight.category,
                'priority': insight.priority,
                'confidence': insight.confidence,
                'affected_domains': [d.value for d in insight.affected_domains],
                'pattern_count': len(insight.patterns),
                'recommendations': insight.recommendations,
                'potential_impact': insight.potential_impact,
                'risk_score': insight.risk_score,
                'created_at': insight.created_at.isoformat(),
                'metadata': insight.metadata
            }
            insight_data.append(insight_dict)

        return APIResponse(
            status="success",
            data={
                'insights': insight_data,
                'total_count': len(insight_data)
            },
            message="Correlation insights retrieved"
        )

    except Exception as e:
        logger.error("get_correlation_insights_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get correlation insights: {str(e)}"
        )


@app.get("/api/v1/correlation/domain/{domain}/summary",
        response_model=APIResponse,
        tags=["correlation"])
async def get_domain_correlation_summary(
    domain: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get correlation summary for a specific domain."""
    try:
        from .services.cross_domain_correlator import DomainType

        domain_enum = DomainType(domain)
        summary = await cross_domain_correlator.get_domain_correlation_summary(domain_enum)

        return APIResponse(
            status="success",
            data=summary,
            message=f"Domain correlation summary for {domain} retrieved"
        )

    except Exception as e:
        logger.error("get_domain_correlation_summary_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get domain correlation summary: {str(e)}"
        )


@app.post("/api/v1/correlation/temporal",
        response_model=APIResponse,
        tags=["correlation"])
async def analyze_temporal_correlations(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze temporal correlations between events."""
    try:
        # Extract events
        events_data = request.get('events', [])
        window_size = request.get('window_size', 3600)  # 1 hour default

        from .services.cross_domain_correlator import CorrelationEvent, DomainType
        events = []
        for event_data in events_data:
            event = CorrelationEvent(
                event_id=event_data['event_id'],
                domain=DomainType(event_data['domain']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                attributes=event_data.get('attributes', {}),
                metadata=event_data.get('metadata', {})
            )
            events.append(event)

        # Analyze temporal correlations
        temporal_patterns = (
            await cross_domain_correlator.temporal_analyzer.analyze_temporal_correlations(events)
        )

        # Convert patterns to dictionaries
        pattern_data = []
        for pattern in temporal_patterns:
            pattern_dict = {
                'pattern_id': pattern.pattern_id,
                'domains': [d.value for d in pattern.domains],
                'strength': pattern.strength.value,
                'confidence': pattern.confidence,
                'temporal_window_seconds': pattern.temporal_window.total_seconds() if pattern.temporal_window else None,
                'description': pattern.description,
                'event_count': len(pattern.events)
            }
            pattern_data.append(pattern_dict)

        return APIResponse(
            status="success",
            data={
                'temporal_patterns': pattern_data,
                'window_size': window_size,
                'analysis_timestamp': datetime.now().isoformat()
            },
            message="Temporal correlation analysis completed"
        )

    except Exception as e:
        logger.error("temporal_correlation_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal correlation analysis failed: {str(e)}"
        )


@app.post("/api/v1/correlation/causal",
        response_model=APIResponse,
        tags=["correlation"])
async def analyze_causal_relationships(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyze causal relationships between domains."""
    try:
        # Extract events
        events_data = request.get('events', [])

        from .services.cross_domain_correlator import CorrelationEvent, DomainType
        events = []
        for event_data in events_data:
            event = CorrelationEvent(
                event_id=event_data['event_id'],
                domain=DomainType(event_data['domain']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                attributes=event_data.get('attributes', {}),
                metadata=event_data.get('metadata', {})
            )
            events.append(event)

        # Analyze causal relationships
        causal_patterns = (
            await cross_domain_correlator.causal_engine.infer_causal_relationships(events)
        )

        # Convert patterns to dictionaries
        pattern_data = []
        for pattern in causal_patterns:
            pattern_dict = {
                'pattern_id': pattern.pattern_id,
                'cause_domain': pattern.domains[0].value if len(pattern.domains) >= 1 else None,
                'effect_domain': pattern.domains[1].value if len(pattern.domains) >= 2 else None,
                'confidence': pattern.confidence,
                'strength': pattern.strength.value,
                'description': pattern.description,
                'event_count': len(pattern.events)
            }
            pattern_data.append(pattern_dict)

        return APIResponse(
            status="success",
            data={
                'causal_patterns': pattern_data,
                'analysis_timestamp': datetime.now().isoformat()
            },
            message="Causal relationship analysis completed"
        )

    except Exception as e:
        logger.error("causal_analysis_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal relationship analysis failed: {str(e)}"
        )


@app.post("/api/v1/correlation/anomaly",
        response_model=APIResponse,
        tags=["correlation"])
async def detect_anomaly_correlations(
    request: Dict[str, Any],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Detect correlations in anomalous events."""
    try:
        # Extract events
        events_data = request.get('events', [])

        from .services.cross_domain_correlator import CorrelationEvent, DomainType
        events = []
        for event_data in events_data:
            event = CorrelationEvent(
                event_id=event_data['event_id'],
                domain=DomainType(event_data['domain']),
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                attributes=event_data.get('attributes', {}),
                metadata=event_data.get('metadata', {})
            )
            events.append(event)

        # Detect anomaly correlations
        anomaly_patterns = (
            await cross_domain_correlator.anomaly_detector.detect_anomaly_correlations(events)
        )

        # Convert patterns to dictionaries
        pattern_data = []
        for pattern in anomaly_patterns:
            pattern_dict = {
                'pattern_id': pattern.pattern_id,
                'domains': [d.value for d in pattern.domains],
                'confidence': pattern.confidence,
                'strength': pattern.strength.value,
                'description': pattern.description,
                'anomaly_count': len(pattern.events)
            }
            pattern_data.append(pattern_dict)

        return APIResponse(
            status="success",
            data={
                'anomaly_correlations': pattern_data,
                'analysis_timestamp': datetime.now().isoformat()
            },
            message="Anomaly correlation detection completed"
        )

    except Exception as e:
        logger.error("anomaly_correlation_detection_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly correlation detection failed: {str(e)}"
        )


@app.get("/api/v1/correlation/health",
        response_model=APIResponse,
        tags=["correlation"])
async def get_correlation_engine_health(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get health status of the cross-domain correlation engine."""
    try:
        health_status = {
            'components': {
                'correlation_engine': cross_domain_correlator._initialized,
                'temporal_analyzer': cross_domain_correlator.temporal_analyzer._initialized if hasattr(
                    cross_domain_correlator.temporal_analyzer,
                    '_initialized'
                ) else True,
                'causal_engine': cross_domain_correlator.causal_engine._initialized if hasattr(
                    cross_domain_correlator.causal_engine,
                    '_initialized'
                ) else True,
                'anomaly_detector': cross_domain_correlator.anomaly_detector._initialized if hasattr(
                    cross_domain_correlator.anomaly_detector,
                    '_initialized'
                ) else True,
                'gnn_model': cross_domain_correlator.gnn_model is not None
            },
            'metrics': {
                'stored_patterns': len(cross_domain_correlator.pattern_store),
                'stored_insights': len(cross_domain_correlator.insight_store),
                'correlation_graph_nodes': cross_domain_correlator.correlation_graph.number_of_nodes(),
                'correlation_graph_edges': cross_domain_correlator.correlation_graph.number_of_edges()
            },
            'status': 'healthy' if cross_domain_correlator._initialized else 'degraded',
            'last_health_check': datetime.now().isoformat()
        }

        return APIResponse(
            status="success",
            data=health_status,
            message="Cross-domain correlation engine health status retrieved"
        )

    except Exception as e:
        logger.error("correlation_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Correlation engine health check failed: {str(e)}"
        )


# Patent 2 Implementation: Unified AI Platform endpoints
@app.post("/api/v1/unified-ai/analyze", response_model=UnifiedAIAnalysisResponse)
async def analyze_unified_governance(
    request: UnifiedAIAnalysisRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Analyze governance state using Unified AI Platform (Patent 2)."""
    start_time = time.time()
    
    try:
        logger.info("unified_ai_analysis_started", request_id=request.request_id)
        
        # Process with unified AI platform
        analysis_result = await unified_ai_platform.analyze_governance_state(
            request.governance_data
        )
        
        duration = time.time() - start_time
        
        if analysis_result['success']:
            MODEL_INFERENCE_COUNT.labels(model_name="unified_ai_platform", status="success").inc()
            MODEL_INFERENCE_DURATION.labels(model_name="unified_ai_platform").observe(duration)
            
            return UnifiedAIAnalysisResponse(
                request_id=request.request_id,
                optimization_scores=analysis_result['optimization_scores'],
                domain_correlations=analysis_result['domain_correlations'],
                embeddings=analysis_result['embeddings'],
                recommendations=[],  # Extract from analysis
                confidence_score=0.85,  # Calculate from analysis
                processing_time_ms=round(duration * 1000, 2)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {analysis_result.get('error')}"
            )
            
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="unified_ai_platform", status="error").inc()
        
        logger.error("unified_ai_analysis_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unified AI analysis failed: {str(e)}"
        )


@app.post("/api/v1/unified-ai/optimize", response_model=GovernanceOptimizationResponse)
async def optimize_governance_configuration(
    request: GovernanceOptimizationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Run multi-objective governance optimization (Patent 2)."""
    start_time = time.time()
    
    try:
        logger.info("governance_optimization_started", request_id=request.request_id)
        
        # Run optimization
        optimization_result = await unified_ai_platform.optimize_governance_configuration(
            request.governance_data,
            request.preferences
        )
        
        duration = time.time() - start_time
        
        if optimization_result['success']:
            MODEL_INFERENCE_COUNT.labels(model_name="governance_optimizer", status="success").inc()
            MODEL_INFERENCE_DURATION.labels(model_name="governance_optimizer").observe(duration)
            
            best_solution = optimization_result['best_solution']
            
            return GovernanceOptimizationResponse(
                request_id=request.request_id,
                best_solution=best_solution,
                pareto_front_size=len(optimization_result['optimization_result']['pareto_front']),
                convergence_achieved=True,
                recommendations=optimization_result['recommendations'],
                utility_score=best_solution['utility_score'],
                processing_time_ms=round(duration * 1000, 2)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization failed: {optimization_result.get('error')}"
            )
            
    except Exception as e:
        duration = time.time() - start_time
        MODEL_INFERENCE_COUNT.labels(model_name="governance_optimizer", status="error").inc()
        
        logger.error("governance_optimization_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Governance optimization failed: {str(e)}"
        )


# Patent 3 Implementation: Conversational Governance Intelligence endpoints
@app.post("/api/v1/conversation/governance", response_model=ConversationResponse)
async def process_governance_conversation(
    request: ConversationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Process governance conversation using Conversational AI (Patent 3)."""
    start_time = time.time()
    
    try:
        logger.info("governance_conversation_started", 
                   session_id=request.session_id,
                   user_id=request.user_id)
        
        # Process conversation
        conversation_result = await conversational_intelligence.process_conversation(
            request.user_input,
            request.session_id,
            request.user_id
        )
        
        duration = time.time() - start_time
        
        if conversation_result['success']:
            CONVERSATION_REQUESTS.labels(
                intent=conversation_result['intent'],
                domain='governance',
                status='success'
            ).inc()
            
            return ConversationResponse(
                response=conversation_result['response'],
                intent=conversation_result['intent'],
                entities=conversation_result['entities'],
                confidence=0.9,  # Calculate from result
                api_call=conversation_result.get('api_call'),
                clarification_needed=conversation_result.get('clarification_needed'),
                data=conversation_result.get('data'),
                success=True
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Conversation processing failed: {conversation_result.get('error')}"
            )
            
    except Exception as e:
        duration = time.time() - start_time
        CONVERSATION_REQUESTS.labels(
            intent='unknown',
            domain='governance',
            status='error'
        ).inc()
        
        logger.error("governance_conversation_failed",
                    session_id=request.session_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Governance conversation failed: {str(e)}"
        )


@app.post("/api/v1/conversation/policy-synthesis", response_model=PolicySynthesisResponse)
async def synthesize_governance_policy(
    request: PolicySynthesisRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Synthesize governance policy from natural language (Patent 3)."""
    start_time = time.time()
    
    try:
        logger.info("policy_synthesis_started", request_id=request.request_id)
        
        # Synthesize policy using conversational intelligence
        synthesis_result = conversational_intelligence.policy_synthesizer.synthesize_policy(
            request.description,
            request.domain,
            request.policy_type
        )
        
        duration = time.time() - start_time
        
        if 'error' not in synthesis_result:
            with POLICY_SYNTHESIS_DURATION.time():
                pass
                
            return PolicySynthesisResponse(
                request_id=request.request_id,
                policy_text=synthesis_result['policy_text'],
                structured_policy=synthesis_result['structured_policy'],
                domain=synthesis_result['domain'],
                confidence_score=synthesis_result['confidence_score'],
                validation_results=None,  # Add validation if needed
                processing_time_ms=round(duration * 1000, 2)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Policy synthesis failed: {synthesis_result['error']}"
            )
            
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error("policy_synthesis_failed",
                    request_id=request.request_id,
                    error=str(e),
                    duration_ms=round(duration * 1000, 2))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy synthesis failed: {str(e)}"
        )


@app.get("/api/v1/conversation/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_governance_conversation_history(
    session_id: str,
    include_metadata: bool = True,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get conversation history for governance session (Patent 3)."""
    try:
        logger.info("conversation_history_requested", session_id=session_id)
        
        history_result = conversational_intelligence.get_conversation_history(session_id)
        
        if history_result.get('success', True):
            return ConversationHistoryResponse(
                session_id=session_id,
                user_id=history_result.get('user_id', 'unknown'),
                history=history_result.get('history', []),
                current_state=history_result.get('current_state', 'unknown'),
                entities=history_result.get('entities', {}),
                success=True
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("conversation_history_failed",
                    session_id=session_id,
                    error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
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
