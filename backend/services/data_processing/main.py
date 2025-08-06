"""
Data Processing Service for PolicyCortex.
Handles ETL pipelines, stream processing, data transformation, and quality checks.
"""

import time
import uuid
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from shared.config import get_settings
from shared.database import DatabaseUtils
from shared.database import get_async_db
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

from .auth import AuthManager
from .models import APIResponse
from .models import DataAggregationRequest
from .models import DataAggregationResponse
from .models import DataExportRequest
from .models import DataExportResponse
from .models import DataLineageRequest
from .models import DataLineageResponse
from .models import DataTransformationRequest
from .models import DataTransformationResponse
from .models import DataValidationRequest
from .models import DataValidationResponse
from .models import ErrorResponse
from .models import ETLPipelineRequest
from .models import ETLPipelineResponse
from .models import HealthResponse
from .models import PipelineStatus
from .models import ProcessingMetrics
from .models import StreamProcessingRequest
from .models import StreamProcessingResponse
from .services.azure_connectors import AzureConnectorService
from .services.data_aggregator import DataAggregatorService
from .services.data_exporter import DataExporterService
from .services.data_pipeline import DataPipeline
from .services.data_pipeline import DataSourceType
from .services.data_transformer import DataTransformerService
from .services.data_validator import DataValidatorService
from .services.etl_pipeline import ETLPipelineService
from .services.lineage_tracker import LineageTrackerService
from .services.stream_processor import StreamProcessorService

# Configuration
settings = get_settings()
logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'data_processing_requests_total',
    'Total API requests',
    ['method',
    'endpoint',
    'status']
)
REQUEST_DURATION = Histogram('data_processing_request_duration_seconds', 'Request duration')
PIPELINE_EXECUTIONS = Counter(
    'data_processing_pipeline_executions_total',
    'Pipeline executions',
    ['pipeline_type',
    'status']
)
PROCESSING_LATENCY = Histogram(
    'data_processing_latency_seconds',
    'Processing latency',
    ['operation']
)

# FastAPI app
app = FastAPI(
    title="PolicyCortex Data Processing Service",
    description="Data processing microservice for ETL, stream processing, and
        data quality management",
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
azure_connector = AzureConnectorService()
etl_pipeline = ETLPipelineService()
stream_processor = StreamProcessorService()
data_transformer = DataTransformerService()
data_validator = DataValidatorService()
data_aggregator = DataAggregatorService()
lineage_tracker = LineageTrackerService()
data_exporter = DataExporterService()
data_pipeline = DataPipeline(settings)


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


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        await data_pipeline.initialize()
        await data_pipeline.start_processing()
        logger.info("data_processing_service_startup_completed")
    except Exception as e:
        logger.error("data_processing_service_startup_failed", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown."""
    try:
        await data_pipeline.cleanup()
        logger.info("data_processing_service_shutdown_completed")
    except Exception as e:
        logger.error("data_processing_service_shutdown_failed", error=str(e))


async def verify_authentication(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Verify authentication for protected endpoints."""

    # Skip authentication for health checks and metrics
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
        service="data-processing",
        version=settings.service.service_version
    )


@app.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check endpoint."""
    try:
        # Check database connectivity
        db = await get_async_db()
        await db.execute("SELECT 1")

        # Check Azure services connectivity
        azure_health = await azure_connector.health_check()

        if azure_health["status"] == "healthy":
            return HealthResponse(
                status="ready",
                timestamp=datetime.utcnow(),
                service="data-processing",
                version=settings.service.service_version,
                details={"azure_services": azure_health}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Azure services not ready"
            )

    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest())


# ETL Pipeline endpoints
@app.post("/api/v1/etl/pipeline", response_model=ETLPipelineResponse)
async def create_etl_pipeline(
    request: ETLPipelineRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Create and execute ETL pipeline."""
    try:
        start_time = time.time()

        # Create pipeline
        pipeline_id = await etl_pipeline.create_pipeline(
            source_config=request.source_config,
            target_config=request.target_config,
            transformation_rules=request.transformation_rules,
            schedule=request.schedule,
            user_id=user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_pipeline_creation(
            pipeline_id=pipeline_id,
            source_config=request.source_config,
            target_config=request.target_config,
            user_id=user.get("id") if user else None
        )

        # Execute pipeline if immediate execution is requested
        if request.execute_immediately:
            background_tasks.add_task(
                etl_pipeline.execute_pipeline,
                pipeline_id,
                user.get("id") if user else None
            )

        # Update metrics
        PIPELINE_EXECUTIONS.labels(pipeline_type="etl", status="created").inc()
        PROCESSING_LATENCY.labels(operation="pipeline_creation").observe(time.time() - start_time)

        return ETLPipelineResponse(
            pipeline_id=pipeline_id,
            status = (
                PipelineStatus.CREATED if not request.execute_immediately else PipelineStatus.RUNNING,
            )
            message="ETL pipeline created successfully"
        )

    except Exception as e:
        logger.error("etl_pipeline_creation_failed", error=str(e))
        PIPELINE_EXECUTIONS.labels(pipeline_type="etl", status="failed").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ETL pipeline: {str(e)}"
        )


@app.get("/api/v1/etl/pipeline/{pipeline_id}", response_model=ETLPipelineResponse)
async def get_etl_pipeline(
    pipeline_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get ETL pipeline status and details."""
    try:
        pipeline_info = await etl_pipeline.get_pipeline_info(pipeline_id)

        return ETLPipelineResponse(
            pipeline_id=pipeline_id,
            status=pipeline_info["status"],
            message=pipeline_info.get("message", ""),
            details=pipeline_info.get("details", {})
        )

    except Exception as e:
        logger.error("etl_pipeline_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found: {str(e)}"
        )


@app.delete("/api/v1/etl/pipeline/{pipeline_id}")
async def delete_etl_pipeline(
    pipeline_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Delete ETL pipeline."""
    try:
        await etl_pipeline.delete_pipeline(pipeline_id)

        return APIResponse(
            success=True,
            message="ETL pipeline deleted successfully"
        )

    except Exception as e:
        logger.error("etl_pipeline_deletion_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete ETL pipeline: {str(e)}"
        )


# Stream Processing endpoints
@app.post("/api/v1/stream/processor", response_model=StreamProcessingResponse)
async def create_stream_processor(
    request: StreamProcessingRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Create and start stream processor."""
    try:
        start_time = time.time()

        # Create stream processor
        processor_id = await stream_processor.create_processor(
            source_config=request.source_config,
            processing_rules=request.processing_rules,
            output_config=request.output_config,
            user_id=user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_stream_processor_creation(
            processor_id=processor_id,
            source_config=request.source_config,
            output_config=request.output_config,
            user_id=user.get("id") if user else None
        )

        # Update metrics
        PIPELINE_EXECUTIONS.labels(pipeline_type="stream", status="created").inc()
        PROCESSING_LATENCY.labels(operation = (
            "stream_processor_creation").observe(time.time() - start_time)
        )

        return StreamProcessingResponse(
            processor_id=processor_id,
            status=PipelineStatus.RUNNING,
            message="Stream processor created and started successfully"
        )

    except Exception as e:
        logger.error("stream_processor_creation_failed", error=str(e))
        PIPELINE_EXECUTIONS.labels(pipeline_type="stream", status="failed").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create stream processor: {str(e)}"
        )


@app.get("/api/v1/stream/processor/{processor_id}", response_model=StreamProcessingResponse)
async def get_stream_processor(
    processor_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get stream processor status and details."""
    try:
        processor_info = await stream_processor.get_processor_info(processor_id)

        return StreamProcessingResponse(
            processor_id=processor_id,
            status=processor_info["status"],
            message=processor_info.get("message", ""),
            details=processor_info.get("details", {})
        )

    except Exception as e:
        logger.error("stream_processor_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream processor not found: {str(e)}"
        )


# Data Transformation endpoints
@app.post("/api/v1/transform/data", response_model=DataTransformationResponse)
async def transform_data(
    request: DataTransformationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Transform data using specified rules."""
    try:
        start_time = time.time()

        # Transform data
        result = await data_transformer.transform_data(
            data=request.data,
            transformation_rules=request.transformation_rules,
            output_format=request.output_format,
            user_id=user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_data_transformation(
            transformation_id=result["transformation_id"],
            input_data=request.data,
            rules=request.transformation_rules,
            output_data=result["transformed_data"],
            user_id=user.get("id") if user else None
        )

        # Update metrics
        PROCESSING_LATENCY.labels(operation="data_transformation").observe(time.time() - start_time)

        return DataTransformationResponse(
            transformation_id=result["transformation_id"],
            transformed_data=result["transformed_data"],
            status="completed",
            message="Data transformation completed successfully"
        )

    except Exception as e:
        logger.error("data_transformation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transform data: {str(e)}"
        )


# Data Validation endpoints
@app.post("/api/v1/validate/data", response_model=DataValidationResponse)
async def validate_data(
    request: DataValidationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Validate data quality and integrity."""
    try:
        start_time = time.time()

        # Validate data
        result = await data_validator.validate_data(
            data=request.data,
            validation_rules=request.validation_rules,
            quality_threshold=request.quality_threshold,
            user_id=user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_data_validation(
            validation_id=result["validation_id"],
            data=request.data,
            rules=request.validation_rules,
            results=result["validation_results"],
            user_id=user.get("id") if user else None
        )

        # Update metrics
        PROCESSING_LATENCY.labels(operation="data_validation").observe(time.time() - start_time)

        return DataValidationResponse(
            validation_id=result["validation_id"],
            validation_results=result["validation_results"],
            quality_score=result["quality_score"],
            status="completed",
            message="Data validation completed successfully"
        )

    except Exception as e:
        logger.error("data_validation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate data: {str(e)}"
        )


# Data Aggregation endpoints
@app.post("/api/v1/aggregate/data", response_model=DataAggregationResponse)
async def aggregate_data(
    request: DataAggregationRequest,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Aggregate data using specified rules."""
    try:
        start_time = time.time()

        # Aggregate data
        result = await data_aggregator.aggregate_data(
            data=request.data,
            aggregation_rules=request.aggregation_rules,
            group_by_fields=request.group_by_fields,
            user_id=user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_data_aggregation(
            aggregation_id=result["aggregation_id"],
            input_data=request.data,
            rules=request.aggregation_rules,
            output_data=result["aggregated_data"],
            user_id=user.get("id") if user else None
        )

        # Update metrics
        PROCESSING_LATENCY.labels(operation="data_aggregation").observe(time.time() - start_time)

        return DataAggregationResponse(
            aggregation_id=result["aggregation_id"],
            aggregated_data=result["aggregated_data"],
            status="completed",
            message="Data aggregation completed successfully"
        )

    except Exception as e:
        logger.error("data_aggregation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to aggregate data: {str(e)}"
        )


# Data Lineage endpoints
@app.get("/api/v1/lineage/{entity_id}", response_model=DataLineageResponse)
async def get_data_lineage(
    entity_id: str,
    entity_type: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get data lineage for a specific entity."""
    try:
        lineage_info = await lineage_tracker.get_lineage(
            entity_id=entity_id,
            entity_type=entity_type
        )

        return DataLineageResponse(
            entity_id=entity_id,
            entity_type=entity_type,
            lineage_graph=lineage_info["lineage_graph"],
            upstream_dependencies=lineage_info["upstream_dependencies"],
            downstream_dependencies=lineage_info["downstream_dependencies"],
            status="completed",
            message="Data lineage retrieved successfully"
        )

    except Exception as e:
        logger.error("data_lineage_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data lineage not found: {str(e)}"
        )


# Data Export endpoints
@app.post("/api/v1/export/data", response_model=DataExportResponse)
async def export_data(
    request: DataExportRequest,
    background_tasks: BackgroundTasks,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Export data to specified destination."""
    try:
        start_time = time.time()

        # Create export job
        export_id = await data_exporter.create_export_job(
            source_config=request.source_config,
            destination_config=request.destination_config,
            export_format=request.export_format,
            filters=request.filters,
            user_id=user.get("id") if user else None
        )

        # Execute export in background
        background_tasks.add_task(
            data_exporter.execute_export,
            export_id,
            user.get("id") if user else None
        )

        # Track lineage
        await lineage_tracker.track_data_export(
            export_id=export_id,
            source_config=request.source_config,
            destination_config=request.destination_config,
            user_id=user.get("id") if user else None
        )

        # Update metrics
        PROCESSING_LATENCY.labels(operation = (
            "data_export_creation").observe(time.time() - start_time)
        )

        return DataExportResponse(
            export_id=export_id,
            status="started",
            message="Data export job created and started successfully"
        )

    except Exception as e:
        logger.error("data_export_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create data export: {str(e)}"
        )


@app.get("/api/v1/export/{export_id}", response_model=DataExportResponse)
async def get_export_status(
    export_id: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get data export status and details."""
    try:
        export_info = await data_exporter.get_export_info(export_id)

        return DataExportResponse(
            export_id=export_id,
            status=export_info["status"],
            message=export_info.get("message", ""),
            details=export_info.get("details", {})
        )

    except Exception as e:
        logger.error("data_export_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Export job not found: {str(e)}"
        )


# Data Pipeline endpoints
@app.post("/api/v1/pipeline/process")
async def process_pipeline_data(
    data: List[Dict[str, Any]],
    source_type: str,
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Process data through the main data pipeline."""
    try:
        # Convert string to enum
        data_source = DataSourceType(source_type)

        # Process data through pipeline
        result = await data_pipeline.process_data(data, data_source)

        return {
            "processing_result": {
                "source_type": result.source_type.value,
                "records_processed": result.records_processed,
                "records_transformed": result.records_transformed,
                "records_stored": result.records_stored,
                "processing_time_ms": result.processing_time_ms,
                "quality_metrics": {
                    "total_records": result.quality_metrics.total_records,
                    "valid_records": result.quality_metrics.valid_records,
                    "invalid_records": result.quality_metrics.invalid_records,
                    "duplicate_records": result.quality_metrics.duplicate_records,
                    "completeness_score": result.quality_metrics.completeness_score,
                    "accuracy_score": result.quality_metrics.accuracy_score,
                    "consistency_score": result.quality_metrics.consistency_score
                },
                "errors": result.errors
            }
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = (
                f"Invalid source type: {source_type}. Valid types: {[t.value for t in DataSourceType]}"
            )
        )
    except Exception as e:
        logger.error("pipeline_data_processing_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process pipeline data: {str(e)}"
        )


@app.get("/api/v1/pipeline/statistics")
async def get_pipeline_statistics(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get data pipeline processing statistics."""
    try:
        stats = await data_pipeline.get_processing_statistics()
        return {"pipeline_statistics": stats}

    except Exception as e:
        logger.error("pipeline_statistics_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pipeline statistics: {str(e)}"
        )


@app.get("/api/v1/pipeline/status")
async def get_pipeline_status(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get data pipeline status."""
    try:
        return {
            "pipeline_status": {
                "is_running": data_pipeline.is_running,
                "active_tasks": len([t for t in data_pipeline.processing_tasks.values() if not t.done()]),
                "total_tasks": len(data_pipeline.processing_tasks)
            }
        }

    except Exception as e:
        logger.error("pipeline_status_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pipeline status: {str(e)}"
        )


# Metrics and monitoring endpoints
@app.get("/api/v1/metrics/processing", response_model=ProcessingMetrics)
async def get_processing_metrics(
    user: Optional[Dict[str, Any]] = Depends(verify_authentication)
):
    """Get data processing metrics."""
    try:
        metrics = await etl_pipeline.get_processing_metrics()

        return ProcessingMetrics(
            active_pipelines=metrics["active_pipelines"],
            completed_pipelines=metrics["completed_pipelines"],
            failed_pipelines=metrics["failed_pipelines"],
            data_volume_processed=metrics["data_volume_processed"],
            average_processing_time=metrics["average_processing_time"],
            quality_score_average=metrics["quality_score_average"]
        )

    except Exception as e:
        logger.error("processing_metrics_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve processing metrics: {str(e)}"
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
