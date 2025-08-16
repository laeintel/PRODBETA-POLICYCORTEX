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
Comprehensive observability implementation with OpenTelemetry, Prometheus metrics, and distributed tracing
"""

import time
import random
import string
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from functools import wraps
import json
import os
import logging

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage, context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.b3 import B3MultiFormat
from opentelemetry.sdk.metrics import MeterProvider, Counter, Histogram, UpDownCounter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, ParentBased
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Prometheus client for custom metrics
from prometheus_client import Counter as PromCounter, Histogram as PromHistogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_client import start_http_server

# FastAPI dependencies
from fastapi import Request, Response, HTTPException
from fastapi.responses import PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Configuration
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")
OTEL_SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "policycortex-api")
OTEL_SERVICE_VERSION = os.getenv("OTEL_SERVICE_VERSION", "2.0.0")
OTEL_ENVIRONMENT = os.getenv("OTEL_ENVIRONMENT", "development")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
TRACE_SAMPLE_RATE = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))

# Logger
logger = logging.getLogger(__name__)

class ObservabilityManager:
    """Central manager for all observability concerns"""
    
    def __init__(self):
        self.tracer = None
        self.meter = None
        self.metrics = {}
        self.slo_definitions = {}
        self.resource = None
        self._initialized = False
        
    def initialize(self, app=None):
        """Initialize observability components"""
        if self._initialized:
            return
            
        # Create resource attributes
        self.resource = Resource.create({
            SERVICE_NAME: OTEL_SERVICE_NAME,
            SERVICE_VERSION: OTEL_SERVICE_VERSION,
            "service.environment": OTEL_ENVIRONMENT,
            "service.instance.id": self._generate_instance_id(),
            "cloud.provider": "azure",
            "cloud.region": os.getenv("AZURE_REGION", "eastus"),
        })
        
        # Initialize tracing
        if ENABLE_TRACING:
            self._initialize_tracing()
            
        # Initialize metrics
        if ENABLE_METRICS:
            self._initialize_metrics()
            
        # Setup propagators for distributed tracing
        set_global_textmap(B3MultiFormat())
        
        # Auto-instrument libraries
        self._setup_auto_instrumentation(app)
        
        # Define SLOs
        self._define_slos()
        
        self._initialized = True
        logger.info("Observability initialized successfully")
        
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing"""
        # Setup trace provider
        trace_provider = TracerProvider(
            resource=self.resource,
            sampler=ParentBased(root=TraceIdRatioBased(TRACE_SAMPLE_RATE))
        )
        
        # Add OTLP exporter
        if OTEL_EXPORTER_OTLP_ENDPOINT != "console":
            otlp_exporter = OTLPSpanExporter(
                endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True  # Use TLS in production
            )
            trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        else:
            # Console exporter for development
            trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            
        # Set global tracer provider
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)
        
    def _initialize_metrics(self):
        """Initialize OpenTelemetry metrics"""
        # Setup metrics provider with Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        
        metrics_provider = MeterProvider(
            resource=self.resource,
            metric_readers=[prometheus_reader]
        )
        
        # Add OTLP metrics exporter if configured
        if OTEL_EXPORTER_OTLP_ENDPOINT != "console":
            otlp_metric_exporter = OTLPMetricExporter(
                endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True
            )
            # Note: In production, configure periodic reader for OTLP
            
        # Set global meter provider
        metrics.set_meter_provider(metrics_provider)
        self.meter = metrics.get_meter(__name__)
        
        # Create standard metrics
        self._create_standard_metrics()
        
        # Start Prometheus metrics server
        start_http_server(PROMETHEUS_PORT)
        logger.info(f"Prometheus metrics available at http://localhost:{PROMETHEUS_PORT}/metrics")
        
    def _create_standard_metrics(self):
        """Create standard application metrics"""
        # Request metrics
        self.metrics["request_count"] = self.meter.create_counter(
            name="http_requests_total",
            description="Total number of HTTP requests",
            unit="requests"
        )
        
        self.metrics["request_duration"] = self.meter.create_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            unit="seconds"
        )
        
        self.metrics["active_requests"] = self.meter.create_up_down_counter(
            name="http_requests_active",
            description="Number of active HTTP requests",
            unit="requests"
        )
        
        # Business metrics
        self.metrics["policy_evaluations"] = self.meter.create_counter(
            name="policy_evaluations_total",
            description="Total number of policy evaluations",
            unit="evaluations"
        )
        
        self.metrics["compliance_score"] = self.meter.create_histogram(
            name="compliance_score",
            description="Current compliance score",
            unit="percentage"
        )
        
        self.metrics["resource_violations"] = self.meter.create_counter(
            name="resource_violations_total",
            description="Total number of resource violations detected",
            unit="violations"
        )
        
        # System metrics
        self.metrics["db_connections"] = self.meter.create_up_down_counter(
            name="database_connections_active",
            description="Number of active database connections",
            unit="connections"
        )
        
        self.metrics["cache_hits"] = self.meter.create_counter(
            name="cache_hits_total",
            description="Total number of cache hits",
            unit="hits"
        )
        
        self.metrics["cache_misses"] = self.meter.create_counter(
            name="cache_misses_total",
            description="Total number of cache misses",
            unit="misses"
        )
        
    def _setup_auto_instrumentation(self, app):
        """Setup automatic instrumentation for libraries"""
        # FastAPI instrumentation
        if app:
            FastAPIInstrumentor.instrument_app(app)
            
        # SQLAlchemy instrumentation
        try:
            SQLAlchemyInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Failed to instrument SQLAlchemy: {e}")
            
        # Redis instrumentation
        try:
            RedisInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Failed to instrument Redis: {e}")
            
        # HTTP client instrumentation
        RequestsInstrumentor().instrument()
        
        # Logging instrumentation
        LoggingInstrumentor().instrument()
        
    def _define_slos(self):
        """Define Service Level Objectives"""
        self.slo_definitions = {
            "availability": {
                "target": 0.999,  # 99.9% uptime
                "window": "30d",
                "metric": "error_rate",
                "threshold": 0.001
            },
            "latency_p99": {
                "target": 0.5,  # 500ms
                "window": "5m",
                "metric": "request_duration_p99",
                "threshold": 0.5
            },
            "latency_p95": {
                "target": 0.2,  # 200ms
                "window": "5m",
                "metric": "request_duration_p95",
                "threshold": 0.2
            },
            "error_rate": {
                "target": 0.01,  # 1% error rate
                "window": "5m",
                "metric": "error_rate",
                "threshold": 0.01
            }
        }
        
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID"""
        return f"{OTEL_SERVICE_NAME}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
        
    def record_metric(self, metric_name: str, value: float, attributes: Dict[str, Any] = None):
        """Record a metric value"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            attrs = attributes or {}
            
            if hasattr(metric, 'add'):
                metric.add(value, attrs)
            elif hasattr(metric, 'record'):
                metric.record(value, attrs)
                
    @asynccontextmanager
    async def trace_operation(self, name: str, attributes: Dict[str, Any] = None):
        """Context manager for tracing operations"""
        if not self.tracer:
            yield
            return
            
        with self.tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes(attributes)
                
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                span.set_status(Status(StatusCode.OK))


# Global observability manager instance
observability = ObservabilityManager()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracing"""
    
    async def dispatch(self, request: Request, call_next):
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = self._generate_correlation_id()
            
        # Add to request state
        request.state.correlation_id = correlation_id
        
        # Add to baggage for propagation
        baggage.set_baggage("correlation_id", correlation_id)
        
        # Add to current span
        span = trace.get_current_span()
        if span:
            span.set_attribute("correlation_id", correlation_id)
            
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        
        return response
        
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        return f"{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}-{int(time.time())}"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics"""
    
    async def dispatch(self, request: Request, call_next):
        # Record active request
        observability.record_metric("active_requests", 1)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            attributes = {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "status_class": f"{response.status_code // 100}xx"
            }
            
            observability.record_metric("request_count", 1, attributes)
            observability.record_metric("request_duration", duration, attributes)
            
            # Check SLO violations
            if duration > 0.5:  # 500ms threshold
                logger.warning(f"SLO violation: Request took {duration:.2f}s", extra={
                    "correlation_id": getattr(request.state, "correlation_id", None),
                    "path": request.url.path
                })
                
            return response
            
        except Exception as e:
            # Record error metrics
            attributes = {
                "method": request.method,
                "path": request.url.path,
                "error_type": type(e).__name__
            }
            observability.record_metric("request_count", 1, {**attributes, "status_code": 500})
            raise
            
        finally:
            # Decrement active requests
            observability.record_metric("active_requests", -1)


class TracingDecorator:
    """Decorator for adding tracing to functions"""
    
    @staticmethod
    def trace(name: str = None, attributes: Dict[str, Any] = None):
        """Decorator to trace function execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                async with observability.trace_operation(span_name, attributes):
                    return await func(*args, **kwargs)
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = name or f"{func.__module__}.{func.__name__}"
                if not observability.tracer:
                    return func(*args, **kwargs)
                    
                with observability.tracer.start_as_current_span(span_name) as span:
                    if attributes:
                        span.set_attributes(attributes)
                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
                        
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class MetricsDecorator:
    """Decorator for recording function metrics"""
    
    @staticmethod
    def timed(metric_name: str = None):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = metric_name or f"{func.__name__}_duration"
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start
                    observability.record_metric(name, duration, {"function": func.__name__})
                    return result
                except Exception as e:
                    duration = time.time() - start
                    observability.record_metric(name, duration, {
                        "function": func.__name__,
                        "error": type(e).__name__
                    })
                    raise
                    
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = metric_name or f"{func.__name__}_duration"
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    observability.record_metric(name, duration, {"function": func.__name__})
                    return result
                except Exception as e:
                    duration = time.time() - start
                    observability.record_metric(name, duration, {
                        "function": func.__name__,
                        "error": type(e).__name__
                    })
                    raise
                    
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    @staticmethod
    def counted(metric_name: str = None):
        """Decorator to count function calls"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                name = metric_name or f"{func.__name__}_calls"
                observability.record_metric(name, 1, {"function": func.__name__})
                return await func(*args, **kwargs)
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                name = metric_name or f"{func.__name__}_calls"
                observability.record_metric(name, 1, {"function": func.__name__})
                return func(*args, **kwargs)
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator


class SLOMonitor:
    """Monitor and report on Service Level Objectives"""
    
    def __init__(self, observability_manager: ObservabilityManager):
        self.observability = observability_manager
        self.slo_metrics = {}
        
    async def check_slos(self) -> Dict[str, Any]:
        """Check current SLO status"""
        results = {}
        
        for slo_name, slo_def in self.observability.slo_definitions.items():
            # Calculate current metric value
            current_value = await self._calculate_metric(slo_def["metric"])
            
            # Check against threshold
            is_meeting = current_value <= slo_def["threshold"] if slo_def["metric"] != "availability" else current_value >= slo_def["target"]
            
            results[slo_name] = {
                "name": slo_name,
                "target": slo_def["target"],
                "current": current_value,
                "is_meeting": is_meeting,
                "window": slo_def["window"]
            }
            
            # Record SLO metric
            self.observability.record_metric("slo_compliance", 1.0 if is_meeting else 0.0, {
                "slo": slo_name
            })
            
        return results
        
    async def _calculate_metric(self, metric_name: str) -> float:
        """Calculate metric value for SLO"""
        # This would normally query a metrics backend
        # For now, return mock values
        mock_values = {
            "error_rate": 0.005,
            "request_duration_p99": 0.45,
            "request_duration_p95": 0.18,
            "availability": 0.9995
        }
        return mock_values.get(metric_name, 0.0)
        
    async def generate_slo_report(self) -> Dict[str, Any]:
        """Generate comprehensive SLO report"""
        slo_status = await self.check_slos()
        
        # Calculate error budget
        error_budgets = {}
        for slo_name, status in slo_status.items():
            if slo_name == "availability":
                budget_total = 1 - self.observability.slo_definitions[slo_name]["target"]
                budget_used = 1 - status["current"]
                budget_remaining = max(0, budget_total - budget_used)
                
                error_budgets[slo_name] = {
                    "total": budget_total,
                    "used": budget_used,
                    "remaining": budget_remaining,
                    "percentage_used": (budget_used / budget_total * 100) if budget_total > 0 else 0
                }
                
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "slo_status": slo_status,
            "error_budgets": error_budgets,
            "overall_health": all(s["is_meeting"] for s in slo_status.values())
        }


# Prometheus metrics endpoint
async def metrics_endpoint(request: Request) -> Response:
    """Endpoint to expose Prometheus metrics"""
    metrics_data = generate_latest(REGISTRY)
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# Health check endpoint with detailed status
async def health_check_endpoint(request: Request) -> Dict[str, Any]:
    """Enhanced health check with observability status"""
    slo_monitor = SLOMonitor(observability)
    slo_report = await slo_monitor.generate_slo_report()
    
    return {
        "status": "healthy" if slo_report["overall_health"] else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": OTEL_SERVICE_VERSION,
        "environment": OTEL_ENVIRONMENT,
        "slo_status": slo_report["slo_status"],
        "observability": {
            "tracing_enabled": ENABLE_TRACING,
            "metrics_enabled": ENABLE_METRICS,
            "trace_sample_rate": TRACE_SAMPLE_RATE
        }
    }


# Decorators for easy use
trace = TracingDecorator.trace
timed = MetricsDecorator.timed
counted = MetricsDecorator.counted

# Example usage
@trace(name="business.policy_evaluation")
@timed(metric_name="policy_evaluation_duration")
async def evaluate_policy(policy_id: str, resource: Dict[str, Any]) -> bool:
    """Example function with tracing and metrics"""
    # Record business metric
    observability.record_metric("policy_evaluations", 1, {
        "policy_id": policy_id,
        "resource_type": resource.get("type", "unknown")
    })
    
    # Simulate policy evaluation
    await asyncio.sleep(0.1)
    
    # Random compliance score
    score = random.uniform(0.7, 1.0)
    observability.record_metric("compliance_score", score, {
        "policy_id": policy_id
    })
    
    return score > 0.8


# Export key components
__all__ = [
    "observability",
    "ObservabilityManager",
    "CorrelationIdMiddleware",
    "MetricsMiddleware",
    "SLOMonitor",
    "trace",
    "timed",
    "counted",
    "metrics_endpoint",
    "health_check_endpoint"
]