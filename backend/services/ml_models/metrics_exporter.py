"""
Prometheus Metrics Exporter for ML Monitoring
Exports model performance, inference, and drift metrics
Author: PolicyCortex ML Team
Date: January 2025
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    start_http_server, push_to_gateway
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import logging
from dataclasses import dataclass
import threading
from queue import Queue
import psutil
import GPUtil

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# =============================================================================
# ML Model Metrics
# =============================================================================

# Prediction metrics
prediction_counter = Counter(
    'ml_predictions_total',
    'Total number of predictions made',
    ['model_id', 'tenant_id', 'model_type'],
    registry=registry
)

prediction_errors = Counter(
    'ml_prediction_errors_total',
    'Total number of prediction errors',
    ['model_id', 'error_type'],
    registry=registry
)

# Latency metrics
inference_latency = Histogram(
    'ml_inference_duration_seconds',
    'Model inference latency in seconds',
    ['model_id', 'model_type'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 1.0],
    registry=registry
)

feature_extraction_latency = Histogram(
    'ml_feature_extraction_duration_seconds',
    'Feature extraction latency in seconds',
    ['feature_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=registry
)

# Model performance metrics
model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_id', 'model_version', 'tenant_id'],
    registry=registry
)

model_precision = Gauge(
    'ml_model_precision',
    'Current model precision',
    ['model_id', 'model_version', 'tenant_id'],
    registry=registry
)

model_recall = Gauge(
    'ml_model_recall',
    'Current model recall',
    ['model_id', 'model_version', 'tenant_id'],
    registry=registry
)

model_f1_score = Gauge(
    'ml_model_f1_score',
    'Current model F1 score',
    ['model_id', 'model_version', 'tenant_id'],
    registry=registry
)

false_positive_rate = Gauge(
    'ml_model_false_positive_rate',
    'Current model false positive rate',
    ['model_id', 'model_version', 'tenant_id'],
    registry=registry
)

# Confidence scoring metrics
confidence_score_distribution = Histogram(
    'ml_confidence_score',
    'Distribution of confidence scores',
    ['model_id'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
    registry=registry
)

# Risk level distribution
risk_level_counter = Counter(
    'ml_risk_level_total',
    'Count of predictions by risk level',
    ['risk_level', 'tenant_id'],
    registry=registry
)

# =============================================================================
# Drift Detection Metrics
# =============================================================================

drift_score = Gauge(
    'ml_drift_score',
    'Current drift score',
    ['model_id', 'drift_type'],
    registry=registry
)

drift_alerts = Counter(
    'ml_drift_alerts_total',
    'Total number of drift alerts',
    ['model_id', 'alert_level'],
    registry=registry
)

reconstruction_error = Histogram(
    'ml_vae_reconstruction_error',
    'VAE reconstruction error distribution',
    ['model_id'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=registry
)

# =============================================================================
# Training Metrics
# =============================================================================

training_duration = Histogram(
    'ml_training_duration_seconds',
    'Model training duration in seconds',
    ['model_type', 'tenant_id'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400],  # 1min to 4hours
    registry=registry
)

training_samples = Counter(
    'ml_training_samples_total',
    'Total number of training samples processed',
    ['model_type', 'tenant_id'],
    registry=registry
)

training_loss = Gauge(
    'ml_training_loss',
    'Current training loss',
    ['model_id', 'epoch'],
    registry=registry
)

# =============================================================================
# Model Versioning Metrics
# =============================================================================

model_versions = Gauge(
    'ml_model_versions_total',
    'Total number of model versions',
    ['tenant_id', 'model_type', 'status'],
    registry=registry
)

model_deployments = Counter(
    'ml_model_deployments_total',
    'Total number of model deployments',
    ['tenant_id', 'environment'],
    registry=registry
)

model_rollbacks = Counter(
    'ml_model_rollbacks_total',
    'Total number of model rollbacks',
    ['tenant_id', 'reason'],
    registry=registry
)

# =============================================================================
# Resource Utilization Metrics
# =============================================================================

gpu_utilization = Gauge(
    'ml_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id', 'gpu_name'],
    registry=registry
)

gpu_memory_used = Gauge(
    'ml_gpu_memory_used_mb',
    'GPU memory used in MB',
    ['gpu_id', 'gpu_name'],
    registry=registry
)

model_memory_usage = Gauge(
    'ml_model_memory_usage_mb',
    'Model memory usage in MB',
    ['model_id'],
    registry=registry
)

# =============================================================================
# WebSocket Metrics
# =============================================================================

websocket_connections = Gauge(
    'ml_websocket_connections_active',
    'Number of active WebSocket connections',
    ['tenant_id'],
    registry=registry
)

websocket_messages = Counter(
    'ml_websocket_messages_total',
    'Total WebSocket messages sent',
    ['message_type', 'tenant_id'],
    registry=registry
)

# =============================================================================
# Business Metrics
# =============================================================================

violations_predicted = Counter(
    'ml_violations_predicted_total',
    'Total violations predicted',
    ['tenant_id', 'time_window'],
    registry=registry
)

remediation_success_rate = Gauge(
    'ml_remediation_success_rate',
    'Automated remediation success rate',
    ['tenant_id'],
    registry=registry
)

compliance_score = Gauge(
    'ml_compliance_score',
    'Overall compliance score',
    ['tenant_id', 'resource_type'],
    registry=registry
)

# =============================================================================
# Patent Requirement Metrics
# =============================================================================

patent_accuracy_requirement = Gauge(
    'ml_patent_accuracy_met',
    'Whether patent accuracy requirement (99.2%) is met',
    ['model_id'],
    registry=registry
)

patent_fpr_requirement = Gauge(
    'ml_patent_fpr_met',
    'Whether patent FPR requirement (<2%) is met',
    ['model_id'],
    registry=registry
)

patent_latency_requirement = Gauge(
    'ml_patent_latency_met',
    'Whether patent latency requirement (<100ms) is met',
    ['model_id'],
    registry=registry
)


class MLMetricsCollector:
    """Custom collector for ML-specific metrics"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.metrics_queue = Queue()
        self.last_collection = datetime.now()
        
    def collect(self):
        """Collect current metrics from various sources"""
        metrics = []
        
        # Collect GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                metrics.append(GaugeMetricFamily(
                    'ml_gpu_utilization_percent',
                    'GPU utilization percentage',
                    value=gpu.load * 100,
                    labels={'gpu_id': str(gpu.id), 'gpu_name': gpu.name}
                ))
                
                metrics.append(GaugeMetricFamily(
                    'ml_gpu_memory_used_mb',
                    'GPU memory used in MB',
                    value=gpu.memoryUsed,
                    labels={'gpu_id': str(gpu.id), 'gpu_name': gpu.name}
                ))
        except:
            pass  # No GPU available
        
        # Collect CPU and memory metrics
        metrics.append(GaugeMetricFamily(
            'ml_cpu_utilization_percent',
            'CPU utilization percentage',
            value=psutil.cpu_percent()
        ))
        
        metrics.append(GaugeMetricFamily(
            'ml_memory_utilization_percent',
            'Memory utilization percentage',
            value=psutil.virtual_memory().percent
        ))
        
        return metrics


class MetricsExporter:
    """Main metrics exporter for ML system"""
    
    def __init__(self, port: int = 9090, push_gateway: Optional[str] = None):
        self.port = port
        self.push_gateway = push_gateway
        self.running = False
        self.metrics_thread = None
        
    def start(self):
        """Start metrics HTTP server"""
        start_http_server(self.port, registry=registry)
        logger.info(f"Metrics server started on port {self.port}")
        
        self.running = True
        
        # Start background metrics updater
        self.metrics_thread = threading.Thread(target=self._update_metrics_loop)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
    
    def stop(self):
        """Stop metrics exporter"""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=5)
    
    def _update_metrics_loop(self):
        """Background loop to update metrics"""
        while self.running:
            try:
                # Update GPU metrics
                self._update_gpu_metrics()
                
                # Push to gateway if configured
                if self.push_gateway:
                    push_to_gateway(self.push_gateway, job='ml_metrics', registry=registry)
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                time.sleep(60)
    
    def _update_gpu_metrics(self):
        """Update GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_utilization.labels(
                    gpu_id=str(gpu.id),
                    gpu_name=gpu.name
                ).set(gpu.load * 100)
                
                gpu_memory_used.labels(
                    gpu_id=str(gpu.id),
                    gpu_name=gpu.name
                ).set(gpu.memoryUsed)
        except:
            pass  # No GPU available
    
    @staticmethod
    def record_prediction(model_id: str, tenant_id: str, model_type: str,
                         latency_ms: float, risk_level: str, confidence: float):
        """Record prediction metrics"""
        # Increment prediction counter
        prediction_counter.labels(
            model_id=model_id,
            tenant_id=tenant_id,
            model_type=model_type
        ).inc()
        
        # Record latency
        inference_latency.labels(
            model_id=model_id,
            model_type=model_type
        ).observe(latency_ms / 1000.0)
        
        # Record risk level
        risk_level_counter.labels(
            risk_level=risk_level,
            tenant_id=tenant_id
        ).inc()
        
        # Record confidence score
        confidence_score_distribution.labels(
            model_id=model_id
        ).observe(confidence)
    
    @staticmethod
    def record_drift(model_id: str, drift_type: str, score: float, alert_level: Optional[str] = None):
        """Record drift detection metrics"""
        drift_score.labels(
            model_id=model_id,
            drift_type=drift_type
        ).set(score)
        
        if alert_level:
            drift_alerts.labels(
                model_id=model_id,
                alert_level=alert_level
            ).inc()
    
    @staticmethod
    def record_training(model_type: str, tenant_id: str, duration_seconds: float,
                       num_samples: int, final_loss: float):
        """Record training metrics"""
        training_duration.labels(
            model_type=model_type,
            tenant_id=tenant_id
        ).observe(duration_seconds)
        
        training_samples.labels(
            model_type=model_type,
            tenant_id=tenant_id
        ).inc(num_samples)
        
        training_loss.labels(
            model_id=f"{model_type}_{tenant_id}",
            epoch="final"
        ).set(final_loss)
    
    @staticmethod
    def update_model_performance(model_id: str, model_version: str, tenant_id: str,
                                accuracy: float, precision: float, recall: float,
                                f1: float, fpr: float):
        """Update model performance metrics"""
        model_accuracy.labels(
            model_id=model_id,
            model_version=model_version,
            tenant_id=tenant_id
        ).set(accuracy)
        
        model_precision.labels(
            model_id=model_id,
            model_version=model_version,
            tenant_id=tenant_id
        ).set(precision)
        
        model_recall.labels(
            model_id=model_id,
            model_version=model_version,
            tenant_id=tenant_id
        ).set(recall)
        
        model_f1_score.labels(
            model_id=model_id,
            model_version=model_version,
            tenant_id=tenant_id
        ).set(f1)
        
        false_positive_rate.labels(
            model_id=model_id,
            model_version=model_version,
            tenant_id=tenant_id
        ).set(fpr)
        
        # Check patent requirements
        patent_accuracy_requirement.labels(model_id=model_id).set(1 if accuracy >= 0.992 else 0)
        patent_fpr_requirement.labels(model_id=model_id).set(1 if fpr < 0.02 else 0)
    
    @staticmethod
    def record_websocket_activity(tenant_id: str, message_type: str, active_connections: int):
        """Record WebSocket metrics"""
        websocket_connections.labels(tenant_id=tenant_id).set(active_connections)
        websocket_messages.labels(
            message_type=message_type,
            tenant_id=tenant_id
        ).inc()
    
    @staticmethod
    def record_business_metrics(tenant_id: str, violations_count: int,
                               remediation_rate: float, overall_compliance: float):
        """Record business-level metrics"""
        violations_predicted.labels(
            tenant_id=tenant_id,
            time_window="72h"
        ).inc(violations_count)
        
        remediation_success_rate.labels(tenant_id=tenant_id).set(remediation_rate)
        compliance_score.labels(
            tenant_id=tenant_id,
            resource_type="all"
        ).set(overall_compliance)


# Singleton instance
metrics_exporter = MetricsExporter()


def get_metrics():
    """Get current metrics in Prometheus format"""
    return generate_latest(registry)


if __name__ == "__main__":
    # Start metrics server
    exporter = MetricsExporter(port=9090)
    exporter.start()
    
    # Simulate some metrics
    import random
    
    while True:
        # Simulate predictions
        MetricsExporter.record_prediction(
            model_id="model_001",
            tenant_id="tenant_001",
            model_type="ensemble",
            latency_ms=random.uniform(30, 120),
            risk_level=random.choice(["low", "medium", "high", "critical"]),
            confidence=random.uniform(0.7, 0.99)
        )
        
        # Simulate drift
        MetricsExporter.record_drift(
            model_id="model_001",
            drift_type="data",
            score=random.uniform(0, 5),
            alert_level="warning" if random.random() > 0.8 else None
        )
        
        time.sleep(5)