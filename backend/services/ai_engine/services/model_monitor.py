"""
Model Monitoring Service for AI Engine.
Handles model performance monitoring, drift detection, and health checks.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import structlog

from ....shared.config import get_settings
from ..models import ModelMetrics

settings = get_settings()
logger = structlog.get_logger(__name__)


@dataclass
class ModelDriftAlert:
    """Represents a model drift alert."""
    model_name: str
    drift_type: str
    drift_score: float
    threshold: float
    detected_at: datetime
    severity: str
    description: str
    recommended_actions: List[str]


class ModelMonitor:
    """Model monitoring service for performance and drift detection."""

    def __init__(self):
        self.settings = settings
        self.monitoring_data = {}
        self.drift_detectors = {}
        self.performance_metrics = {}
        self.alert_thresholds = self._load_alert_thresholds()
        self.monitoring_config = self._load_monitoring_config()

    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds for different metrics."""
        return {
            "accuracy": {
                "warning": 0.1,    # 10% decrease
                "critical": 0.2    # 20% decrease
            },
            "precision": {
                "warning": 0.1,
                "critical": 0.2
            },
            "recall": {
                "warning": 0.1,
                "critical": 0.2
            },
            "f1_score": {
                "warning": 0.1,
                "critical": 0.2
            },
            "drift_score": {
                "warning": 0.1,
                "critical": 0.2
            },
            "inference_time": {
                "warning": 2.0,    # 2x increase
                "critical": 5.0    # 5x increase
            },
            "error_rate": {
                "warning": 0.05,   # 5% error rate
                "critical": 0.1    # 10% error rate
            }
        }

    def _load_monitoring_config(self) -> Dict[str, Any]:
        """Load monitoring configuration."""
        return {
            "drift_detection": {
                "methods": ["statistical", "ks_test", "psi"],
                "window_size": 1000,
                "reference_period": 30,  # days
                "detection_frequency": 24  # hours
            },
            "performance_monitoring": {
                "metrics": ["accuracy", "precision", "recall", "f1_score", "inference_time"],
                "collection_frequency": 1,  # hours
                "retention_period": 90  # days
            },
            "alerting": {
                "channels": ["email", "slack", "webhook"],
                "escalation_levels": ["info", "warning", "critical"],
                "cooldown_period": 60  # minutes
            }
        }

    async def initialize(self) -> None:
        """Initialize the model monitor."""
        try:
            logger.info("Initializing model monitor")

            # Initialize drift detectors
            await self._initialize_drift_detectors()

            # Initialize performance trackers
            await self._initialize_performance_trackers()

            logger.info("Model monitor initialized successfully")

        except Exception as e:
            logger.error("Model monitor initialization failed", error=str(e))
            raise

    async def _initialize_drift_detectors(self) -> None:
        """Initialize drift detection methods."""
        try:
            self.drift_detectors = {
                "statistical": self._detect_statistical_drift,
                "ks_test": self._detect_ks_drift,
                "psi": self._detect_psi_drift
            }

            logger.info("Drift detectors initialized")

        except Exception as e:
            logger.error("Drift detector initialization failed", error=str(e))

    async def _initialize_performance_trackers(self) -> None:
        """Initialize performance tracking."""
        try:
            self.performance_metrics = {
                "accuracy_tracker": [],
                "precision_tracker": [],
                "recall_tracker": [],
                "f1_score_tracker": [],
                "inference_time_tracker": [],
                "error_rate_tracker": []
            }

            logger.info("Performance trackers initialized")

        except Exception as e:
            logger.error("Performance tracker initialization failed", error=str(e))

    async def record_prediction(self, model_name: str, input_data: Dict[str, Any],
                              prediction: Any, actual: Any = None,
                              inference_time: float = None) -> None:
        """Record a model prediction for monitoring."""
        try:
            timestamp = datetime.utcnow()

            # Initialize model data if not exists
            if model_name not in self.monitoring_data:
                self.monitoring_data[model_name] = {
                    "predictions": [],
                    "performance_history": [],
                    "drift_history": [],
                    "last_updated": timestamp
                }

            # Record prediction
            prediction_record = {
                "timestamp": timestamp,
                "input_data": input_data,
                "prediction": prediction,
                "actual": actual,
                "inference_time": inference_time
            }

            self.monitoring_data[model_name]["predictions"].append(prediction_record)

            # Keep only recent predictions (last 10000)
            if len(self.monitoring_data[model_name]["predictions"]) > 10000:
                self.monitoring_data[model_name]["predictions"] = \
                    self.monitoring_data[model_name]["predictions"][-10000:]

            # Update performance metrics if actual value is provided
            if actual is not None:
                await self._update_performance_metrics(
                    model_name,
                    prediction,
                    actual,
                    inference_time
                )

            # Check for drift periodically
            await self._check_drift_if_needed(model_name)

        except Exception as e:
            logger.error("Prediction recording failed", model_name=model_name, error=str(e))

    async def _update_performance_metrics(self, model_name: str, prediction: Any,
                                        actual: Any, inference_time: float = None) -> None:
        """Update performance metrics for a model."""
        try:
            timestamp = datetime.utcnow()

            # Calculate metrics based on prediction type
            if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                # Regression metrics
                error = abs(prediction - actual)
                metrics = {
                    "mae": error,
                    "mse": error ** 2,
                    "timestamp": timestamp
                }
            else:
                # Classification metrics (simplified)
                accuracy = 1.0 if prediction == actual else 0.0
                metrics = {
                    "accuracy": accuracy,
                    "timestamp": timestamp
                }

            # Add inference time if provided
            if inference_time is not None:
                metrics["inference_time"] = inference_time

            # Store metrics
            self.monitoring_data[model_name]["performance_history"].append(metrics)

            # Keep only recent metrics (last 1000)
            if len(self.monitoring_data[model_name]["performance_history"]) > 1000:
                self.monitoring_data[model_name]["performance_history"] = \
                    self.monitoring_data[model_name]["performance_history"][-1000:]

        except Exception as e:
            logger.error("Performance metrics update failed", model_name=model_name, error=str(e))

    async def _check_drift_if_needed(self, model_name: str) -> None:
        """Check for drift if it's time to do so."""
        try:
            model_data = self.monitoring_data[model_name]
            last_drift_check = model_data.get("last_drift_check", datetime.min)

            # Check if enough time has passed since last drift check
            time_since_check = datetime.utcnow() - last_drift_check
            detection_frequency = self.monitoring_config["drift_detection"]["detection_frequency"]

            if time_since_check.total_seconds() >= detection_frequency * 3600:
                await self._detect_drift(model_name)
                model_data["last_drift_check"] = datetime.utcnow()

        except Exception as e:
            logger.error("Drift check failed", model_name=model_name, error=str(e))

    async def _detect_drift(self, model_name: str) -> None:
        """Detect drift in model predictions."""
        try:
            model_data = self.monitoring_data[model_name]
            predictions = model_data["predictions"]

            if len(predictions) < 100:  # Need minimum data for drift detection
                return

            # Get reference and current data
            reference_size = min(500, len(predictions) // 2)
            current_size = min(500, len(predictions) // 2)

            reference_data = predictions[:reference_size]
            current_data = predictions[-current_size:]

            # Extract features for drift detection
            reference_features = self._extract_drift_features(reference_data)
            current_features = self._extract_drift_features(current_data)

            # Apply drift detection methods
            for method_name, method_func in self.drift_detectors.items():
                drift_score = await method_func(reference_features, current_features)

                # Check if drift exceeds threshold
                threshold = self.alert_thresholds["drift_score"]["warning"]
                if drift_score > threshold:
                    await self._handle_drift_alert(model_name, method_name, drift_score, threshold)

        except Exception as e:
            logger.error("Drift detection failed", model_name=model_name, error=str(e))

    def _extract_drift_features(self, prediction_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for drift detection."""
        try:
            features = []

            for record in prediction_data:
                # Extract numerical features from input data
                feature_vector = []
                input_data = record.get("input_data", {})

                for key, value in input_data.items():
                    if isinstance(value, (int, float)):
                        feature_vector.append(value)
                    elif isinstance(value, str):
                        # Simple string encoding
                        feature_vector.append(len(value))

                if feature_vector:
                    features.append(feature_vector)

            return np.array(features) if features else np.array([])

        except Exception as e:
            logger.error("Feature extraction for drift detection failed", error=str(e))
            return np.array([])

    async def _detect_statistical_drift(self, reference_features: np.ndarray,
                                      current_features: np.ndarray) -> float:
        """Detect drift using statistical methods."""
        try:
            if len(reference_features) == 0 or len(current_features) == 0:
                return 0.0

            # Calculate mean and std for each feature
            ref_means = np.mean(reference_features, axis=0)
            ref_stds = np.std(reference_features, axis=0)
            curr_means = np.mean(current_features, axis=0)
            curr_stds = np.std(current_features, axis=0)

            # Calculate drift score as normalized difference
            drift_scores = []
            for i in range(len(ref_means)):
                if ref_stds[i] > 0:
                    drift_score = abs(curr_means[i] - ref_means[i]) / ref_stds[i]
                    drift_scores.append(drift_score)

            return np.mean(drift_scores) if drift_scores else 0.0

        except Exception as e:
            logger.error("Statistical drift detection failed", error=str(e))
            return 0.0

    async def _detect_ks_drift(self, reference_features: np.ndarray,
                             current_features: np.ndarray) -> float:
        """Detect drift using Kolmogorov-Smirnov test."""
        try:
            if len(reference_features) == 0 or len(current_features) == 0:
                return 0.0

            # Simple KS test implementation
            drift_scores = []

            for i in range(reference_features.shape[1]):
                ref_feature = reference_features[:, i]
                curr_feature = current_features[:, i]

                # Calculate empirical CDFs
                ref_sorted = np.sort(ref_feature)
                curr_sorted = np.sort(curr_feature)

                # Simple KS statistic approximation
                combined = np.concatenate([ref_sorted, curr_sorted])
                combined_sorted = np.sort(combined)

                ref_cdf = np.searchsorted(
                    ref_sorted,
                    combined_sorted,
                    side='right') / len(ref_sorted
                )
                curr_cdf = np.searchsorted(
                    curr_sorted,
                    combined_sorted,
                    side='right') / len(curr_sorted
                )

                ks_stat = np.max(np.abs(ref_cdf - curr_cdf))
                drift_scores.append(ks_stat)

            return np.mean(drift_scores) if drift_scores else 0.0

        except Exception as e:
            logger.error("KS drift detection failed", error=str(e))
            return 0.0

    async def _detect_psi_drift(self, reference_features: np.ndarray,
                              current_features: np.ndarray) -> float:
        """Detect drift using Population Stability Index."""
        try:
            if len(reference_features) == 0 or len(current_features) == 0:
                return 0.0

            psi_scores = []

            for i in range(reference_features.shape[1]):
                ref_feature = reference_features[:, i]
                curr_feature = current_features[:, i]

                # Create bins
                min_val = min(np.min(ref_feature), np.min(curr_feature))
                max_val = max(np.max(ref_feature), np.max(curr_feature))

                if max_val == min_val:
                    continue

                bins = np.linspace(min_val, max_val, 11)  # 10 bins

                # Calculate distributions
                ref_hist, _ = np.histogram(ref_feature, bins=bins)
                curr_hist, _ = np.histogram(curr_feature, bins=bins)

                # Normalize
                ref_dist = ref_hist / np.sum(ref_hist)
                curr_dist = curr_hist / np.sum(curr_hist)

                # Calculate PSI
                psi = 0.0
                for j in range(len(ref_dist)):
                    if ref_dist[j] > 0 and curr_dist[j] > 0:
                        psi += (curr_dist[j] - ref_dist[j]) * np.log(curr_dist[j] / ref_dist[j])

                psi_scores.append(psi)

            return np.mean(psi_scores) if psi_scores else 0.0

        except Exception as e:
            logger.error("PSI drift detection failed", error=str(e))
            return 0.0

    async def _handle_drift_alert(self, model_name: str, method_name: str,
                                drift_score: float, threshold: float) -> None:
        """Handle drift alert."""
        try:
            severity = "warning"
            if drift_score > self.alert_thresholds["drift_score"]["critical"]:
                severity = "critical"

            alert = ModelDriftAlert(
                model_name=model_name,
                drift_type=method_name,
                drift_score=drift_score,
                threshold=threshold,
                detected_at=datetime.utcnow(),
                severity=severity,
                description=f"Model drift detected using {method_name} method",
                recommended_actions=[
                    "Investigate data distribution changes",
                    "Consider model retraining",
                    "Review feature engineering pipeline",
                    "Check data quality"
                ]
            )

            # Store alert
            if model_name not in self.monitoring_data:
                self.monitoring_data[model_name] = {"drift_history": []}

            self.monitoring_data[model_name]["drift_history"].append(alert)

            logger.warning("Model drift detected",
                          model_name=model_name,
                          method=method_name,
                          drift_score=drift_score,
                          severity=severity)

        except Exception as e:
            logger.error("Drift alert handling failed", error=str(e))

    async def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get current metrics for a model."""
        try:
            if model_name not in self.monitoring_data:
                return None

            model_data = self.monitoring_data[model_name]
            performance_history = model_data.get("performance_history", [])

            if not performance_history:
                return None

            # Calculate current metrics
            recent_metrics = performance_history[-100:]  # Last 100 predictions

            # Calculate averages
            accuracy = np.mean([m.get("accuracy", 0) for m in recent_metrics if "accuracy" in m])
            precision = 0.8  # Placeholder
            recall = 0.8     # Placeholder
            f1_score = 0.8   # Placeholder

            inference_times = [m.get(
                "inference_time",
                0
            ) for m in recent_metrics if "inference_time" in m]
            avg_inference_time = np.mean(inference_times) if inference_times else 0.0

            total_predictions = len(model_data.get("predictions", []))

            # Get drift score
            drift_history = model_data.get("drift_history", [])
            drift_score = drift_history[-1].drift_score if drift_history else 0.0

            return ModelMetrics(
                model_name=model_name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                inference_time_ms=avg_inference_time * 1000,
                total_predictions=total_predictions,
                drift_score=drift_score,
                last_updated=datetime.utcnow()
            )

        except Exception as e:
            logger.error("Model metrics retrieval failed", model_name=model_name, error=str(e))
            return None

    async def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get overall health status for a model."""
        try:
            if model_name not in self.monitoring_data:
                return {"status": "unknown", "message": "No monitoring data available"}

            model_data = self.monitoring_data[model_name]
            health_status = {"status": "healthy", "issues": [], "recommendations": []}

            # Check performance degradation
            performance_history = model_data.get("performance_history", [])
            if len(performance_history) > 10:
                recent_accuracy = np.mean([m.get("accuracy", 0) for m in performance_history[-10:]])
                baseline_accuracy = np.mean(
                    [m.get("accuracy",
                    0) for m in performance_history[:10]]
                )

                if baseline_accuracy > 0 and recent_accuracy < baseline_accuracy * 0.9:
                    health_status["status"] = "degraded"
                    health_status["issues"].append("Performance degradation detected")
                    health_status["recommendations"].append("Consider model retraining")

            # Check drift alerts
            drift_history = model_data.get("drift_history", [])
            recent_drift = [alert for alert in drift_history
                          if (datetime.utcnow() - alert.detected_at).days <= 7]

            if recent_drift:
                critical_drift = [alert for alert in recent_drift if alert.severity == "critical"]
                if critical_drift:
                    health_status["status"] = "unhealthy"
                    health_status["issues"].append("Critical drift detected")
                    health_status["recommendations"].append("Immediate model retraining required")
                elif health_status["status"] == "healthy":
                    health_status["status"] = "warning"
                    health_status["issues"].append("Drift detected")
                    health_status["recommendations"].append("Monitor closely and
                        consider retraining")

            return health_status

        except Exception as e:
            logger.error("Model health check failed", model_name=model_name, error=str(e))
            return {"status": "error", "message": str(e)}

    async def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored models."""
        try:
            summary = {
                "total_models": len(self.monitoring_data),
                "healthy_models": 0,
                "degraded_models": 0,
                "unhealthy_models": 0,
                "models_with_drift": 0,
                "total_predictions": 0,
                "avg_inference_time": 0.0
            }

            inference_times = []

            for model_name, model_data in self.monitoring_data.items():
                # Count predictions
                summary["total_predictions"] += len(model_data.get("predictions", []))

                # Collect inference times
                performance_history = model_data.get("performance_history", [])
                for metrics in performance_history:
                    if "inference_time" in metrics:
                        inference_times.append(metrics["inference_time"])

                # Check model health
                health = await self.get_model_health(model_name)
                if health["status"] == "healthy":
                    summary["healthy_models"] += 1
                elif health["status"] == "degraded" or health["status"] == "warning":
                    summary["degraded_models"] += 1
                else:
                    summary["unhealthy_models"] += 1

                # Check drift
                drift_history = model_data.get("drift_history", [])
                if drift_history:
                    summary["models_with_drift"] += 1

            # Calculate average inference time
            if inference_times:
                summary["avg_inference_time"] = np.mean(inference_times)

            return summary

        except Exception as e:
            logger.error("Monitoring summary failed", error=str(e))
            return {}

    def is_ready(self) -> bool:
        """Check if model monitor is ready."""
        return len(self.drift_detectors) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Clear monitoring data
            self.monitoring_data.clear()
            self.drift_detectors.clear()
            self.performance_metrics.clear()

            logger.info("Model monitor cleanup completed")

        except Exception as e:
            logger.error("Model monitor cleanup failed", error=str(e))
