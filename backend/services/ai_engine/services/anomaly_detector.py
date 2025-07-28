"""
Anomaly Detection Service for AI Engine.
Handles anomaly detection for Azure resources and infrastructure.
"""

import json
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from azure.monitor.query.aio import LogsQueryClient
from azure.identity.aio import DefaultAzureCredential

from ....shared.config import get_settings
from ..models import DetectionType

settings = get_settings()
logger = structlog.get_logger(__name__)


@dataclass
class AnomalyResult:
    """Represents an anomaly detection result."""
    timestamp: datetime
    resource_id: str
    anomaly_type: str
    severity: str
    confidence: float
    baseline_value: float
    observed_value: float
    deviation: float
    description: str
    recommended_actions: List[str]


class AnomalyDetector:
    """Anomaly detection service for Azure resources."""

    def __init__(self):
        self.settings = settings
        self.logs_client = None
        self.azure_credential = None
        self.detection_models = {}
        self.baseline_cache = {}
        self.anomaly_thresholds = self._load_anomaly_thresholds()
        self.detection_rules = self._load_detection_rules()

    def _load_anomaly_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load anomaly detection thresholds for different metrics."""
        return {
            "cpu_usage": {
                "low": 0.1,      # 10% CPU
                "medium": 0.5,   # 50% CPU
                "high": 0.8,     # 80% CPU
                "critical": 0.95 # 95% CPU
            },
            "memory_usage": {
                "low": 0.2,      # 20% Memory
                "medium": 0.6,   # 60% Memory
                "high": 0.85,    # 85% Memory
                "critical": 0.95 # 95% Memory
            },
            "disk_usage": {
                "low": 0.3,      # 30% Disk
                "medium": 0.7,   # 70% Disk
                "high": 0.9,     # 90% Disk
                "critical": 0.98 # 98% Disk
            },
            "network_latency": {
                "low": 50,       # 50ms
                "medium": 200,   # 200ms
                "high": 500,     # 500ms
                "critical": 1000 # 1000ms
            },
            "error_rate": {
                "low": 0.001,    # 0.1%
                "medium": 0.01,  # 1%
                "high": 0.05,    # 5%
                "critical": 0.1  # 10%
            },
            "cost_variance": {
                "low": 0.1,      # 10% variance
                "medium": 0.25,  # 25% variance
                "high": 0.5,     # 50% variance
                "critical": 1.0  # 100% variance
            }
        }

    def _load_detection_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load detection rules for different anomaly types."""
        return {
            "resource_usage": {
                "algorithm": "isolation_forest",
                "window_size": 24,  # hours
                "min_samples": 10,
                "contamination": 0.1,
                "features": ["cpu_usage", "memory_usage", "disk_usage", "network_io"]
            },
            "cost_anomaly": {
                "algorithm": "statistical",
                "window_size": 30,  # days
                "min_samples": 7,
                "std_threshold": 2.0,
                "features": ["daily_cost", "resource_count", "usage_hours"]
            },
            "security_anomaly": {
                "algorithm": "rule_based",
                "window_size": 1,   # hours
                "min_samples": 1,
                "features": ["login_attempts", "privilege_escalation", "data_access"]
            },
            "performance_anomaly": {
                "algorithm": "time_series",
                "window_size": 12,  # hours
                "min_samples": 20,
                "seasonality": True,
                "features": ["response_time", "throughput", "error_rate"]
            },
            "compliance_anomaly": {
                "algorithm": "rule_based",
                "window_size": 24,  # hours
                "min_samples": 1,
                "features": ["policy_violations", "access_violations", "configuration_drift"]
            }
        }

    async def initialize(self) -> None:
        """Initialize the anomaly detector."""
        try:
            logger.info("Initializing anomaly detector")

            # Initialize Azure clients
            if self.settings.is_production():
                await self._initialize_azure_clients()

            # Initialize detection models
            await self._initialize_detection_models()

            logger.info("Anomaly detector initialized successfully")

        except Exception as e:
            logger.error("Anomaly detector initialization failed", error=str(e))
            raise

    async def _initialize_azure_clients(self) -> None:
        """Initialize Azure clients for data collection."""
        try:
            self.azure_credential = DefaultAzureCredential()

            # Initialize Azure Monitor Logs client
            self.logs_client = LogsQueryClient(self.azure_credential)

            logger.info("Azure clients initialized for anomaly detection")

        except Exception as e:
            logger.warning("Failed to initialize Azure clients", error=str(e))

    async def _initialize_detection_models(self) -> None:
        """Initialize anomaly detection models."""
        try:
            # Initialize different detection models
            for detection_type, config in self.detection_rules.items():
                if config["algorithm"] == "isolation_forest":
                    self.detection_models[detection_type] = (
                        await self._create_isolation_forest_model(config)
                    )
                elif config["algorithm"] == "statistical":
                    self.detection_models[detection_type] = (
                        await self._create_statistical_model(config)
                    )
                elif config["algorithm"] == "rule_based":
                    self.detection_models[detection_type] = (
                        await self._create_rule_based_model(config)
                    )
                elif config["algorithm"] == "time_series":
                    self.detection_models[detection_type] = (
                        await self._create_time_series_model(config)
                    )

            logger.info("Detection models initialized", model_count=len(self.detection_models))

        except Exception as e:
            logger.error("Failed to initialize detection models", error=str(e))

    async def _create_isolation_forest_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an Isolation Forest model for anomaly detection."""
        try:
            # Simulated Isolation Forest model
            model = {
                "type": "isolation_forest",
                "contamination": config.get("contamination", 0.1),
                "n_estimators": config.get("n_estimators", 100),
                "features": config.get("features", []),
                "window_size": config.get("window_size", 24),
                "min_samples": config.get("min_samples", 10),
                "trained": False,
                "last_updated": datetime.utcnow()
            }

            return model

        except Exception as e:
            logger.error("Failed to create isolation forest model", error=str(e))
            raise

    async def _create_statistical_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a statistical model for anomaly detection."""
        try:
            model = {
                "type": "statistical",
                "std_threshold": config.get("std_threshold", 2.0),
                "features": config.get("features", []),
                "window_size": config.get("window_size", 30),
                "min_samples": config.get("min_samples", 7),
                "baseline_stats": {},
                "last_updated": datetime.utcnow()
            }

            return model

        except Exception as e:
            logger.error("Failed to create statistical model", error=str(e))
            raise

    async def _create_rule_based_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rule-based model for anomaly detection."""
        try:
            model = {
                "type": "rule_based",
                "features": config.get("features", []),
                "window_size": config.get("window_size", 1),
                "rules": self._define_detection_rules(config.get("features", [])),
                "last_updated": datetime.utcnow()
            }

            return model

        except Exception as e:
            logger.error("Failed to create rule-based model", error=str(e))
            raise

    async def _create_time_series_model(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a time series model for anomaly detection."""
        try:
            model = {
                "type": "time_series",
                "features": config.get("features", []),
                "window_size": config.get("window_size", 12),
                "min_samples": config.get("min_samples", 20),
                "seasonality": config.get("seasonality", True),
                "forecast_horizon": 24,  # hours
                "last_updated": datetime.utcnow()
            }

            return model

        except Exception as e:
            logger.error("Failed to create time series model", error=str(e))
            raise

    def _define_detection_rules(self, features: List[str]) -> List[Dict[str, Any]]:
        """Define detection rules for rule-based models."""
        rules = []

        if "login_attempts" in features:
            rules.append({
                "name": "excessive_login_attempts",
                "condition": "login_attempts > 10",
                "severity": "high",
                "description": "Excessive login attempts detected"
            })

        if "privilege_escalation" in features:
            rules.append({
                "name": "unauthorized_privilege_escalation",
                "condition": "privilege_escalation > 0",
                "severity": "critical",
                "description": "Unauthorized privilege escalation attempt"
            })

        if "policy_violations" in features:
            rules.append({
                "name": "policy_violation",
                "condition": "policy_violations > 0",
                "severity": "medium",
                "description": "Policy violation detected"
            })

        if "configuration_drift" in features:
            rules.append({
                "name": "configuration_drift",
                "condition": "configuration_drift > 0",
                "severity": "medium",
                "description": "Configuration drift detected"
            })

        return rules

    async def detect_anomalies(self, resource_data: Dict[str, Any],
                             detection_type: str, threshold: float) -> Dict[str, Any]:
        """Detect anomalies in resource data."""
        try:
            logger.info("Starting anomaly detection",
                       detection_type=detection_type,
                       threshold=threshold)

            # Get detection model
            model = self.detection_models.get(detection_type)
            if not model:
                raise ValueError(f"No model available for detection type: {detection_type}")

            # Extract features from resource data
            features = await self._extract_features(resource_data, model["features"])

            # Perform anomaly detection based on model type
            if model["type"] == "isolation_forest":
                anomalies = await self._detect_isolation_forest(features, model, threshold)
            elif model["type"] == "statistical":
                anomalies = await self._detect_statistical(features, model, threshold)
            elif model["type"] == "rule_based":
                anomalies = await self._detect_rule_based(features, model, threshold)
            elif model["type"] == "time_series":
                anomalies = await self._detect_time_series(features, model, threshold)
            else:
                raise ValueError(f"Unknown model type: {model['type']}")

            # Process and format results
            results = await self._process_anomalies(anomalies, resource_data)

            logger.info("Anomaly detection completed",
                       detection_type=detection_type,
                       anomalies_found=len(results["anomalies"]))

            return results

        except Exception as e:
            logger.error("Anomaly detection failed", error=str(e))
            raise

    async def _extract_features(self, resource_data: Dict[str, Any],
                               feature_names: List[str]) -> Dict[str, Any]:
        """Extract features from resource data."""
        try:
            features = {}

            for feature_name in feature_names:
                if feature_name in resource_data:
                    features[feature_name] = resource_data[feature_name]
                else:
                    # Try to extract from nested data
                    for key, value in resource_data.items():
                        if isinstance(value, dict) and feature_name in value:
                            features[feature_name] = value[feature_name]
                            break
                    else:
                        # Set default value
                        features[feature_name] = 0

            return features

        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            return {}

    async def _detect_isolation_forest(self, features: Dict[str, Any],
                                     model: Dict[str, Any], threshold: float) -> List[AnomalyResult]:
        """Detect anomalies using Isolation Forest algorithm."""
        try:
            anomalies = []

            # Simulated Isolation Forest detection
            for feature_name, value in features.items():
                # Simple outlier detection based on thresholds
                if feature_name in self.anomaly_thresholds:
                    thresholds = self.anomaly_thresholds[feature_name]

                    if isinstance(value, (int, float)):
                        if value > thresholds["critical"]:
                            severity = "critical"
                            confidence = 0.95
                        elif value > thresholds["high"]:
                            severity = "high"
                            confidence = 0.85
                        elif value > thresholds["medium"]:
                            severity = "medium"
                            confidence = 0.75
                        else:
                            continue

                        if confidence >= threshold:
                            anomaly = AnomalyResult(
                                timestamp=datetime.utcnow(),
                                resource_id=features.get("resource_id", "unknown"),
                                anomaly_type=f"{feature_name}_anomaly",
                                severity=severity,
                                confidence=confidence,
                                baseline_value=thresholds["medium"],
                                observed_value=value,
                                deviation=abs(value - thresholds["medium"]),
                                description=f"Anomalous {feature_name} detected: {value}",
                                recommended_actions=[
                                    f"Investigate {feature_name} spike",
                                    f"Check resource scaling for {feature_name}",
                                    f"Review {feature_name} usage patterns"
                                ]
                            )
                            anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error("Isolation forest detection failed", error=str(e))
            return []

    async def _detect_statistical(self, features: Dict[str, Any],
                                model: Dict[str, Any], threshold: float) -> List[AnomalyResult]:
        """Detect anomalies using statistical methods."""
        try:
            anomalies = []

            # Get baseline statistics
            baseline_stats = model.get("baseline_stats", {})
            std_threshold = model.get("std_threshold", 2.0)

            for feature_name, value in features.items():
                if feature_name in baseline_stats:
                    stats = baseline_stats[feature_name]
                    mean = stats.get("mean", 0)
                    std = stats.get("std", 1)

                    # Calculate z-score
                    z_score = abs(value - mean) / std if std > 0 else 0

                    if z_score > std_threshold:
                        confidence = min(z_score / std_threshold, 1.0)

                        if confidence >= threshold:
                            severity = "high" if z_score > 3 else "medium"

                            anomaly = AnomalyResult(
                                timestamp=datetime.utcnow(),
                                resource_id=features.get("resource_id", "unknown"),
                                anomaly_type=f"{feature_name}_statistical_anomaly",
                                severity=severity,
                                confidence=confidence,
                                baseline_value=mean,
                                observed_value=value,
                                deviation=abs(value - mean),
                                description = (
                                    f"Statistical anomaly in {feature_name}: z-score {z_score:.2f}",
                                )
                                recommended_actions=[
                                    f"Investigate {feature_name} deviation",
                                    f"Update baseline for {feature_name}",
                                    f"Check data quality for {feature_name}"
                                ]
                            )
                            anomalies.append(anomaly)
                else:
                    # Initialize baseline stats
                    baseline_stats[feature_name] = {
                        "mean": value,
                        "std": 0,
                        "min": value,
                        "max": value,
                        "count": 1
                    }

            # Update model with new baseline stats
            model["baseline_stats"] = baseline_stats

            return anomalies

        except Exception as e:
            logger.error("Statistical detection failed", error=str(e))
            return []

    async def _detect_rule_based(self, features: Dict[str, Any],
                               model: Dict[str, Any], threshold: float) -> List[AnomalyResult]:
        """Detect anomalies using rule-based methods."""
        try:
            anomalies = []
            rules = model.get("rules", [])

            for rule in rules:
                condition = rule["condition"]

                # Simple condition evaluation
                if self._evaluate_condition(condition, features):
                    confidence = 1.0  # Rule-based detection has high confidence

                    if confidence >= threshold:
                        anomaly = AnomalyResult(
                            timestamp=datetime.utcnow(),
                            resource_id=features.get("resource_id", "unknown"),
                            anomaly_type=rule["name"],
                            severity=rule["severity"],
                            confidence=confidence,
                            baseline_value=0,
                            observed_value=1,
                            deviation=1,
                            description=rule["description"],
                            recommended_actions=[
                                f"Investigate {rule['name']}",
                                f"Review security logs",
                                f"Check compliance status"
                            ]
                        )
                        anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error("Rule-based detection failed", error=str(e))
            return []

    async def _detect_time_series(self, features: Dict[str, Any],
                                model: Dict[str, Any], threshold: float) -> List[AnomalyResult]:
        """Detect anomalies using time series analysis."""
        try:
            anomalies = []

            # Simulated time series anomaly detection
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    # Simple trend analysis
                    expected_range = self._calculate_expected_range(feature_name, model)

                    if value < expected_range["min"] or value > expected_range["max"]:
                        confidence = 0.8

                        if confidence >= threshold:
                            severity = "high" if value > expected_range["max"] * 2 else "medium"

                            anomaly = AnomalyResult(
                                timestamp=datetime.utcnow(),
                                resource_id=features.get("resource_id", "unknown"),
                                anomaly_type=f"{feature_name}_time_series_anomaly",
                                severity=severity,
                                confidence=confidence,
                                baseline_value=(expected_range["min"] + expected_range["max"]) / 2,
                                observed_value=value,
                                deviation=min(abs(value - expected_range["min"]),
                                            abs(value - expected_range["max"])),
                                description=f"Time series anomaly in {feature_name}",
                                recommended_actions=[
                                    f"Analyze {feature_name} trends",
                                    f"Check seasonal patterns",
                                    f"Review forecasting model"
                                ]
                            )
                            anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error("Time series detection failed", error=str(e))
            return []

    def _evaluate_condition(self, condition: str, features: Dict[str, Any]) -> bool:
        """Evaluate a rule condition."""
        try:
            # Simple condition evaluation (production would use safe evaluation)
            for feature_name, value in features.items():
                condition = condition.replace(feature_name, str(value))

            # Basic condition evaluation
            if " > " in condition:
                parts = condition.split(" > ")
                if len(parts) == 2:
                    try:
                        left = float(parts[0])
                        right = float(parts[1])
                        return left > right
                    except ValueError:
                        return False

            return False

        except Exception as e:
            logger.error("Condition evaluation failed", error=str(e))
            return False

    def _calculate_expected_range(
        self,
        feature_name: str,
        model: Dict[str,
        Any]
    ) -> Dict[str, float]:
        """Calculate expected range for time series feature."""
        try:
            # Simple expected range calculation
            if feature_name in self.anomaly_thresholds:
                thresholds = self.anomaly_thresholds[feature_name]
                return {
                    "min": thresholds["low"],
                    "max": thresholds["high"]
                }
            else:
                return {"min": 0, "max": 100}

        except Exception as e:
            logger.error("Expected range calculation failed", error=str(e))
            return {"min": 0, "max": 100}

    async def _process_anomalies(self, anomalies: List[AnomalyResult],
                               resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format anomaly results."""
        try:
            # Convert anomalies to dictionary format
            anomaly_dicts = []
            for anomaly in anomalies:
                anomaly_dict = {
                    "timestamp": anomaly.timestamp.isoformat(),
                    "resource_id": anomaly.resource_id,
                    "anomaly_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "confidence": anomaly.confidence,
                    "baseline_value": anomaly.baseline_value,
                    "observed_value": anomaly.observed_value,
                    "deviation": anomaly.deviation,
                    "description": anomaly.description,
                    "recommended_actions": anomaly.recommended_actions
                }
                anomaly_dicts.append(anomaly_dict)

            # Calculate severity distribution
            severity_counts = {}
            for anomaly in anomalies:
                severity = anomaly.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Calculate overall confidence
            if anomalies:
                overall_confidence = sum(a.confidence for a in anomalies) / len(anomalies)
            else:
                overall_confidence = 0.0

            # Generate summary
            summary = {
                "total_anomalies": len(anomalies),
                "severity_distribution": severity_counts,
                "overall_confidence": overall_confidence,
                "detection_timestamp": datetime.utcnow().isoformat(),
                "resource_count": len(set(a.resource_id for a in anomalies))
            }

            return {
                "anomalies": anomaly_dicts,
                "summary": summary,
                "confidence": overall_confidence
            }

        except Exception as e:
            logger.error("Anomaly processing failed", error=str(e))
            return {"anomalies": [], "summary": {}, "confidence": 0.0}

    def is_ready(self) -> bool:
        """Check if anomaly detector is ready."""
        return len(self.detection_models) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Clear caches
            self.detection_models.clear()
            self.baseline_cache.clear()

            # Close Azure clients
            if self.logs_client:
                await self.logs_client.close()

            logger.info("Anomaly detector cleanup completed")

        except Exception as e:
            logger.error("Anomaly detector cleanup failed", error=str(e))
