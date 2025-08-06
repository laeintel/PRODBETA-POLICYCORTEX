"""
Feature Engineering Service for AI Engine.
Handles feature extraction, transformation, and engineering for ML models.
"""

import json
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

settings = get_settings()
logger = structlog.get_logger(__name__)


class FeatureEngineer:
    """Feature engineering service for ML pipelines."""

    def __init__(self):
        self.settings = settings
        self.feature_extractors = {}
        self.transformers = {}
        self.feature_definitions = self._load_feature_definitions()
        self.preprocessing_pipelines = self._load_preprocessing_pipelines()

    def _load_feature_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load feature definitions for different data types."""
        return {
            "resource_metrics": {
                "cpu_utilization": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "transformations": ["normalize", "smooth"],
                },
                "memory_utilization": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "transformations": ["normalize", "smooth"],
                },
                "disk_io": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": None,
                    "transformations": ["log_transform", "normalize"],
                },
                "network_io": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": None,
                    "transformations": ["log_transform", "normalize"],
                },
            },
            "cost_data": {
                "daily_cost": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": None,
                    "transformations": ["smooth", "normalize"],
                },
                "resource_count": {
                    "type": "numerical",
                    "min_value": 0,
                    "max_value": None,
                    "transformations": ["normalize"],
                },
                "service_category": {
                    "type": "categorical",
                    "categories": ["compute", "storage", "networking", "database", "ai"],
                    "transformations": ["one_hot_encode"],
                },
            },
            "security_events": {
                "login_attempts": {
                    "type": "numerical",
                    "min_value": 0,
                    "max_value": None,
                    "transformations": ["clip_outliers", "normalize"],
                },
                "failed_logins": {
                    "type": "numerical",
                    "min_value": 0,
                    "max_value": None,
                    "transformations": ["clip_outliers", "normalize"],
                },
                "ip_address": {"type": "categorical", "transformations": ["ip_to_features"]},
                "user_agent": {"type": "text", "transformations": ["text_to_features"]},
            },
            "compliance_data": {
                "policy_violations": {
                    "type": "numerical",
                    "min_value": 0,
                    "max_value": None,
                    "transformations": ["smooth", "normalize"],
                },
                "compliance_score": {
                    "type": "numerical",
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "transformations": ["smooth"],
                },
                "policy_type": {
                    "type": "categorical",
                    "categories": ["security", "cost", "governance", "compliance"],
                    "transformations": ["one_hot_encode"],
                },
            },
        }

    def _load_preprocessing_pipelines(self) -> Dict[str, List[str]]:
        """Load preprocessing pipelines for different use cases."""
        return {
            "anomaly_detection": [
                "handle_missing_values",
                "remove_outliers",
                "normalize_features",
                "create_time_features",
                "create_aggregate_features",
            ],
            "cost_optimization": [
                "handle_missing_values",
                "create_cost_features",
                "create_usage_features",
                "create_time_features",
                "normalize_features",
            ],
            "predictive_analytics": [
                "handle_missing_values",
                "create_time_features",
                "create_lag_features",
                "create_rolling_features",
                "normalize_features",
            ],
            "sentiment_analysis": [
                "clean_text",
                "create_text_features",
                "create_sentiment_features",
                "normalize_features",
            ],
        }

    async def initialize(self) -> None:
        """Initialize the feature engineer."""
        try:
            logger.info("Initializing feature engineer")

            # Initialize feature extractors
            await self._initialize_extractors()

            # Initialize transformers
            await self._initialize_transformers()

            logger.info("Feature engineer initialized successfully")

        except Exception as e:
            logger.error("Feature engineer initialization failed", error=str(e))
            raise

    async def _initialize_extractors(self) -> None:
        """Initialize feature extractors."""
        try:
            self.feature_extractors = {
                "time_features": self._extract_time_features,
                "aggregate_features": self._extract_aggregate_features,
                "lag_features": self._extract_lag_features,
                "rolling_features": self._extract_rolling_features,
                "text_features": self._extract_text_features,
                "ip_features": self._extract_ip_features,
                "cost_features": self._extract_cost_features,
                "usage_features": self._extract_usage_features,
            }

            logger.info("Feature extractors initialized")

        except Exception as e:
            logger.error("Feature extractor initialization failed", error=str(e))

    async def _initialize_transformers(self) -> None:
        """Initialize feature transformers."""
        try:
            self.transformers = {
                "normalize": self._normalize_features,
                "smooth": self._smooth_features,
                "log_transform": self._log_transform_features,
                "clip_outliers": self._clip_outliers,
                "one_hot_encode": self._one_hot_encode,
                "handle_missing_values": self._handle_missing_values,
                "remove_outliers": self._remove_outliers,
            }

            logger.info("Feature transformers initialized")

        except Exception as e:
            logger.error("Feature transformer initialization failed", error=str(e))

    async def engineer_features(
        self,
        raw_data: Dict[str, Any],
        feature_types: List[str],
        preprocessing_steps: Optional[List[str]] = None,
        target_variable: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Engineer features from raw data."""
        try:
            logger.info(
                "Starting feature engineering",
                feature_types=feature_types,
                preprocessing_steps=preprocessing_steps,
            )

            # Initialize results
            results = {
                "engineered_features": {},
                "feature_importance": {},
                "preprocessing_summary": {},
                "feature_statistics": {},
                "processing_metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "feature_types": feature_types,
                    "preprocessing_steps": preprocessing_steps or [],
                    "target_variable": target_variable,
                },
            }

            # Extract features based on types
            for feature_type in feature_types:
                if feature_type in self.feature_extractors:
                    features = await self.feature_extractors[feature_type](raw_data)
                    results["engineered_features"].update(features)

            # Apply preprocessing steps
            if preprocessing_steps:
                for step in preprocessing_steps:
                    if step in self.transformers:
                        results["engineered_features"] = await self.transformers[step](
                            results["engineered_features"]
                        )
                        results["preprocessing_summary"][step] = "applied"
                    else:
                        logger.warning(f"Unknown preprocessing step: {step}")

            # Calculate feature statistics
            results["feature_statistics"] = await self._calculate_feature_statistics(
                results["engineered_features"]
            )

            # Calculate feature importance if target variable is provided
            if target_variable and target_variable in results["engineered_features"]:
                results["feature_importance"] = await self._calculate_feature_importance(
                    results["engineered_features"], target_variable
                )

            logger.info(
                "Feature engineering completed",
                features_count=len(results["engineered_features"]),
                preprocessing_steps_applied=len(results["preprocessing_summary"]),
            )

            return results

        except Exception as e:
            logger.error("Feature engineering failed", error=str(e))
            raise

    async def _extract_time_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time-based features."""
        try:
            features = {}

            # Look for timestamp fields
            for key, value in data.items():
                if "timestamp" in key.lower() or "time" in key.lower():
                    if isinstance(value, str):
                        try:
                            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        except:
                            continue
                    elif isinstance(value, datetime):
                        dt = value
                    else:
                        continue

                    # Extract time features
                    prefix = f"{key}_"
                    features[f"{prefix}hour"] = dt.hour
                    features[f"{prefix}day_of_week"] = dt.weekday()
                    features[f"{prefix}day_of_month"] = dt.day
                    features[f"{prefix}month"] = dt.month
                    features[f"{prefix}quarter"] = (dt.month - 1) // 3 + 1
                    features[f"{prefix}is_weekend"] = 1 if dt.weekday() >= 5 else 0
                    features[f"{prefix}is_business_hour"] = 1 if 9 <= dt.hour <= 17 else 0

            return features

        except Exception as e:
            logger.error("Time feature extraction failed", error=str(e))
            return {}

    async def _extract_aggregate_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aggregate features from data."""
        try:
            features = {}

            # Find numerical arrays/lists for aggregation
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(v, (int, float)) for v in value):
                        features[f"{key}_mean"] = np.mean(value)
                        features[f"{key}_std"] = np.std(value)
                        features[f"{key}_min"] = np.min(value)
                        features[f"{key}_max"] = np.max(value)
                        features[f"{key}_sum"] = np.sum(value)
                        features[f"{key}_count"] = len(value)

                        # Percentiles
                        features[f"{key}_p25"] = np.percentile(value, 25)
                        features[f"{key}_p50"] = np.percentile(value, 50)
                        features[f"{key}_p75"] = np.percentile(value, 75)
                        features[f"{key}_p90"] = np.percentile(value, 90)
                        features[f"{key}_p95"] = np.percentile(value, 95)

            return features

        except Exception as e:
            logger.error("Aggregate feature extraction failed", error=str(e))
            return {}

    async def _extract_lag_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract lag features for time series data."""
        try:
            features = {}

            # Create lag features for time series
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 1:
                    if all(isinstance(v, (int, float)) for v in value):
                        # Create lag features
                        for lag in [1, 2, 3, 6, 12, 24]:
                            if len(value) > lag:
                                features[f"{key}_lag_{lag}"] = value[-lag - 1]

            return features

        except Exception as e:
            logger.error("Lag feature extraction failed", error=str(e))
            return {}

    async def _extract_rolling_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract rolling window features."""
        try:
            features = {}

            # Create rolling window features
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 1:
                    if all(isinstance(v, (int, float)) for v in value):
                        # Different window sizes
                        for window in [3, 6, 12, 24]:
                            if len(value) >= window:
                                window_data = value[-window:]
                                features[f"{key}_rolling_{window}_mean"] = np.mean(window_data)
                                features[f"{key}_rolling_{window}_std"] = np.std(window_data)
                                features[f"{key}_rolling_{window}_min"] = np.min(window_data)
                                features[f"{key}_rolling_{window}_max"] = np.max(window_data)

            return features

        except Exception as e:
            logger.error("Rolling feature extraction failed", error=str(e))
            return {}

    async def _extract_text_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from text data."""
        try:
            features = {}

            # Process text fields
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 0:
                    # Basic text features
                    features[f"{key}_length"] = len(value)
                    features[f"{key}_word_count"] = len(value.split())
                    features[f"{key}_char_count"] = len(value)
                    features[f"{key}_uppercase_ratio"] = sum(c.isupper() for c in value) / len(
                        value
                    )
                    features[f"{key}_digit_ratio"] = sum(c.isdigit() for c in value) / len(value)
                    features[f"{key}_special_char_ratio"] = sum(
                        not c.isalnum() and not c.isspace() for c in value
                    ) / len(value)

                    # Sentiment indicators
                    positive_words = ["good", "great", "excellent", "secure", "compliant"]
                    negative_words = ["bad", "poor", "insecure", "violation", "failed"]

                    value_lower = value.lower()
                    features[f"{key}_positive_words"] = sum(
                        word in value_lower for word in positive_words
                    )
                    features[f"{key}_negative_words"] = sum(
                        word in value_lower for word in negative_words
                    )

            return features

        except Exception as e:
            logger.error("Text feature extraction failed", error=str(e))
            return {}

    async def _extract_ip_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from IP addresses."""
        try:
            features = {}

            # Process IP address fields
            for key, value in data.items():
                if isinstance(value, str) and "ip" in key.lower():
                    # Basic IP features
                    ip_parts = value.split(".")
                    if len(ip_parts) == 4:
                        try:
                            features[f"{key}_first_octet"] = int(ip_parts[0])
                            features[f"{key}_second_octet"] = int(ip_parts[1])
                            features[f"{key}_third_octet"] = int(ip_parts[2])
                            features[f"{key}_fourth_octet"] = int(ip_parts[3])

                            # Network class
                            first_octet = int(ip_parts[0])
                            if 1 <= first_octet <= 126:
                                features[f"{key}_class"] = 1  # Class A
                            elif 128 <= first_octet <= 191:
                                features[f"{key}_class"] = 2  # Class B
                            elif 192 <= first_octet <= 223:
                                features[f"{key}_class"] = 3  # Class C
                            else:
                                features[f"{key}_class"] = 4  # Other

                            # Private IP
                            is_private = (
                                (first_octet == 10)
                                or (first_octet == 172 and 16 <= int(ip_parts[1]) <= 31)
                                or (first_octet == 192 and int(ip_parts[1]) == 168)
                            )
                            features[f"{key}_is_private"] = 1 if is_private else 0

                        except ValueError:
                            pass

            return features

        except Exception as e:
            logger.error("IP feature extraction failed", error=str(e))
            return {}

    async def _extract_cost_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cost-related features."""
        try:
            features = {}

            # Look for cost-related fields
            cost_fields = []
            for key, value in data.items():
                if "cost" in key.lower() or "price" in key.lower() or "billing" in key.lower():
                    if isinstance(value, (int, float)):
                        cost_fields.append(value)
                        features[f"{key}_normalized"] = value

            # Calculate cost ratios and comparisons
            if len(cost_fields) > 1:
                total_cost = sum(cost_fields)
                features["total_cost"] = total_cost
                features["avg_cost"] = total_cost / len(cost_fields)
                features["max_cost"] = max(cost_fields)
                features["min_cost"] = min(cost_fields)
                features["cost_variance"] = np.var(cost_fields)

            return features

        except Exception as e:
            logger.error("Cost feature extraction failed", error=str(e))
            return {}

    async def _extract_usage_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract usage-related features."""
        try:
            features = {}

            # Look for usage-related fields
            usage_fields = []
            for key, value in data.items():
                if any(
                    term in key.lower()
                    for term in ["usage", "utilization", "consumption", "activity"]
                ):
                    if isinstance(value, (int, float)):
                        usage_fields.append(value)
                        features[f"{key}_normalized"] = value

            # Calculate usage statistics
            if len(usage_fields) > 1:
                features["total_usage"] = sum(usage_fields)
                features["avg_usage"] = sum(usage_fields) / len(usage_fields)
                features["max_usage"] = max(usage_fields)
                features["min_usage"] = min(usage_fields)
                features["usage_variance"] = np.var(usage_fields)

            return features

        except Exception as e:
            logger.error("Usage feature extraction failed", error=str(e))
            return {}

    async def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize numerical features."""
        try:
            normalized_features = {}

            # Collect numerical features
            numerical_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numerical_features[key] = value
                else:
                    normalized_features[key] = value

            if numerical_features:
                values = list(numerical_features.values())
                min_val = min(values)
                max_val = max(values)

                if max_val > min_val:
                    for key, value in numerical_features.items():
                        normalized_features[key] = (value - min_val) / (max_val - min_val)
                else:
                    normalized_features.update(numerical_features)

            return normalized_features

        except Exception as e:
            logger.error("Feature normalization failed", error=str(e))
            return features

    async def _smooth_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply smoothing to features."""
        try:
            # Simple smoothing for now
            return features

        except Exception as e:
            logger.error("Feature smoothing failed", error=str(e))
            return features

    async def _log_transform_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply log transformation to features."""
        try:
            transformed_features = {}

            for key, value in features.items():
                if isinstance(value, (int, float)) and value > 0:
                    transformed_features[key] = np.log1p(value)  # log(1 + x)
                else:
                    transformed_features[key] = value

            return transformed_features

        except Exception as e:
            logger.error("Log transformation failed", error=str(e))
            return features

    async def _clip_outliers(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Clip outliers in features."""
        try:
            clipped_features = {}

            # Collect numerical features
            numerical_features = []
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    numerical_features.append(value)

            if numerical_features:
                q25 = np.percentile(numerical_features, 25)
                q75 = np.percentile(numerical_features, 75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr

                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        clipped_features[key] = np.clip(value, lower_bound, upper_bound)
                    else:
                        clipped_features[key] = value
            else:
                clipped_features = features

            return clipped_features

        except Exception as e:
            logger.error("Outlier clipping failed", error=str(e))
            return features

    async def _one_hot_encode(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply one-hot encoding to categorical features."""
        try:
            encoded_features = {}

            for key, value in features.items():
                if isinstance(value, str):
                    # Simple one-hot encoding
                    categories = ["compute", "storage", "networking", "database", "ai", "security"]
                    for category in categories:
                        encoded_features[f"{key}_{category}"] = 1 if value == category else 0
                else:
                    encoded_features[key] = value

            return encoded_features

        except Exception as e:
            logger.error("One-hot encoding failed", error=str(e))
            return features

    async def _handle_missing_values(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in features."""
        try:
            handled_features = {}

            for key, value in features.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    # Fill with appropriate default
                    if "count" in key.lower():
                        handled_features[key] = 0
                    elif "ratio" in key.lower():
                        handled_features[key] = 0.0
                    elif "score" in key.lower():
                        handled_features[key] = 0.5
                    else:
                        handled_features[key] = 0
                else:
                    handled_features[key] = value

            return handled_features

        except Exception as e:
            logger.error("Missing value handling failed", error=str(e))
            return features

    async def _remove_outliers(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Remove outliers from features."""
        try:
            # For now, just return features as-is
            # In production, would implement proper outlier removal
            return features

        except Exception as e:
            logger.error("Outlier removal failed", error=str(e))
            return features

    async def _calculate_feature_statistics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for engineered features."""
        try:
            stats = {
                "total_features": len(features),
                "numerical_features": 0,
                "categorical_features": 0,
                "text_features": 0,
                "missing_values": 0,
            }

            for key, value in features.items():
                if isinstance(value, (int, float)):
                    stats["numerical_features"] += 1
                elif isinstance(value, str):
                    if len(value) > 50:
                        stats["text_features"] += 1
                    else:
                        stats["categorical_features"] += 1

                if value is None:
                    stats["missing_values"] += 1

            return stats

        except Exception as e:
            logger.error("Feature statistics calculation failed", error=str(e))
            return {}

    async def _calculate_feature_importance(
        self, features: Dict[str, Any], target_variable: str
    ) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            importance_scores = {}

            if target_variable not in features:
                return importance_scores

            target_value = features[target_variable]

            # Simple correlation-based importance
            for key, value in features.items():
                if key != target_variable and isinstance(value, (int, float)):
                    # Simple correlation approximation
                    if isinstance(target_value, (int, float)):
                        importance_scores[key] = abs(np.corrcoef([value], [target_value])[0, 1])
                    else:
                        importance_scores[key] = 0.0

            return importance_scores

        except Exception as e:
            logger.error("Feature importance calculation failed", error=str(e))
            return {}

    def is_ready(self) -> bool:
        """Check if feature engineer is ready."""
        return len(self.feature_extractors) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Clear extractors and transformers
            self.feature_extractors.clear()
            self.transformers.clear()

            logger.info("Feature engineer cleanup completed")

        except Exception as e:
            logger.error("Feature engineer cleanup failed", error=str(e))
