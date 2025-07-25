"""
Predictive Analytics Service for AI Engine.
Provides AI-driven predictive analytics for Azure resource usage and trends.
Enhanced with advanced ML models for governance prediction and forecasting.
"""

import json
import numpy as np
import pandas as pd
import asyncio
import pickle
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import structlog
from azure.monitor.query.aio import LogsQueryClient
from azure.identity.aio import DefaultAzureCredential
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
import joblib

from ....shared.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


class AdvancedPredictionType(Enum):
    """Advanced prediction types for governance scenarios."""
    COMPLIANCE_VIOLATION = "compliance_violation"
    COST_FORECAST = "cost_forecast"
    RESOURCE_UTILIZATION = "resource_utilization"
    SECURITY_RISK = "security_risk"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    POLICY_DRIFT = "policy_drift"
    RBAC_ANOMALY = "rbac_anomaly"


@dataclass
class PredictionResult:
    """Represents a prediction result."""
    timestamp: datetime
    metric_name: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    seasonality: Optional[str] = None
    anomaly_probability: float = 0.0


@dataclass
class GovernancePredictionRequest:
    """Request for advanced governance prediction."""
    prediction_type: AdvancedPredictionType
    data: Dict[str, Any]
    time_horizon: int  # hours
    confidence_threshold: float
    context: Dict[str, Any] = None


@dataclass
class GovernancePredictionResult:
    """Result of advanced governance prediction."""
    prediction_id: str
    prediction_type: AdvancedPredictionType
    predicted_value: Any
    confidence_score: float
    time_horizon: int
    factors: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime
    risk_level: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    rmse: float
    mae: float
    last_trained: datetime
    training_samples: int


class PredictiveAnalyticsService:
    """Predictive analytics service for resource usage forecasting."""
    
    def __init__(self):
        self.settings = settings
        self.logs_client = None
        self.azure_credential = None
        self.forecasting_models = {}
        self.trend_analyzers = {}
        self.seasonality_patterns = self._load_seasonality_patterns()
        self.prediction_config = self._load_prediction_config()
        
        # Advanced governance prediction components
        self.governance_models = {}
        self.feature_scalers = {}
        self.label_encoders = {}
        self.model_metrics = {}
        self.prediction_cache = {}
        self.is_advanced_initialized = False
    
    def _load_seasonality_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load seasonality patterns for different metrics."""
        return {
            "cpu_usage": {
                "daily": {"peak_hours": [9, 10, 11, 14, 15, 16], "low_hours": [0, 1, 2, 3, 4, 5]},
                "weekly": {"peak_days": [1, 2, 3, 4, 5], "low_days": [0, 6]},  # Monday-Friday peak
                "monthly": {"peak_periods": [1, 2, 3], "low_periods": [4]},  # End of month peak
                "seasonal": {"peak_seasons": ["winter", "summer"], "low_seasons": ["spring", "fall"]}
            },
            "memory_usage": {
                "daily": {"peak_hours": [10, 11, 14, 15, 16], "low_hours": [0, 1, 2, 3, 4, 5, 6]},
                "weekly": {"peak_days": [1, 2, 3, 4, 5], "low_days": [0, 6]},
                "monthly": {"peak_periods": [1, 2, 3], "low_periods": [4]},
                "seasonal": {"peak_seasons": ["winter"], "low_seasons": ["summer"]}
            },
            "network_traffic": {
                "daily": {"peak_hours": [8, 9, 10, 13, 14, 15, 16, 17], "low_hours": [0, 1, 2, 3, 4, 5, 6, 7]},
                "weekly": {"peak_days": [1, 2, 3, 4, 5], "low_days": [0, 6]},
                "monthly": {"peak_periods": [1, 2, 3], "low_periods": [4]},
                "seasonal": {"peak_seasons": ["winter", "summer"], "low_seasons": ["spring", "fall"]}
            },
            "storage_usage": {
                "daily": {"peak_hours": [9, 10, 11, 14, 15, 16, 17], "low_hours": [0, 1, 2, 3, 4, 5, 6]},
                "weekly": {"peak_days": [1, 2, 3, 4, 5], "low_days": [0, 6]},
                "monthly": {"peak_periods": [1, 2, 3, 4], "low_periods": []},
                "seasonal": {"peak_seasons": ["winter"], "low_seasons": ["summer"]}
            },
            "cost": {
                "daily": {"peak_hours": [10, 11, 14, 15, 16], "low_hours": [0, 1, 2, 3, 4, 5, 6]},
                "weekly": {"peak_days": [1, 2, 3, 4, 5], "low_days": [0, 6]},
                "monthly": {"peak_periods": [1, 2, 3, 4], "low_periods": []},
                "seasonal": {"peak_seasons": ["winter", "summer"], "low_seasons": ["spring", "fall"]}
            }
        }
    
    def _load_prediction_config(self) -> Dict[str, Any]:
        """Load prediction configuration parameters."""
        return {
            "algorithms": {
                "linear_regression": {
                    "suitable_for": ["linear_trends", "simple_patterns"],
                    "min_data_points": 10,
                    "accuracy_threshold": 0.7
                },
                "arima": {
                    "suitable_for": ["time_series", "seasonal_patterns"],
                    "min_data_points": 50,
                    "accuracy_threshold": 0.8
                },
                "prophet": {
                    "suitable_for": ["seasonal_patterns", "holidays", "long_term_trends"],
                    "min_data_points": 100,
                    "accuracy_threshold": 0.85
                },
                "exponential_smoothing": {
                    "suitable_for": ["short_term_trends", "seasonal_patterns"],
                    "min_data_points": 20,
                    "accuracy_threshold": 0.75
                }
            },
            "forecast_horizons": {
                "short_term": {"hours": 24, "confidence": 0.9},
                "medium_term": {"days": 7, "confidence": 0.8},
                "long_term": {"days": 30, "confidence": 0.7}
            },
            "confidence_levels": {
                "high": 0.95,
                "medium": 0.80,
                "low": 0.60
            }
        }
    
    async def initialize(self) -> None:
        """Initialize the predictive analytics service."""
        try:
            logger.info("Initializing predictive analytics service")
            
            # Initialize Azure clients
            if self.settings.is_production():
                await self._initialize_azure_clients()
            
            # Initialize forecasting models
            await self._initialize_forecasting_models()
            
            logger.info("Predictive analytics service initialized successfully")
            
        except Exception as e:
            logger.error("Predictive analytics service initialization failed", error=str(e))
            raise
    
    async def _initialize_azure_clients(self) -> None:
        """Initialize Azure clients for data collection."""
        try:
            self.azure_credential = DefaultAzureCredential()
            
            # Initialize Azure Monitor Logs client
            self.logs_client = LogsQueryClient(self.azure_credential)
            
            logger.info("Azure clients initialized for predictive analytics")
            
        except Exception as e:
            logger.warning("Failed to initialize Azure clients", error=str(e))
    
    async def _initialize_forecasting_models(self) -> None:
        """Initialize forecasting models for different metrics."""
        try:
            algorithms = self.prediction_config["algorithms"]
            
            for metric in ["cpu_usage", "memory_usage", "network_traffic", "storage_usage", "cost"]:
                self.forecasting_models[metric] = {
                    "algorithm": "prophet",  # Default algorithm
                    "model": None,
                    "last_trained": None,
                    "accuracy": 0.0,
                    "parameters": algorithms["prophet"]
                }
                
                self.trend_analyzers[metric] = {
                    "current_trend": "stable",
                    "trend_strength": 0.0,
                    "last_analyzed": None
                }
            
            logger.info("Forecasting models initialized", model_count=len(self.forecasting_models))
            
        except Exception as e:
            logger.error("Failed to initialize forecasting models", error=str(e))
    
    async def predict_usage(self, historical_data: Dict[str, Any], 
                          prediction_horizon: str, metrics: List[str]) -> Dict[str, Any]:
        """Predict resource usage patterns."""
        try:
            logger.info("Starting predictive analytics", 
                       horizon=prediction_horizon,
                       metrics=metrics)
            
            # Initialize results
            results = {
                "prediction_horizon": prediction_horizon,
                "metrics": metrics,
                "predictions": [],
                "trends": {},
                "forecast_accuracy": {},
                "confidence_intervals": {},
                "risk_factors": [],
                "seasonality_analysis": {}
            }
            
            # Process each metric
            for metric in metrics:
                if metric not in self.forecasting_models:
                    logger.warning(f"No forecasting model for metric: {metric}")
                    continue
                
                # Extract historical data for this metric
                metric_data = await self._extract_metric_data(historical_data, metric)
                
                if len(metric_data) < 10:  # Minimum data points
                    logger.warning(f"Insufficient data for metric: {metric}")
                    continue
                
                # Perform prediction
                predictions = await self._predict_metric(metric, metric_data, prediction_horizon)
                results["predictions"].extend(predictions)
                
                # Analyze trends
                trend_analysis = await self._analyze_trends(metric, metric_data)
                results["trends"][metric] = trend_analysis
                
                # Calculate forecast accuracy
                accuracy = await self._calculate_forecast_accuracy(metric, metric_data)
                results["forecast_accuracy"][metric] = accuracy
                
                # Calculate confidence intervals
                confidence_intervals = await self._calculate_confidence_intervals(metric, predictions)
                results["confidence_intervals"][metric] = confidence_intervals
                
                # Analyze seasonality
                seasonality = await self._analyze_seasonality(metric, metric_data)
                results["seasonality_analysis"][metric] = seasonality
            
            # Identify risk factors
            results["risk_factors"] = await self._identify_risk_factors(results["predictions"])
            
            logger.info("Predictive analytics completed",
                       predictions_count=len(results["predictions"]),
                       metrics_analyzed=len(results["trends"]))
            
            return results
            
        except Exception as e:
            logger.error("Predictive analytics failed", error=str(e))
            raise
    
    async def _extract_metric_data(self, historical_data: Dict[str, Any], 
                                 metric: str) -> List[Dict[str, Any]]:
        """Extract historical data for a specific metric."""
        try:
            metric_data = []
            
            # Extract from different data sources
            if "timeseries" in historical_data:
                timeseries = historical_data["timeseries"]
                for datapoint in timeseries:
                    if metric in datapoint:
                        metric_data.append({
                            "timestamp": datapoint.get("timestamp", datetime.utcnow()),
                            "value": datapoint[metric],
                            "metadata": datapoint.get("metadata", {})
                        })
            
            # Extract from resource metrics
            if "resources" in historical_data:
                for resource in historical_data["resources"]:
                    if "metrics" in resource and metric in resource["metrics"]:
                        metric_data.append({
                            "timestamp": resource.get("timestamp", datetime.utcnow()),
                            "value": resource["metrics"][metric],
                            "resource_id": resource.get("id", "unknown"),
                            "metadata": resource.get("metadata", {})
                        })
            
            # Sort by timestamp
            metric_data.sort(key=lambda x: x["timestamp"])
            
            return metric_data
            
        except Exception as e:
            logger.error("Metric data extraction failed", metric=metric, error=str(e))
            return []
    
    async def _predict_metric(self, metric: str, metric_data: List[Dict[str, Any]], 
                            horizon: str) -> List[Dict[str, Any]]:
        """Predict values for a specific metric."""
        try:
            model_config = self.forecasting_models[metric]
            algorithm = model_config["algorithm"]
            
            # Prepare data
            timestamps = [d["timestamp"] for d in metric_data]
            values = [d["value"] for d in metric_data]
            
            # Choose prediction method based on algorithm
            if algorithm == "linear_regression":
                predictions = await self._predict_linear_regression(metric, timestamps, values, horizon)
            elif algorithm == "arima":
                predictions = await self._predict_arima(metric, timestamps, values, horizon)
            elif algorithm == "prophet":
                predictions = await self._predict_prophet(metric, timestamps, values, horizon)
            elif algorithm == "exponential_smoothing":
                predictions = await self._predict_exponential_smoothing(metric, timestamps, values, horizon)
            else:
                predictions = await self._predict_simple_average(metric, timestamps, values, horizon)
            
            return predictions
            
        except Exception as e:
            logger.error("Metric prediction failed", metric=metric, error=str(e))
            return []
    
    async def _predict_linear_regression(self, metric: str, timestamps: List[datetime], 
                                       values: List[float], horizon: str) -> List[Dict[str, Any]]:
        """Predict using linear regression."""
        try:
            predictions = []
            
            # Simple linear regression implementation
            if len(values) < 2:
                return predictions
            
            # Convert timestamps to numeric values
            base_time = timestamps[0]
            x_values = [(ts - base_time).total_seconds() / 3600 for ts in timestamps]  # Hours
            
            # Calculate linear regression parameters
            n = len(x_values)
            x_mean = sum(x_values) / n
            y_mean = sum(values) / n
            
            numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator
            
            intercept = y_mean - slope * x_mean
            
            # Generate predictions
            last_time = timestamps[-1]
            horizon_hours = self._get_horizon_hours(horizon)
            
            for hour in range(1, horizon_hours + 1):
                future_time = last_time + timedelta(hours=hour)
                x_future = (future_time - base_time).total_seconds() / 3600
                predicted_value = slope * x_future + intercept
                
                # Ensure non-negative values
                predicted_value = max(0, predicted_value)
                
                predictions.append({
                    "timestamp": future_time,
                    "metric": metric,
                    "predicted_value": predicted_value,
                    "confidence": 0.7,
                    "algorithm": "linear_regression"
                })
            
            return predictions
            
        except Exception as e:
            logger.error("Linear regression prediction failed", error=str(e))
            return []
    
    async def _predict_prophet(self, metric: str, timestamps: List[datetime], 
                             values: List[float], horizon: str) -> List[Dict[str, Any]]:
        """Predict using Prophet algorithm (simplified implementation)."""
        try:
            predictions = []
            
            if len(values) < 10:
                return predictions
            
            # Simple Prophet-like implementation
            # Calculate trend
            trend = self._calculate_trend(values)
            
            # Calculate seasonality
            seasonality = self._calculate_seasonality(metric, timestamps, values)
            
            # Generate predictions
            last_time = timestamps[-1]
            last_value = values[-1]
            horizon_hours = self._get_horizon_hours(horizon)
            
            for hour in range(1, horizon_hours + 1):
                future_time = last_time + timedelta(hours=hour)
                
                # Apply trend
                trend_component = trend * hour
                
                # Apply seasonality
                seasonality_component = self._get_seasonality_component(metric, future_time, seasonality)
                
                # Combine components
                predicted_value = last_value + trend_component + seasonality_component
                
                # Ensure non-negative values
                predicted_value = max(0, predicted_value)
                
                predictions.append({
                    "timestamp": future_time,
                    "metric": metric,
                    "predicted_value": predicted_value,
                    "confidence": 0.85,
                    "algorithm": "prophet"
                })
            
            return predictions
            
        except Exception as e:
            logger.error("Prophet prediction failed", error=str(e))
            return []
    
    async def _predict_exponential_smoothing(self, metric: str, timestamps: List[datetime], 
                                           values: List[float], horizon: str) -> List[Dict[str, Any]]:
        """Predict using exponential smoothing."""
        try:
            predictions = []
            
            if len(values) < 5:
                return predictions
            
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            
            # Calculate smoothed values
            smoothed = [values[0]]
            for i in range(1, len(values)):
                smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i-1]
                smoothed.append(smoothed_value)
            
            # Generate predictions
            last_time = timestamps[-1]
            last_smoothed = smoothed[-1]
            horizon_hours = self._get_horizon_hours(horizon)
            
            for hour in range(1, horizon_hours + 1):
                future_time = last_time + timedelta(hours=hour)
                
                # For exponential smoothing, prediction is the last smoothed value
                predicted_value = last_smoothed
                
                predictions.append({
                    "timestamp": future_time,
                    "metric": metric,
                    "predicted_value": predicted_value,
                    "confidence": 0.75,
                    "algorithm": "exponential_smoothing"
                })
            
            return predictions
            
        except Exception as e:
            logger.error("Exponential smoothing prediction failed", error=str(e))
            return []
    
    async def _predict_simple_average(self, metric: str, timestamps: List[datetime], 
                                    values: List[float], horizon: str) -> List[Dict[str, Any]]:
        """Predict using simple average (fallback method)."""
        try:
            predictions = []
            
            if not values:
                return predictions
            
            # Calculate simple average
            avg_value = sum(values) / len(values)
            
            # Generate predictions
            last_time = timestamps[-1]
            horizon_hours = self._get_horizon_hours(horizon)
            
            for hour in range(1, horizon_hours + 1):
                future_time = last_time + timedelta(hours=hour)
                
                predictions.append({
                    "timestamp": future_time,
                    "metric": metric,
                    "predicted_value": avg_value,
                    "confidence": 0.6,
                    "algorithm": "simple_average"
                })
            
            return predictions
            
        except Exception as e:
            logger.error("Simple average prediction failed", error=str(e))
            return []
    
    def _get_horizon_hours(self, horizon: str) -> int:
        """Convert horizon string to hours."""
        horizon_map = {
            "1h": 1, "6h": 6, "12h": 12, "1d": 24, "3d": 72,
            "1w": 168, "2w": 336, "1m": 720, "3m": 2160
        }
        return horizon_map.get(horizon, 24)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend component."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend
        n = len(values)
        x_values = list(range(n))
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_seasonality(self, metric: str, timestamps: List[datetime], 
                             values: List[float]) -> Dict[str, float]:
        """Calculate seasonality components."""
        seasonality = {"hourly": 0.0, "daily": 0.0, "weekly": 0.0}
        
        if len(values) < 24:  # Need at least 24 hours for daily seasonality
            return seasonality
        
        # Calculate hourly seasonality
        hourly_values = {}
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            if hour not in hourly_values:
                hourly_values[hour] = []
            hourly_values[hour].append(values[i])
        
        if len(hourly_values) > 1:
            hourly_means = {hour: sum(vals) / len(vals) for hour, vals in hourly_values.items()}
            overall_mean = sum(values) / len(values)
            seasonality["hourly"] = max(hourly_means.values()) - min(hourly_means.values())
        
        return seasonality
    
    def _get_seasonality_component(self, metric: str, future_time: datetime, 
                                 seasonality: Dict[str, float]) -> float:
        """Get seasonality component for a future time."""
        component = 0.0
        
        # Simple seasonality based on hour of day
        patterns = self.seasonality_patterns.get(metric, {})
        daily_pattern = patterns.get("daily", {})
        
        if future_time.hour in daily_pattern.get("peak_hours", []):
            component += seasonality.get("hourly", 0) * 0.3
        elif future_time.hour in daily_pattern.get("low_hours", []):
            component -= seasonality.get("hourly", 0) * 0.3
        
        return component
    
    async def _analyze_trends(self, metric: str, metric_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends for a metric."""
        try:
            if len(metric_data) < 2:
                return {"trend": "stable", "strength": 0.0, "direction": "none"}
            
            values = [d["value"] for d in metric_data]
            trend_slope = self._calculate_trend(values)
            
            # Determine trend direction and strength
            if abs(trend_slope) < 0.01:
                trend_direction = "stable"
                trend_strength = 0.0
            elif trend_slope > 0:
                trend_direction = "increasing"
                trend_strength = min(abs(trend_slope), 1.0)
            else:
                trend_direction = "decreasing"
                trend_strength = min(abs(trend_slope), 1.0)
            
            return {
                "trend": trend_direction,
                "strength": trend_strength,
                "slope": trend_slope,
                "direction": trend_direction
            }
            
        except Exception as e:
            logger.error("Trend analysis failed", error=str(e))
            return {"trend": "stable", "strength": 0.0, "direction": "none"}
    
    async def _calculate_forecast_accuracy(self, metric: str, 
                                         metric_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        try:
            # Simple accuracy calculation
            if len(metric_data) < 10:
                return {"accuracy": 0.5, "mae": 0.0, "mse": 0.0}
            
            values = [d["value"] for d in metric_data]
            
            # Calculate mean absolute error (simplified)
            mae = sum(abs(values[i] - values[i-1]) for i in range(1, len(values))) / (len(values) - 1)
            
            # Calculate mean squared error (simplified)
            mse = sum((values[i] - values[i-1])**2 for i in range(1, len(values))) / (len(values) - 1)
            
            # Simple accuracy score
            mean_value = sum(values) / len(values)
            accuracy = max(0, 1 - (mae / mean_value)) if mean_value > 0 else 0.5
            
            return {"accuracy": accuracy, "mae": mae, "mse": mse}
            
        except Exception as e:
            logger.error("Forecast accuracy calculation failed", error=str(e))
            return {"accuracy": 0.5, "mae": 0.0, "mse": 0.0}
    
    async def _calculate_confidence_intervals(self, metric: str, 
                                           predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence intervals for predictions."""
        try:
            if not predictions:
                return {"lower_bound": [], "upper_bound": [], "confidence_level": 0.8}
            
            confidence_intervals = {"lower_bound": [], "upper_bound": [], "confidence_level": 0.8}
            
            for pred in predictions:
                predicted_value = pred["predicted_value"]
                confidence = pred["confidence"]
                
                # Simple confidence interval calculation
                error_margin = predicted_value * (1 - confidence) * 0.5
                
                confidence_intervals["lower_bound"].append(max(0, predicted_value - error_margin))
                confidence_intervals["upper_bound"].append(predicted_value + error_margin)
            
            return confidence_intervals
            
        except Exception as e:
            logger.error("Confidence interval calculation failed", error=str(e))
            return {"lower_bound": [], "upper_bound": [], "confidence_level": 0.8}
    
    async def _analyze_seasonality(self, metric: str, 
                                 metric_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        try:
            if len(metric_data) < 24:
                return {"has_seasonality": False, "patterns": {}}
            
            timestamps = [d["timestamp"] for d in metric_data]
            values = [d["value"] for d in metric_data]
            
            seasonality_analysis = {"has_seasonality": False, "patterns": {}}
            
            # Analyze hourly patterns
            hourly_patterns = {}
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                if hour not in hourly_patterns:
                    hourly_patterns[hour] = []
                hourly_patterns[hour].append(values[i])
            
            # Check if there's significant hourly variation
            if len(hourly_patterns) > 1:
                hourly_means = {hour: sum(vals) / len(vals) for hour, vals in hourly_patterns.items()}
                max_mean = max(hourly_means.values())
                min_mean = min(hourly_means.values())
                
                if max_mean > min_mean * 1.2:  # 20% variation threshold
                    seasonality_analysis["has_seasonality"] = True
                    seasonality_analysis["patterns"]["hourly"] = {
                        "peak_hours": [hour for hour, mean in hourly_means.items() if mean > max_mean * 0.8],
                        "low_hours": [hour for hour, mean in hourly_means.items() if mean < min_mean * 1.2]
                    }
            
            return seasonality_analysis
            
        except Exception as e:
            logger.error("Seasonality analysis failed", error=str(e))
            return {"has_seasonality": False, "patterns": {}}
    
    async def _identify_risk_factors(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify risk factors in predictions."""
        try:
            risk_factors = []
            
            # Group predictions by metric
            metrics_predictions = {}
            for pred in predictions:
                metric = pred["metric"]
                if metric not in metrics_predictions:
                    metrics_predictions[metric] = []
                metrics_predictions[metric].append(pred)
            
            # Analyze each metric for risks
            for metric, metric_predictions in metrics_predictions.items():
                values = [p["predicted_value"] for p in metric_predictions]
                
                # Check for sudden spikes
                if len(values) > 1:
                    max_increase = max(values[i] - values[i-1] for i in range(1, len(values)))
                    avg_value = sum(values) / len(values)
                    
                    if max_increase > avg_value * 0.5:  # 50% increase
                        risk_factors.append({
                            "type": "sudden_spike",
                            "metric": metric,
                            "severity": "high",
                            "description": f"Predicted sudden spike in {metric}",
                            "mitigation": f"Monitor {metric} closely and prepare scaling"
                        })
                
                # Check for capacity limits
                if metric in ["cpu_usage", "memory_usage"] and max(values) > 0.8:
                    risk_factors.append({
                        "type": "capacity_limit",
                        "metric": metric,
                        "severity": "medium",
                        "description": f"Predicted {metric} approaching capacity limits",
                        "mitigation": f"Consider scaling up resources for {metric}"
                    })
            
            return risk_factors
            
        except Exception as e:
            logger.error("Risk factor identification failed", error=str(e))
            return []
    
    async def initialize_advanced_models(self):
        """Initialize advanced governance prediction models."""
        try:
            logger.info("initializing_advanced_governance_models")
            
            # Initialize models for each advanced prediction type
            for prediction_type in AdvancedPredictionType:
                await self._initialize_governance_model(prediction_type)
            
            self.is_advanced_initialized = True
            logger.info("advanced_governance_models_initialized",
                       model_count=len(self.governance_models))
        
        except Exception as e:
            logger.error("advanced_model_initialization_failed", error=str(e))
            raise
    
    async def _initialize_governance_model(self, prediction_type: AdvancedPredictionType):
        """Initialize a specific governance model."""
        try:
            if prediction_type in [AdvancedPredictionType.COMPLIANCE_VIOLATION, 
                                 AdvancedPredictionType.SECURITY_RISK,
                                 AdvancedPredictionType.PERFORMANCE_DEGRADATION,
                                 AdvancedPredictionType.POLICY_DRIFT,
                                 AdvancedPredictionType.RBAC_ANOMALY]:
                # Classification models
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            else:
                # Regression models for cost and resource utilization
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            
            self.governance_models[prediction_type] = {
                'model': model,
                'is_trained': False,
                'version': '1.0',
                'last_trained': None
            }
            
            # Initialize feature scaling
            self.feature_scalers[prediction_type] = StandardScaler()
            
            logger.debug("governance_model_initialized",
                        prediction_type=prediction_type.value,
                        model_type=type(model).__name__)
        
        except Exception as e:
            logger.error("governance_model_initialization_failed",
                        prediction_type=prediction_type.value,
                        error=str(e))
            raise
    
    async def predict_governance_scenario(self, request: GovernancePredictionRequest) -> GovernancePredictionResult:
        """Make an advanced governance prediction."""
        try:
            if not self.is_advanced_initialized:
                await self.initialize_advanced_models()
            
            prediction_id = f"gov_pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request.prediction_type.value}"
            
            logger.info("making_governance_prediction",
                       prediction_id=prediction_id,
                       prediction_type=request.prediction_type.value)
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if (datetime.utcnow() - cached_result.created_at).seconds < 300:  # 5 minutes
                    return cached_result
            
            # Get model
            model_info = self.governance_models.get(request.prediction_type)
            if not model_info or not model_info['is_trained']:
                # Use rule-based prediction as fallback
                return await self._rule_based_prediction(request, prediction_id)
            
            # Prepare features
            features = await self._prepare_governance_features(request)
            
            if not features:
                return await self._rule_based_prediction(request, prediction_id)
            
            # Make prediction
            model = model_info['model']
            scaler = self.feature_scalers[request.prediction_type]
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Predict
            if hasattr(model, 'predict_proba'):  # Classification
                prediction_proba = model.predict_proba(features_scaled)[0]
                predicted_value = model.predict(features_scaled)[0]
                confidence_score = max(prediction_proba)
            else:  # Regression
                predicted_value = model.predict(features_scaled)[0]
                confidence_score = 0.85  # Default confidence for regression
            
            # Determine risk level
            risk_level = self._determine_risk_level(request.prediction_type, predicted_value, confidence_score)
            
            # Generate factors
            factors = await self._generate_governance_factors(request.prediction_type, features, predicted_value)
            
            # Generate recommendations
            recommendations = await self._generate_governance_recommendations(
                request.prediction_type, predicted_value, risk_level, factors
            )
            
            result = GovernancePredictionResult(
                prediction_id=prediction_id,
                prediction_type=request.prediction_type,
                predicted_value=predicted_value,
                confidence_score=confidence_score,
                time_horizon=request.time_horizon,
                factors=factors,
                recommendations=recommendations,
                created_at=datetime.utcnow(),
                risk_level=risk_level
            )
            
            # Cache result
            self.prediction_cache[cache_key] = result
            
            logger.info("governance_prediction_completed",
                       prediction_id=prediction_id,
                       predicted_value=predicted_value,
                       risk_level=risk_level)
            
            return result
        
        except Exception as e:
            logger.error("governance_prediction_failed",
                        prediction_type=request.prediction_type.value,
                        error=str(e))
            # Return fallback prediction
            return await self._rule_based_prediction(request, f"fallback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    
    async def _prepare_governance_features(self, request: GovernancePredictionRequest) -> Optional[List[float]]:
        """Prepare features for governance prediction."""
        try:
            data = request.data
            features = []
            
            if request.prediction_type == AdvancedPredictionType.COMPLIANCE_VIOLATION:
                features = [
                    data.get('policy_count', 0),
                    data.get('resource_count', 0),
                    data.get('violation_history', 0),
                    data.get('policy_complexity', 0),
                    data.get('resource_diversity', 0),
                    data.get('change_frequency', 0),
                    data.get('user_activity', 0),
                    data.get('time_since_last_review', 0)
                ]
            
            elif request.prediction_type == AdvancedPredictionType.SECURITY_RISK:
                features = [
                    data.get('failed_login_rate', 0),
                    data.get('unusual_access_patterns', 0),
                    data.get('privilege_escalations', 0),
                    data.get('network_anomalies', 0),
                    data.get('outdated_resources', 0),
                    data.get('security_violations', 0),
                    data.get('external_threats', 0),
                    data.get('vulnerability_score', 0)
                ]
            
            elif request.prediction_type == AdvancedPredictionType.COST_FORECAST:
                features = [
                    data.get('current_cost', 0),
                    data.get('resource_growth_rate', 0),
                    data.get('usage_trends', 0),
                    data.get('seasonal_patterns', 0),
                    data.get('optimization_opportunities', 0),
                    data.get('reserved_instances', 0),
                    data.get('spot_instance_usage', 0),
                    data.get('auto_scaling_efficiency', 0)
                ]
            
            elif request.prediction_type == AdvancedPredictionType.RESOURCE_UTILIZATION:
                features = [
                    data.get('cpu_utilization', 0),
                    data.get('memory_utilization', 0),
                    data.get('network_utilization', 0),
                    data.get('storage_utilization', 0),
                    data.get('request_rate', 0),
                    data.get('error_rate', 0),
                    data.get('response_time', 0),
                    data.get('throughput', 0)
                ]
            
            elif request.prediction_type == AdvancedPredictionType.PERFORMANCE_DEGRADATION:
                features = [
                    data.get('response_time_trend', 0),
                    data.get('error_rate_trend', 0),
                    data.get('throughput_trend', 0),
                    data.get('resource_saturation', 0),
                    data.get('dependency_issues', 0),
                    data.get('configuration_changes', 0),
                    data.get('external_factors', 0),
                    data.get('historical_incidents', 0)
                ]
            
            else:
                # Default feature set
                features = [
                    data.get('metric1', 0),
                    data.get('metric2', 0),
                    data.get('metric3', 0),
                    data.get('metric4', 0)
                ]
            
            return features if features else None
        
        except Exception as e:
            logger.error("feature_preparation_failed", error=str(e))
            return None
    
    async def _rule_based_prediction(self, request: GovernancePredictionRequest, prediction_id: str) -> GovernancePredictionResult:
        """Rule-based prediction fallback."""
        try:
            data = request.data
            
            # Simple rule-based logic
            if request.prediction_type == AdvancedPredictionType.COMPLIANCE_VIOLATION:
                violation_score = data.get('violation_history', 0) * 0.4 + data.get('policy_complexity', 0) * 0.3
                predicted_value = min(violation_score / 100, 1.0)
                risk_level = 'high' if predicted_value > 0.7 else 'medium' if predicted_value > 0.4 else 'low'
            
            elif request.prediction_type == AdvancedPredictionType.SECURITY_RISK:
                risk_score = data.get('failed_login_rate', 0) * 0.3 + data.get('security_violations', 0) * 0.4
                predicted_value = min(risk_score / 100, 1.0)
                risk_level = 'critical' if predicted_value > 0.8 else 'high' if predicted_value > 0.6 else 'medium'
            
            elif request.prediction_type == AdvancedPredictionType.COST_FORECAST:
                growth_rate = data.get('resource_growth_rate', 0)
                predicted_value = data.get('current_cost', 0) * (1 + growth_rate / 100)
                risk_level = 'high' if growth_rate > 20 else 'medium' if growth_rate > 10 else 'low'
            
            else:
                predicted_value = 0.5
                risk_level = 'medium'
            
            return GovernancePredictionResult(
                prediction_id=prediction_id,
                prediction_type=request.prediction_type,
                predicted_value=predicted_value,
                confidence_score=0.6,  # Lower confidence for rule-based
                time_horizon=request.time_horizon,
                factors=[{"factor": "rule_based", "importance": 1.0, "value": predicted_value}],
                recommendations=["Consider training ML models with more data for better predictions"],
                created_at=datetime.utcnow(),
                risk_level=risk_level
            )
        
        except Exception as e:
            logger.error("rule_based_prediction_failed", error=str(e))
            # Return safe default
            return GovernancePredictionResult(
                prediction_id=prediction_id,
                prediction_type=request.prediction_type,
                predicted_value=0.0,
                confidence_score=0.5,
                time_horizon=request.time_horizon,
                factors=[],
                recommendations=["Unable to generate prediction, please check data quality"],
                created_at=datetime.utcnow(),
                risk_level='low'
            )
    
    def _determine_risk_level(self, prediction_type: AdvancedPredictionType, 
                             predicted_value: Any, confidence_score: float) -> str:
        """Determine risk level based on prediction."""
        try:
            if prediction_type in [AdvancedPredictionType.SECURITY_RISK]:
                if predicted_value > 0.8:
                    return 'critical'
                elif predicted_value > 0.6:
                    return 'high'
                elif predicted_value > 0.4:
                    return 'medium'
                else:
                    return 'low'
            
            elif prediction_type in [AdvancedPredictionType.COMPLIANCE_VIOLATION, 
                                   AdvancedPredictionType.PERFORMANCE_DEGRADATION]:
                if predicted_value > 0.7:
                    return 'high'
                elif predicted_value > 0.4:
                    return 'medium'
                else:
                    return 'low'
            
            elif prediction_type == AdvancedPredictionType.COST_FORECAST:
                # For cost, consider growth rate
                if predicted_value > 10000:  # High cost threshold
                    return 'high'
                elif predicted_value > 5000:
                    return 'medium'
                else:
                    return 'low'
            
            else:
                return 'medium'  # Default
        
        except Exception:
            return 'low'  # Safe default
    
    async def _generate_governance_factors(self, prediction_type: AdvancedPredictionType,
                                         features: List[float], predicted_value: Any) -> List[Dict[str, Any]]:
        """Generate factors that influenced the governance prediction."""
        factors = []
        
        try:
            feature_names = self._get_feature_names(prediction_type)
            
            for i, (name, value) in enumerate(zip(feature_names, features)):
                # Simple importance calculation
                importance = abs(value) / (sum(abs(f) for f in features) + 1e-6)
                
                factors.append({
                    "factor": name,
                    "importance": importance,
                    "value": value,
                    "description": f"{name}: {value}"
                })
            
            # Sort by importance
            factors.sort(key=lambda x: x["importance"], reverse=True)
            return factors[:5]  # Top 5 factors
        
        except Exception as e:
            logger.error("factor_generation_failed", error=str(e))
            return []
    
    def _get_feature_names(self, prediction_type: AdvancedPredictionType) -> List[str]:
        """Get feature names for a prediction type."""
        feature_map = {
            AdvancedPredictionType.COMPLIANCE_VIOLATION: [
                'policy_count', 'resource_count', 'violation_history', 'policy_complexity',
                'resource_diversity', 'change_frequency', 'user_activity', 'time_since_last_review'
            ],
            AdvancedPredictionType.SECURITY_RISK: [
                'failed_login_rate', 'unusual_access_patterns', 'privilege_escalations', 'network_anomalies',
                'outdated_resources', 'security_violations', 'external_threats', 'vulnerability_score'
            ],
            AdvancedPredictionType.COST_FORECAST: [
                'current_cost', 'resource_growth_rate', 'usage_trends', 'seasonal_patterns',
                'optimization_opportunities', 'reserved_instances', 'spot_instance_usage', 'auto_scaling_efficiency'
            ],
            AdvancedPredictionType.RESOURCE_UTILIZATION: [
                'cpu_utilization', 'memory_utilization', 'network_utilization', 'storage_utilization',
                'request_rate', 'error_rate', 'response_time', 'throughput'
            ]
        }
        
        return feature_map.get(prediction_type, ['feature1', 'feature2', 'feature3', 'feature4'])
    
    async def _generate_governance_recommendations(self, prediction_type: AdvancedPredictionType,
                                                 predicted_value: Any, risk_level: str,
                                                 factors: List[Dict[str, Any]]) -> List[str]:
        """Generate governance-specific recommendations."""
        recommendations = []
        
        try:
            if prediction_type == AdvancedPredictionType.COMPLIANCE_VIOLATION:
                if risk_level in ['high', 'critical']:
                    recommendations.extend([
                        "Immediately review and update non-compliant policies",
                        "Implement automated compliance monitoring",
                        "Schedule regular policy compliance audits"
                    ])
                elif risk_level == 'medium':
                    recommendations.extend([
                        "Monitor policy compliance more frequently",
                        "Review policy configurations for potential improvements"
                    ])
            
            elif prediction_type == AdvancedPredictionType.SECURITY_RISK:
                if risk_level == 'critical':
                    recommendations.extend([
                        "Immediate security assessment required",
                        "Enable emergency security monitoring",
                        "Review and strengthen all access controls"
                    ])
                elif risk_level == 'high':
                    recommendations.extend([
                        "Enhance security monitoring and alerting",
                        "Implement additional authentication factors",
                        "Conduct security audit"
                    ])
            
            elif prediction_type == AdvancedPredictionType.COST_FORECAST:
                if risk_level == 'high':
                    recommendations.extend([
                        "Implement immediate cost optimization measures",
                        "Review resource utilization for right-sizing opportunities",
                        "Consider reserved instances for predictable workloads"
                    ])
            
            # Add factor-specific recommendations
            for factor in factors[:2]:  # Top 2 factors
                factor_rec = self._get_factor_recommendation(factor["factor"], factor["value"])
                if factor_rec:
                    recommendations.append(factor_rec)
            
            return recommendations[:5]  # Limit to 5 recommendations
        
        except Exception as e:
            logger.error("recommendation_generation_failed", error=str(e))
            return ["Review system configuration and monitoring"]
    
    def _get_factor_recommendation(self, factor_name: str, value: float) -> Optional[str]:
        """Get recommendation based on a specific factor."""
        if "violation" in factor_name and value > 5:
            return "Focus on reducing policy violations through automated remediation"
        elif "cost" in factor_name and value > 1000:
            return "Implement cost optimization strategies for high-cost resources"
        elif "utilization" in factor_name and value > 80:
            return "Consider scaling resources to handle high utilization"
        elif "security" in factor_name and value > 0.1:
            return "Strengthen security controls and monitoring"
        elif "error" in factor_name and value > 0.05:
            return "Investigate and resolve error sources"
        
        return None
    
    def _generate_cache_key(self, request: GovernancePredictionRequest) -> str:
        """Generate cache key for governance prediction request."""
        import hashlib
        
        data_str = json.dumps(request.data, sort_keys=True, default=str)
        key_data = f"{request.prediction_type.value}_{data_str}_{request.time_horizon}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def train_governance_model(self, prediction_type: AdvancedPredictionType,
                                   training_data: pd.DataFrame,
                                   target_column: str) -> ModelMetrics:
        """Train a governance prediction model."""
        try:
            logger.info("training_governance_model",
                       prediction_type=prediction_type.value,
                       data_shape=training_data.shape)
            
            # Prepare features
            feature_columns = [col for col in training_data.columns if col != target_column]
            X = training_data[feature_columns].fillna(0)
            y = training_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = self.feature_scalers[prediction_type]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = self.governance_models[prediction_type]['model']
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            if hasattr(model, 'predict_proba'):  # Classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                rmse = 0.0
                mae = 0.0
            else:  # Regression
                from sklearn.metrics import mean_absolute_error
                accuracy = 0.0
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
            
            # Update model status
            self.governance_models[prediction_type]['is_trained'] = True
            self.governance_models[prediction_type]['last_trained'] = datetime.utcnow()
            
            # Store metrics
            metrics = ModelMetrics(
                model_name=f"{prediction_type.value}_governance_model",
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                rmse=rmse,
                mae=mae,
                last_trained=datetime.utcnow(),
                training_samples=len(training_data)
            )
            
            self.model_metrics[prediction_type] = metrics
            
            logger.info("governance_model_training_completed",
                       prediction_type=prediction_type.value,
                       accuracy=accuracy,
                       rmse=rmse)
            
            return metrics
        
        except Exception as e:
            logger.error("governance_model_training_failed",
                        prediction_type=prediction_type.value,
                        error=str(e))
            raise
    
    async def get_governance_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of governance analytics capabilities."""
        return {
            "advanced_models_initialized": self.is_advanced_initialized,
            "available_prediction_types": [pt.value for pt in AdvancedPredictionType],
            "trained_models": {
                pt.value: self.governance_models[pt]['is_trained'] 
                for pt in AdvancedPredictionType if pt in self.governance_models
            },
            "model_performance": {
                pt.value: {
                    "accuracy": metrics.accuracy,
                    "last_trained": metrics.last_trained.isoformat(),
                    "training_samples": metrics.training_samples
                }
                for pt, metrics in self.model_metrics.items()
            },
            "cached_predictions": len(self.prediction_cache)
        }
    
    def is_ready(self) -> bool:
        """Check if predictive analytics service is ready."""
        return len(self.forecasting_models) > 0
    
    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Clear models
            self.forecasting_models.clear()
            self.trend_analyzers.clear()
            
            # Clear advanced governance components
            self.governance_models.clear()
            self.feature_scalers.clear()
            self.label_encoders.clear()
            self.model_metrics.clear()
            self.prediction_cache.clear()
            
            # Close Azure clients
            if self.logs_client:
                await self.logs_client.close()
            
            self.is_advanced_initialized = False
            
            logger.info("Predictive analytics service cleanup completed")
            
        except Exception as e:
            logger.error("Predictive analytics service cleanup failed", error=str(e))