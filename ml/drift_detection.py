"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# Drift Detection System for PolicyCortex
# Defense #7: Detect model and data drift in production

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
import json
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class DriftType(Enum):
    """Types of drift detected"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"

class DriftSeverity(Enum):
    """Severity levels for drift"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DriftMetrics:
    """Metrics for drift detection"""
    drift_score: float
    p_value: float
    severity: DriftSeverity
    drift_type: DriftType
    features_affected: List[str]
    timestamp: datetime
    recommendations: List[str]

@dataclass
class BaselineProfile:
    """Baseline profile for comparison"""
    feature_distributions: Dict[str, Any]
    label_distribution: Dict[str, float]
    prediction_distribution: Dict[str, float]
    feature_statistics: Dict[str, Dict[str, float]]
    correlation_matrix: np.ndarray
    timestamp: datetime
    sample_size: int

class DriftDetector:
    """Advanced drift detection for ML models"""
    
    def __init__(self, 
                 sensitivity: float = 0.05,
                 window_size: int = 1000,
                 update_frequency: str = 'hourly'):
        self.sensitivity = sensitivity  # p-value threshold
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.baseline = None
        self.drift_history = []
        self.alert_thresholds = {
            DriftSeverity.LOW: 0.1,
            DriftSeverity.MEDIUM: 0.05,
            DriftSeverity.HIGH: 0.01,
            DriftSeverity.CRITICAL: 0.001
        }
    
    def create_baseline(self, 
                       data: pd.DataFrame,
                       features: List[str],
                       labels: Optional[pd.Series] = None,
                       predictions: Optional[pd.Series] = None) -> BaselineProfile:
        """Create baseline profile from training data"""
        
        profile = BaselineProfile(
            feature_distributions={},
            label_distribution={},
            prediction_distribution={},
            feature_statistics={},
            correlation_matrix=None,
            timestamp=datetime.now(),
            sample_size=len(data)
        )
        
        # Calculate feature distributions
        for feature in features:
            if feature in data.columns:
                if data[feature].dtype in ['float64', 'int64']:
                    # Numeric features
                    profile.feature_distributions[feature] = {
                        'type': 'numeric',
                        'mean': data[feature].mean(),
                        'std': data[feature].std(),
                        'median': data[feature].median(),
                        'min': data[feature].min(),
                        'max': data[feature].max(),
                        'quantiles': data[feature].quantile([0.25, 0.5, 0.75]).to_dict()
                    }
                    
                    profile.feature_statistics[feature] = {
                        'mean': data[feature].mean(),
                        'std': data[feature].std(),
                        'skewness': data[feature].skew(),
                        'kurtosis': data[feature].kurtosis()
                    }
                else:
                    # Categorical features
                    value_counts = data[feature].value_counts(normalize=True)
                    profile.feature_distributions[feature] = {
                        'type': 'categorical',
                        'distribution': value_counts.to_dict(),
                        'unique_values': len(value_counts)
                    }
        
        # Calculate label distribution if provided
        if labels is not None:
            if labels.dtype in ['float64', 'int64']:
                profile.label_distribution = {
                    'mean': labels.mean(),
                    'std': labels.std()
                }
            else:
                profile.label_distribution = labels.value_counts(normalize=True).to_dict()
        
        # Calculate prediction distribution if provided
        if predictions is not None:
            if predictions.dtype in ['float64', 'int64']:
                profile.prediction_distribution = {
                    'mean': predictions.mean(),
                    'std': predictions.std()
                }
            else:
                profile.prediction_distribution = predictions.value_counts(normalize=True).to_dict()
        
        # Calculate correlation matrix for numeric features
        numeric_features = [f for f in features if f in data.columns and data[f].dtype in ['float64', 'int64']]
        if numeric_features:
            profile.correlation_matrix = data[numeric_features].corr().values
        
        self.baseline = profile
        return profile
    
    def detect_drift(self,
                    current_data: pd.DataFrame,
                    features: List[str],
                    labels: Optional[pd.Series] = None,
                    predictions: Optional[pd.Series] = None) -> DriftMetrics:
        """Detect drift in current data compared to baseline"""
        
        if self.baseline is None:
            raise ValueError("Baseline not set. Call create_baseline first.")
        
        drift_scores = {}
        p_values = {}
        affected_features = []
        
        # Check feature drift
        for feature in features:
            if feature in current_data.columns and feature in self.baseline.feature_distributions:
                baseline_dist = self.baseline.feature_distributions[feature]
                
                if baseline_dist['type'] == 'numeric':
                    drift_score, p_value = self._detect_numeric_drift(
                        current_data[feature],
                        baseline_dist
                    )
                else:
                    drift_score, p_value = self._detect_categorical_drift(
                        current_data[feature],
                        baseline_dist
                    )
                
                drift_scores[feature] = drift_score
                p_values[feature] = p_value
                
                if p_value < self.sensitivity:
                    affected_features.append(feature)
        
        # Check label drift if provided
        label_drift_score = 0
        if labels is not None and self.baseline.label_distribution:
            label_drift_score, label_p_value = self._detect_label_drift(
                labels,
                self.baseline.label_distribution
            )
            drift_scores['label'] = label_drift_score
            p_values['label'] = label_p_value
        
        # Check prediction drift if provided
        prediction_drift_score = 0
        if predictions is not None and self.baseline.prediction_distribution:
            prediction_drift_score, pred_p_value = self._detect_prediction_drift(
                predictions,
                self.baseline.prediction_distribution
            )
            drift_scores['prediction'] = prediction_drift_score
            p_values['prediction'] = pred_p_value
        
        # Calculate overall drift score
        overall_drift_score = np.mean(list(drift_scores.values()))
        min_p_value = min(p_values.values()) if p_values else 1.0
        
        # Determine drift type and severity
        drift_type = self._determine_drift_type(
            drift_scores,
            label_drift_score,
            prediction_drift_score
        )
        severity = self._determine_severity(min_p_value)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            drift_type,
            severity,
            affected_features
        )
        
        # Create drift metrics
        metrics = DriftMetrics(
            drift_score=overall_drift_score,
            p_value=min_p_value,
            severity=severity,
            drift_type=drift_type,
            features_affected=affected_features,
            timestamp=datetime.now(),
            recommendations=recommendations
        )
        
        # Store in history
        self.drift_history.append(metrics)
        
        return metrics
    
    def _detect_numeric_drift(self,
                             current_values: pd.Series,
                             baseline_dist: Dict) -> Tuple[float, float]:
        """Detect drift in numeric features using KS test"""
        
        # Generate baseline samples from distribution parameters
        baseline_samples = np.random.normal(
            baseline_dist['mean'],
            baseline_dist['std'],
            size=len(current_values)
        )
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(current_values.dropna(), baseline_samples)
        
        # Calculate drift score (0-1 scale)
        drift_score = ks_statistic
        
        return drift_score, p_value
    
    def _detect_categorical_drift(self,
                                 current_values: pd.Series,
                                 baseline_dist: Dict) -> Tuple[float, float]:
        """Detect drift in categorical features using chi-square test"""
        
        baseline_distribution = baseline_dist['distribution']
        current_distribution = current_values.value_counts(normalize=True).to_dict()
        
        # Align categories
        all_categories = set(baseline_distribution.keys()) | set(current_distribution.keys())
        
        baseline_freq = []
        current_freq = []
        
        for cat in all_categories:
            baseline_freq.append(baseline_distribution.get(cat, 0) * len(current_values))
            current_freq.append(current_distribution.get(cat, 0) * len(current_values))
        
        # Chi-square test
        if len(all_categories) > 1:
            chi2, p_value = stats.chisquare(current_freq, baseline_freq)
            drift_score = min(chi2 / len(all_categories), 1.0)  # Normalize
        else:
            drift_score = 0
            p_value = 1.0
        
        return drift_score, p_value
    
    def _detect_label_drift(self,
                           current_labels: pd.Series,
                           baseline_dist: Dict) -> Tuple[float, float]:
        """Detect drift in label distribution"""
        
        if isinstance(baseline_dist, dict) and 'mean' in baseline_dist:
            # Numeric labels
            baseline_mean = baseline_dist['mean']
            baseline_std = baseline_dist['std']
            
            current_mean = current_labels.mean()
            current_std = current_labels.std()
            
            # Z-test for mean difference
            z_score = abs(current_mean - baseline_mean) / (baseline_std / np.sqrt(len(current_labels)))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            drift_score = min(abs(current_mean - baseline_mean) / baseline_mean, 1.0)
        else:
            # Categorical labels
            return self._detect_categorical_drift(
                current_labels,
                {'distribution': baseline_dist, 'type': 'categorical'}
            )
        
        return drift_score, p_value
    
    def _detect_prediction_drift(self,
                                current_predictions: pd.Series,
                                baseline_dist: Dict) -> Tuple[float, float]:
        """Detect drift in model predictions"""
        
        # Similar to label drift detection
        return self._detect_label_drift(current_predictions, baseline_dist)
    
    def _determine_drift_type(self,
                             drift_scores: Dict[str, float],
                             label_drift: float,
                             prediction_drift: float) -> DriftType:
        """Determine the primary type of drift"""
        
        feature_drift_avg = np.mean([
            score for key, score in drift_scores.items() 
            if key not in ['label', 'prediction']
        ])
        
        if prediction_drift > feature_drift_avg and prediction_drift > label_drift:
            return DriftType.PREDICTION_DRIFT
        elif label_drift > feature_drift_avg and label_drift > prediction_drift:
            return DriftType.LABEL_DRIFT
        elif feature_drift_avg > 0.3:
            return DriftType.DATA_DRIFT
        else:
            return DriftType.FEATURE_DRIFT
    
    def _determine_severity(self, p_value: float) -> DriftSeverity:
        """Determine drift severity based on p-value"""
        
        if p_value > 0.1:
            return DriftSeverity.NONE
        elif p_value > 0.05:
            return DriftSeverity.LOW
        elif p_value > 0.01:
            return DriftSeverity.MEDIUM
        elif p_value > 0.001:
            return DriftSeverity.HIGH
        else:
            return DriftSeverity.CRITICAL
    
    def _generate_recommendations(self,
                                 drift_type: DriftType,
                                 severity: DriftSeverity,
                                 affected_features: List[str]) -> List[str]:
        """Generate recommendations based on drift detection"""
        
        recommendations = []
        
        if severity == DriftSeverity.NONE:
            recommendations.append("No significant drift detected. Continue monitoring.")
            return recommendations
        
        # General recommendations based on severity
        if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("URGENT: Significant drift detected. Immediate action required.")
            recommendations.append("Consider pausing model predictions until investigation complete.")
        
        # Specific recommendations based on drift type
        if drift_type == DriftType.DATA_DRIFT:
            recommendations.append("Investigate changes in data collection or preprocessing.")
            recommendations.append("Check for upstream system changes.")
            recommendations.append("Consider retraining model with recent data.")
        
        elif drift_type == DriftType.CONCEPT_DRIFT:
            recommendations.append("Business logic or relationships may have changed.")
            recommendations.append("Review model assumptions and update if necessary.")
            recommendations.append("Consider incremental learning or model adaptation.")
        
        elif drift_type == DriftType.PREDICTION_DRIFT:
            recommendations.append("Model predictions shifting from baseline.")
            recommendations.append("Validate model performance on recent data.")
            recommendations.append("Check for model decay or degradation.")
        
        elif drift_type == DriftType.FEATURE_DRIFT:
            if affected_features:
                recommendations.append(f"Features with drift: {', '.join(affected_features[:5])}")
            recommendations.append("Investigate feature engineering pipeline.")
            recommendations.append("Check for data quality issues in affected features.")
        
        elif drift_type == DriftType.LABEL_DRIFT:
            recommendations.append("Target variable distribution has changed.")
            recommendations.append("Review labeling process and criteria.")
            recommendations.append("Consider stratified retraining.")
        
        return recommendations

class AdaptiveDriftHandler:
    """Handle drift with adaptive strategies"""
    
    def __init__(self, detector: DriftDetector):
        self.detector = detector
        self.adaptation_strategies = {
            DriftSeverity.LOW: self._handle_low_drift,
            DriftSeverity.MEDIUM: self._handle_medium_drift,
            DriftSeverity.HIGH: self._handle_high_drift,
            DriftSeverity.CRITICAL: self._handle_critical_drift
        }
    
    async def handle_drift(self, metrics: DriftMetrics) -> Dict[str, Any]:
        """Handle detected drift with appropriate strategy"""
        
        if metrics.severity == DriftSeverity.NONE:
            return {"action": "none", "message": "No drift detected"}
        
        handler = self.adaptation_strategies.get(metrics.severity)
        if handler:
            return await handler(metrics)
        
        return {"action": "monitor", "message": "Drift detected, monitoring"}
    
    async def _handle_low_drift(self, metrics: DriftMetrics) -> Dict[str, Any]:
        """Handle low severity drift"""
        
        return {
            "action": "monitor",
            "message": "Low drift detected",
            "steps": [
                "Increase monitoring frequency",
                "Log drift metrics for analysis",
                "No immediate action required"
            ],
            "monitoring_interval": "30 minutes"
        }
    
    async def _handle_medium_drift(self, metrics: DriftMetrics) -> Dict[str, Any]:
        """Handle medium severity drift"""
        
        return {
            "action": "investigate",
            "message": "Medium drift detected",
            "steps": [
                "Trigger automated investigation",
                "Collect additional metrics",
                "Prepare for potential retraining",
                "Alert data science team"
            ],
            "investigation_scope": metrics.features_affected,
            "retraining_readiness": "standby"
        }
    
    async def _handle_high_drift(self, metrics: DriftMetrics) -> Dict[str, Any]:
        """Handle high severity drift"""
        
        return {
            "action": "mitigate",
            "message": "High drift detected - mitigation required",
            "steps": [
                "Switch to fallback model if available",
                "Initiate emergency retraining",
                "Increase prediction uncertainty estimates",
                "Alert stakeholders"
            ],
            "fallback_model": "ensemble_conservative",
            "retraining_priority": "high",
            "uncertainty_multiplier": 1.5
        }
    
    async def _handle_critical_drift(self, metrics: DriftMetrics) -> Dict[str, Any]:
        """Handle critical severity drift"""
        
        return {
            "action": "emergency",
            "message": "CRITICAL drift detected - emergency response",
            "steps": [
                "Pause model predictions",
                "Switch to rule-based fallback",
                "Immediate investigation required",
                "Executive escalation",
                "Prepare incident report"
            ],
            "model_status": "paused",
            "fallback": "rule_based_system",
            "escalation_level": "executive",
            "incident_id": f"DRIFT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }

class DriftMonitor:
    """Continuous drift monitoring service"""
    
    def __init__(self,
                 detector: DriftDetector,
                 handler: AdaptiveDriftHandler,
                 check_interval: int = 3600):  # seconds
        self.detector = detector
        self.handler = handler
        self.check_interval = check_interval
        self.is_running = False
        self.drift_log = []
    
    async def start_monitoring(self):
        """Start continuous drift monitoring"""
        
        self.is_running = True
        logger.info("Drift monitoring started")
        
        while self.is_running:
            try:
                # Get current data window
                current_data = await self._get_current_data_window()
                
                if current_data is not None and len(current_data) > 0:
                    # Detect drift
                    features = list(current_data.columns)
                    metrics = self.detector.detect_drift(
                        current_data,
                        features
                    )
                    
                    # Log metrics
                    self._log_drift_metrics(metrics)
                    
                    # Handle drift if detected
                    if metrics.severity != DriftSeverity.NONE:
                        response = await self.handler.handle_drift(metrics)
                        logger.warning(f"Drift handled: {response}")
                    
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _get_current_data_window(self) -> Optional[pd.DataFrame]:
        """Get current data window for drift detection"""
        # In production, this would fetch from data pipeline
        # For now, return None to indicate no new data
        return None
    
    def _log_drift_metrics(self, metrics: DriftMetrics):
        """Log drift metrics for analysis"""
        
        log_entry = {
            'timestamp': metrics.timestamp.isoformat(),
            'drift_score': metrics.drift_score,
            'p_value': metrics.p_value,
            'severity': metrics.severity.value,
            'drift_type': metrics.drift_type.value,
            'features_affected': metrics.features_affected,
            'recommendations': metrics.recommendations
        }
        
        self.drift_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.drift_log) > 1000:
            self.drift_log = self.drift_log[-1000:]
        
        # Log to file or monitoring system
        logger.info(f"Drift metrics: {json.dumps(log_entry)}")
    
    def stop_monitoring(self):
        """Stop drift monitoring"""
        self.is_running = False
        logger.info("Drift monitoring stopped")
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results"""
        
        if not self.drift_log:
            return {"message": "No drift data available"}
        
        recent_logs = self.drift_log[-100:]  # Last 100 entries
        
        severities = [log['severity'] for log in recent_logs]
        drift_types = [log['drift_type'] for log in recent_logs]
        
        summary = {
            'total_checks': len(recent_logs),
            'drift_detected': sum(1 for s in severities if s != 'none'),
            'severity_distribution': {
                severity: severities.count(severity) 
                for severity in set(severities)
            },
            'drift_type_distribution': {
                dtype: drift_types.count(dtype)
                for dtype in set(drift_types)
            },
            'average_drift_score': np.mean([log['drift_score'] for log in recent_logs]),
            'last_check': recent_logs[-1]['timestamp'] if recent_logs else None
        }
        
        return summary

# Export main components
__all__ = [
    'DriftDetector',
    'AdaptiveDriftHandler',
    'DriftMonitor',
    'DriftMetrics',
    'BaselineProfile',
    'DriftType',
    'DriftSeverity'
]