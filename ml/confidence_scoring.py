"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/confidence_scoring.py
# Confidence Scoring System for PolicyCortex Predictions

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV
import scipy.stats as stats

logger = logging.getLogger(__name__)

@dataclass
class PredictionWithConfidence:
    """Prediction result with confidence metrics"""
    prediction: Any
    confidence: float
    uncertainty: float
    explanation: Dict[str, Any]
    contributing_factors: List[Dict[str, float]]
    reliability_score: float

class ConfidenceScorer:
    """Advanced confidence scoring for predictions"""
    
    def __init__(self):
        self.ensemble_models = []
        self.calibrator = None
        self.feature_quality_thresholds = {}
        self.historical_accuracy = {}
        self.uncertainty_estimator = UncertaintyEstimator()
        
    def calculate_confidence(self, prediction: Any, features: Dict[str, Any]) -> PredictionWithConfidence:
        """Calculate comprehensive confidence score for a prediction"""
        
        # 1. Ensemble disagreement as confidence measure
        ensemble_confidence = self._calculate_ensemble_confidence(features)
        
        # 2. Feature quality assessment
        feature_confidence = self._assess_feature_quality(features)
        
        # 3. Prediction margin (distance from decision boundary)
        margin_confidence = self._calculate_prediction_margin(features)
        
        # 4. Historical accuracy for similar predictions
        historical_confidence = self._get_historical_confidence(features)
        
        # 5. Uncertainty quantification
        uncertainty = self.uncertainty_estimator.estimate(features)
        
        # Weighted combination of confidence factors
        weights = {
            'ensemble': 0.3,
            'features': 0.25,
            'margin': 0.25,
            'historical': 0.2
        }
        
        overall_confidence = (
            weights['ensemble'] * ensemble_confidence +
            weights['features'] * feature_confidence +
            weights['margin'] * margin_confidence +
            weights['historical'] * historical_confidence
        )
        
        # Adjust for uncertainty
        adjusted_confidence = overall_confidence * (1 - uncertainty * 0.5)
        
        # Generate explanation
        explanation = self._generate_confidence_explanation(
            ensemble_confidence, feature_confidence, 
            margin_confidence, historical_confidence, uncertainty
        )
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features)
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(
            features, adjusted_confidence, uncertainty
        )
        
        return PredictionWithConfidence(
            prediction=prediction,
            confidence=adjusted_confidence,
            uncertainty=uncertainty,
            explanation=explanation,
            contributing_factors=contributing_factors,
            reliability_score=reliability_score
        )
    
    def _calculate_ensemble_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate confidence based on ensemble model agreement"""
        if not self.ensemble_models:
            return 0.5
        
        # Get predictions from all models
        feature_array = self._features_to_array(features)
        predictions = []
        
        for model in self.ensemble_models:
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(feature_array.reshape(1, -1))[0]
                    predictions.append(prob)
                else:
                    pred = model.predict(feature_array.reshape(1, -1))[0]
                    predictions.append([1-pred, pred] if pred in [0, 1] else [0.5, 0.5])
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                predictions.append([0.5, 0.5])
        
        if not predictions:
            return 0.5
        
        # Calculate variance across predictions
        predictions = np.array(predictions)
        variance = np.var(predictions[:, 1])  # Variance of positive class probability
        
        # Lower variance = higher confidence
        confidence = 1.0 - min(variance * 2, 1.0)  # Scale variance to [0, 1]
        
        return confidence
    
    def _assess_feature_quality(self, features: Dict[str, Any]) -> float:
        """Assess quality and completeness of input features"""
        quality_score = 1.0
        
        # Check for missing features
        if self.has_missing_features(features):
            quality_score *= 0.8
        
        # Check for outliers
        outlier_score = self._detect_outliers(features)
        quality_score *= (1.0 - outlier_score * 0.3)
        
        # Check data freshness
        if 'timestamp' in features:
            age_hours = (datetime.now() - features['timestamp']).total_seconds() / 3600
            if age_hours > 24:
                quality_score *= 0.9
            if age_hours > 168:  # One week
                quality_score *= 0.7
        
        # Check feature reliability
        for feature, value in features.items():
            if feature in self.feature_quality_thresholds:
                threshold = self.feature_quality_thresholds[feature]
                if not self._meets_quality_threshold(value, threshold):
                    quality_score *= 0.95
        
        return max(quality_score, 0.1)
    
    def _calculate_prediction_margin(self, features: Dict[str, Any]) -> float:
        """Calculate confidence based on distance from decision boundary"""
        if not self.ensemble_models:
            return 0.5
        
        feature_array = self._features_to_array(features)
        margins = []
        
        for model in self.ensemble_models:
            if hasattr(model, 'decision_function'):
                try:
                    # Get distance from decision boundary
                    margin = abs(model.decision_function(feature_array.reshape(1, -1))[0])
                    margins.append(margin)
                except:
                    pass
            elif hasattr(model, 'predict_proba'):
                try:
                    # Use probability difference as proxy for margin
                    probs = model.predict_proba(feature_array.reshape(1, -1))[0]
                    margin = abs(probs[1] - probs[0])
                    margins.append(margin)
                except:
                    pass
        
        if not margins:
            return 0.5
        
        # Average margin across models
        avg_margin = np.mean(margins)
        
        # Normalize to [0, 1] using sigmoid
        confidence = 1 / (1 + np.exp(-2 * avg_margin))
        
        return confidence
    
    def _get_historical_confidence(self, features: Dict[str, Any]) -> float:
        """Get confidence based on historical accuracy for similar predictions"""
        # Find similar historical cases
        feature_hash = self._hash_features(features)
        
        if feature_hash in self.historical_accuracy:
            accuracy_data = self.historical_accuracy[feature_hash]
            if accuracy_data['count'] >= 10:
                return accuracy_data['accuracy']
        
        # Default confidence based on overall model performance
        return 0.75
    
    def has_missing_features(self, features: Dict[str, Any]) -> bool:
        """Check if important features are missing"""
        required_features = [
            'resource_type', 'compliance_score', 'risk_level',
            'last_modified', 'policy_violations'
        ]
        
        missing = sum(1 for f in required_features if f not in features or features[f] is None)
        return missing > 0
    
    def _detect_outliers(self, features: Dict[str, Any]) -> float:
        """Detect outliers in feature values"""
        numeric_features = [v for v in features.values() if isinstance(v, (int, float))]
        
        if not numeric_features:
            return 0.0
        
        # Use z-score for outlier detection
        z_scores = np.abs(stats.zscore(numeric_features))
        outlier_ratio = sum(z > 3 for z in z_scores) / len(z_scores)
        
        return outlier_ratio
    
    def _meets_quality_threshold(self, value: Any, threshold: Dict[str, Any]) -> bool:
        """Check if a feature value meets quality thresholds"""
        if 'min' in threshold and isinstance(value, (int, float)):
            if value < threshold['min']:
                return False
        if 'max' in threshold and isinstance(value, (int, float)):
            if value > threshold['max']:
                return False
        if 'allowed_values' in threshold:
            if value not in threshold['allowed_values']:
                return False
        return True
    
    def _generate_confidence_explanation(
        self, ensemble: float, features: float, 
        margin: float, historical: float, uncertainty: float
    ) -> Dict[str, Any]:
        """Generate human-readable explanation of confidence score"""
        explanation = {
            'summary': self._get_confidence_summary(
                (ensemble + features + margin + historical) / 4
            ),
            'factors': {
                'model_agreement': {
                    'score': ensemble,
                    'description': self._describe_ensemble_confidence(ensemble)
                },
                'data_quality': {
                    'score': features,
                    'description': self._describe_feature_quality(features)
                },
                'prediction_strength': {
                    'score': margin,
                    'description': self._describe_margin_confidence(margin)
                },
                'historical_performance': {
                    'score': historical,
                    'description': self._describe_historical_confidence(historical)
                }
            },
            'uncertainty': {
                'level': uncertainty,
                'description': self._describe_uncertainty(uncertainty)
            }
        }
        
        return explanation
    
    def _get_confidence_summary(self, confidence: float) -> str:
        """Get summary description of confidence level"""
        if confidence >= 0.9:
            return "Very high confidence in this prediction"
        elif confidence >= 0.75:
            return "High confidence in this prediction"
        elif confidence >= 0.6:
            return "Moderate confidence in this prediction"
        elif confidence >= 0.4:
            return "Low confidence in this prediction"
        else:
            return "Very low confidence - prediction uncertain"
    
    def _describe_ensemble_confidence(self, score: float) -> str:
        """Describe ensemble agreement level"""
        if score >= 0.9:
            return "All models strongly agree on this prediction"
        elif score >= 0.7:
            return "Most models agree on this prediction"
        elif score >= 0.5:
            return "Models show some disagreement"
        else:
            return "Significant model disagreement - results may vary"
    
    def _describe_feature_quality(self, score: float) -> str:
        """Describe feature quality assessment"""
        if score >= 0.9:
            return "Excellent data quality with all features present"
        elif score >= 0.75:
            return "Good data quality with minor gaps"
        elif score >= 0.6:
            return "Acceptable data quality with some limitations"
        else:
            return "Poor data quality affecting prediction reliability"
    
    def _describe_margin_confidence(self, score: float) -> str:
        """Describe prediction margin strength"""
        if score >= 0.85:
            return "Clear and decisive prediction"
        elif score >= 0.7:
            return "Reasonably strong prediction"
        elif score >= 0.55:
            return "Prediction near decision boundary"
        else:
            return "Very close to decision boundary - could go either way"
    
    def _describe_historical_confidence(self, score: float) -> str:
        """Describe historical performance"""
        if score >= 0.9:
            return "Excellent historical accuracy for similar cases"
        elif score >= 0.75:
            return "Good historical performance on similar predictions"
        elif score >= 0.6:
            return "Mixed historical results for this type of prediction"
        else:
            return "Limited historical data or poor past performance"
    
    def _describe_uncertainty(self, uncertainty: float) -> str:
        """Describe uncertainty level"""
        if uncertainty < 0.1:
            return "Very low uncertainty"
        elif uncertainty < 0.25:
            return "Low uncertainty"
        elif uncertainty < 0.4:
            return "Moderate uncertainty"
        elif uncertainty < 0.6:
            return "High uncertainty"
        else:
            return "Very high uncertainty - use with caution"
    
    def _identify_contributing_factors(self, features: Dict[str, Any]) -> List[Dict[str, float]]:
        """Identify top factors contributing to the prediction"""
        factors = []
        
        # In production, would use SHAP or similar
        # For now, use feature importance if available
        if self.ensemble_models and hasattr(self.ensemble_models[0], 'feature_importances_'):
            feature_names = list(features.keys())
            importances = self.ensemble_models[0].feature_importances_
            
            for i, importance in enumerate(importances[:5]):  # Top 5 factors
                if i < len(feature_names):
                    factors.append({
                        'feature': feature_names[i],
                        'importance': float(importance),
                        'value': features.get(feature_names[i])
                    })
        
        return factors
    
    def _calculate_reliability_score(
        self, features: Dict[str, Any], 
        confidence: float, uncertainty: float
    ) -> float:
        """Calculate overall reliability score"""
        # Combine confidence and uncertainty into reliability metric
        reliability = confidence * (1 - uncertainty)
        
        # Adjust for data quality
        if self.has_missing_features(features):
            reliability *= 0.9
        
        # Adjust for outliers
        outlier_score = self._detect_outliers(features)
        reliability *= (1 - outlier_score * 0.2)
        
        return max(min(reliability, 1.0), 0.0)
    
    def _features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        # In production, would use proper feature encoding
        numeric_features = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                numeric_features.append(value)
            elif isinstance(value, bool):
                numeric_features.append(1 if value else 0)
            elif value is None:
                numeric_features.append(0)
        
        return np.array(numeric_features)
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Create hash of feature values for similarity matching"""
        # Simplified hashing for demonstration
        feature_str = ''.join(f"{k}:{v}" for k, v in sorted(features.items()))
        return str(hash(feature_str))
    
    def add_ensemble_model(self, model):
        """Add a model to the ensemble"""
        self.ensemble_models.append(model)
    
    def update_historical_accuracy(self, features: Dict[str, Any], was_correct: bool):
        """Update historical accuracy for similar predictions"""
        feature_hash = self._hash_features(features)
        
        if feature_hash not in self.historical_accuracy:
            self.historical_accuracy[feature_hash] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.5
            }
        
        data = self.historical_accuracy[feature_hash]
        data['total'] += 1
        if was_correct:
            data['correct'] += 1
        data['accuracy'] = data['correct'] / data['total']

class UncertaintyEstimator:
    """Estimate prediction uncertainty using various methods"""
    
    def __init__(self):
        self.methods = ['entropy', 'variance', 'mutual_information']
        
    def estimate(self, features: Dict[str, Any]) -> float:
        """Estimate uncertainty for given features"""
        uncertainties = []
        
        # Entropy-based uncertainty
        entropy_uncertainty = self._calculate_entropy_uncertainty(features)
        if entropy_uncertainty is not None:
            uncertainties.append(entropy_uncertainty)
        
        # Variance-based uncertainty
        variance_uncertainty = self._calculate_variance_uncertainty(features)
        if variance_uncertainty is not None:
            uncertainties.append(variance_uncertainty)
        
        # Feature completeness uncertainty
        completeness_uncertainty = self._calculate_completeness_uncertainty(features)
        uncertainties.append(completeness_uncertainty)
        
        # Average uncertainties
        if uncertainties:
            return np.mean(uncertainties)
        return 0.5
    
    def _calculate_entropy_uncertainty(self, features: Dict[str, Any]) -> Optional[float]:
        """Calculate entropy-based uncertainty"""
        # In production, would calculate actual entropy
        # For now, simulate based on feature variance
        numeric_features = [v for v in features.values() if isinstance(v, (int, float))]
        if numeric_features:
            variance = np.var(numeric_features)
            # Map variance to uncertainty
            return min(variance / 10, 1.0)
        return None
    
    def _calculate_variance_uncertainty(self, features: Dict[str, Any]) -> Optional[float]:
        """Calculate variance-based uncertainty"""
        # Would use Monte Carlo dropout or similar in production
        return None
    
    def _calculate_completeness_uncertainty(self, features: Dict[str, Any]) -> float:
        """Calculate uncertainty based on feature completeness"""
        expected_features = 20  # Expected number of features
        actual_features = len([v for v in features.values() if v is not None])
        
        completeness = actual_features / expected_features
        uncertainty = 1.0 - completeness
        
        return max(min(uncertainty, 1.0), 0.0)

class CalibrationSystem:
    """Calibrate prediction probabilities for better confidence estimates"""
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrators = {}
        
    def calibrate_model(self, model, X_cal, y_cal):
        """Calibrate a model using calibration data"""
        calibrated = CalibratedClassifierCV(
            model, 
            method=self.method,
            cv='prefit'
        )
        calibrated.fit(X_cal, y_cal)
        return calibrated
    
    def calibrate_probability(self, prob: float, model_name: str) -> float:
        """Calibrate a single probability value"""
        if model_name in self.calibrators:
            # Apply calibration mapping
            return self.calibrators[model_name].transform([[prob]])[0][0]
        return prob

# Export main components
__all__ = [
    'ConfidenceScorer',
    'PredictionWithConfidence',
    'UncertaintyEstimator',
    'CalibrationSystem'
]