"""
Patent #4: Confidence Scoring Module
Uncertainty quantification and risk-based decision making
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- Violation probability calculation within specified time windows
- Confidence intervals using Monte Carlo dropout
- Bayesian uncertainty quantification
- Risk-adjusted impact scoring
- Calibrated confidence scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    """Container for risk assessment results"""
    violation_probability: float
    time_window_hours: int
    confidence_score: float
    confidence_interval: Tuple[float, float]
    risk_level: str  # 'critical', 'high', 'medium', 'low'
    impact_score: float
    uncertainty_sources: Dict[str, float]
    recommendations: List[str]


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty quantification
    Patent Requirement: Confidence intervals using MC dropout
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        
    def predict_with_uncertainty(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Generate predictions with uncertainty estimates
        Returns mean prediction, std deviation, and confidence intervals
        """
        # Enable dropout during inference
        self.model.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Forward pass with dropout
                output = self.model(X)
                
                if isinstance(output, dict):
                    pred = output['prediction']
                else:
                    pred = output
                
                # Apply softmax if not already applied
                if pred.dim() > 1 and pred.size(-1) > 1:
                    pred = F.softmax(pred, dim=-1)
                
                predictions.append(pred.cpu().numpy())
        
        # Back to eval mode
        self.model.eval()
        
        # Stack predictions
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals (95%)
        lower_ci = np.percentile(predictions, 2.5, axis=0)
        upper_ci = np.percentile(predictions, 97.5, axis=0)
        
        # Calculate predictive entropy (uncertainty measure)
        mean_probs = np.mean(predictions, axis=0)
        predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
        
        # Calculate mutual information (epistemic uncertainty)
        expected_entropy = np.mean(
            -np.sum(predictions * np.log(predictions + 1e-8), axis=-1),
            axis=0
        )
        mutual_information = predictive_entropy - expected_entropy
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'confidence_interval_lower': lower_ci,
            'confidence_interval_upper': upper_ci,
            'predictive_entropy': predictive_entropy,
            'mutual_information': mutual_information,
            'epistemic_uncertainty': mutual_information,
            'aleatoric_uncertainty': expected_entropy
        }


class BayesianUncertainty:
    """
    Bayesian uncertainty quantification for probabilistic predictions
    Patent Requirement: Bayesian inference for uncertainty
    """
    
    def __init__(self, prior_mean: float = 0.5, prior_variance: float = 0.25):
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_samples = []
        
    def update_posterior(self, observations: np.ndarray, outcomes: np.ndarray):
        """Update posterior distribution with new observations"""
        # Using conjugate prior (Beta distribution for binary outcomes)
        alpha = self.prior_mean * (1/self.prior_variance - 1)
        beta = (1 - self.prior_mean) * (1/self.prior_variance - 1)
        
        # Update with observations
        successes = np.sum(outcomes)
        failures = len(outcomes) - successes
        
        # Posterior parameters
        posterior_alpha = alpha + successes
        posterior_beta = beta + failures
        
        # Sample from posterior
        self.posterior_samples = np.random.beta(
            posterior_alpha, posterior_beta, size=1000
        )
        
        return posterior_alpha, posterior_beta
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Dict[str, float]:
        """Generate prediction with Bayesian uncertainty"""
        if len(self.posterior_samples) == 0:
            # Use prior if no updates
            samples = np.random.beta(
                self.prior_mean * 10,
                (1 - self.prior_mean) * 10,
                size=1000
            )
        else:
            samples = self.posterior_samples
        
        # Calculate statistics
        mean_pred = np.mean(samples)
        std_pred = np.std(samples)
        
        # Credible interval (95%)
        lower_ci = np.percentile(samples, 2.5)
        upper_ci = np.percentile(samples, 97.5)
        
        # Calculate entropy as uncertainty measure
        hist, _ = np.histogram(samples, bins=50, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        entropy = -np.sum(hist * np.log(hist)) / len(hist)
        
        return {
            'mean': float(mean_pred),
            'std': float(std_pred),
            'lower_ci': float(lower_ci),
            'upper_ci': float(upper_ci),
            'entropy': float(entropy),
            'confidence': float(1.0 - std_pred)  # Simple confidence measure
        }


class CalibrationModule:
    """
    Calibrate model confidence scores to ensure reliability
    Patent Requirement: Calibrated confidence scores
    """
    
    def __init__(self, method: str = 'isotonic'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, predictions: np.ndarray, labels: np.ndarray):
        """Fit calibration mapping"""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'sigmoid':
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        # Fit calibrator
        self.calibrator.fit(predictions.reshape(-1, 1), labels)
        self.is_fitted = True
        
        # Calculate calibration metrics
        calibration_error = self._calculate_calibration_error(predictions, labels)
        logger.info(f"Calibration error: {calibration_error:.4f}")
    
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """Apply calibration to predictions"""
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning uncalibrated predictions")
            return predictions
        
        calibrated = self.calibrator.predict(predictions.reshape(-1, 1))
        return calibrated
    
    def _calculate_calibration_error(self, predictions: np.ndarray, 
                                    labels: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class RiskScorer:
    """
    Calculate risk-adjusted impact scores
    Patent Requirement: Risk-adjusted scoring for decision making
    """
    
    def __init__(self):
        self.impact_weights = {
            'security': 0.4,
            'compliance': 0.3,
            'operational': 0.2,
            'financial': 0.1
        }
        self.severity_multipliers = {
            'critical': 10.0,
            'high': 5.0,
            'medium': 2.0,
            'low': 1.0
        }
        
    def calculate_risk_score(self, 
                            violation_probability: float,
                            impact_factors: Dict[str, float],
                            time_to_violation: Optional[float] = None) -> float:
        """
        Calculate comprehensive risk score
        Combines probability, impact, and time factors
        """
        # Base risk from violation probability
        base_risk = violation_probability
        
        # Calculate weighted impact score
        impact_score = 0
        for factor, weight in self.impact_weights.items():
            impact_score += impact_factors.get(factor, 0) * weight
        
        # Time-based urgency factor
        if time_to_violation is not None:
            # Urgent if violation expected within 24 hours
            urgency = np.exp(-time_to_violation / 24)
        else:
            urgency = 0.5
        
        # Combined risk score
        risk_score = base_risk * impact_score * urgency
        
        # Normalize to [0, 1]
        risk_score = np.tanh(risk_score)
        
        return float(risk_score)
    
    def classify_risk_level(self, risk_score: float) -> str:
        """Classify risk into categories"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def calculate_impact_factors(self, configuration: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact factors from configuration context"""
        factors = {}
        
        # Security impact
        factors['security'] = 0.5  # Base score
        if configuration.get('encryption', {}).get('enabled', False):
            factors['security'] -= 0.2
        if configuration.get('public_access', False):
            factors['security'] += 0.3
        if configuration.get('mfa', {}).get('enabled', False):
            factors['security'] -= 0.1
        
        # Compliance impact
        factors['compliance'] = 0.5
        compliance_tags = configuration.get('compliance_tags', [])
        if 'HIPAA' in compliance_tags or 'PCI-DSS' in compliance_tags:
            factors['compliance'] += 0.3
        if 'SOC2' in compliance_tags:
            factors['compliance'] += 0.2
        
        # Operational impact
        factors['operational'] = 0.3
        criticality = configuration.get('criticality', 'medium')
        if criticality == 'critical':
            factors['operational'] = 0.9
        elif criticality == 'high':
            factors['operational'] = 0.7
        
        # Financial impact
        factors['financial'] = 0.2
        monthly_cost = configuration.get('monthly_cost', 0)
        if monthly_cost > 10000:
            factors['financial'] = 0.8
        elif monthly_cost > 5000:
            factors['financial'] = 0.5
        
        # Normalize all factors to [0, 1]
        for key in factors:
            factors[key] = min(max(factors[key], 0), 1)
        
        return factors


class TimeWindowPredictor:
    """
    Calculate violation probability within specified time windows
    Patent Requirement: Time-based violation probability
    """
    
    def __init__(self):
        self.time_windows = [24, 48, 72]  # Hours
        self.decay_rate = 0.1  # Exponential decay rate
        
    def predict_time_windows(self, 
                            current_probability: float,
                            drift_velocity: float,
                            historical_trends: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Predict violation probability for different time windows
        Returns probability for each time window
        """
        predictions = {}
        
        for window in self.time_windows:
            # Linear extrapolation with drift velocity
            linear_prob = current_probability + (drift_velocity * window / 24)
            
            # Exponential growth if trending upward
            if drift_velocity > 0:
                exp_prob = current_probability * np.exp(self.decay_rate * window / 24)
            else:
                exp_prob = current_probability * np.exp(-self.decay_rate * window / 24)
            
            # Combine predictions (weighted average)
            combined_prob = 0.7 * linear_prob + 0.3 * exp_prob
            
            # Apply bounds
            predictions[window] = float(min(max(combined_prob, 0), 1))
        
        # Adjust based on historical trends if available
        if historical_trends is not None and len(historical_trends) > 24:
            trend_adjustment = self._calculate_trend_adjustment(historical_trends)
            for window in predictions:
                predictions[window] *= (1 + trend_adjustment * window / 72)
                predictions[window] = float(min(max(predictions[window], 0), 1))
        
        return predictions
    
    def _calculate_trend_adjustment(self, historical: np.ndarray) -> float:
        """Calculate adjustment factor based on historical trends"""
        # Fit linear trend
        x = np.arange(len(historical))
        slope, _ = np.polyfit(x, historical, 1)
        
        # Normalize slope to [-0.5, 0.5] range
        trend_adjustment = np.tanh(slope * 100)
        return float(trend_adjustment)


class ConfidenceScoringEngine:
    """
    Main confidence scoring engine orchestrating all uncertainty components
    Provides unified interface for risk assessment and confidence scoring
    """
    
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model
        
        # Initialize components
        self.mc_dropout = MonteCarloDropout(model) if model else None
        self.bayesian = BayesianUncertainty()
        self.calibration = CalibrationModule()
        self.risk_scorer = RiskScorer()
        self.time_predictor = TimeWindowPredictor()
        
        self.assessment_cache = {}
        
    def assess_risk(self, 
                   X: torch.Tensor,
                   configuration: Dict[str, Any],
                   drift_metrics: Optional[Dict[str, float]] = None) -> RiskAssessment:
        """
        Comprehensive risk assessment with uncertainty quantification
        Patent Requirements: All confidence scoring requirements
        """
        # Get MC dropout predictions if model available
        if self.mc_dropout:
            mc_results = self.mc_dropout.predict_with_uncertainty(X)
            base_probability = float(mc_results['mean_prediction'][0, 1])  # Binary class 1
            epistemic_uncertainty = float(mc_results['epistemic_uncertainty'][0])
            aleatoric_uncertainty = float(mc_results['aleatoric_uncertainty'][0])
            confidence_interval = (
                float(mc_results['confidence_interval_lower'][0, 1]),
                float(mc_results['confidence_interval_upper'][0, 1])
            )
        else:
            # Fallback values
            base_probability = 0.5
            epistemic_uncertainty = 0.1
            aleatoric_uncertainty = 0.1
            confidence_interval = (0.3, 0.7)
        
        # Get Bayesian uncertainty
        bayes_results = self.bayesian.predict_with_uncertainty(X.numpy() if isinstance(X, torch.Tensor) else X)
        
        # Combine uncertainties
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        confidence_score = 1.0 / (1.0 + total_uncertainty)
        
        # Calibrate probability
        if self.calibration.is_fitted:
            calibrated_prob = self.calibration.calibrate(np.array([base_probability]))[0]
        else:
            calibrated_prob = base_probability
        
        # Calculate impact factors
        impact_factors = self.risk_scorer.calculate_impact_factors(configuration)
        
        # Get drift velocity for time predictions
        drift_velocity = drift_metrics.get('drift_velocity', 0) if drift_metrics else 0
        
        # Predict time windows
        time_predictions = self.time_predictor.predict_time_windows(
            calibrated_prob, drift_velocity
        )
        
        # Use 72-hour window as primary
        primary_window = 72
        primary_probability = time_predictions.get(primary_window, calibrated_prob)
        
        # Calculate risk score
        risk_score = self.risk_scorer.calculate_risk_score(
            primary_probability,
            impact_factors,
            primary_window
        )
        
        # Classify risk level
        risk_level = self.risk_scorer.classify_risk_level(risk_score)
        
        # Calculate overall impact score
        impact_score = sum(impact_factors.values()) / len(impact_factors)
        
        # Uncertainty sources breakdown
        uncertainty_sources = {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'model': bayes_results['entropy'],
            'calibration': abs(calibrated_prob - base_probability)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, primary_probability, impact_factors, uncertainty_sources
        )
        
        # Create assessment result
        assessment = RiskAssessment(
            violation_probability=primary_probability,
            time_window_hours=primary_window,
            confidence_score=confidence_score,
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            impact_score=impact_score,
            uncertainty_sources=uncertainty_sources,
            recommendations=recommendations
        )
        
        # Cache result
        cache_key = hashlib.md5(str(configuration).encode()).hexdigest()
        self.assessment_cache[cache_key] = assessment
        
        return assessment
    
    def _generate_recommendations(self,
                                risk_level: str,
                                probability: float,
                                impact_factors: Dict[str, float],
                                uncertainty: Dict[str, float]) -> List[str]:
        """Generate risk-based recommendations"""
        recommendations = []
        
        # Risk level recommendations
        if risk_level == 'critical':
            recommendations.append("IMMEDIATE ACTION REQUIRED: Critical violation risk detected")
            recommendations.append("Initiate emergency remediation procedures")
        elif risk_level == 'high':
            recommendations.append("HIGH PRIORITY: Schedule immediate configuration review")
            recommendations.append("Implement preventive controls within 24 hours")
        elif risk_level == 'medium':
            recommendations.append("MODERATE RISK: Plan remediation within 72 hours")
        
        # High uncertainty recommendations
        if sum(uncertainty.values()) > 0.5:
            recommendations.append("High uncertainty detected - gather additional data for better assessment")
        
        # Impact-specific recommendations
        if impact_factors.get('security', 0) > 0.7:
            recommendations.append("Critical security impact - review access controls and encryption")
        
        if impact_factors.get('compliance', 0) > 0.7:
            recommendations.append("Compliance violation risk - verify regulatory requirements")
        
        if impact_factors.get('operational', 0) > 0.7:
            recommendations.append("High operational impact - ensure backup and recovery procedures")
        
        return recommendations[:5]  # Limit to top 5
    
    def update_calibration(self, predictions: np.ndarray, outcomes: np.ndarray):
        """Update calibration with new data"""
        self.calibration.fit(predictions, outcomes)
        logger.info("Calibration updated with new outcome data")
    
    def update_bayesian_prior(self, observations: np.ndarray, outcomes: np.ndarray):
        """Update Bayesian prior with observations"""
        self.bayesian.update_posterior(observations, outcomes)
        logger.info("Bayesian posterior updated")
    
    def get_confidence_report(self) -> Dict[str, Any]:
        """Generate confidence scoring report"""
        if not self.assessment_cache:
            return {'status': 'no_assessments'}
        
        # Aggregate statistics from cache
        all_scores = [a.confidence_score for a in self.assessment_cache.values()]
        all_probs = [a.violation_probability for a in self.assessment_cache.values()]
        risk_levels = [a.risk_level for a in self.assessment_cache.values()]
        
        return {
            'total_assessments': len(self.assessment_cache),
            'average_confidence': float(np.mean(all_scores)),
            'confidence_std': float(np.std(all_scores)),
            'average_violation_probability': float(np.mean(all_probs)),
            'risk_distribution': {
                'critical': risk_levels.count('critical') / len(risk_levels),
                'high': risk_levels.count('high') / len(risk_levels),
                'medium': risk_levels.count('medium') / len(risk_levels),
                'low': risk_levels.count('low') / len(risk_levels)
            },
            'calibration_fitted': self.calibration.is_fitted,
            'timestamp': datetime.now().isoformat()
        }