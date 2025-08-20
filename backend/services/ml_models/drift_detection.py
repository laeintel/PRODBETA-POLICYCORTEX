"""
Patent #4: Drift Detection Subsystem
VAE + Reconstruction Error + Statistical Process Control
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- VAE Latent Space: 128 dimensions
- Reconstruction Error Thresholds: Dynamic based on historical variance
- SPC Control Limits: 3-sigma rules
- Bayesian uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu
import logging

logger = logging.getLogger(__name__)

@dataclass
class DriftMetrics:
    """Container for drift detection metrics"""
    drift_score: float
    drift_velocity: float
    time_to_violation: Optional[float]
    confidence: float
    property_scores: Dict[str, float]
    recommendations: List[str]


class VAEDriftDetector(nn.Module):
    """
    Variational Autoencoder for configuration drift detection
    Patent Spec: Latent space dimensionality of 128
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim)
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        VAE loss = Reconstruction loss + KL divergence
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl_divergence': kl_loss
        }
    
    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for drift detection"""
        with torch.no_grad():
            reconstruction, _, _ = self.forward(x)
            error = F.mse_loss(reconstruction, x, reduction='none')
            return error.mean(dim=1)
    
    def compute_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian uncertainty quantification through multiple forward passes
        Returns: (mean_prediction, uncertainty)
        """
        predictions = []
        
        self.train()  # Enable dropout for uncertainty estimation
        with torch.no_grad():
            for _ in range(n_samples):
                reconstruction, _, _ = self.forward(x)
                predictions.append(reconstruction)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        self.eval()
        return mean_pred, uncertainty


class SPCMonitor:
    """
    Statistical Process Control for configuration monitoring
    Patent Spec: 3-sigma control limits with Western Electric rules
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.control_limits = {}
        self.baselines = {}
        self.history = []
        
    def establish_baseline(self, data: np.ndarray, feature_names: List[str]):
        """Establish baseline statistics and control limits"""
        if len(data) < self.window_size:
            logger.warning(f"Insufficient data for baseline. Need {self.window_size}, got {len(data)}")
            data = np.pad(data, ((0, self.window_size - len(data)), (0, 0)), mode='mean')
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # Calculate statistics
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            
            # 3-sigma control limits
            self.control_limits[feature] = {
                'ucl': mean + 3 * std,  # Upper control limit
                'uwl': mean + 2 * std,  # Upper warning limit
                'center': mean,
                'lwl': mean - 2 * std,  # Lower warning limit
                'lcl': mean - 3 * std   # Lower control limit
            }
            
            self.baselines[feature] = {
                'mean': mean,
                'std': std,
                'median': np.median(feature_data),
                'q1': np.percentile(feature_data, 25),
                'q3': np.percentile(feature_data, 75)
            }
        
        logger.info(f"Established baseline for {len(feature_names)} features")
    
    def check_control_rules(self, data: np.ndarray, feature_names: List[str]) -> Dict[str, List[str]]:
        """
        Apply Western Electric rules for out-of-control detection
        Returns violations for each feature
        """
        violations = {feature: [] for feature in feature_names}
        
        for i, feature in enumerate(feature_names):
            if feature not in self.control_limits:
                continue
            
            limits = self.control_limits[feature]
            values = data[:, i]
            
            # Rule 1: One point outside 3-sigma limits
            if np.any(values > limits['ucl']) or np.any(values < limits['lcl']):
                violations[feature].append("Rule 1: Point outside 3-sigma limits")
            
            # Rule 2: Two out of three consecutive points outside 2-sigma limits
            for j in range(len(values) - 2):
                subset = values[j:j+3]
                outside_2sigma = np.sum((subset > limits['uwl']) | (subset < limits['lwl']))
                if outside_2sigma >= 2:
                    violations[feature].append("Rule 2: 2/3 points outside 2-sigma")
                    break
            
            # Rule 3: Four out of five consecutive points outside 1-sigma limits
            for j in range(len(values) - 4):
                subset = values[j:j+5]
                one_sigma_upper = limits['center'] + self.baselines[feature]['std']
                one_sigma_lower = limits['center'] - self.baselines[feature]['std']
                outside_1sigma = np.sum((subset > one_sigma_upper) | (subset < one_sigma_lower))
                if outside_1sigma >= 4:
                    violations[feature].append("Rule 3: 4/5 points outside 1-sigma")
                    break
            
            # Rule 4: Eight consecutive points on one side of centerline
            for j in range(len(values) - 7):
                subset = values[j:j+8]
                if np.all(subset > limits['center']) or np.all(subset < limits['center']):
                    violations[feature].append("Rule 4: 8 points on one side")
                    break
            
            # Rule 5: Six points in a row steadily increasing or decreasing
            for j in range(len(values) - 5):
                subset = values[j:j+6]
                diffs = np.diff(subset)
                if np.all(diffs > 0) or np.all(diffs < 0):
                    violations[feature].append("Rule 5: 6 points trending")
                    break
        
        return violations
    
    def calculate_cusum(self, data: np.ndarray, feature_idx: int, k: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate CUSUM (Cumulative Sum) for detecting small shifts
        Returns: (cusum_upper, cusum_lower)
        """
        feature_name = list(self.baselines.keys())[feature_idx]
        target = self.baselines[feature_name]['mean']
        sigma = self.baselines[feature_name]['std']
        
        values = data[:, feature_idx]
        n = len(values)
        
        cusum_upper = np.zeros(n)
        cusum_lower = np.zeros(n)
        
        for i in range(1, n):
            cusum_upper[i] = max(0, values[i] - (target + k * sigma) + cusum_upper[i-1])
            cusum_lower[i] = max(0, (target - k * sigma) - values[i] + cusum_lower[i-1])
        
        return cusum_upper, cusum_lower
    
    def detect_distribution_shift(self, current_data: np.ndarray, 
                                 historical_data: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Detect distribution shifts using statistical tests
        Returns p-values for each feature and test
        """
        results = {}
        
        for i, feature in enumerate(feature_names):
            current = current_data[:, i]
            historical = historical_data[:, i]
            
            # Kolmogorov-Smirnov test for distribution change
            ks_stat, ks_pvalue = ks_2samp(current, historical)
            
            # Mann-Whitney U test for median shift
            mw_stat, mw_pvalue = mannwhitneyu(current, historical, alternative='two-sided')
            
            # Anderson-Darling test would go here if needed
            
            results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'significant_shift': ks_pvalue < 0.05 or mw_pvalue < 0.05
            }
        
        return results


class DriftAnalyzer:
    """
    Comprehensive drift analysis combining VAE and SPC
    Calculates drift scores, velocity, and time-to-violation predictions
    """
    
    def __init__(self, input_dim: int):
        self.vae = VAEDriftDetector(input_dim)
        self.spc = SPCMonitor()
        self.historical_drifts = []
        self.reconstruction_thresholds = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, feature_names: List[str], epochs: int = 100):
        """Train VAE and establish SPC baselines"""
        logger.info("Training Drift Detection System...")
        
        # Train VAE
        self._train_vae(X, epochs)
        
        # Establish SPC baselines
        self.spc.establish_baseline(X, feature_names)
        
        # Calculate reconstruction error thresholds
        self._calculate_thresholds(X)
        
        self.is_fitted = True
        logger.info("Drift detection system ready")
    
    def _train_vae(self, X: np.ndarray, epochs: int):
        """Train the VAE model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = self.vae.to(device)
        
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X).to(device)
        
        self.vae.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            reconstruction, mu, logvar = self.vae(X_tensor)
            losses = self.vae.loss_function(reconstruction, X_tensor, mu, logvar)
            
            losses['total'].backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {losses['total'].item():.4f}")
        
        self.vae.eval()
    
    def _calculate_thresholds(self, X: np.ndarray):
        """Calculate dynamic reconstruction error thresholds"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Get reconstruction errors
        errors = self.vae.compute_reconstruction_error(X_tensor).cpu().numpy()
        
        # Calculate percentile-based thresholds
        self.reconstruction_thresholds = {
            'normal': np.percentile(errors, 95),
            'warning': np.percentile(errors, 98),
            'critical': np.percentile(errors, 99.5)
        }
        
        # Store historical variance for dynamic adjustment
        self.error_variance = np.var(errors)
        self.error_mean = np.mean(errors)
    
    def analyze_drift(self, current_config: np.ndarray, 
                     historical_configs: np.ndarray,
                     feature_names: List[str]) -> DriftMetrics:
        """
        Comprehensive drift analysis
        Returns drift metrics including score, velocity, and predictions
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted before use")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # VAE-based drift detection
        current_tensor = torch.FloatTensor(current_config).to(device)
        reconstruction_error = self.vae.compute_reconstruction_error(current_tensor).cpu().numpy()
        
        # Normalize drift score
        drift_score = (reconstruction_error - self.error_mean) / (np.sqrt(self.error_variance) + 1e-8)
        drift_score = float(np.mean(drift_score))
        
        # Calculate drift velocity (rate of change)
        if len(self.historical_drifts) > 0:
            recent_drifts = self.historical_drifts[-10:]
            drift_velocity = (drift_score - np.mean(recent_drifts)) / max(len(recent_drifts), 1)
        else:
            drift_velocity = 0.0
        
        self.historical_drifts.append(drift_score)
        
        # SPC-based analysis
        spc_violations = self.spc.check_control_rules(current_config.reshape(1, -1), feature_names)
        
        # Distribution shift detection
        if len(historical_configs) > 0:
            distribution_shifts = self.spc.detect_distribution_shift(
                current_config.reshape(1, -1),
                historical_configs,
                feature_names
            )
        else:
            distribution_shifts = {}
        
        # Property-specific drift scores
        property_scores = {}
        for i, feature in enumerate(feature_names):
            feature_error = float(reconstruction_error[i] if i < len(reconstruction_error) else 0)
            property_scores[feature] = min(feature_error / (self.reconstruction_thresholds['normal'] + 1e-8), 1.0)
        
        # Predict time to violation
        time_to_violation = self._predict_time_to_violation(drift_score, drift_velocity)
        
        # Bayesian uncertainty quantification
        _, uncertainty = self.vae.compute_uncertainty(current_tensor)
        confidence = 1.0 / (1.0 + float(uncertainty.mean()))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            drift_score, spc_violations, distribution_shifts, property_scores
        )
        
        return DriftMetrics(
            drift_score=drift_score,
            drift_velocity=drift_velocity,
            time_to_violation=time_to_violation,
            confidence=confidence,
            property_scores=property_scores,
            recommendations=recommendations
        )
    
    def _predict_time_to_violation(self, current_drift: float, velocity: float) -> Optional[float]:
        """
        Predict time to violation using linear and exponential extrapolation
        Returns hours until predicted violation
        """
        if velocity <= 0:
            return None  # No violation predicted if drift is decreasing
        
        # Violation threshold (normalized score)
        violation_threshold = 3.0  # 3 standard deviations
        
        if current_drift >= violation_threshold:
            return 0.0  # Already in violation
        
        # Linear extrapolation
        if velocity > 0:
            hours_to_violation = (violation_threshold - current_drift) / (velocity * 24)
            
            # Cap at 72 hours (3 days) for practical purposes
            return min(hours_to_violation, 72.0)
        
        return None
    
    def _generate_recommendations(self, drift_score: float,
                                 spc_violations: Dict[str, List[str]],
                                 distribution_shifts: Dict[str, Dict[str, float]],
                                 property_scores: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        # Overall drift recommendations
        if drift_score > 2.0:
            recommendations.append("CRITICAL: Significant configuration drift detected. Immediate review required.")
        elif drift_score > 1.0:
            recommendations.append("WARNING: Moderate drift detected. Schedule configuration review.")
        
        # SPC violation recommendations
        features_with_violations = [f for f, v in spc_violations.items() if v]
        if features_with_violations:
            recommendations.append(f"Review SPC violations in: {', '.join(features_with_violations[:3])}")
        
        # Distribution shift recommendations
        shifted_features = [f for f, v in distribution_shifts.items() if v.get('significant_shift', False)]
        if shifted_features:
            recommendations.append(f"Investigate distribution changes in: {', '.join(shifted_features[:3])}")
        
        # Property-specific recommendations
        critical_properties = [f for f, s in property_scores.items() if s > 0.8]
        if critical_properties:
            recommendations.append(f"Critical drift in properties: {', '.join(critical_properties[:3])}")
        
        # Remediation suggestions
        if drift_score > 1.5:
            recommendations.extend([
                "Consider rolling back recent configuration changes",
                "Verify compliance with baseline security policies",
                "Run automated compliance validation"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations


class ConfigurationDriftEngine:
    """
    Main drift detection engine orchestrating VAE and SPC components
    Provides unified interface for drift detection and analysis
    """
    
    def __init__(self, input_dim: int, feature_names: List[str]):
        self.input_dim = input_dim
        self.feature_names = feature_names
        self.analyzer = DriftAnalyzer(input_dim)
        self.drift_history = []
        self.violation_predictions = []
        
    def initialize(self, baseline_configs: np.ndarray):
        """Initialize drift detection with baseline configurations"""
        logger.info("Initializing Configuration Drift Engine...")
        self.analyzer.fit(baseline_configs, self.feature_names)
        logger.info("Drift engine initialized successfully")
    
    def detect_drift(self, current_config: np.ndarray) -> Dict[str, Any]:
        """
        Main drift detection interface
        Returns comprehensive drift analysis results
        """
        # Get historical configs for comparison
        historical = np.array(self.drift_history[-100:]) if self.drift_history else np.array([])
        
        # Perform drift analysis
        metrics = self.analyzer.analyze_drift(
            current_config,
            historical.reshape(-1, self.input_dim) if len(historical) > 0 else np.array([]),
            self.feature_names
        )
        
        # Store current config in history
        self.drift_history.append(current_config.flatten())
        
        # Track violation predictions
        if metrics.time_to_violation is not None:
            self.violation_predictions.append({
                'timestamp': datetime.now(),
                'predicted_violation_time': datetime.now() + timedelta(hours=metrics.time_to_violation),
                'confidence': metrics.confidence
            })
        
        # Compile results
        result = {
            'drift_detected': metrics.drift_score > 1.0,
            'drift_score': metrics.drift_score,
            'drift_velocity': metrics.drift_velocity,
            'time_to_violation_hours': metrics.time_to_violation,
            'confidence': metrics.confidence,
            'property_drift_scores': metrics.property_scores,
            'recommendations': metrics.recommendations,
            'severity': self._classify_severity(metrics.drift_score),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _classify_severity(self, drift_score: float) -> str:
        """Classify drift severity based on score"""
        if drift_score < 0.5:
            return 'none'
        elif drift_score < 1.0:
            return 'low'
        elif drift_score < 2.0:
            return 'medium'
        elif drift_score < 3.0:
            return 'high'
        else:
            return 'critical'
    
    def get_drift_trend(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get drift trend analysis over specified time window"""
        if not self.drift_history:
            return {'trend': 'insufficient_data', 'samples': 0}
        
        # Calculate trend statistics
        recent_drifts = self.drift_history[-window_hours:]
        if len(recent_drifts) < 2:
            return {'trend': 'insufficient_data', 'samples': len(recent_drifts)}
        
        # Perform trend analysis
        drift_scores = [self.analyzer.analyze_drift(
            np.array(config).reshape(1, -1),
            np.array([]),
            self.feature_names
        ).drift_score for config in recent_drifts]
        
        # Calculate trend direction
        trend_coefficient = np.polyfit(range(len(drift_scores)), drift_scores, 1)[0]
        
        trend_direction = 'increasing' if trend_coefficient > 0.01 else \
                         'decreasing' if trend_coefficient < -0.01 else 'stable'
        
        return {
            'trend': trend_direction,
            'trend_coefficient': float(trend_coefficient),
            'mean_drift': float(np.mean(drift_scores)),
            'max_drift': float(np.max(drift_scores)),
            'min_drift': float(np.min(drift_scores)),
            'samples': len(drift_scores)
        }