"""
Patent #4: Predictive Policy Compliance Engine
Variational Autoencoder (VAE) Drift Detection with Statistical Process Control

This module implements the VAE-based drift detection subsystem as specified in Patent #4,
with 128-dimensional latent space, reconstruction error analysis, and SPC control limits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
from scipy import stats
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionConfig:
    """Configuration for VAE drift detection and SPC monitoring."""
    latent_dim: int = 128  # Patent specification
    hidden_dims: List[int] = None
    input_dim: int = 256
    beta: float = 1.0  # Beta-VAE weight for KL divergence
    reconstruction_weight: float = 1.0
    spc_window_size: int = 100
    control_limit_sigma: float = 3.0  # Patent specification: 3-sigma rules
    drift_threshold: float = 0.01
    trend_window: int = 10
    run_rules_enabled: bool = True
    

class VAEDriftDetector(nn.Module):
    """
    Variational Autoencoder for configuration drift detection.
    Patent Specifications:
    - Latent space dimensionality: 128
    - Reconstruction error thresholds: Dynamic
    - Bayesian uncertainty quantification
    """
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        super(VAEDriftDetector, self).__init__()
        
        self.config = config or DriftDetectionConfig()
        
        if self.config.hidden_dims is None:
            self.config.hidden_dims = [512, 256]
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)
        self.fc_var = nn.Linear(self.config.hidden_dims[-1], self.config.latent_dim)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Drift tracking
        self.baseline_mu = None
        self.baseline_var = None
        self.drift_history = deque(maxlen=self.config.spc_window_size)
        
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder network."""
        layers = []
        in_dim = self.config.input_dim
        
        for h_dim in self.config.hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            )
            in_dim = h_dim
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder network."""
        layers = []
        hidden_dims_reversed = self.config.hidden_dims[::-1]
        
        # First layer from latent
        layers.append(
            nn.Sequential(
                nn.Linear(self.config.latent_dim, hidden_dims_reversed[0]),
                nn.BatchNorm1d(hidden_dims_reversed[0]),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        )
        
        # Hidden layers
        for i in range(len(hidden_dims_reversed) - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dims_reversed[i], hidden_dims_reversed[i + 1]),
                    nn.BatchNorm1d(hidden_dims_reversed[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
            )
        
        # Output layer
        layers.append(nn.Linear(hidden_dims_reversed[-1], self.config.input_dim))
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mu, log_var) for latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with reconstruction, latent parameters, and loss components
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def loss_function(self, x: torch.Tensor, output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with reconstruction and KL divergence.
        
        Args:
            x: Original input
            output: VAE forward pass output
            
        Returns:
            Dictionary with loss components
        """
        reconstruction = output['reconstruction']
        mu = output['mu']
        log_var = output['log_var']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='none')
        recon_loss = recon_loss.sum(dim=1).mean()
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        
        # Total loss with beta weighting
        total_loss = self.config.reconstruction_weight * recon_loss + self.config.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def calculate_drift_score(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate drift score for input configurations.
        
        Args:
            x: Input configurations
            
        Returns:
            Dictionary with drift scores and components
        """
        self.eval()
        with torch.no_grad():
            # Get latent representation
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            reconstruction = self.decode(z)
            
            # Reconstruction error
            reconstruction_error = F.mse_loss(reconstruction, x, reduction='none')
            reconstruction_score = reconstruction_error.mean(dim=1)
            
            # Latent space drift
            if self.baseline_mu is not None:
                # Calculate Wasserstein distance in latent space
                latent_drift = torch.sqrt(
                    ((mu - self.baseline_mu) ** 2).sum(dim=1)
                )
                
                # Calculate uncertainty-weighted drift
                uncertainty = torch.exp(0.5 * log_var).mean(dim=1)
                weighted_drift = latent_drift / (1 + uncertainty)
            else:
                latent_drift = torch.zeros(x.size(0))
                weighted_drift = torch.zeros(x.size(0))
            
            # Combined drift score
            drift_score = 0.5 * reconstruction_score + 0.5 * weighted_drift
            
            # Property-specific drift (for interpretability)
            property_drift = reconstruction_error  # Per-feature drift
            
            return {
                'drift_score': drift_score,
                'reconstruction_score': reconstruction_score,
                'latent_drift': latent_drift,
                'weighted_drift': weighted_drift,
                'property_drift': property_drift,
                'uncertainty': torch.exp(0.5 * log_var).mean(dim=1)
            }
    
    def set_baseline(self, x: torch.Tensor):
        """
        Set baseline configuration for drift detection.
        
        Args:
            x: Baseline configurations
        """
        self.eval()
        with torch.no_grad():
            mu, log_var = self.encode(x)
            self.baseline_mu = mu.mean(dim=0)
            self.baseline_var = torch.exp(log_var).mean(dim=0)
        
        logger.info(f"Baseline set with latent mu shape: {self.baseline_mu.shape}")
    
    def predict_time_to_violation(self, drift_scores: np.ndarray, 
                                 threshold: float) -> Dict[str, Union[float, str]]:
        """
        Predict time to violation based on drift velocity.
        
        Args:
            drift_scores: Historical drift scores
            threshold: Violation threshold
            
        Returns:
            Dictionary with prediction and confidence
        """
        if len(drift_scores) < 2:
            return {'time_to_violation': np.inf, 'method': 'insufficient_data'}
        
        # Calculate drift velocity
        time_points = np.arange(len(drift_scores))
        
        # Linear extrapolation
        slope, intercept, r_value, _, _ = stats.linregress(time_points, drift_scores)
        
        if slope <= 0:
            return {'time_to_violation': np.inf, 'method': 'no_positive_trend'}
        
        # Time when drift reaches threshold
        time_to_violation_linear = (threshold - drift_scores[-1]) / slope
        
        # Exponential extrapolation for accelerating drift
        if len(drift_scores) > 5:
            try:
                # Fit exponential: y = a * exp(b * x)
                log_scores = np.log(drift_scores + 1e-8)
                exp_slope, exp_intercept, _, _, _ = stats.linregress(time_points, log_scores)
                
                if exp_slope > 0:
                    # Solve for time when exp(exp_intercept + exp_slope * t) = threshold
                    time_to_violation_exp = (np.log(threshold) - exp_intercept) / exp_slope - len(drift_scores)
                else:
                    time_to_violation_exp = np.inf
            except:
                time_to_violation_exp = np.inf
        else:
            time_to_violation_exp = np.inf
        
        # Use minimum of linear and exponential predictions
        time_to_violation = min(time_to_violation_linear, time_to_violation_exp)
        
        return {
            'time_to_violation': max(0, time_to_violation),
            'method': 'linear' if time_to_violation == time_to_violation_linear else 'exponential',
            'confidence': abs(r_value),
            'current_drift': drift_scores[-1],
            'drift_velocity': slope
        }


class StatisticalProcessControl:
    """
    Statistical Process Control for drift monitoring.
    Patent Specification: 3-sigma control limits with Western Electric rules.
    """
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        
        # Control chart parameters
        self.center_line = None
        self.upper_control_limit = None
        self.lower_control_limit = None
        self.moving_range = deque(maxlen=self.config.spc_window_size)
        
        # Historical data
        self.data_history = deque(maxlen=self.config.spc_window_size)
        
        # Rule violation tracking
        self.rule_violations = {
            'rule1': False,  # Single point beyond control limits
            'rule2': False,  # 9 points on same side of centerline
            'rule3': False,  # 6 points trending up or down
            'rule4': False,  # 14 points alternating up and down
            'rule5': False,  # 2 of 3 points beyond 2-sigma
            'rule6': False,  # 4 of 5 points beyond 1-sigma
            'rule7': False,  # 15 points within 1-sigma
            'rule8': False   # 8 points beyond 1-sigma on both sides
        }
        
    def update_control_limits(self, data: np.ndarray):
        """
        Update control limits based on historical data.
        
        Args:
            data: New data points
        """
        # Add to history
        self.data_history.extend(data.tolist())
        
        if len(self.data_history) < 20:
            return  # Need minimum data for reliable limits
        
        # Calculate statistics
        data_array = np.array(self.data_history)
        self.center_line = np.mean(data_array)
        std_dev = np.std(data_array)
        
        # Set control limits (3-sigma)
        self.upper_control_limit = self.center_line + self.config.control_limit_sigma * std_dev
        self.lower_control_limit = self.center_line - self.config.control_limit_sigma * std_dev
        
        # Calculate moving range for improved limits
        if len(self.data_history) > 1:
            for i in range(1, len(self.data_history)):
                self.moving_range.append(abs(self.data_history[i] - self.data_history[i-1]))
            
            # Use average moving range for more robust limits
            if self.moving_range:
                avg_moving_range = np.mean(self.moving_range)
                # d2 constant for n=2 (successive differences)
                d2 = 1.128
                estimated_sigma = avg_moving_range / d2
                
                # Update limits with estimated sigma
                self.upper_control_limit = self.center_line + self.config.control_limit_sigma * estimated_sigma
                self.lower_control_limit = self.center_line - self.config.control_limit_sigma * estimated_sigma
    
    def check_western_electric_rules(self) -> Dict[str, bool]:
        """
        Check Western Electric rules for out-of-control conditions.
        
        Returns:
            Dictionary of rule violations
        """
        if len(self.data_history) < 20:
            return self.rule_violations
        
        data = np.array(self.data_history)
        n = len(data)
        
        # Calculate sigma levels
        std_dev = np.std(data)
        one_sigma_upper = self.center_line + std_dev
        one_sigma_lower = self.center_line - std_dev
        two_sigma_upper = self.center_line + 2 * std_dev
        two_sigma_lower = self.center_line - 2 * std_dev
        
        # Rule 1: Single point beyond 3-sigma limits
        self.rule_violations['rule1'] = bool(
            np.any(data > self.upper_control_limit) or 
            np.any(data < self.lower_control_limit)
        )
        
        # Rule 2: 9 consecutive points on same side of centerline
        if n >= 9:
            for i in range(n - 8):
                subset = data[i:i+9]
                if np.all(subset > self.center_line) or np.all(subset < self.center_line):
                    self.rule_violations['rule2'] = True
                    break
        
        # Rule 3: 6 consecutive points trending up or down
        if n >= 6:
            for i in range(n - 5):
                subset = data[i:i+6]
                diffs = np.diff(subset)
                if np.all(diffs > 0) or np.all(diffs < 0):
                    self.rule_violations['rule3'] = True
                    break
        
        # Rule 4: 14 consecutive points alternating up and down
        if n >= 14:
            for i in range(n - 13):
                subset = data[i:i+14]
                diffs = np.diff(subset)
                signs = np.sign(diffs)
                if np.all(signs[:-1] != signs[1:]):
                    self.rule_violations['rule4'] = True
                    break
        
        # Rule 5: 2 out of 3 consecutive points beyond 2-sigma
        if n >= 3:
            for i in range(n - 2):
                subset = data[i:i+3]
                beyond_2sigma = np.sum(
                    (subset > two_sigma_upper) | (subset < two_sigma_lower)
                )
                if beyond_2sigma >= 2:
                    self.rule_violations['rule5'] = True
                    break
        
        # Rule 6: 4 out of 5 consecutive points beyond 1-sigma
        if n >= 5:
            for i in range(n - 4):
                subset = data[i:i+5]
                beyond_1sigma = np.sum(
                    (subset > one_sigma_upper) | (subset < one_sigma_lower)
                )
                if beyond_1sigma >= 4:
                    self.rule_violations['rule6'] = True
                    break
        
        # Rule 7: 15 consecutive points within 1-sigma
        if n >= 15:
            for i in range(n - 14):
                subset = data[i:i+15]
                within_1sigma = np.all(
                    (subset <= one_sigma_upper) & (subset >= one_sigma_lower)
                )
                if within_1sigma:
                    self.rule_violations['rule7'] = True
                    break
        
        # Rule 8: 8 consecutive points beyond 1-sigma on both sides
        if n >= 8:
            for i in range(n - 7):
                subset = data[i:i+8]
                beyond_1sigma = (subset > one_sigma_upper) | (subset < one_sigma_lower)
                if np.all(beyond_1sigma):
                    self.rule_violations['rule8'] = True
                    break
        
        return self.rule_violations
    
    def detect_drift(self, value: float) -> Dict[str, Union[bool, float, str]]:
        """
        Detect drift using SPC rules.
        
        Args:
            value: New measurement value
            
        Returns:
            Dictionary with drift detection results
        """
        # Add to history
        self.data_history.append(value)
        
        # Update control limits
        self.update_control_limits(np.array([value]))
        
        # Check for out-of-control conditions
        drift_detected = False
        drift_type = 'none'
        
        if self.center_line is not None:
            # Check basic control limits
            if value > self.upper_control_limit:
                drift_detected = True
                drift_type = 'upper_limit_violation'
            elif value < self.lower_control_limit:
                drift_detected = True
                drift_type = 'lower_limit_violation'
            
            # Check Western Electric rules
            if self.config.run_rules_enabled:
                rule_violations = self.check_western_electric_rules()
                if any(rule_violations.values()):
                    drift_detected = True
                    violated_rules = [rule for rule, violated in rule_violations.items() if violated]
                    drift_type = f"rule_violation: {', '.join(violated_rules)}"
        
        return {
            'drift_detected': drift_detected,
            'drift_type': drift_type,
            'value': value,
            'center_line': self.center_line,
            'upper_control_limit': self.upper_control_limit,
            'lower_control_limit': self.lower_control_limit,
            'rule_violations': self.rule_violations.copy()
        }


class IntegratedDriftDetectionSystem:
    """
    Integrated system combining VAE drift detection with Statistical Process Control.
    This implements the complete drift detection subsystem from Patent #4.
    """
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        self.config = config or DriftDetectionConfig()
        
        # Initialize components
        self.vae = VAEDriftDetector(self.config)
        self.spc = StatisticalProcessControl(self.config)
        
        # Drift tracking
        self.drift_scores_history = deque(maxlen=self.config.spc_window_size)
        self.violations_history = []
        
    def train_vae(self, data: torch.Tensor, epochs: int = 100, 
                  batch_size: int = 32, learning_rate: float = 1e-3):
        """
        Train the VAE on baseline configuration data.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        """
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=learning_rate)
        
        n_batches = len(data) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                batch = data[i*batch_size:(i+1)*batch_size]
                
                optimizer.zero_grad()
                output = self.vae(batch)
                losses = self.vae.loss_function(batch, output)
                losses['total_loss'].backward()
                optimizer.step()
                
                epoch_loss += losses['total_loss'].item()
            
            if epoch % 10 == 0:
                logger.info(f"VAE Training - Epoch {epoch}/{epochs}, "
                          f"Loss: {epoch_loss/n_batches:.4f}")
        
        # Set baseline after training
        self.vae.set_baseline(data)
        
    def detect_configuration_drift(self, configuration: torch.Tensor) -> Dict[str, any]:
        """
        Comprehensive drift detection combining VAE and SPC.
        
        Args:
            configuration: Current configuration to check for drift
            
        Returns:
            Comprehensive drift detection results
        """
        # Get VAE drift scores
        vae_results = self.vae.calculate_drift_score(configuration)
        drift_score = vae_results['drift_score'].mean().item()
        
        # Add to history
        self.drift_scores_history.append(drift_score)
        
        # SPC analysis
        spc_results = self.spc.detect_drift(drift_score)
        
        # Time to violation prediction
        if len(self.drift_scores_history) > 5:
            ttv_results = self.vae.predict_time_to_violation(
                np.array(self.drift_scores_history),
                threshold=self.spc.upper_control_limit or self.config.drift_threshold
            )
        else:
            ttv_results = {'time_to_violation': np.inf, 'method': 'insufficient_data'}
        
        # Generate remediation recommendations
        recommendations = self._generate_remediation_recommendations(
            vae_results, spc_results, ttv_results
        )
        
        # Compile results
        results = {
            'drift_detected': spc_results['drift_detected'],
            'drift_score': drift_score,
            'reconstruction_score': vae_results['reconstruction_score'].mean().item(),
            'latent_drift': vae_results['latent_drift'].mean().item(),
            'uncertainty': vae_results['uncertainty'].mean().item(),
            'spc_status': spc_results,
            'time_to_violation': ttv_results,
            'property_drift': vae_results['property_drift'].mean(dim=0).tolist(),
            'recommendations': recommendations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Track violations
        if results['drift_detected']:
            self.violations_history.append(results)
        
        return results
    
    def _generate_remediation_recommendations(self, vae_results: Dict, 
                                             spc_results: Dict, 
                                             ttv_results: Dict) -> List[str]:
        """
        Generate remediation recommendations based on drift analysis.
        
        Args:
            vae_results: VAE drift detection results
            spc_results: SPC analysis results
            ttv_results: Time to violation prediction
            
        Returns:
            List of remediation recommendations
        """
        recommendations = []
        
        # High drift score recommendation
        if vae_results['drift_score'].mean() > self.config.drift_threshold:
            recommendations.append(
                "Review and update configuration baseline - significant drift detected"
            )
        
        # Property-specific recommendations
        property_drift = vae_results['property_drift'].mean(dim=0)
        top_drifting_properties = torch.topk(property_drift, k=min(5, len(property_drift)))[1]
        if len(top_drifting_properties) > 0:
            recommendations.append(
                f"Focus on top drifting configuration properties: {top_drifting_properties.tolist()}"
            )
        
        # SPC rule violations
        if spc_results.get('rule_violations'):
            violated_rules = [rule for rule, violated in spc_results['rule_violations'].items() if violated]
            if violated_rules:
                recommendations.append(
                    f"Statistical anomalies detected: {', '.join(violated_rules)}. "
                    "Investigate systematic configuration changes."
                )
        
        # Time to violation urgency
        if ttv_results['time_to_violation'] < 24:  # Less than 24 time units
            recommendations.append(
                f"URGENT: Predicted violation in {ttv_results['time_to_violation']:.1f} time units. "
                "Immediate remediation required."
            )
        elif ttv_results['time_to_violation'] < 72:  # Less than 72 time units
            recommendations.append(
                f"WARNING: Violation predicted in {ttv_results['time_to_violation']:.1f} time units. "
                "Schedule remediation within next maintenance window."
            )
        
        # High uncertainty recommendation
        if vae_results['uncertainty'].mean() > 0.5:
            recommendations.append(
                "High uncertainty in drift detection - consider validating configuration "
                "against known good baseline"
            )
        
        return recommendations
    
    def get_drift_report(self) -> Dict[str, any]:
        """
        Generate comprehensive drift detection report.
        
        Returns:
            Drift detection report with statistics and trends
        """
        report = {
            'total_evaluations': len(self.drift_scores_history),
            'violations_count': len(self.violations_history),
            'current_drift_score': self.drift_scores_history[-1] if self.drift_scores_history else 0,
            'average_drift_score': np.mean(self.drift_scores_history) if self.drift_scores_history else 0,
            'max_drift_score': np.max(self.drift_scores_history) if self.drift_scores_history else 0,
            'trend': self._calculate_trend(),
            'spc_status': {
                'center_line': self.spc.center_line,
                'upper_control_limit': self.spc.upper_control_limit,
                'lower_control_limit': self.spc.lower_control_limit,
                'current_violations': self.spc.rule_violations
            },
            'recent_violations': self.violations_history[-5:] if self.violations_history else [],
            'report_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report
    
    def _calculate_trend(self) -> str:
        """Calculate drift trend over recent history."""
        if len(self.drift_scores_history) < 10:
            return 'insufficient_data'
        
        recent_scores = list(self.drift_scores_history)[-10:]
        slope, _, r_value, _, _ = stats.linregress(range(len(recent_scores)), recent_scores)
        
        if abs(r_value) < 0.5:
            return 'stable'
        elif slope > 0.001:
            return 'increasing'
        elif slope < -0.001:
            return 'decreasing'
        else:
            return 'stable'


if __name__ == "__main__":
    # Test the integrated drift detection system
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic configuration data
    n_samples = 500
    input_dim = 256
    
    # Normal baseline configurations
    baseline_data = torch.randn(n_samples, input_dim) * 0.5
    
    # Create drift detection system
    drift_system = IntegratedDriftDetectionSystem()
    
    # Train VAE on baseline
    print("Training VAE on baseline configurations...")
    drift_system.train_vae(baseline_data, epochs=20)
    
    # Test with normal configuration
    normal_config = torch.randn(1, input_dim) * 0.5
    normal_results = drift_system.detect_configuration_drift(normal_config)
    print(f"\nNormal Configuration Results:")
    print(f"  Drift detected: {normal_results['drift_detected']}")
    print(f"  Drift score: {normal_results['drift_score']:.4f}")
    print(f"  Reconstruction score: {normal_results['reconstruction_score']:.4f}")
    
    # Test with drifted configuration
    drifted_config = torch.randn(1, input_dim) * 2.0  # Higher variance = drift
    drift_results = drift_system.detect_configuration_drift(drifted_config)
    print(f"\nDrifted Configuration Results:")
    print(f"  Drift detected: {drift_results['drift_detected']}")
    print(f"  Drift score: {drift_results['drift_score']:.4f}")
    print(f"  Reconstruction score: {drift_results['reconstruction_score']:.4f}")
    print(f"  Recommendations: {drift_results['recommendations']}")
    
    # Simulate gradual drift over time
    print(f"\nSimulating gradual drift over 50 time steps...")
    for t in range(50):
        # Gradually increase drift
        drift_factor = 0.5 + (t / 50) * 1.5
        config = torch.randn(1, input_dim) * drift_factor
        results = drift_system.detect_configuration_drift(config)
        
        if t % 10 == 0:
            print(f"  Time {t}: Drift score={results['drift_score']:.4f}, "
                  f"Detected={results['drift_detected']}, "
                  f"Time to violation={results['time_to_violation'].get('time_to_violation', 'N/A')}")
    
    # Generate final report
    report = drift_system.get_drift_report()
    print(f"\nFinal Drift Report:")
    print(f"  Total evaluations: {report['total_evaluations']}")
    print(f"  Violations count: {report['violations_count']}")
    print(f"  Average drift score: {report['average_drift_score']:.4f}")
    print(f"  Max drift score: {report['max_drift_score']:.4f}")
    print(f"  Trend: {report['trend']}")
    print(f"  SPC Status: {report['spc_status']}")