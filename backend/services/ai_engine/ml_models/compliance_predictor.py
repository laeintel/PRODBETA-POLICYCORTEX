"""
Predictive Policy Compliance Engine for PolicyCortex.
Implements Patent 1: Machine Learning System for Temporal Predictive Cloud Policy Compliance Analysis.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from statsmodels.tsa.seasonal import STL
from scipy import stats

logger = structlog.get_logger(__name__)


class ComplianceDriftDetector:
    """
    Configuration drift detection using Variational Autoencoders (VAEs).
    Detects movements toward non-compliant states.
    """
    
    def __init__(self, latent_dim: int = 32, input_dim: int = 128):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.vae = self._build_vae()
        self.baseline_distributions = {}
        self.drift_threshold = 2.0  # Standard deviations
        
    def _build_vae(self):
        """Build Variational Autoencoder for drift detection."""
        
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(VAE, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                
                self.fc_mu = nn.Linear(128, latent_dim)
                self.fc_var = nn.Linear(128, latent_dim)
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, input_dim),
                    nn.Sigmoid()
                )
                
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)
            
            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                return self.decode(z), mu, log_var
        
        return VAE(self.input_dim, self.latent_dim)
    
    async def learn_baseline(self, resource_configs: List[Dict[str, Any]]) -> None:
        """Learn baseline configuration distributions."""
        logger.info("learning_baseline_configurations", count=len(resource_configs))
        
        # Convert configurations to feature vectors
        features = self._extract_features(resource_configs)
        
        # Train VAE on baseline configurations
        self.vae.train()
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        
        dataset = torch.FloatTensor(features)
        for epoch in range(100):
            optimizer.zero_grad()
            recon, mu, log_var = self.vae(dataset)
            
            # VAE loss
            recon_loss = nn.functional.mse_loss(recon, dataset)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.001 * kl_loss
            
            loss.backward()
            optimizer.step()
        
        # Store baseline latent space distributions
        self.vae.eval()
        with torch.no_grad():
            mu, log_var = self.vae.encode(dataset)
            self.baseline_distributions = {
                'mean': mu.mean(dim=0).numpy(),
                'std': mu.std(dim=0).numpy()
            }
    
    async def detect_drift(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect configuration drift from baseline."""
        features = self._extract_features([current_config])[0]
        
        self.vae.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            mu, _ = self.vae.encode(input_tensor)
            current_latent = mu.numpy()[0]
        
        # Calculate drift score using Mahalanobis distance
        drift_score = self._calculate_mahalanobis_distance(
            current_latent,
            self.baseline_distributions['mean'],
            self.baseline_distributions['std']
        )
        
        # Calculate drift velocity if we have history
        drift_velocity = 0.0  # Placeholder for actual velocity calculation
        
        # Determine drift classification
        if drift_score > self.drift_threshold * 2:
            drift_class = "significant"
        elif drift_score > self.drift_threshold:
            drift_class = "moderate"
        else:
            drift_class = "none"
        
        return {
            'drift_score': float(drift_score),
            'drift_class': drift_class,
            'drift_velocity': drift_velocity,
            'baseline_distance': float(np.linalg.norm(
                current_latent - self.baseline_distributions['mean']
            ))
        }
    
    def _extract_features(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from configurations."""
        # Simplified feature extraction - in production, this would be more sophisticated
        features = []
        for config in configs:
            feature_vector = []
            
            # Extract numerical values
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(value)
                elif isinstance(value, bool):
                    feature_vector.append(1.0 if value else 0.0)
                elif isinstance(value, str) and value.isdigit():
                    feature_vector.append(float(value))
            
            # Pad or truncate to fixed size
            if len(feature_vector) < self.input_dim:
                feature_vector.extend([0.0] * (self.input_dim - len(feature_vector)))
            else:
                feature_vector = feature_vector[:self.input_dim]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_mahalanobis_distance(self, x: np.ndarray, mean: np.ndarray, 
                                       std: np.ndarray) -> float:
        """Calculate Mahalanobis distance for drift detection."""
        # Avoid division by zero
        std_safe = np.where(std > 0, std, 1e-6)
        normalized = (x - mean) / std_safe
        return np.sqrt(np.sum(normalized ** 2))


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in compliance data using STL decomposition,
    motif discovery, and regime change detection.
    """
    
    def __init__(self):
        self.seasonal_period = 24 * 7  # Weekly seasonality (hourly data)
        self.motif_length = 24  # Daily patterns
        
    async def decompose_time_series(self, compliance_data: pd.Series) -> Dict[str, Any]:
        """Decompose compliance time series into components."""
        try:
            # Perform STL decomposition
            stl = STL(compliance_data, seasonal=self.seasonal_period)
            result = stl.fit()
            
            return {
                'trend': result.trend.tolist(),
                'seasonal': result.seasonal.tolist(),
                'residual': result.resid.tolist(),
                'strength': {
                    'trend': self._calculate_trend_strength(result),
                    'seasonal': self._calculate_seasonal_strength(result)
                }
            }
        except Exception as e:
            logger.error("stl_decomposition_failed", error=str(e))
            return {
                'trend': compliance_data.tolist(),
                'seasonal': [],
                'residual': [],
                'strength': {'trend': 0.0, 'seasonal': 0.0}
            }
    
    async def discover_motifs(self, time_series: np.ndarray) -> List[Dict[str, Any]]:
        """Discover recurring patterns using matrix profile."""
        motifs = []
        
        # Simplified motif discovery - in production, use matrix profile
        window_size = self.motif_length
        for i in range(len(time_series) - window_size * 2):
            pattern = time_series[i:i + window_size]
            
            # Find similar patterns
            for j in range(i + window_size, len(time_series) - window_size):
                candidate = time_series[j:j + window_size]
                distance = np.linalg.norm(pattern - candidate)
                
                if distance < 0.1:  # Threshold for similarity
                    motifs.append({
                        'start_idx': i,
                        'pattern': pattern.tolist(),
                        'occurrences': [i, j],
                        'distance': float(distance)
                    })
        
        return motifs[:10]  # Return top 10 motifs
    
    async def detect_regime_changes(self, time_series: np.ndarray) -> List[Dict[str, Any]]:
        """Detect regime changes using change point detection."""
        changes = []
        
        # Simple change point detection using rolling statistics
        window = 24
        for i in range(window, len(time_series) - window):
            before = time_series[i - window:i]
            after = time_series[i:i + window]
            
            # Test for mean shift
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # Significant change
                changes.append({
                    'index': i,
                    'timestamp': i,  # Would be actual timestamp in production
                    'before_mean': float(np.mean(before)),
                    'after_mean': float(np.mean(after)),
                    'p_value': float(p_value),
                    'change_magnitude': float(abs(np.mean(after) - np.mean(before)))
                })
        
        return changes
    
    def _calculate_trend_strength(self, stl_result) -> float:
        """Calculate trend strength metric."""
        var_residual = np.var(stl_result.resid)
        var_detrended = np.var(stl_result.seasonal + stl_result.resid)
        return max(0, 1 - var_residual / var_detrended) if var_detrended > 0 else 0
    
    def _calculate_seasonal_strength(self, stl_result) -> float:
        """Calculate seasonal strength metric."""
        var_residual = np.var(stl_result.resid)
        var_deseasonalized = np.var(stl_result.trend + stl_result.resid)
        return max(0, 1 - var_residual / var_deseasonalized) if var_deseasonalized > 0 else 0


class CompliancePredictor:
    """
    Main predictive compliance engine combining drift detection, temporal analysis,
    and ensemble prediction models.
    """
    
    def __init__(self):
        self.drift_detector = ComplianceDriftDetector()
        self.pattern_analyzer = TemporalPatternAnalyzer()
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the compliance predictor."""
        logger.info("initializing_compliance_predictor")
        
        # Initialize ensemble models
        self.ensemble_models = {
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5
            ),
            'isolation_forest': IsolationForest(
                contamination=0.1,
                random_state=42
            )
        }
        
        self.is_initialized = True
        logger.info("compliance_predictor_initialized")
    
    async def train(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the compliance prediction models."""
        logger.info("training_compliance_models")
        
        try:
            # Extract features and labels
            features, labels = self._prepare_training_data(historical_data)
            
            # Train drift detector baseline
            await self.drift_detector.learn_baseline(
                historical_data.get('baseline_configs', [])
            )
            
            # Fit scaler
            self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            
            # Train ensemble models
            for name, model in self.ensemble_models.items():
                if name != 'isolation_forest':  # Isolation forest is unsupervised
                    model.fit(scaled_features, labels)
                else:
                    model.fit(scaled_features)
            
            # Calculate training metrics
            train_scores = {}
            for name, model in self.ensemble_models.items():
                if name != 'isolation_forest':
                    predictions = model.predict(scaled_features)
                    accuracy = np.mean(predictions == labels)
                    train_scores[name] = float(accuracy)
            
            return {
                'status': 'success',
                'models_trained': list(self.ensemble_models.keys()),
                'training_scores': train_scores,
                'samples_used': len(features)
            }
            
        except Exception as e:
            logger.error("training_failed", error=str(e))
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def predict_compliance(self, resource_data: Dict[str, Any], 
                                horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict compliance violations with specified time horizon."""
        logger.info("predicting_compliance", horizon_hours=horizon_hours)
        
        try:
            # Detect configuration drift
            drift_results = await self.drift_detector.detect_drift(
                resource_data.get('current_config', {})
            )
            
            # Analyze temporal patterns
            if 'time_series' in resource_data:
                ts_data = pd.Series(resource_data['time_series'])
                temporal_analysis = await self.pattern_analyzer.decompose_time_series(ts_data)
                motifs = await self.pattern_analyzer.discover_motifs(ts_data.values)
                regime_changes = await self.pattern_analyzer.detect_regime_changes(ts_data.values)
            else:
                temporal_analysis = {}
                motifs = []
                regime_changes = []
            
            # Prepare features for prediction
            features = self._extract_prediction_features(resource_data, drift_results)
            scaled_features = self.scaler.transform([features])
            
            # Get predictions from ensemble
            predictions = {}
            confidence_scores = []
            
            for name, model in self.ensemble_models.items():
                if name == 'isolation_forest':
                    # Anomaly detection
                    anomaly_score = model.decision_function(scaled_features)[0]
                    predictions[name] = float(anomaly_score < 0)  # Negative = anomaly
                    confidence_scores.append(abs(anomaly_score))
                else:
                    # Classification
                    pred_proba = model.predict_proba(scaled_features)[0]
                    predictions[name] = float(pred_proba[1])  # Probability of violation
                    confidence_scores.append(pred_proba[1])
            
            # Combine predictions using weighted average
            weights = {'xgboost': 0.45, 'random_forest': 0.35, 'isolation_forest': 0.20}
            final_prediction = sum(
                predictions.get(name, 0) * weights.get(name, 0)
                for name in weights
            )
            
            # Calculate confidence interval
            confidence_mean = np.mean(confidence_scores)
            confidence_std = np.std(confidence_scores)
            
            # Risk assessment
            risk_score = self._calculate_risk_score(
                final_prediction,
                drift_results,
                regime_changes
            )
            
            return {
                'prediction': {
                    'violation_probability': float(final_prediction),
                    'confidence': float(confidence_mean),
                    'confidence_interval': [
                        float(max(0, confidence_mean - 1.96 * confidence_std)),
                        float(min(1, confidence_mean + 1.96 * confidence_std))
                    ],
                    'horizon_hours': horizon_hours
                },
                'drift_analysis': drift_results,
                'temporal_patterns': {
                    'decomposition': temporal_analysis,
                    'motifs_found': len(motifs),
                    'regime_changes': len(regime_changes)
                },
                'risk_assessment': {
                    'risk_score': float(risk_score),
                    'risk_level': self._classify_risk(risk_score),
                    'contributing_factors': self._identify_risk_factors(
                        drift_results, regime_changes
                    )
                },
                'ensemble_predictions': predictions
            }
            
        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            return {
                'prediction': {
                    'violation_probability': 0.5,
                    'confidence': 0.0,
                    'error': str(e)
                }
            }
    
    def _prepare_training_data(self, historical_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        # Simplified feature extraction - in production, this would be more sophisticated
        samples = historical_data.get('samples', [])
        features = []
        labels = []
        
        for sample in samples:
            feature_vector = [
                sample.get('resource_count', 0),
                sample.get('policy_count', 0),
                sample.get('violation_count', 0),
                sample.get('days_since_last_violation', 999),
                sample.get('config_change_frequency', 0),
                sample.get('resource_age_days', 0),
                sample.get('cost_trend', 0),
                sample.get('access_frequency', 0)
            ]
            features.append(feature_vector)
            labels.append(1 if sample.get('had_violation', False) else 0)
        
        return np.array(features), np.array(labels)
    
    def _extract_prediction_features(self, resource_data: Dict[str, Any],
                                   drift_results: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction."""
        return np.array([
            resource_data.get('resource_count', 0),
            resource_data.get('policy_count', 0),
            resource_data.get('recent_violations', 0),
            resource_data.get('days_since_last_violation', 999),
            drift_results.get('drift_score', 0),
            resource_data.get('config_changes_24h', 0),
            resource_data.get('cost_increase_pct', 0),
            resource_data.get('access_anomaly_score', 0)
        ])
    
    def _calculate_risk_score(self, violation_prob: float, drift_results: Dict[str, Any],
                            regime_changes: List[Dict[str, Any]]) -> float:
        """Calculate comprehensive risk score."""
        # Base risk from violation probability
        base_risk = violation_prob * 0.5
        
        # Add drift component
        drift_score = drift_results.get('drift_score', 0)
        drift_component = min(drift_score / 10, 0.3)  # Cap at 0.3
        
        # Add regime change component
        regime_component = min(len(regime_changes) * 0.05, 0.2)  # Cap at 0.2
        
        return base_risk + drift_component + regime_component
    
    def _classify_risk(self, risk_score: float) -> str:
        """Classify risk level based on score."""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def _identify_risk_factors(self, drift_results: Dict[str, Any],
                             regime_changes: List[Dict[str, Any]]) -> List[str]:
        """Identify contributing risk factors."""
        factors = []
        
        if drift_results.get('drift_class') in ['moderate', 'significant']:
            factors.append(f"Configuration drift detected ({drift_results['drift_class']})")
        
        if len(regime_changes) > 0:
            factors.append(f"Detected {len(regime_changes)} regime changes")
        
        if drift_results.get('drift_velocity', 0) > 0.1:
            factors.append("Rapid configuration changes detected")
        
        return factors