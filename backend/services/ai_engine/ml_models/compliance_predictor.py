"""
Predictive Policy Compliance Engine for PolicyCortex.
Implements Patent 1: Machine Learning System for Temporal Predictive Cloud Policy Compliance Analysis.

Advanced ensemble learning approach combining:
- XGBoost for gradient boosting
- LSTM with attention mechanism for temporal patterns
- Prophet for seasonal pattern recognition
- Fuzzy logic for risk assessment
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import structlog
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from statsmodels.tsa.seasonal import STL
from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

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


class ComplianceLSTM(nn.Module):
    """
    LSTM with attention mechanism for sequential pattern recognition in compliance data.
    """
    
    def __init__(self, sequence_length=30, feature_dim=50, lstm_units=128, num_classes=5):
        super(ComplianceLSTM, self).__init__()
        
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        
        # Multi-layer LSTM with attention
        self.lstm1 = nn.LSTM(feature_dim, lstm_units, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_units, num_heads=8, dropout=0.1)
        
        # Dense layers for prediction
        self.dense1 = nn.Linear(lstm_units, 64)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # Sequential processing through LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        
        # Transpose for attention (seq_len, batch, features)
        lstm_out3_t = lstm_out3.transpose(0, 1)
        
        # Apply attention mechanism
        attention_output, attention_weights = self.attention(
            lstm_out3_t, lstm_out3_t, lstm_out3_t
        )
        
        # Transpose back and take mean over sequence
        attention_output = attention_output.transpose(0, 1)
        pooled = torch.mean(attention_output, dim=1)
        
        # Dense layers for final prediction
        x = F.relu(self.dense1(pooled))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        output = F.softmax(self.output_layer(x), dim=1)
        
        return output, attention_weights


class GovernanceProphet:
    """
    Prophet model implementation for governance seasonality and trend analysis.
    """
    
    def __init__(self):
        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available, seasonal analysis will be limited")
            self.model = None
            return
            
        self.model = Prophet(
            growth='logistic',
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays_prior_scale=10.0,
            seasonality_prior_scale=10.0,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities for business patterns
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )
        
        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=3
        )
        
        # Add external regressors
        self.model.add_regressor('policy_changes')
        self.model.add_regressor('resource_changes')
        self.model.add_regressor('maintenance_windows')
        
    def prepare_data(self, violation_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for Prophet model."""
        if not PROPHET_AVAILABLE:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'ds': pd.to_datetime(violation_data['timestamps']),
            'y': violation_data['violation_counts'],
            'cap': violation_data.get('max_possible_violations', 
                                    np.max(violation_data['violation_counts']) * 1.2)
        })
        
        # Add regressors for external factors
        df['policy_changes'] = violation_data.get('policy_change_counts', 0)
        df['resource_changes'] = violation_data.get('resource_change_counts', 0)
        df['maintenance_windows'] = violation_data.get('maintenance_indicators', 0)
        
        return df
        
    def fit(self, violation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit Prophet model to historical data."""
        if not PROPHET_AVAILABLE:
            return {'status': 'failed', 'error': 'Prophet not available'}
            
        try:
            df = self.prepare_data(violation_data)
            self.model.fit(df)
            
            return {
                'status': 'success',
                'samples_used': len(df),
                'changepoints_detected': len(self.model.changepoints)
            }
        except Exception as e:
            logger.error("prophet_fitting_failed", error=str(e))
            return {'status': 'failed', 'error': str(e)}
    
    def predict(self, future_periods: int = 24) -> Dict[str, Any]:
        """Generate predictions for future periods."""
        if not PROPHET_AVAILABLE or self.model is None:
            return {'predictions': [], 'trend': [], 'seasonal': []}
            
        try:
            future = self.model.make_future_dataframe(periods=future_periods, freq='h')
            
            # Add future regressor values (simplified - would use forecasts in production)
            future['policy_changes'].fillna(0, inplace=True)
            future['resource_changes'].fillna(0, inplace=True)
            future['maintenance_windows'].fillna(0, inplace=True)
            
            forecast = self.model.predict(future)
            
            return {
                'predictions': forecast['yhat'].tolist()[-future_periods:],
                'trend': forecast['trend'].tolist()[-future_periods:],
                'seasonal': forecast['seasonal'].tolist()[-future_periods:],
                'uncertainty_lower': forecast['yhat_lower'].tolist()[-future_periods:],
                'uncertainty_upper': forecast['yhat_upper'].tolist()[-future_periods:]
            }
        except Exception as e:
            logger.error("prophet_prediction_failed", error=str(e))
            return {'predictions': [], 'trend': [], 'seasonal': []}


class FuzzyRiskAssessment:
    """
    Fuzzy logic implementation for risk assessment as specified in the documentation.
    """
    
    def __init__(self):
        try:
            self.setup_fuzzy_system()
            self.system_ready = True
        except Exception as e:
            logger.error("fuzzy_system_setup_failed", error=str(e))
            self.system_ready = False
    
    def setup_fuzzy_system(self):
        """Setup fuzzy logic system for risk assessment."""
        
        # Define input variables
        self.violation_probability = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'violation_probability')
        self.business_impact = ctrl.Antecedent(np.arange(0, 11, 1), 'business_impact')
        self.remediation_complexity = ctrl.Antecedent(np.arange(0, 11, 1), 'remediation_complexity')
        
        # Define output variable
        self.risk_score = ctrl.Consequent(np.arange(0, 11, 1), 'risk_score')
        
        # Define membership functions
        self.violation_probability['low'] = fuzz.trimf(self.violation_probability.universe, [0, 0, 0.3])
        self.violation_probability['medium'] = fuzz.trimf(self.violation_probability.universe, [0.2, 0.5, 0.8])
        self.violation_probability['high'] = fuzz.trimf(self.violation_probability.universe, [0.7, 1.0, 1.0])
        
        self.business_impact['low'] = fuzz.trimf(self.business_impact.universe, [0, 0, 3])
        self.business_impact['medium'] = fuzz.trimf(self.business_impact.universe, [2, 5, 8])
        self.business_impact['high'] = fuzz.trimf(self.business_impact.universe, [7, 10, 10])
        
        self.remediation_complexity['low'] = fuzz.trimf(self.remediation_complexity.universe, [0, 0, 3])
        self.remediation_complexity['medium'] = fuzz.trimf(self.remediation_complexity.universe, [2, 5, 8])
        self.remediation_complexity['high'] = fuzz.trimf(self.remediation_complexity.universe, [7, 10, 10])
        
        self.risk_score['very_low'] = fuzz.trimf(self.risk_score.universe, [0, 0, 2])
        self.risk_score['low'] = fuzz.trimf(self.risk_score.universe, [1, 3, 5])
        self.risk_score['medium'] = fuzz.trimf(self.risk_score.universe, [4, 5, 6])
        self.risk_score['high'] = fuzz.trimf(self.risk_score.universe, [5, 7, 9])
        self.risk_score['very_high'] = fuzz.trimf(self.risk_score.universe, [8, 10, 10])
        
        # Define fuzzy rules
        self.rules = [
            ctrl.Rule(self.violation_probability['low'] & self.business_impact['low'], self.risk_score['very_low']),
            ctrl.Rule(self.violation_probability['low'] & self.business_impact['medium'], self.risk_score['low']),
            ctrl.Rule(self.violation_probability['low'] & self.business_impact['high'], self.risk_score['medium']),
            ctrl.Rule(self.violation_probability['medium'] & self.business_impact['low'], self.risk_score['low']),
            ctrl.Rule(self.violation_probability['medium'] & self.business_impact['medium'], self.risk_score['medium']),
            ctrl.Rule(self.violation_probability['medium'] & self.business_impact['high'], self.risk_score['high']),
            ctrl.Rule(self.violation_probability['high'] & self.business_impact['low'], self.risk_score['medium']),
            ctrl.Rule(self.violation_probability['high'] & self.business_impact['medium'], self.risk_score['high']),
            ctrl.Rule(self.violation_probability['high'] & self.business_impact['high'], self.risk_score['very_high']),
        ]
        
        # Create control system
        self.risk_ctrl = ctrl.ControlSystem(self.rules)
        self.risk_simulation = ctrl.ControlSystemSimulation(self.risk_ctrl)
    
    def assess_risk(self, violation_prob: float, impact_score: float, complexity_score: float) -> Dict[str, Any]:
        """Assess risk using fuzzy logic."""
        if not self.system_ready:
            return {
                'risk_score': violation_prob * impact_score / 10,
                'risk_level': 'MEDIUM',
                'confidence': 0.5,
                'error': 'Fuzzy system not available'
            }
        
        try:
            # Set input values
            self.risk_simulation.input['violation_probability'] = violation_prob
            self.risk_simulation.input['business_impact'] = impact_score
            self.risk_simulation.input['remediation_complexity'] = complexity_score
            
            # Compute risk score
            self.risk_simulation.compute()
            
            risk_score = self.risk_simulation.output['risk_score']
            
            return {
                'risk_score': float(risk_score),
                'risk_level': self.get_risk_level(risk_score),
                'confidence': self.calculate_confidence(violation_prob, impact_score, complexity_score)
            }
        except Exception as e:
            logger.error("fuzzy_risk_assessment_failed", error=str(e))
            return {
                'risk_score': violation_prob * impact_score / 10,
                'risk_level': 'MEDIUM',
                'confidence': 0.5,
                'error': str(e)
            }
    
    def get_risk_level(self, score: float) -> str:
        """Convert numeric risk score to categorical level."""
        if score <= 2:
            return 'VERY_LOW'
        elif score <= 4:
            return 'LOW'
        elif score <= 6:
            return 'MEDIUM'
        elif score <= 8:
            return 'HIGH'
        else:
            return 'VERY_HIGH'
    
    def calculate_confidence(self, violation_prob: float, impact_score: float, 
                           complexity_score: float) -> float:
        """Calculate confidence in fuzzy assessment."""
        # Confidence based on input certainty
        prob_certainty = 1.0 - abs(violation_prob - 0.5) * 2  # Higher certainty at extremes
        impact_certainty = min(impact_score / 10, (10 - impact_score) / 10) * 2
        complexity_certainty = min(complexity_score / 10, (10 - complexity_score) / 10) * 2
        
        return float(np.mean([prob_certainty, impact_certainty, complexity_certainty]))


class ComplianceEnsemble:
    """
    Ensemble voting classifier that combines XGBoost, LSTM, and Prophet predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.weights = None
        self.performance_history = {
            'xgboost': [],
            'lstm': [],
            'prophet': []
        }
        
    def create_ensemble(self, xgb_model, lstm_model, prophet_model):
        """Create weighted ensemble of all models."""
        self.models = {
            'xgboost': xgb_model,
            'lstm': lstm_model,
            'prophet': prophet_model
        }
        
        # Dynamic weight calculation based on recent performance
        self.weights = self.calculate_dynamic_weights()
        
        logger.info("ensemble_created", weights=self.weights)
        
    def calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate dynamic weights based on recent model performance."""
        # Default weights if no performance history
        default_weights = {'xgboost': 0.45, 'lstm': 0.35, 'prophet': 0.20}
        
        if not any(self.performance_history.values()):
            return default_weights
        
        # Calculate recent performance (last 30 evaluations)
        recent_performance = {}
        for model_name, scores in self.performance_history.items():
            if scores:
                recent_scores = scores[-30:]  # Last 30 evaluations
                recent_performance[model_name] = np.mean(recent_scores)
            else:
                recent_performance[model_name] = 0.5  # Default performance
        
        # Normalize weights
        total_performance = sum(recent_performance.values())
        if total_performance > 0:
            weights = {name: perf / total_performance for name, perf in recent_performance.items()}
        else:
            weights = default_weights
        
        return weights
    
    def predict_with_confidence(self, features: np.ndarray, 
                               sequence_data: Optional[np.ndarray] = None,
                               prophet_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make predictions with confidence intervals."""
        predictions = {}
        probabilities = {}
        
        try:
            # XGBoost prediction
            if 'xgboost' in self.models and self.models['xgboost'] is not None:
                xgb_proba = self.models['xgboost'].predict_proba(features)
                predictions['xgboost'] = self.models['xgboost'].predict(features)
                probabilities['xgboost'] = xgb_proba[:, 1] if xgb_proba.shape[1] > 1 else xgb_proba[:, 0]
            
            # LSTM prediction
            if 'lstm' in self.models and self.models['lstm'] is not None and sequence_data is not None:
                self.models['lstm'].eval()
                with torch.no_grad():
                    lstm_input = torch.FloatTensor(sequence_data)
                    lstm_output, _ = self.models['lstm'](lstm_input)
                    lstm_proba = lstm_output.numpy()
                    predictions['lstm'] = (lstm_proba[:, 1] > 0.5).astype(int) if lstm_proba.shape[1] > 1 else (lstm_proba > 0.5).astype(int)
                    probabilities['lstm'] = lstm_proba[:, 1] if lstm_proba.shape[1] > 1 else lstm_proba.flatten()
            
            # Prophet prediction
            if 'prophet' in self.models and self.models['prophet'] is not None and prophet_data is not None:
                prophet_result = self.models['prophet'].predict(future_periods=1)
                if prophet_result['predictions']:
                    prophet_value = prophet_result['predictions'][0]
                    # Convert Prophet output to probability
                    prophet_prob = np.clip(prophet_value / 10, 0, 1)  # Normalize to [0,1]
                    predictions['prophet'] = [1 if prophet_prob > 0.5 else 0]
                    probabilities['prophet'] = [prophet_prob]
            
            # Combine predictions using weighted average
            if probabilities:
                weighted_proba = np.zeros(len(list(probabilities.values())[0]))
                total_weight = 0
                
                for model_name, proba in probabilities.items():
                    weight = self.weights.get(model_name, 0)
                    weighted_proba += np.array(proba) * weight
                    total_weight += weight
                
                if total_weight > 0:
                    weighted_proba /= total_weight
                
                # Calculate confidence
                confidence_scores = self._calculate_confidence(probabilities)
                
                return {
                    'predictions': (weighted_proba > 0.5).astype(int),
                    'probabilities': weighted_proba,
                    'confidence_scores': confidence_scores,
                    'individual_predictions': predictions,
                    'individual_probabilities': probabilities
                }
            else:
                return {
                    'predictions': np.array([0]),
                    'probabilities': np.array([0.5]),
                    'confidence_scores': np.array([0.0]),
                    'error': 'No valid predictions from any model'
                }
                
        except Exception as e:
            logger.error("ensemble_prediction_failed", error=str(e))
            return {
                'predictions': np.array([0]),
                'probabilities': np.array([0.5]),
                'confidence_scores': np.array([0.0]),
                'error': str(e)
            }
    
    def _calculate_confidence(self, probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate prediction confidence based on model agreement."""
        if not probabilities:
            return np.array([0.0])
        
        # Convert to numpy arrays
        prob_arrays = [np.array(proba) for proba in probabilities.values()]
        
        # Calculate standard deviation as confidence measure (lower std = higher confidence)
        prob_std = np.std(prob_arrays, axis=0)
        confidence = 1.0 - np.clip(prob_std * 2, 0, 1)  # Invert and scale
        
        return confidence
    
    def update_performance(self, model_name: str, accuracy: float):
        """Update performance history for dynamic weighting."""
        if model_name in self.performance_history:
            self.performance_history[model_name].append(accuracy)
            # Keep only last 100 evaluations
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name] = self.performance_history[model_name][-100:]


class CompliancePredictor:
    """
    Main predictive compliance engine combining drift detection, temporal analysis,
    and ensemble prediction models implementing Patent 1 specifications.
    """
    
    def __init__(self):
        self.drift_detector = ComplianceDriftDetector()
        self.pattern_analyzer = TemporalPatternAnalyzer()
        self.fuzzy_risk_assessment = FuzzyRiskAssessment()
        
        # Advanced ensemble components
        self.ensemble = ComplianceEnsemble()
        self.xgb_model = None
        self.lstm_model = None
        self.prophet_model = GovernanceProphet()
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.sequence_scaler = MinMaxScaler()
        
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the compliance predictor with advanced ensemble models."""
        logger.info("initializing_advanced_compliance_predictor")
        
        try:
            # Initialize XGBoost with optimized parameters
            self.xgb_model = XGBClassifier(
                objective='multi:softprob',
                num_class=5,  # 5 violation severity levels
                max_depth=8,
                learning_rate=0.1,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                early_stopping_rounds=50
            )
            
            # Initialize LSTM model
            self.lstm_model = ComplianceLSTM(
                sequence_length=30,
                feature_dim=50,
                lstm_units=128,
                num_classes=5
            )
            
            # Prophet model is already initialized in __init__
            
            # Create ensemble
            self.ensemble.create_ensemble(
                self.xgb_model,
                self.lstm_model,
                self.prophet_model
            )
            
            self.is_initialized = True
            logger.info("advanced_compliance_predictor_initialized", 
                       models=['xgboost', 'lstm', 'prophet', 'fuzzy_logic'])
            
        except Exception as e:
            logger.error("compliance_predictor_initialization_failed", error=str(e))
            raise
    
    async def train(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the advanced ensemble compliance prediction models."""
        logger.info("training_advanced_compliance_models")
        
        try:
            # Extract features and labels for tabular models
            features, labels = self._prepare_training_data(historical_data)
            sequence_data, sequence_labels = self._prepare_sequence_data(historical_data)
            
            # Train drift detector baseline
            await self.drift_detector.learn_baseline(
                historical_data.get('baseline_configs', [])
            )
            
            # Fit scalers
            self.feature_scaler.fit(features)
            scaled_features = self.feature_scaler.transform(features)
            
            if len(sequence_data) > 0:
                sequence_reshaped = sequence_data.reshape(-1, sequence_data.shape[-1])
                self.sequence_scaler.fit(sequence_reshaped)
                scaled_sequences = self.sequence_scaler.transform(sequence_reshaped)
                scaled_sequences = scaled_sequences.reshape(sequence_data.shape)
            else:
                scaled_sequences = sequence_data
            
            training_results = {}
            
            # Train XGBoost
            if self.xgb_model is not None:
                self.xgb_model.fit(scaled_features, labels)
                xgb_predictions = self.xgb_model.predict(scaled_features)
                xgb_accuracy = accuracy_score(labels, xgb_predictions)
                training_results['xgboost'] = {
                    'accuracy': float(xgb_accuracy),
                    'precision': float(precision_score(labels, xgb_predictions, average='weighted', zero_division=0)),
                    'recall': float(recall_score(labels, xgb_predictions, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(labels, xgb_predictions, average='weighted', zero_division=0))
                }
                self.ensemble.update_performance('xgboost', xgb_accuracy)
            
            # Train LSTM
            if self.lstm_model is not None and len(scaled_sequences) > 0:
                lstm_results = await self._train_lstm_model(scaled_sequences, sequence_labels)
                training_results['lstm'] = lstm_results
            
            # Train Prophet
            if self.prophet_model is not None and 'time_series_data' in historical_data:
                prophet_results = self.prophet_model.fit(historical_data['time_series_data'])
                training_results['prophet'] = prophet_results
            
            # Update ensemble weights based on training performance
            self.ensemble.weights = self.ensemble.calculate_dynamic_weights()
            
            return {
                'status': 'success',
                'models_trained': list(training_results.keys()),
                'training_results': training_results,
                'ensemble_weights': self.ensemble.weights,
                'samples_used': {
                    'tabular': len(features),
                    'sequence': len(scaled_sequences)
                }
            }
            
        except Exception as e:
            logger.error("advanced_training_failed", error=str(e))
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _train_lstm_model(self, sequence_data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Train LSTM model with PyTorch."""
        try:
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(sequence_data)
            y_tensor = torch.LongTensor(labels)
            
            # Training parameters
            num_epochs = 100
            batch_size = 32
            learning_rate = 0.001
            
            # Loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=learning_rate)
            
            # Training loop
            self.lstm_model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i + batch_size]
                    batch_y = y_tensor[i:i + batch_size]
                    
                    optimizer.zero_grad()
                    outputs, _ = self.lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if epoch % 20 == 0:
                    logger.info(f"LSTM training epoch {epoch}, loss: {total_loss:.4f}")
            
            # Evaluate on training data
            self.lstm_model.eval()
            with torch.no_grad():
                train_outputs, _ = self.lstm_model(X_tensor)
                _, predicted = torch.max(train_outputs.data, 1)
                accuracy = (predicted == y_tensor).float().mean().item()
            
            self.ensemble.update_performance('lstm', accuracy)
            
            return {
                'accuracy': float(accuracy),
                'epochs_trained': num_epochs,
                'final_loss': float(total_loss)
            }
            
        except Exception as e:
            logger.error("lstm_training_failed", error=str(e))
            return {
                'accuracy': 0.0,
                'error': str(e)
            }
    
    def _prepare_sequence_data(self, historical_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training."""
        try:
            time_series_samples = historical_data.get('time_series_samples', [])
            if not time_series_samples:
                return np.array([]), np.array([])
            
            sequences = []
            labels = []
            
            for sample in time_series_samples:
                sequence = sample.get('sequence', [])
                label = sample.get('label', 0)
                
                if len(sequence) >= 30:  # Ensure minimum sequence length
                    # Pad or truncate to fixed length
                    if len(sequence) > 30:
                        sequence = sequence[-30:]  # Take last 30 points
                    else:
                        sequence = sequence + [0] * (30 - len(sequence))  # Pad with zeros
                    
                    # Convert to feature matrix (30 timesteps x 50 features)
                    feature_matrix = []
                    for i, value in enumerate(sequence):
                        # Create feature vector for each timestep
                        feature_vector = [
                            value,  # Main value
                            i / 30,  # Position in sequence
                            np.sin(2 * np.pi * i / 7),  # Weekly pattern
                            np.cos(2 * np.pi * i / 7),  # Weekly pattern
                            np.sin(2 * np.pi * i / 24),  # Daily pattern
                            np.cos(2 * np.pi * i / 24),  # Daily pattern
                        ]
                        # Pad to 50 features
                        feature_vector.extend([0.0] * (50 - len(feature_vector)))
                        feature_matrix.append(feature_vector)
                    
                    sequences.append(feature_matrix)
                    labels.append(label)
            
            return np.array(sequences), np.array(labels)
            
        except Exception as e:
            logger.error("sequence_data_preparation_failed", error=str(e))
            return np.array([]), np.array([])
    
    async def predict_compliance(self, resource_data: Dict[str, Any], 
                                horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict compliance violations using advanced ensemble approach."""
        logger.info("predicting_compliance_advanced", horizon_hours=horizon_hours)
        
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
            
            # Prepare features for different models
            tabular_features = self._extract_prediction_features(resource_data, drift_results)
            scaled_tabular_features = self.feature_scaler.transform([tabular_features])
            
            # Prepare sequence data for LSTM
            sequence_data = self._prepare_prediction_sequence(resource_data)
            scaled_sequence_data = None
            if len(sequence_data) > 0:
                sequence_reshaped = sequence_data.reshape(-1, sequence_data.shape[-1])
                scaled_sequence_reshaped = self.sequence_scaler.transform(sequence_reshaped)
                scaled_sequence_data = scaled_sequence_reshaped.reshape(sequence_data.shape)
            
            # Prepare Prophet data
            prophet_data = resource_data.get('prophet_data', None)
            
            # Get ensemble predictions
            ensemble_results = self.ensemble.predict_with_confidence(
                features=scaled_tabular_features,
                sequence_data=scaled_sequence_data,
                prophet_data=prophet_data
            )
            
            # Extract main prediction
            final_prediction = ensemble_results['probabilities'][0] if len(ensemble_results['probabilities']) > 0 else 0.5
            confidence_score = ensemble_results['confidence_scores'][0] if len(ensemble_results['confidence_scores']) > 0 else 0.0
            
            # Calculate business impact and remediation complexity for fuzzy logic
            business_impact = self._calculate_business_impact(resource_data)
            remediation_complexity = self._calculate_remediation_complexity(resource_data, drift_results)
            
            # Fuzzy logic risk assessment
            fuzzy_risk = self.fuzzy_risk_assessment.assess_risk(
                violation_prob=final_prediction,
                impact_score=business_impact,
                complexity_score=remediation_complexity
            )
            
            # Enhanced risk assessment combining multiple factors
            comprehensive_risk = self._calculate_comprehensive_risk(
                final_prediction,
                drift_results,
                regime_changes,
                fuzzy_risk,
                temporal_analysis
            )
            
            # Generate violation likelihood categories
            violation_categories = self._categorize_violations(final_prediction, drift_results)
            
            # Calculate confidence intervals
            confidence_interval = self._calculate_confidence_interval(
                ensemble_results, confidence_score
            )
            
            return {
                'prediction': {
                    'violation_probability': float(final_prediction),
                    'confidence': float(confidence_score),
                    'confidence_interval': confidence_interval,
                    'horizon_hours': horizon_hours,
                    'violation_categories': violation_categories,
                    'ensemble_agreement': self._calculate_ensemble_agreement(ensemble_results)
                },
                'drift_analysis': drift_results,
                'temporal_patterns': {
                    'decomposition': temporal_analysis,
                    'motifs_found': len(motifs),
                    'regime_changes': len(regime_changes),
                    'seasonal_strength': temporal_analysis.get('strength', {}).get('seasonal', 0),
                    'trend_strength': temporal_analysis.get('strength', {}).get('trend', 0)
                },
                'risk_assessment': {
                    'comprehensive_risk': comprehensive_risk,
                    'fuzzy_risk': fuzzy_risk,
                    'business_impact': float(business_impact),
                    'remediation_complexity': float(remediation_complexity),
                    'contributing_factors': self._identify_risk_factors(drift_results, regime_changes)
                },
                'ensemble_details': {
                    'individual_predictions': ensemble_results.get('individual_predictions', {}),
                    'individual_probabilities': ensemble_results.get('individual_probabilities', {}),
                    'model_weights': self.ensemble.weights,
                    'performance_history': self._get_recent_performance()
                }
            }
            
        except Exception as e:
            logger.error("advanced_prediction_failed", error=str(e))
            return {
                'prediction': {
                    'violation_probability': 0.5,
                    'confidence': 0.0,
                    'error': str(e),
                    'horizon_hours': horizon_hours
                },
                'error_details': str(e)
            }
    
    def _prepare_prediction_sequence(self, resource_data: Dict[str, Any]) -> np.ndarray:
        """Prepare sequence data for LSTM prediction."""
        try:
            time_series = resource_data.get('time_series', [])
            if len(time_series) < 30:
                # Pad with zeros if not enough data
                time_series = [0] * (30 - len(time_series)) + time_series
            elif len(time_series) > 30:
                # Take last 30 points
                time_series = time_series[-30:]
            
            # Convert to feature matrix (1 x 30 x 50)
            feature_matrix = []
            for i, value in enumerate(time_series):
                feature_vector = [
                    value,  # Main value
                    i / 30,  # Position in sequence
                    np.sin(2 * np.pi * i / 7),  # Weekly pattern
                    np.cos(2 * np.pi * i / 7),  # Weekly pattern
                    np.sin(2 * np.pi * i / 24),  # Daily pattern
                    np.cos(2 * np.pi * i / 24),  # Daily pattern
                ]
                # Pad to 50 features
                feature_vector.extend([0.0] * (50 - len(feature_vector)))
                feature_matrix.append(feature_vector)
            
            return np.array([feature_matrix])  # Add batch dimension
            
        except Exception as e:
            logger.error("sequence_preparation_failed", error=str(e))
            return np.array([])
    
    def _calculate_business_impact(self, resource_data: Dict[str, Any]) -> float:
        """Calculate business impact score for fuzzy logic (0-10 scale)."""
        impact_factors = {
            'resource_criticality': resource_data.get('criticality_level', 5),
            'cost_impact': min(resource_data.get('monthly_cost', 0) / 1000, 10),
            'user_impact': resource_data.get('user_count', 0) / 100,
            'compliance_importance': resource_data.get('compliance_importance', 5)
        }
        
        # Weighted average
        weights = {'resource_criticality': 0.3, 'cost_impact': 0.25, 'user_impact': 0.25, 'compliance_importance': 0.2}
        business_impact = sum(impact_factors[factor] * weights[factor] for factor in weights)
        
        return min(max(business_impact, 0), 10)  # Clamp to 0-10 range
    
    def _calculate_remediation_complexity(self, resource_data: Dict[str, Any], 
                                        drift_results: Dict[str, Any]) -> float:
        """Calculate remediation complexity score for fuzzy logic (0-10 scale)."""
        complexity_factors = {
            'resource_dependencies': min(resource_data.get('dependency_count', 0), 10),
            'configuration_complexity': drift_results.get('drift_score', 0) * 2,
            'automation_level': 10 - resource_data.get('automation_level', 5),
            'team_expertise': 10 - resource_data.get('team_expertise_level', 5)
        }
        
        # Weighted average
        weights = {'resource_dependencies': 0.3, 'configuration_complexity': 0.3, 
                  'automation_level': 0.2, 'team_expertise': 0.2}
        complexity = sum(complexity_factors[factor] * weights[factor] for factor in weights)
        
        return min(max(complexity, 0), 10)  # Clamp to 0-10 range
    
    def _calculate_comprehensive_risk(self, violation_prob: float, drift_results: Dict[str, Any],
                                    regime_changes: List[Dict[str, Any]], fuzzy_risk: Dict[str, Any],
                                    temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment combining multiple factors."""
        
        # Base risk from violation probability
        base_risk = violation_prob * 0.4
        
        # Drift risk component
        drift_score = drift_results.get('drift_score', 0)
        drift_risk = min(drift_score / 5, 0.2)  # Cap at 0.2
        
        # Temporal instability risk
        regime_change_risk = min(len(regime_changes) * 0.02, 0.1)  # Cap at 0.1
        
        # Seasonal uncertainty risk
        seasonal_strength = temporal_analysis.get('strength', {}).get('seasonal', 0)
        seasonal_risk = (1 - seasonal_strength) * 0.1  # Higher risk if low seasonality
        
        # Fuzzy logic risk (normalized to 0-1)
        fuzzy_risk_norm = fuzzy_risk.get('risk_score', 5) / 10 * 0.2
        
        # Combine all risk components
        total_risk = base_risk + drift_risk + regime_change_risk + seasonal_risk + fuzzy_risk_norm
        total_risk = min(total_risk, 1.0)  # Cap at 1.0
        
        return {
            'total_risk_score': float(total_risk),
            'risk_level': self._classify_risk(total_risk),
            'risk_components': {
                'base_risk': float(base_risk),
                'drift_risk': float(drift_risk),
                'regime_change_risk': float(regime_change_risk),
                'seasonal_risk': float(seasonal_risk),
                'fuzzy_risk': float(fuzzy_risk_norm)
            },
            'risk_confidence': fuzzy_risk.get('confidence', 0.5)
        }
    
    def _categorize_violations(self, violation_prob: float, drift_results: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize potential violations by type and severity."""
        categories = {
            'security': {
                'probability': violation_prob * 0.8 if drift_results.get('drift_class') == 'significant' else violation_prob * 0.3,
                'severity': 'high' if violation_prob > 0.7 else 'medium' if violation_prob > 0.4 else 'low'
            },
            'compliance': {
                'probability': violation_prob * 0.9,  # Compliance is primary focus
                'severity': 'critical' if violation_prob > 0.8 else 'high' if violation_prob > 0.6 else 'medium'
            },
            'cost': {
                'probability': violation_prob * 0.6,
                'severity': 'medium' if violation_prob > 0.5 else 'low'
            },
            'performance': {
                'probability': violation_prob * 0.4,
                'severity': 'low'
            }
        }
        
        return categories
    
    def _calculate_confidence_interval(self, ensemble_results: Dict[str, Any], 
                                     confidence_score: float) -> List[float]:
        """Calculate confidence interval for predictions."""
        if 'probabilities' not in ensemble_results or len(ensemble_results['probabilities']) == 0:
            return [0.0, 1.0]
        
        prediction = ensemble_results['probabilities'][0]
        
        # Calculate standard error based on ensemble agreement
        std_error = (1 - confidence_score) * 0.1  # Lower confidence = higher std error
        
        # 95% confidence interval
        margin_of_error = 1.96 * std_error
        
        lower_bound = max(0.0, prediction - margin_of_error)
        upper_bound = min(1.0, prediction + margin_of_error)
        
        return [float(lower_bound), float(upper_bound)]
    
    def _calculate_ensemble_agreement(self, ensemble_results: Dict[str, Any]) -> float:
        """Calculate agreement between ensemble models."""
        if 'individual_probabilities' not in ensemble_results:
            return 0.0
        
        probabilities = list(ensemble_results['individual_probabilities'].values())
        if len(probabilities) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        prob_arrays = [np.array(p) for p in probabilities if len(p) > 0]
        if not prob_arrays:
            return 0.0
        
        mean_prob = np.mean(prob_arrays, axis=0)
        std_prob = np.std(prob_arrays, axis=0)
        
        # Agreement is inverse of coefficient of variation
        cv = std_prob / (mean_prob + 1e-8)  # Add small epsilon to avoid division by zero
        agreement = 1.0 / (1.0 + cv)
        
        return float(np.mean(agreement))
    
    def _get_recent_performance(self) -> Dict[str, List[float]]:
        """Get recent performance metrics for each model."""
        recent_performance = {}
        for model_name, history in self.ensemble.performance_history.items():
            recent_performance[model_name] = history[-10:] if len(history) >= 10 else history
        return recent_performance
    
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