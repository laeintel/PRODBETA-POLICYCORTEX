"""
Patent #4: Ensemble Machine Learning Engine
LSTM + Attention + Gradient Boosting + Prophet for Predictive Compliance
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- LSTM: 512 hidden dims, 3 layers, 0.2 dropout, 8 attention heads
- Ensemble Weights: Isolation Forest (40%), LSTM (30%), Autoencoder (30%)
- Prediction Accuracy: 99.2%
- False Positive Rate: <2%
- Inference Latency: <100ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for LSTM networks"""
    
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight tracking
        Returns: (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and reshape
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.fc_out(attention_output)
        
        return output, attention_weights


class PolicyCompliancePredictor(nn.Module):
    """
    LSTM network with attention for policy compliance prediction
    Patent Spec: 512 hidden dims, 3 layers, 0.2 dropout, 8 attention heads
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 3, 
                 dropout: float = 0.2, num_heads: int = 8, sequence_length: int = 100):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Feature extraction layers (256→512→1024→512)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim * 2, num_heads, dropout)
        
        # Prediction layers (512→256→128→2)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Binary classification: compliant/non-compliant
        )
        
        # Confidence scorer (512→1)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        Returns: {
            'prediction': class probabilities,
            'confidence': confidence score,
            'attention_weights': attention weights for explainability
        }
        """
        batch_size = x.size(0)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Reshape for LSTM if needed
        if len(features.shape) == 2:
            features = features.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Apply attention
        attended_output, attention_weights = self.attention(lstm_out, mask)
        
        # Global pooling
        pooled = torch.mean(attended_output, dim=1)
        
        # Predictions
        predictions = self.predictor(pooled)
        confidence = self.confidence_scorer(pooled)
        
        return {
            'prediction': F.softmax(predictions, dim=-1),
            'confidence': confidence,
            'attention_weights': attention_weights
        }


class LSTMAnomalyDetector(nn.Module):
    """LSTM-based sequence-to-sequence autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )
        
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            dropout=0.2,
            batch_first=True,
            bidirectional=True
        )
        
        self.decoder_fc = nn.Linear(hidden_dim * 2, input_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder
        Returns: (reconstruction, latent_representation)
        """
        # Encode
        encoder_out, (hidden, cell) = self.encoder_lstm(x)
        latent = self.encoder_fc(encoder_out[:, -1, :])
        
        # Decode
        batch_size, seq_len, _ = x.shape
        latent_expanded = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder_lstm(latent_expanded)
        reconstruction = self.decoder_fc(decoder_out)
        
        return reconstruction, latent
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score"""
        reconstruction, _ = self.forward(x)
        mse = F.mse_loss(reconstruction, x, reduction='none')
        anomaly_score = torch.mean(mse, dim=(1, 2))
        return anomaly_score


class AutoencoderAnomalyDetector(nn.Module):
    """Deep autoencoder for configuration anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        decoding_dims = encoding_dims[::-1][1:] + [input_dim]
        prev_dim = encoding_dims[-1]
        for dim in decoding_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU() if dim != input_dim else nn.Identity(),
                nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity()
            ])
            prev_dim = dim
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error as anomaly score"""
        reconstruction, _ = self.forward(x)
        anomaly_score = F.mse_loss(reconstruction, x, reduction='none').mean(dim=1)
        return anomaly_score


class AnomalyDetectionPipeline:
    """
    Ensemble anomaly detection pipeline
    Patent Spec: Isolation Forest (40%), LSTM Detector (30%), Autoencoder (30%)
    """
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
        # Initialize ensemble components
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.02,  # 2% expected anomaly rate
            random_state=42
        )
        
        self.lstm_detector = LSTMAnomalyDetector(input_dim)
        self.autoencoder = AutoencoderAnomalyDetector(input_dim)
        
        # Ensemble weights (Patent requirement)
        self.weights = {
            'isolation_forest': 0.4,
            'lstm': 0.3,
            'autoencoder': 0.3
        }
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, epochs: int = 50) -> None:
        """Train all ensemble components"""
        logger.info("Training Anomaly Detection Ensemble...")
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        # Train LSTM Detector
        logger.info("Training LSTM Anomaly Detector...")
        self._train_neural_model(self.lstm_detector, X_scaled, epochs)
        
        # Train Autoencoder
        logger.info("Training Autoencoder...")
        self._train_neural_model(self.autoencoder, X_scaled, epochs)
        
        self.is_fitted = True
        logger.info("Ensemble training complete")
        
    def _train_neural_model(self, model: nn.Module, X: np.ndarray, epochs: int):
        """Train a neural network model"""
        # Use CPU for compatibility during testing
        device = torch.device('cpu')
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            if hasattr(model, 'compute_anomaly_score'):
                # For models with anomaly score computation
                scores = model.compute_anomaly_score(X_tensor)
                loss = scores.mean()
            else:
                # For standard autoencoders
                output, _ = model(X_tensor)
                loss = criterion(output, X_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        model.eval()
        
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Make ensemble predictions with confidence scores
        Returns anomaly scores, predictions, and confidence
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        start_time = time.time()
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each component
        scores = {}
        
        # Isolation Forest scores
        if_scores = self.isolation_forest.score_samples(X_scaled)
        # Normalize to [0, 1] where 1 is most anomalous
        if_scores = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        scores['isolation_forest'] = if_scores
        
        # LSTM scores
        device = torch.device('cpu')
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            lstm_scores = self.lstm_detector.compute_anomaly_score(X_tensor).cpu().numpy()
        scores['lstm'] = lstm_scores
        
        # Autoencoder scores
        with torch.no_grad():
            ae_scores = self.autoencoder.compute_anomaly_score(X_tensor).cpu().numpy()
        scores['autoencoder'] = ae_scores
        
        # Compute weighted ensemble score
        ensemble_score = np.zeros(len(X))
        for model_name, weight in self.weights.items():
            ensemble_score += weight * scores[model_name]
        
        # Binary predictions (threshold at 0.5)
        predictions = (ensemble_score > 0.5).astype(int)
        
        # Confidence based on model agreement
        model_predictions = {
            name: (score > 0.5).astype(int) 
            for name, score in scores.items()
        }
        
        agreement = np.std([pred for pred in model_predictions.values()], axis=0)
        confidence = 1 - agreement  # High confidence when models agree
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'anomaly_scores': ensemble_score,
            'predictions': predictions,
            'confidence': confidence,
            'component_scores': scores,
            'inference_time_ms': inference_time
        }


class GradientBoostingPredictor:
    """Gradient boosting for structured compliance prediction"""
    
    def __init__(self, use_xgboost: bool = True):
        self.use_xgboost = use_xgboost
        
        if use_xgboost:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
        else:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary',
                random_state=42
            )
        
        self.feature_importance = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None):
        """Train gradient boosting model"""
        if validation_data:
            X_val, y_val = validation_data
            eval_set = [(X_val, y_val)]
            self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=10, verbose=False)
        else:
            self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions"""
        return self.model.predict(X)


class ProphetForecaster:
    """Time series forecasting for compliance trends using Prophet"""
    
    def __init__(self, forecast_horizon: int = 72):
        """
        Initialize Prophet forecaster
        Args:
            forecast_horizon: Hours to forecast ahead (default 72 for 3 days)
        """
        self.forecast_horizon = forecast_horizon
        self.models = {}  # Store multiple models for different metrics
        
    def fit(self, df: pd.DataFrame, target_column: str = 'violations'):
        """
        Fit Prophet model on time series data
        Args:
            df: DataFrame with 'ds' (datetime) and target column
            target_column: Name of column to forecast
        """
        # Prepare data for Prophet
        prophet_df = df[['ds', target_column]].rename(columns={target_column: 'y'})
        
        # Initialize and fit model
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays_prior_scale=10,
            seasonality_mode='multiplicative',
            weekly_seasonality=True,
            daily_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        model.fit(prophet_df)
        self.models[target_column] = model
        
    def predict(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Generate forecasts
        Args:
            periods: Number of periods to forecast (uses horizon if not specified)
        """
        if not self.models:
            raise ValueError("Model must be fitted before prediction")
        
        periods = periods or self.forecast_horizon
        predictions = {}
        
        for target, model in self.models.items():
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='H')
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract relevant columns
            predictions[target] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        return predictions


class EnsembleComplianceEngine:
    """
    Main ensemble engine combining all models
    Orchestrates LSTM, Anomaly Detection, Gradient Boosting, and Prophet
    """
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
        # Initialize all components
        self.compliance_predictor = PolicyCompliancePredictor(input_dim)
        self.anomaly_pipeline = AnomalyDetectionPipeline(input_dim)
        self.gradient_booster = GradientBoostingPredictor()
        self.prophet_forecaster = ProphetForecaster()
        
        # Model weights for final ensemble
        self.model_weights = {
            'lstm_compliance': 0.3,
            'anomaly_detection': 0.3,
            'gradient_boosting': 0.25,
            'prophet_forecast': 0.15
        }
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, time_series_df: Optional[pd.DataFrame] = None):
        """Train all ensemble components"""
        logger.info("Training Ensemble Compliance Engine...")
        
        # Train LSTM compliance predictor
        self._train_compliance_predictor(X, y)
        
        # Train anomaly detection pipeline
        self.anomaly_pipeline.fit(X)
        
        # Train gradient boosting
        self.gradient_booster.fit(X, y)
        
        # Train Prophet if time series data provided
        if time_series_df is not None:
            self.prophet_forecaster.fit(time_series_df)
        
        self.is_fitted = True
        logger.info("Ensemble training complete")
        
    def _train_compliance_predictor(self, X: np.ndarray, y: np.ndarray, epochs: int = 100):
        """Train the LSTM compliance predictor"""
        # Use CPU for compatibility during testing
        device = torch.device('cpu')
        self.compliance_predictor = self.compliance_predictor.to(device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.compliance_predictor.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.compliance_predictor.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            outputs = self.compliance_predictor(X_tensor)
            loss = criterion(outputs['prediction'], y_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                accuracy = (outputs['prediction'].argmax(dim=1) == y_tensor).float().mean()
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        
        self.compliance_predictor.eval()
        
    def predict(self, X: np.ndarray, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Generate ensemble predictions
        Patent Requirement: <100ms inference latency
        """
        if not self.is_fitted:
            raise ValueError("Engine must be fitted before prediction")
        
        start_time = time.time()
        predictions = {}
        
        # Get LSTM compliance predictions
        device = torch.device('cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            lstm_output = self.compliance_predictor(X_tensor)
            predictions['lstm_compliance'] = lstm_output['prediction'].cpu().numpy()
            attention_weights = lstm_output['attention_weights'].cpu().numpy()
        
        # Get anomaly detection predictions
        anomaly_output = self.anomaly_pipeline.predict(X)
        predictions['anomaly_detection'] = 1 - anomaly_output['anomaly_scores']  # Invert for compliance
        
        # Get gradient boosting predictions
        gb_proba = self.gradient_booster.predict_proba(X)
        predictions['gradient_boosting'] = gb_proba[:, 1]  # Probability of compliance
        
        # Combine predictions using weighted average
        final_scores = np.zeros(len(X))
        for model_name, weight in self.model_weights.items():
            if model_name in predictions:
                final_scores += weight * predictions[model_name]
        
        # Normalize weights if Prophet not available
        if 'prophet_forecast' not in predictions:
            weight_sum = sum(v for k, v in self.model_weights.items() if k != 'prophet_forecast')
            final_scores = final_scores / weight_sum
        
        # Binary predictions
        final_predictions = (final_scores > 0.5).astype(int)
        
        # Calculate confidence
        model_std = np.std([pred for pred in predictions.values()], axis=0)
        confidence = 1 - model_std
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        result = {
            'predictions': final_predictions,
            'compliance_probability': final_scores,
            'confidence': confidence,
            'inference_time_ms': inference_time,
            'attention_weights': attention_weights
        }
        
        if return_all_scores:
            result['component_predictions'] = predictions
        
        # Verify latency requirement (<100ms)
        if inference_time > 100:
            logger.warning(f"Inference time {inference_time:.2f}ms exceeds 100ms requirement")
        
        return result
    
    def validate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Validate model performance against patent requirements
        Requirements: 99.2% accuracy, <2% false positive rate
        """
        predictions = self.predict(X_test)
        y_pred = predictions['predictions']
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        
        # False positive rate
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        true_negatives = np.sum((y_pred == 0) & (y_test == 0))
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        
        # Check patent requirements
        meets_accuracy = accuracy >= 0.992
        meets_fpr = fpr < 0.02
        meets_latency = predictions['inference_time_ms'] < 100
        
        return {
            'accuracy': accuracy,
            'false_positive_rate': fpr,
            'inference_time_ms': predictions['inference_time_ms'],
            'meets_accuracy_requirement': meets_accuracy,
            'meets_fpr_requirement': meets_fpr,
            'meets_latency_requirement': meets_latency,
            'all_requirements_met': meets_accuracy and meets_fpr and meets_latency
        }