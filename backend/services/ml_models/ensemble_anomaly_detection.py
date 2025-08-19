"""
Patent #4: Predictive Policy Compliance Engine
Ensemble Anomaly Detection Pipeline

This module implements the ensemble anomaly detection system as specified in Patent #4,
combining Isolation Forest (40%), LSTM Detector (30%), and Autoencoder (30%) for
robust anomaly detection in compliance patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Union
import joblib
from dataclasses import dataclass
import logging

from policy_compliance_predictor import LSTMAnomalyDetector, create_anomaly_detector

logger = logging.getLogger(__name__)


@dataclass
class AnomalyDetectionConfig:
    """Configuration for ensemble anomaly detection."""
    isolation_forest_weight: float = 0.4  # Patent specification
    lstm_detector_weight: float = 0.3     # Patent specification
    autoencoder_weight: float = 0.3       # Patent specification
    contamination: float = 0.01
    n_estimators: int = 100
    max_samples: Union[int, str] = 'auto'
    random_state: int = 42
    sequence_length: int = 100
    input_size: int = 256
    hidden_size: int = 512
    latent_size: int = 128  # Patent specification
    dropout: float = 0.2


class AutoencoderAnomalyDetector(nn.Module):
    """
    Deep Autoencoder for anomaly detection in configuration data.
    Part of the ensemble system with 30% weight contribution.
    """
    
    def __init__(self, 
                 input_size: int = 256,
                 hidden_sizes: List[int] = None,
                 latent_size: int = 128,
                 dropout: float = 0.2):
        super(AutoencoderAnomalyDetector, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256]
        
        # Encoder layers
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        encoder_layers.append(nn.Linear(prev_size, latent_size))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (mirror of encoder)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_size, hidden_sizes[-1]))
        
        for i in range(len(hidden_sizes) - 1, 0, -1):
            decoder_layers.extend([
                nn.ReLU(),
                nn.BatchNorm1d(hidden_sizes[i]),
                nn.Dropout(dropout),
                nn.Linear(hidden_sizes[i], hidden_sizes[i-1])
            ])
        
        decoder_layers.extend([
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], input_size)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor [batch_size, input_size] or [batch_size, seq_len, input_size]
            
        Returns:
            Dictionary with reconstruction and latent representation
        """
        # Handle sequence data by flattening
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x = x.reshape(-1, features)
        
        # Encode and decode
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        # Reshape back if needed
        if len(original_shape) == 3:
            reconstructed = reconstructed.reshape(original_shape)
            latent = latent.reshape(batch_size, seq_len, -1)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
        anomaly_scores = reconstruction_error.mean(dim=-1)
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'reconstruction_error': reconstruction_error,
            'anomaly_scores': anomaly_scores
        }


class AnomalyDetectionPipeline:
    """
    Ensemble Anomaly Detection Pipeline combining multiple detection methods.
    Patent Specification: Isolation Forest (40%), LSTM Detector (30%), Autoencoder (30%)
    """
    
    def __init__(self, config: Optional[AnomalyDetectionConfig] = None):
        """
        Initialize the ensemble anomaly detection pipeline.
        
        Args:
            config: Configuration for the ensemble system
        """
        self.config = config or AnomalyDetectionConfig()
        
        # Validate weights sum to 1.0
        total_weight = (self.config.isolation_forest_weight + 
                       self.config.lstm_detector_weight + 
                       self.config.autoencoder_weight)
        assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight}"
        
        # Initialize components
        self._init_isolation_forest()
        self._init_lstm_detector()
        self._init_autoencoder()
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Threshold calibration
        self.threshold = None
        self.is_fitted = False
        
    def _init_isolation_forest(self):
        """Initialize Isolation Forest component (40% weight)."""
        self.isolation_forest = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        logger.info("Initialized Isolation Forest with weight: 40%")
        
    def _init_lstm_detector(self):
        """Initialize LSTM Anomaly Detector component (30% weight)."""
        self.lstm_detector = create_anomaly_detector({
            'input_size': self.config.input_size,
            'hidden_size': self.config.hidden_size,
            'latent_size': self.config.latent_size
        })
        logger.info("Initialized LSTM Detector with weight: 30%")
        
    def _init_autoencoder(self):
        """Initialize Autoencoder component (30% weight)."""
        self.autoencoder = AutoencoderAnomalyDetector(
            input_size=self.config.input_size,
            hidden_sizes=[512, 256],
            latent_size=self.config.latent_size,
            dropout=self.config.dropout
        )
        logger.info("Initialized Autoencoder with weight: 30%")
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
            epochs: int = 100, batch_size: int = 32, 
            learning_rate: float = 1e-3) -> 'AnomalyDetectionPipeline':
        """
        Fit the ensemble anomaly detection pipeline.
        
        Args:
            X: Training data [n_samples, n_features] or [n_samples, seq_len, n_features]
            y: Optional labels for semi-supervised learning
            epochs: Number of training epochs for neural components
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training ensemble on data shape: {X.shape}")
        
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scaler=True)
        
        # Fit Isolation Forest
        self._fit_isolation_forest(X_scaled)
        
        # Fit LSTM Detector
        self._fit_lstm_detector(X_scaled, epochs, batch_size, learning_rate)
        
        # Fit Autoencoder
        self._fit_autoencoder(X_scaled, epochs, batch_size, learning_rate)
        
        # Calibrate threshold
        self._calibrate_threshold(X_scaled)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
        
        return self
        
    def _prepare_data(self, X: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Prepare and normalize data."""
        # Reshape if needed
        if len(X.shape) == 3:
            n_samples, seq_len, n_features = X.shape
            X_flat = X.reshape(-1, n_features)
        else:
            X_flat = X
        
        # Fit or transform scaler
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_flat)
        else:
            X_scaled = self.scaler.transform(X_flat)
        
        # Reshape back if needed
        if len(X.shape) == 3:
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
        
        return X_scaled
        
    def _fit_isolation_forest(self, X: np.ndarray):
        """Fit Isolation Forest component."""
        # Flatten sequences for Isolation Forest
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        self.isolation_forest.fit(X_flat)
        logger.info("Isolation Forest fitted")
        
    def _fit_lstm_detector(self, X: np.ndarray, epochs: int, 
                           batch_size: int, learning_rate: float):
        """Fit LSTM Detector component."""
        # Ensure sequence format
        if len(X.shape) == 2:
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X_seq = X
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq)
        
        # Setup training
        self.lstm_detector.train()
        optimizer = torch.optim.Adam(self.lstm_detector.parameters(), lr=learning_rate)
        
        # Training loop
        n_batches = len(X_tensor) // batch_size
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                batch = X_tensor[i*batch_size:(i+1)*batch_size]
                
                optimizer.zero_grad()
                output = self.lstm_detector(batch)
                loss = output['reconstruction_error'].mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"LSTM Detector - Epoch {epoch}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")
        
        logger.info("LSTM Detector fitted")
        
    def _fit_autoencoder(self, X: np.ndarray, epochs: int, 
                         batch_size: int, learning_rate: float):
        """Fit Autoencoder component."""
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Setup training
        self.autoencoder.train()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        
        # Training loop
        n_batches = len(X_tensor) // batch_size
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(n_batches):
                batch = X_tensor[i*batch_size:(i+1)*batch_size]
                
                optimizer.zero_grad()
                output = self.autoencoder(batch)
                loss = output['reconstruction_error'].mean()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Autoencoder - Epoch {epoch}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")
        
        logger.info("Autoencoder fitted")
        
    def _calibrate_threshold(self, X: np.ndarray):
        """Calibrate anomaly threshold based on training data."""
        scores = self.predict_scores(X)
        
        # Use percentile-based threshold
        percentile = (1 - self.config.contamination) * 100
        self.threshold = np.percentile(scores, percentile)
        
        logger.info(f"Threshold calibrated: {self.threshold:.4f}")
        
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using weighted ensemble.
        
        Args:
            X: Input data [n_samples, n_features] or [n_samples, seq_len, n_features]
            
        Returns:
            Anomaly scores [n_samples]
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scaler=False)
        
        # Get scores from each component
        if_scores = self._get_isolation_forest_scores(X_scaled)
        lstm_scores = self._get_lstm_scores(X_scaled)
        ae_scores = self._get_autoencoder_scores(X_scaled)
        
        # Combine with specified weights
        ensemble_scores = (
            self.config.isolation_forest_weight * if_scores +
            self.config.lstm_detector_weight * lstm_scores +
            self.config.autoencoder_weight * ae_scores
        )
        
        return ensemble_scores
        
    def _get_isolation_forest_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from Isolation Forest."""
        # Flatten sequences if needed
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        
        # Get scores (negative scores indicate anomalies)
        scores = -self.isolation_forest.score_samples(X_flat)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
        
    def _get_lstm_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from LSTM Detector."""
        # Ensure sequence format
        if len(X.shape) == 2:
            X_seq = X.reshape(X.shape[0], 1, X.shape[1])
        else:
            X_seq = X
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq)
        
        # Get scores
        self.lstm_detector.eval()
        with torch.no_grad():
            output = self.lstm_detector(X_tensor)
            scores = output['anomaly_scores'].numpy()
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
        
    def _get_autoencoder_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores from Autoencoder."""
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Get scores
        self.autoencoder.eval()
        with torch.no_grad():
            output = self.autoencoder(X_tensor)
            scores = output['anomaly_scores'].numpy()
        
        # Handle sequence data
        if len(scores.shape) > 1:
            scores = scores.mean(axis=1)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
        
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomalies using the ensemble.
        
        Args:
            X: Input data
            threshold: Optional custom threshold (uses calibrated if not provided)
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        scores = self.predict_scores(X)
        
        if threshold is None:
            if self.threshold is None:
                raise ValueError("No threshold calibrated. Fit the model first or provide threshold.")
            threshold = self.threshold
        
        predictions = (scores > threshold).astype(int)
        
        return predictions
        
    def predict_with_confidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict anomalies with confidence scores and component contributions.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with predictions, scores, and component contributions
        """
        # Prepare data
        X_scaled = self._prepare_data(X, fit_scaler=False)
        
        # Get individual component scores
        if_scores = self._get_isolation_forest_scores(X_scaled)
        lstm_scores = self._get_lstm_scores(X_scaled)
        ae_scores = self._get_autoencoder_scores(X_scaled)
        
        # Calculate ensemble scores
        ensemble_scores = self.predict_scores(X)
        
        # Get predictions
        predictions = self.predict(X)
        
        # Calculate confidence based on component agreement
        component_predictions = np.stack([
            if_scores > np.percentile(if_scores, 99),
            lstm_scores > np.percentile(lstm_scores, 99),
            ae_scores > np.percentile(ae_scores, 99)
        ], axis=1)
        
        agreement_score = component_predictions.mean(axis=1)
        confidence = np.where(predictions == 1, agreement_score, 1 - agreement_score)
        
        return {
            'predictions': predictions,
            'anomaly_scores': ensemble_scores,
            'confidence': confidence,
            'isolation_forest_scores': if_scores,
            'lstm_scores': lstm_scores,
            'autoencoder_scores': ae_scores,
            'component_weights': {
                'isolation_forest': self.config.isolation_forest_weight,
                'lstm_detector': self.config.lstm_detector_weight,
                'autoencoder': self.config.autoencoder_weight
            }
        }
        
    def save(self, path: str):
        """
        Save the ensemble model to disk.
        
        Args:
            path: Directory path to save the model
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        joblib.dump(self.config, os.path.join(path, 'config.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Save Isolation Forest
        joblib.dump(self.isolation_forest, os.path.join(path, 'isolation_forest.pkl'))
        
        # Save LSTM Detector
        torch.save(self.lstm_detector.state_dict(), 
                  os.path.join(path, 'lstm_detector.pth'))
        
        # Save Autoencoder
        torch.save(self.autoencoder.state_dict(), 
                  os.path.join(path, 'autoencoder.pth'))
        
        # Save threshold
        joblib.dump(self.threshold, os.path.join(path, 'threshold.pkl'))
        
        logger.info(f"Ensemble model saved to {path}")
        
    def load(self, path: str):
        """
        Load the ensemble model from disk.
        
        Args:
            path: Directory path to load the model from
        """
        import os
        
        # Load configuration
        self.config = joblib.load(os.path.join(path, 'config.pkl'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        
        # Load Isolation Forest
        self.isolation_forest = joblib.load(os.path.join(path, 'isolation_forest.pkl'))
        
        # Load LSTM Detector
        self._init_lstm_detector()
        self.lstm_detector.load_state_dict(
            torch.load(os.path.join(path, 'lstm_detector.pth'))
        )
        
        # Load Autoencoder
        self._init_autoencoder()
        self.autoencoder.load_state_dict(
            torch.load(os.path.join(path, 'autoencoder.pth'))
        )
        
        # Load threshold
        self.threshold = joblib.load(os.path.join(path, 'threshold.pkl'))
        
        self.is_fitted = True
        logger.info(f"Ensemble model loaded from {path}")


def create_ensemble_pipeline(config: Optional[Dict] = None) -> AnomalyDetectionPipeline:
    """
    Factory function to create an ensemble anomaly detection pipeline.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured AnomalyDetectionPipeline
    """
    if config:
        config_obj = AnomalyDetectionConfig(**config)
    else:
        config_obj = AnomalyDetectionConfig()
    
    pipeline = AnomalyDetectionPipeline(config_obj)
    
    return pipeline


if __name__ == "__main__":
    # Test the ensemble pipeline
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 1000
    seq_len = 100  # Patent specification
    n_features = 256
    
    # Normal data
    X_normal = np.random.randn(n_samples, seq_len, n_features) * 0.5
    
    # Anomalous data (10% contamination)
    n_anomalies = int(n_samples * 0.1)
    X_anomalies = np.random.randn(n_anomalies, seq_len, n_features) * 2.0
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y_true = np.array([0] * n_samples + [1] * n_anomalies)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y_true = y_true[indices]
    
    # Create and train ensemble
    pipeline = create_ensemble_pipeline()
    pipeline.fit(X[:800], epochs=10)  # Use 80% for training
    
    # Test on remaining data
    X_test = X[800:]
    y_test = y_true[800:]
    
    # Get predictions with confidence
    results = pipeline.predict_with_confidence(X_test)
    
    # Calculate metrics
    predictions = results['predictions']
    accuracy = (predictions == y_test).mean()
    
    print(f"Ensemble Anomaly Detection Results:")
    print(f"  Test Accuracy: {accuracy:.2%}")
    print(f"  Anomalies detected: {predictions.sum()}/{len(predictions)}")
    print(f"  Average confidence: {results['confidence'].mean():.3f}")
    print(f"  Component weights: {results['component_weights']}")
    
    # Show score distributions
    print(f"\nScore distributions:")
    print(f"  Isolation Forest: mean={results['isolation_forest_scores'].mean():.3f}, "
          f"std={results['isolation_forest_scores'].std():.3f}")
    print(f"  LSTM Detector: mean={results['lstm_scores'].mean():.3f}, "
          f"std={results['lstm_scores'].std():.3f}")
    print(f"  Autoencoder: mean={results['autoencoder_scores'].mean():.3f}, "
          f"std={results['autoencoder_scores'].std():.3f}")
    print(f"  Ensemble: mean={results['anomaly_scores'].mean():.3f}, "
          f"std={results['anomaly_scores'].std():.3f}")