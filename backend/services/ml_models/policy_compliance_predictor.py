"""
Patent #4: Predictive Policy Compliance Engine
LSTM Network with Attention Mechanisms for Policy Compliance Prediction

This module implements the PolicyCompliancePredictor as specified in Patent #4,
with 512-dimensional hidden states, 3 layers, 0.2 dropout, and 8-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for policy-resource correlation analysis.
    Patent Specification: 8-head multi-head attention
    """
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, dropout: float = 0.2):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape for multi-head
        Q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        output = self.out(context)
        
        return output, attention_weights


class PolicyCompliancePredictor(nn.Module):
    """
    LSTM Network with Attention for Policy Compliance Prediction
    Patent Specifications:
    - Hidden dimensions: 512
    - Layers: 3
    - Dropout rate: 0.2
    - Attention heads: 8
    - Sequence length: 100
    """
    
    def __init__(self, 
                 input_size: int = 256,
                 hidden_size: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 num_heads: int = 8,
                 sequence_length: int = 100,
                 num_classes: int = 2):
        super(PolicyCompliancePredictor, self).__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        
        # Feature extraction layers (256→512→1024→512)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
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
        
        # LSTM layers with specified configuration
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention for policy-resource correlation
        self.attention = MultiHeadAttention(
            hidden_size=hidden_size * 2,  # Bidirectional LSTM
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Policy encoder
        self.policy_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Prediction layers (512→256→128→2)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Confidence scorer (512→1)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
    def forward(self, 
                resource_features: torch.Tensor,
                policy_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            resource_features: [batch_size, sequence_length, input_size]
            policy_features: [batch_size, policy_size]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing predictions, confidence scores, and attention weights
        """
        batch_size = resource_features.size(0)
        seq_len = resource_features.size(1)
        
        # Extract features from resource sequence
        resource_encoded = torch.zeros(batch_size, seq_len, self.hidden_size).to(resource_features.device)
        for t in range(seq_len):
            resource_encoded[:, t, :] = self.feature_extractor(resource_features[:, t, :])
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(resource_encoded)
        
        # Encode policy features
        policy_encoded = self.policy_encoder(policy_features)
        policy_encoded = policy_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine policy encoding with LSTM output for attention
        combined = torch.cat([lstm_out[:, :, :self.hidden_size], 
                             policy_encoded], dim=-1)
        
        # Apply multi-head attention
        attended_output, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out, mask
        )
        
        # Add residual connection and layer normalization
        attended_output = self.layer_norm(attended_output + lstm_out)
        
        # Global pooling over sequence
        pooled_output = torch.mean(attended_output, dim=1)
        
        # Generate predictions
        compliance_prediction = self.predictor(pooled_output)
        confidence_score = self.confidence_scorer(pooled_output)
        
        # Calculate violation probability
        violation_prob = F.softmax(compliance_prediction, dim=-1)[:, 1]
        
        return {
            'predictions': compliance_prediction,
            'violation_probability': violation_prob,
            'confidence_score': confidence_score.squeeze(-1),
            'attention_weights': attention_weights,
            'hidden_states': lstm_out
        }
    
    def predict_with_uncertainty(self, 
                                 resource_features: torch.Tensor,
                                 policy_features: torch.Tensor,
                                 n_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty quantification using Monte Carlo dropout.
        
        Args:
            resource_features: Input resource features
            policy_features: Input policy features
            n_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            Dictionary with mean predictions and uncertainty bounds
        """
        self.train()  # Enable dropout for Monte Carlo sampling
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(resource_features, policy_features)
                predictions.append(output['violation_probability'])
                confidences.append(output['confidence_score'])
        
        predictions = torch.stack(predictions)
        confidences = torch.stack(confidences)
        
        # Calculate statistics
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        mean_confidence = confidences.mean(dim=0)
        
        # Calculate confidence intervals
        lower_bound = mean_prediction - 1.96 * std_prediction
        upper_bound = mean_prediction + 1.96 * std_prediction
        
        return {
            'mean_prediction': mean_prediction,
            'std_prediction': std_prediction,
            'lower_bound': torch.clamp(lower_bound, 0, 1),
            'upper_bound': torch.clamp(upper_bound, 0, 1),
            'mean_confidence': mean_confidence,
            'epistemic_uncertainty': std_prediction
        }


class LSTMAnomalyDetector(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder for temporal anomaly detection.
    Used for configuration drift and anomaly detection in compliance patterns.
    """
    
    def __init__(self,
                 input_size: int = 256,
                 hidden_size: int = 512,
                 latent_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super(LSTMAnomalyDetector, self).__init__()
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Latent space projection
        self.encoder_fc = nn.Linear(hidden_size * 2, latent_size)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            input_size=latent_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Output projection
        self.decoder_fc = nn.Linear(hidden_size * 2, input_size)
        
        # Attention mechanism for important time steps
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to latent representation."""
        lstm_out, (hidden, cell) = self.encoder_lstm(x)
        
        # Apply temporal attention
        attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling
        pooled = torch.mean(attended, dim=1)
        
        # Project to latent space
        latent = self.encoder_fc(pooled)
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation back to sequence."""
        batch_size = latent.size(0)
        
        # Repeat latent vector for each time step
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode through LSTM
        lstm_out, _ = self.decoder_lstm(decoder_input)
        
        # Project to output space
        reconstructed = self.decoder_fc(lstm_out)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_size]
            
        Returns:
            Dictionary with reconstruction and latent representation
        """
        seq_len = x.size(1)
        
        # Encode
        latent = self.encode(x)
        
        # Decode
        reconstructed = self.decode(latent, seq_len)
        
        # Calculate reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, x, reduction='none')
        anomaly_scores = reconstruction_error.mean(dim=[1, 2])
        
        return {
            'reconstructed': reconstructed,
            'latent': latent,
            'reconstruction_error': reconstruction_error,
            'anomaly_scores': anomaly_scores
        }
    
    def detect_anomalies(self, x: torch.Tensor, threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            x: Input sequence
            threshold: Anomaly threshold (if None, uses statistical threshold)
            
        Returns:
            Dictionary with anomaly detection results
        """
        output = self.forward(x)
        anomaly_scores = output['anomaly_scores']
        
        if threshold is None:
            # Use statistical threshold (mean + 3*std)
            mean_score = anomaly_scores.mean()
            std_score = anomaly_scores.std()
            threshold = mean_score + 3 * std_score
        
        anomalies = anomaly_scores > threshold
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies,
            'threshold': threshold,
            'reconstructed': output['reconstructed'],
            'latent': output['latent']
        }


def create_policy_compliance_predictor(config: Dict) -> PolicyCompliancePredictor:
    """
    Factory function to create PolicyCompliancePredictor with patent specifications.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PolicyCompliancePredictor model
    """
    model = PolicyCompliancePredictor(
        input_size=config.get('input_size', 256),
        hidden_size=512,  # Patent specification
        num_layers=3,     # Patent specification
        dropout=0.2,      # Patent specification
        num_heads=8,      # Patent specification
        sequence_length=100,  # Patent specification
        num_classes=config.get('num_classes', 2)
    )
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    return model


def create_anomaly_detector(config: Dict) -> LSTMAnomalyDetector:
    """
    Factory function to create LSTMAnomalyDetector for configuration drift detection.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LSTMAnomalyDetector model
    """
    model = LSTMAnomalyDetector(
        input_size=config.get('input_size', 256),
        hidden_size=512,
        latent_size=128,  # Patent specification
        num_layers=2,
        dropout=0.2
    )
    
    return model


# Training utilities
class ComplianceLoss(nn.Module):
    """
    Custom loss function combining classification loss with confidence calibration.
    """
    
    def __init__(self, alpha: float = 0.5):
        super(ComplianceLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model output dictionary
            targets: Ground truth labels
            
        Returns:
            Combined loss value
        """
        # Classification loss
        classification_loss = self.ce_loss(predictions['predictions'], targets)
        
        # Confidence calibration loss
        # Confidence should be high when prediction is correct
        pred_classes = predictions['predictions'].argmax(dim=1)
        correct = (pred_classes == targets).float()
        confidence_loss = F.mse_loss(predictions['confidence_score'], correct)
        
        # Combined loss
        total_loss = classification_loss + self.alpha * confidence_loss
        
        return total_loss


if __name__ == "__main__":
    # Test the models
    config = {'input_size': 256, 'num_classes': 2}
    
    # Test PolicyCompliancePredictor
    predictor = create_policy_compliance_predictor(config)
    batch_size, seq_len, input_size = 32, 100, 256
    resource_features = torch.randn(batch_size, seq_len, input_size)
    policy_features = torch.randn(batch_size, input_size)
    
    output = predictor(resource_features, policy_features)
    print(f"Prediction output shapes:")
    print(f"  Predictions: {output['predictions'].shape}")
    print(f"  Violation probability: {output['violation_probability'].shape}")
    print(f"  Confidence score: {output['confidence_score'].shape}")
    print(f"  Attention weights: {output['attention_weights'].shape}")
    
    # Test uncertainty quantification
    uncertainty_output = predictor.predict_with_uncertainty(
        resource_features[:1], policy_features[:1], n_samples=10
    )
    print(f"\nUncertainty quantification:")
    print(f"  Mean prediction: {uncertainty_output['mean_prediction'].item():.4f}")
    print(f"  Epistemic uncertainty: {uncertainty_output['epistemic_uncertainty'].item():.4f}")
    
    # Test LSTMAnomalyDetector
    detector = create_anomaly_detector(config)
    anomaly_output = detector.detect_anomalies(resource_features)
    print(f"\nAnomaly detection output:")
    print(f"  Anomaly scores: {anomaly_output['anomaly_scores'].shape}")
    print(f"  Anomalies detected: {anomaly_output['anomalies'].sum().item()}/{batch_size}")