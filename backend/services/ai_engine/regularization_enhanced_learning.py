"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Enhanced Continuous Learning System with Comprehensive Regularization
Prevents overfitting through multiple regularization techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from collections import deque
import random
from datetime import datetime
import math

logger = logging.getLogger(__name__)

# Enhanced configuration with regularization parameters
@dataclass
class RegularizationConfig:
    """Configuration for all regularization techniques"""
    # Dropout rates
    dropout_rate: float = 0.3  # Increased from 0.1
    attention_dropout: float = 0.2
    embedding_dropout: float = 0.15
    
    # Weight decay (L2 regularization)
    weight_decay: float = 0.01
    
    # L1 regularization
    l1_lambda: float = 1e-5
    
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    gradient_clip_value: float = 5.0
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # Mixup augmentation
    mixup_alpha: float = 0.2
    
    # Noise injection
    input_noise_std: float = 0.01
    weight_noise_std: float = 0.005
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    # Validation split
    validation_split: float = 0.2
    
    # Ensemble
    num_models: int = 3
    
    # Batch normalization
    use_batch_norm: bool = True
    
    # Spectral normalization
    use_spectral_norm: bool = True
    
    # Maximum norm constraint
    max_norm: float = 3.0


class DropoutScheduler:
    """Dynamically adjust dropout rate during training"""
    
    def __init__(self, base_rate: float = 0.3, warmup_steps: int = 1000):
        self.base_rate = base_rate
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def get_rate(self) -> float:
        """Get current dropout rate based on training progress"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Start with lower dropout, gradually increase
            return self.base_rate * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (self.current_step - self.warmup_steps) / self.warmup_steps
            return self.base_rate * (1 + 0.5 * math.cos(math.pi * min(1.0, progress)))


class StochasticDepth(nn.Module):
    """Stochastic depth (layer dropout) for regularization"""
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
            
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(torch.full((x.size(0), 1, 1), keep_prob, device=x.device))
        return x * mask / keep_prob


class RegularizedTransformerLayer(nn.Module):
    """Transformer layer with comprehensive regularization"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 config: RegularizationConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention with dropout
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, 
            dropout=config.attention_dropout,
            batch_first=True  # Fix the warning about batch_first
        )
        
        # Feedforward network with regularization
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Apply spectral normalization if enabled
        if config.use_spectral_norm:
            self.linear1 = nn.utils.spectral_norm(self.linear1)
            self.linear2 = nn.utils.spectral_norm(self.linear2)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Batch normalization (optional)
        if config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(d_model)
        
        # Dropout layers
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)
        
        # Stochastic depth
        self.stochastic_depth = StochasticDepth(drop_prob=0.1)
        
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual and dropout
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.stochastic_depth(self.dropout1(src2))
        src = self.norm1(src)
        
        # Feedforward with GELU activation and dropout
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.stochastic_depth(self.dropout2(src2))
        src = self.norm2(src)
        
        # Optional batch normalization
        if self.config.use_batch_norm and self.training:
            batch_size, seq_len, d_model = src.shape
            src = src.reshape(-1, d_model)
            src = self.batch_norm(src)
            src = src.reshape(batch_size, seq_len, d_model)
        
        return src


class RegularizedErrorLearningModel(nn.Module):
    """Enhanced error learning model with comprehensive regularization"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512, 
                 config: RegularizationConfig = None):
        super().__init__()
        self.config = config or RegularizationConfig()
        
        # Embeddings with dropout
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.domain_embedding = nn.Embedding(10, embedding_dim // 4)
        self.severity_embedding = nn.Embedding(5, embedding_dim // 4)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(self.config.embedding_dropout)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=self.config.dropout_rate)
        
        # Transformer encoder with regularization
        self.transformer_layers = nn.ModuleList([
            RegularizedTransformerLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=1024,
                config=self.config
            ) for _ in range(6)
        ])
        
        # Output layers with weight constraints
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, embedding_dim)
        
        # Apply weight constraints
        self._apply_weight_constraints()
        
        # Task-specific heads
        self.error_classifier = nn.Linear(embedding_dim, 100)
        self.solution_generator = nn.Linear(embedding_dim, vocab_size)
        self.severity_predictor = nn.Linear(embedding_dim, 5)
        
        # Additional regularization layers
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Initialize weights with Xavier/He initialization
        self._initialize_weights()
        
    def _apply_weight_constraints(self):
        """Apply max norm constraint to weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = F.normalize(
                    module.weight.data, p=2, dim=0
                ) * min(self.config.max_norm, module.weight.data.norm(2, dim=0).max())
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def add_noise_to_weights(self):
        """Add Gaussian noise to weights for regularization"""
        if self.training:
            for param in self.parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * self.config.weight_noise_std
                    param.data.add_(noise)
    
    def forward(self, tokens: torch.Tensor, domain: torch.Tensor,
                severity: torch.Tensor, mask: Optional[torch.Tensor] = None,
                apply_mixup: bool = False, mixup_lambda: float = None) -> Dict[str, torch.Tensor]:
        
        # Add input noise for regularization
        if self.training and self.config.input_noise_std > 0:
            tokens = tokens + torch.randn_like(tokens.float()) * self.config.input_noise_std
            tokens = tokens.long()
        
        # Embed inputs
        token_emb = self.embedding_dropout(self.token_embedding(tokens))
        domain_emb = self.domain_embedding(domain).unsqueeze(1).expand(-1, tokens.size(1), -1)
        severity_emb = self.severity_embedding(severity).unsqueeze(1).expand(-1, tokens.size(1), -1)
        
        # Combine embeddings
        x = torch.cat([token_emb, domain_emb, severity_emb], dim=-1)
        
        # Project to correct dimension
        projection = nn.Linear(x.size(-1), token_emb.size(-1)).to(x.device)
        x = projection(x)
        
        # Apply mixup if training
        if self.training and apply_mixup and mixup_lambda is not None:
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            x = mixup_lambda * x + (1 - mixup_lambda) * x[index]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer layers with stochastic depth
        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
        
        # Global pooling with attention weights
        attention_weights = F.softmax(x.mean(dim=-1), dim=1).unsqueeze(-1)
        x = (x * attention_weights).sum(dim=1)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Pass through FC layers with dropout
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        features = self.fc2(hidden)
        
        # Generate outputs with label smoothing preparation
        outputs = {
            'features': features,
            'error_class': self.error_classifier(features),
            'solution': self.solution_generator(features),
            'severity': self.severity_predictor(features)
        }
        
        return outputs


class PositionalEncoding(nn.Module):
    """Positional encoding with dropout"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for regularization"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class ValidationMonitor:
    """Monitor validation metrics to detect overfitting"""
    
    def __init__(self, window_size: int = 10):
        self.train_losses = deque(maxlen=window_size)
        self.val_losses = deque(maxlen=window_size)
        self.overfitting_threshold = 0.1
        
    def update(self, train_loss: float, val_loss: float):
        """Update losses and check for overfitting"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
    def is_overfitting(self) -> bool:
        """Check if model is overfitting"""
        if len(self.train_losses) < 5:
            return False
            
        avg_train = np.mean(self.train_losses)
        avg_val = np.mean(self.val_losses)
        
        # Check if validation loss is significantly higher than training
        if avg_val > avg_train * (1 + self.overfitting_threshold):
            return True
            
        # Check if validation loss is increasing while training decreases
        if len(self.val_losses) == self.val_losses.maxlen:
            val_trend = np.polyfit(range(len(self.val_losses)), list(self.val_losses), 1)[0]
            train_trend = np.polyfit(range(len(self.train_losses)), list(self.train_losses), 1)[0]
            
            if val_trend > 0 and train_trend < 0:
                return True
                
        return False


class DataAugmentation:
    """Data augmentation techniques for error text"""
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 2) -> str:
        """Replace n words with synonyms"""
        words = text.split()
        if len(words) < n:
            return text
            
        # Simple replacement with common error synonyms
        synonyms = {
            'error': ['failure', 'exception', 'issue'],
            'failed': ['unsuccessful', 'errored', 'broken'],
            'connection': ['link', 'network', 'socket'],
            'timeout': ['expired', 'deadline', 'limit'],
            'invalid': ['incorrect', 'wrong', 'bad']
        }
        
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()
            if word in synonyms:
                words[idx] = random.choice(synonyms[word])
                
        return ' '.join(words)
    
    @staticmethod
    def random_insertion(text: str, n: int = 2) -> str:
        """Insert n random words"""
        words = text.split()
        technical_words = ['system', 'process', 'service', 'module', 'component']
        
        for _ in range(n):
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(technical_words))
            
        return ' '.join(words)
    
    @staticmethod
    def random_swap(text: str, n: int = 2) -> str:
        """Swap n pairs of words"""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(n):
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    @staticmethod
    def augment(text: str, num_augmentations: int = 1) -> List[str]:
        """Generate augmented versions of text"""
        augmented = [text]
        
        for _ in range(num_augmentations):
            aug_type = random.choice([
                DataAugmentation.synonym_replacement,
                DataAugmentation.random_insertion,
                DataAugmentation.random_swap
            ])
            augmented.append(aug_type(text))
            
        return augmented


class EnsembleLearning:
    """Ensemble of models to reduce overfitting"""
    
    def __init__(self, num_models: int = 3, vocab_size: int = 50000):
        self.models = nn.ModuleList([
            RegularizedErrorLearningModel(
                vocab_size=vocab_size,
                config=RegularizationConfig()
            ) for _ in range(num_models)
        ])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble with weighted averaging"""
        outputs = []
        
        for i, model in enumerate(self.models):
            model_output = model(*args, **kwargs)
            outputs.append(model_output)
        
        # Weighted average of predictions
        ensemble_output = {}
        for key in outputs[0].keys():
            stacked = torch.stack([out[key] for out in outputs])
            weights = F.softmax(self.weights, dim=0).view(-1, 1, 1)
            ensemble_output[key] = (stacked * weights).sum(dim=0)
            
        return ensemble_output
    
    def train_models(self, dataloader, optimizer, criterion, config: RegularizationConfig):
        """Train ensemble with different data subsets"""
        for i, model in enumerate(self.models):
            model.train()
            
            # Use different random subset for each model
            subset_size = int(len(dataloader.dataset) * 0.8)
            indices = torch.randperm(len(dataloader.dataset))[:subset_size]
            subset = torch.utils.data.Subset(dataloader.dataset, indices)
            subset_loader = torch.utils.data.DataLoader(
                subset, batch_size=dataloader.batch_size, shuffle=True
            )
            
            # Train individual model
            for batch in subset_loader:
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = criterion(outputs, batch['labels'])
                
                # Add L1 regularization
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = loss + config.l1_lambda * l1_loss
                
                loss.backward()
                optimizer.step()


class RegularizedOptimizer:
    """Optimizer with advanced regularization techniques"""
    
    def __init__(self, model: nn.Module, config: RegularizationConfig):
        self.config = config
        
        # Main optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.weight_decay
        )
        
        # Learning rate schedulers
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=1000
        )
        
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=100,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.plateau_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        self.current_step = 0
        
    def step(self, loss: torch.Tensor, val_loss: Optional[float] = None):
        """Optimization step with gradient clipping and scheduling"""
        self.current_step += 1
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.optimizer.param_groups[0]['params'],
            self.config.gradient_clip_norm
        )
        torch.nn.utils.clip_grad_value_(
            self.optimizer.param_groups[0]['params'],
            self.config.gradient_clip_value
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update schedulers
        if self.current_step < 1000:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
            
        if val_loss is not None:
            self.plateau_scheduler.step(val_loss)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']


# Export enhanced components
__all__ = [
    'RegularizationConfig',
    'RegularizedErrorLearningModel',
    'LabelSmoothingLoss',
    'EarlyStopping',
    'ValidationMonitor',
    'DataAugmentation',
    'EnsembleLearning',
    'RegularizedOptimizer',
    'DropoutScheduler'
]