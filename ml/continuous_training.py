# File: ml/continuous_training.py
# Real-Time Model Retraining Pipeline for PolicyCortex

import numpy as np
import pickle
import json
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional
import asyncio
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)

class DataBuffer:
    """Buffer for accumulating training data"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.feature_stats = {}
        
    @property
    def size(self) -> int:
        return len(self.buffer)
    
    def add(self, data: Dict[str, Any]):
        """Add new data point to buffer"""
        self.buffer.append({
            'timestamp': datetime.now(),
            'features': data.get('features'),
            'label': data.get('label'),
            'confidence': data.get('confidence', 1.0),
            'source': data.get('source', 'production')
        })
        self._update_feature_stats(data.get('features'))
    
    def _update_feature_stats(self, features: Optional[Dict]):
        """Update running statistics for features"""
        if not features:
            return
        for key, value in features.items():
            if key not in self.feature_stats:
                self.feature_stats[key] = {
                    'mean': 0,
                    'std': 0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'count': 0
                }
            stats = self.feature_stats[key]
            if isinstance(value, (int, float)):
                stats['count'] += 1
                stats['mean'] = (stats['mean'] * (stats['count'] - 1) + value) / stats['count']
                stats['min'] = min(stats['min'], value)
                stats['max'] = max(stats['max'], value)
    
    def get_batch(self, size: int) -> List[Dict]:
        """Get batch of data for training"""
        if size >= self.size:
            return list(self.buffer)
        # Return most recent data
        return list(self.buffer)[-size:]
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()

class PredictiveComplianceModel:
    """Main predictive compliance model with incremental learning"""
    def __init__(self):
        self.model = None
        self.model_type = 'xgboost'
        self.version = 1
        self.metrics_history = []
        self.feature_importance = {}
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model based on type"""
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
    
    def partial_fit(self, new_data: List[Dict]):
        """Incremental training on new data"""
        if not new_data:
            return
        
        # Extract features and labels
        X = np.array([d['features'] for d in new_data if 'features' in d])
        y = np.array([d['label'] for d in new_data if 'label' in d])
        
        if len(X) == 0 or len(y) == 0:
            return
        
        # For XGBoost, we need to retrain with combined data
        if self.model_type == 'xgboost' and hasattr(self.model, 'feature_importances_'):
            # Get existing training data (simplified - in production would store)
            # For now, just train on new data
            pass
        
        # Fit the model
        self.model.fit(X, y)
        
        # Update feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = {
                f'feature_{i}': importance 
                for i, importance in enumerate(self.model.feature_importances_)
            }
        
        logger.info(f"Model partially fitted on {len(X)} samples")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.predict_proba(features)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1': f1_score(y_test, predictions, average='weighted'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def save(self, filepath: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'version': self.version,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'metrics_history': self.metrics_history[-10:]  # Keep last 10 metrics
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.version = model_data.get('version', 1)
        self.model_type = model_data.get('model_type', 'xgboost')
        self.feature_importance = model_data.get('feature_importance', {})
        self.metrics_history = model_data.get('metrics_history', [])
        
        logger.info(f"Model loaded from {filepath}")

class ContinuousTrainingPipeline:
    """Pipeline for continuous model training and deployment"""
    def __init__(self):
        self.model = PredictiveComplianceModel()
        self.data_buffer = DataBuffer(max_size=10000)
        self.validation_set = None
        self.model_registry = {}
        self.current_model_path = "models/current_model.pkl"
        self.training_threshold = 1000  # Minimum data points before retraining
        self.performance_threshold = 0.90  # Minimum accuracy to deploy
        self.is_training = False
        
    async def retrain_on_new_data(self):
        """Retrain model when enough new data is available"""
        if self.is_training or self.data_buffer.size < self.training_threshold:
            return
        
        self.is_training = True
        try:
            # Get training batch
            new_data = self.data_buffer.get_batch(1000)
            
            # Incremental training
            self.model.partial_fit(new_data)
            
            # Validate on holdout set
            if self.validation_set:
                metrics = self.model.evaluate(
                    self.validation_set['X'],
                    self.validation_set['y']
                )
                
                # Deploy if performance is good
                if metrics['accuracy'] >= self.performance_threshold:
                    model_path = f"models/v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                    self.model.save(model_path)
                    await self.deploy_new_model(model_path)
                    logger.info(f"New model deployed with accuracy: {metrics['accuracy']:.3f}")
                else:
                    logger.warning(f"Model accuracy {metrics['accuracy']:.3f} below threshold")
            
            # Clear processed data
            self.data_buffer.clear()
            
        finally:
            self.is_training = False
    
    async def deploy_new_model(self, model_path: str):
        """Deploy new model to production"""
        # In production, this would involve:
        # 1. A/B testing
        # 2. Gradual rollout
        # 3. Monitoring
        # 4. Rollback capability
        
        # For now, simply update the current model path
        self.current_model_path = model_path
        self.model_registry[datetime.now().isoformat()] = model_path
        
        logger.info(f"Model deployed: {model_path}")
    
    def add_training_data(self, features: Dict, label: int, confidence: float = 1.0):
        """Add new training data to buffer"""
        self.data_buffer.add({
            'features': features,
            'label': label,
            'confidence': confidence
        })
    
    def set_validation_data(self, X_val: np.ndarray, y_val: np.ndarray):
        """Set validation dataset"""
        self.validation_set = {'X': X_val, 'y': y_val}
    
    async def monitor_and_retrain(self):
        """Continuous monitoring and retraining loop"""
        while True:
            try:
                await self.retrain_on_new_data()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Error in continuous training: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

class ModelVersionManager:
    """Manage model versions and rollback"""
    def __init__(self):
        self.versions = {}
        self.current_version = None
        self.performance_history = {}
        
    def register_version(self, version: str, model_path: str, metrics: Dict):
        """Register new model version"""
        self.versions[version] = {
            'path': model_path,
            'metrics': metrics,
            'deployed_at': datetime.now(),
            'status': 'staged'
        }
    
    def promote_version(self, version: str):
        """Promote version to production"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        # Demote current version
        if self.current_version:
            self.versions[self.current_version]['status'] = 'archived'
        
        # Promote new version
        self.versions[version]['status'] = 'production'
        self.current_version = version
        
        logger.info(f"Promoted version {version} to production")
    
    def rollback(self, to_version: Optional[str] = None):
        """Rollback to previous version"""
        if to_version:
            target = to_version
        else:
            # Find last stable version
            sorted_versions = sorted(
                self.versions.items(),
                key=lambda x: x[1]['deployed_at'],
                reverse=True
            )
            target = None
            for version, info in sorted_versions:
                if info['status'] == 'archived' and info['metrics'].get('accuracy', 0) > 0.85:
                    target = version
                    break
        
        if target:
            self.promote_version(target)
            logger.info(f"Rolled back to version {target}")
        else:
            logger.error("No suitable version found for rollback")

# Neural Network Model for Advanced Predictions
class NeuralComplianceModel(nn.Module):
    """Neural network model for compliance prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(NeuralComplianceModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class OnlineLearningSystem:
    """Online learning system with drift detection"""
    def __init__(self):
        self.pipeline = ContinuousTrainingPipeline()
        self.version_manager = ModelVersionManager()
        self.drift_threshold = 0.1
        self.performance_window = deque(maxlen=100)
        
    async def process_feedback(self, prediction_id: str, actual_outcome: int):
        """Process user feedback for online learning"""
        # In production, retrieve original features from database
        # For now, simulate with random features
        features = {f'feature_{i}': np.random.random() for i in range(10)}
        
        # Add to training buffer
        self.pipeline.add_training_data(features, actual_outcome)
        
        # Track performance
        self.performance_window.append({
            'timestamp': datetime.now(),
            'correct': True  # Would compare with original prediction
        })
        
        # Check for drift
        if await self.detect_drift():
            logger.warning("Model drift detected, triggering retraining")
            await self.pipeline.retrain_on_new_data()
    
    async def detect_drift(self) -> bool:
        """Detect model drift based on performance"""
        if len(self.performance_window) < 50:
            return False
        
        recent_accuracy = sum(
            1 for p in list(self.performance_window)[-20:] 
            if p.get('correct', False)
        ) / 20
        
        overall_accuracy = sum(
            1 for p in self.performance_window 
            if p.get('correct', False)
        ) / len(self.performance_window)
        
        drift = abs(recent_accuracy - overall_accuracy)
        return drift > self.drift_threshold

# Export main components
__all__ = [
    'ContinuousTrainingPipeline',
    'PredictiveComplianceModel',
    'ModelVersionManager',
    'OnlineLearningSystem',
    'NeuralComplianceModel',
    'DataBuffer'
]