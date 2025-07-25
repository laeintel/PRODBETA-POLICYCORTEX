"""
Advanced Model Ensemble Manager
Coordinates multiple AI models for comprehensive PolicyCortex governance analysis
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import optuna
from hydra import compose, initialize
from omegaconf import DictConfig

from backend.core.config import settings
from backend.core.redis_client import redis_client
from backend.core.exceptions import APIError

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Types of AI models in the ensemble"""
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    ANOMALY_DETECTION = "anomaly_detection"
    NLP_PROCESSING = "nlp_processing"
    COST_OPTIMIZATION = "cost_optimization"
    COMPLIANCE_PREDICTION = "compliance_prediction"
    CORRELATION_ANALYSIS = "correlation_analysis"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"
    CONVERSATION_INTELLIGENCE = "conversation_intelligence"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    FEATURE_ENGINEERING = "feature_engineering"


class EnsembleStrategy(str, Enum):
    """Ensemble combination strategies"""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_SELECTION = "dynamic_selection"
    CONSENSUS = "consensus"


class ModelStatus(str, Enum):
    """Model status indicators"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UPDATING = "updating"
    OFFLINE = "offline"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction_id: str
    predictions: Dict[str, Any]
    confidence_scores: Dict[str, float]
    model_contributions: Dict[str, float]
    ensemble_confidence: float
    strategy_used: EnsembleStrategy
    processing_time_ms: float
    models_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelConfiguration:
    """Model configuration settings"""
    model_id: str
    model_type: ModelType
    weight: float = 1.0
    enabled: bool = True
    max_latency_ms: float = 5000
    min_confidence: float = 0.5
    fallback_models: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    update_frequency: timedelta = field(default_factory=lambda: timedelta(hours=24))


class ModelHealthMonitor:
    """Monitors health and performance of individual models"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.alert_thresholds = {
            'accuracy': 0.7,
            'latency_ms': 10000,
            'memory_usage_mb': 2048,
            'cpu_usage_percent': 80
        }
        self._monitoring = False
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start continuous model health monitoring"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Model health monitoring started")
    
    async def stop_monitoring(self):
        """Stop model health monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Model health monitoring stopped")
    
    async def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self._monitoring:
            try:
                await self._check_all_models()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_models(self):
        """Check health of all registered models"""
        # This would integrate with actual model services
        # For now, simulate health checks
        pass
    
    async def update_model_metrics(self, model_id: str, metrics: ModelMetrics):
        """Update metrics for a specific model"""
        self.model_metrics[model_id] = metrics
        
        # Check for alerts
        alerts = []
        if metrics.accuracy < self.alert_thresholds['accuracy']:
            alerts.append(f"Low accuracy: {metrics.accuracy:.3f}")
        
        if metrics.latency_ms > self.alert_thresholds['latency_ms']:
            alerts.append(f"High latency: {metrics.latency_ms:.1f}ms")
        
        if metrics.memory_usage_mb > self.alert_thresholds['memory_usage_mb']:
            alerts.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Model {model_id} health alerts: {', '.join(alerts)}")
    
    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get current status of a model"""
        if model_id not in self.model_metrics:
            return ModelStatus.OFFLINE
        
        metrics = self.model_metrics[model_id]
        
        # Determine status based on metrics
        if (metrics.accuracy < 0.5 or 
            metrics.latency_ms > 20000 or 
            metrics.memory_usage_mb > 4096):
            return ModelStatus.FAILED
        
        if (metrics.accuracy < 0.7 or 
            metrics.latency_ms > 10000 or 
            metrics.memory_usage_mb > 2048):
            return ModelStatus.DEGRADED
        
        return ModelStatus.HEALTHY


class EnsembleOptimizer:
    """Optimizes ensemble configurations using hyperparameter tuning"""
    
    def __init__(self):
        self.study = None
        self.best_params = {}
        self.optimization_history = []
    
    async def optimize_ensemble(self, 
                              validation_data: List[Dict[str, Any]],
                              optimization_metric: str = 'f1_score',
                              n_trials: int = 100) -> Dict[str, Any]:
        """Optimize ensemble configuration"""
        
        def objective(trial):
            # Suggest hyperparameters
            strategy = trial.suggest_categorical('strategy', [
                'voting', 'stacking', 'weighted_average'
            ])
            
            model_weights = {}
            for model_type in ModelType:
                weight = trial.suggest_float(f'weight_{model_type.value}', 0.0, 2.0)
                model_weights[model_type.value] = weight
            
            confidence_threshold = trial.suggest_float('confidence_threshold', 0.1, 0.9)
            
            # Simulate ensemble performance with these parameters
            # In production, this would run actual validation
            simulated_score = np.random.uniform(0.7, 0.95)
            
            return simulated_score
        
        # Create optimization study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=n_trials)
        
        self.best_params = self.study.best_params
        
        optimization_result = {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': n_trials,
            'optimization_time': datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        
        logger.info(f"Ensemble optimization completed: {self.study.best_value:.4f}")
        
        return optimization_result


class DynamicModelSelector:
    """Dynamically selects best models based on input characteristics"""
    
    def __init__(self):
        self.selection_rules = {}
        self.performance_history = defaultdict(list)
        self.context_models = {}
    
    async def select_models(self, 
                          input_data: Dict[str, Any],
                          task_type: str,
                          available_models: List[str]) -> List[str]:
        """Select optimal models for given input and task"""
        
        # Analyze input characteristics
        input_features = self._extract_input_features(input_data)
        
        # Get model performance for similar inputs
        model_scores = {}
        for model_id in available_models:
            score = self._calculate_model_suitability(model_id, input_features, task_type)
            model_scores[model_id] = score
        
        # Select top models based on scores
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 3-5 models
        selected_models = [model_id for model_id, score in sorted_models[:5] if score > 0.5]
        
        if not selected_models:
            # Fallback to all available models
            selected_models = available_models[:3]
        
        logger.info(f"Selected models for {task_type}: {selected_models}")
        
        return selected_models
    
    def _extract_input_features(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from input data for model selection"""
        features = {
            'data_size': len(str(input_data)),
            'complexity': len(input_data),
            'numeric_ratio': 0.5,  # Placeholder
            'text_ratio': 0.3,     # Placeholder
            'temporal_components': 0.2  # Placeholder
        }
        
        return features
    
    def _calculate_model_suitability(self, 
                                   model_id: str, 
                                   input_features: Dict[str, float],
                                   task_type: str) -> float:
        """Calculate how suitable a model is for given input"""
        
        # Base suitability score
        base_score = 0.7
        
        # Adjust based on model type and task type compatibility
        compatibility_matrix = {
            ('predictive_analytics', 'prediction'): 1.0,
            ('anomaly_detection', 'anomaly'): 1.0,
            ('nlp_processing', 'text_analysis'): 1.0,
            ('correlation_analysis', 'correlation'): 1.0,
        }
        
        model_type = model_id.split('_')[0] if '_' in model_id else model_id
        compatibility_key = (model_type, task_type)
        
        if compatibility_key in compatibility_matrix:
            base_score *= compatibility_matrix[compatibility_key]
        
        # Adjust based on historical performance
        if model_id in self.performance_history:
            recent_performance = np.mean(self.performance_history[model_id][-10:])
            base_score *= recent_performance
        
        return min(base_score, 1.0)
    
    async def update_model_performance(self, 
                                     model_id: str, 
                                     performance_score: float):
        """Update performance history for a model"""
        self.performance_history[model_id].append(performance_score)
        
        # Keep only recent history
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]


class AdvancedModelEnsemble:
    """Main advanced model ensemble manager"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfiguration] = {}
        self.health_monitor = ModelHealthMonitor()
        self.optimizer = EnsembleOptimizer()
        self.selector = DynamicModelSelector()
        self.ensemble_cache: Dict[str, EnsemblePrediction] = {}
        self.prediction_history = deque(maxlen=10000)
        self._initialized = False
        self._model_lock = threading.RLock()
    
    async def initialize(self):
        """Initialize the ensemble manager"""
        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Initialize MLflow for experiment tracking
            try:
                mlflow.set_tracking_uri("http://localhost:5000")
                mlflow.set_experiment("PolicyCortex_Ensemble")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {str(e)}")
            
            self._initialized = True
            logger.info("Advanced model ensemble initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble: {str(e)}")
            raise
    
    async def register_model(self, 
                           model_id: str, 
                           model_instance: Any,
                           config: ModelConfiguration):
        """Register a model with the ensemble"""
        
        with self._model_lock:
            self.models[model_id] = model_instance
            self.model_configs[model_id] = config
        
        logger.info(f"Registered model: {model_id} ({config.model_type.value})")
    
    async def predict_ensemble(self, 
                             input_data: Dict[str, Any],
                             task_type: str,
                             strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
                             max_models: int = 5) -> EnsemblePrediction:
        """Make ensemble prediction using multiple models"""
        
        if not self._initialized:
            raise APIError("Ensemble not initialized", status_code=500)
        
        start_time = datetime.now()
        prediction_id = str(uuid.uuid4())
        
        try:
            # Select appropriate models
            available_models = [
                model_id for model_id, config in self.model_configs.items()
                if (config.enabled and 
                    self.health_monitor.get_model_status(model_id) != ModelStatus.FAILED)
            ]
            
            selected_models = await self.selector.select_models(
                input_data, task_type, available_models
            )
            selected_models = selected_models[:max_models]
            
            if not selected_models:
                raise APIError("No healthy models available", status_code=503)
            
            # Collect predictions from selected models
            model_predictions = {}
            model_confidences = {}
            model_contributions = {}
            
            # Execute predictions in parallel
            tasks = []
            for model_id in selected_models:
                task = asyncio.create_task(
                    self._get_model_prediction(model_id, input_data, task_type)
                )
                tasks.append((model_id, task))
            
            for model_id, task in tasks:
                try:
                    prediction, confidence = await asyncio.wait_for(task, timeout=10.0)
                    model_predictions[model_id] = prediction
                    model_confidences[model_id] = confidence
                    
                    # Calculate contribution weight
                    config = self.model_configs[model_id]
                    health_status = self.health_monitor.get_model_status(model_id)
                    
                    base_weight = config.weight
                    if health_status == ModelStatus.DEGRADED:
                        base_weight *= 0.5
                    elif health_status == ModelStatus.FAILED:
                        base_weight = 0.0
                    
                    model_contributions[model_id] = base_weight * confidence
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Model {model_id} prediction timeout")
                    continue
                except Exception as e:
                    logger.error(f"Model {model_id} prediction failed: {str(e)}")
                    continue
            
            if not model_predictions:
                raise APIError("All model predictions failed", status_code=503)
            
            # Combine predictions using specified strategy
            ensemble_result = await self._combine_predictions(
                model_predictions, 
                model_confidences,
                model_contributions,
                strategy
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create ensemble prediction
            prediction = EnsemblePrediction(
                prediction_id=prediction_id,
                predictions=ensemble_result,
                confidence_scores=model_confidences,
                model_contributions=model_contributions,
                ensemble_confidence=np.mean(list(model_confidences.values())),
                strategy_used=strategy,
                processing_time_ms=processing_time,
                models_used=list(model_predictions.keys())
            )
            
            # Cache prediction
            self.ensemble_cache[prediction_id] = prediction
            self.prediction_history.append(prediction)
            
            # Update model performance
            for model_id in model_predictions.keys():
                await self.selector.update_model_performance(
                    model_id, 
                    model_confidences[model_id]
                )
            
            # Log to MLflow if available
            try:
                with mlflow.start_run():
                    mlflow.log_param("strategy", strategy.value)
                    mlflow.log_param("models_used", len(model_predictions))
                    mlflow.log_metric("ensemble_confidence", prediction.ensemble_confidence)
                    mlflow.log_metric("processing_time_ms", processing_time)
            except Exception as e:
                logger.debug(f"MLflow logging failed: {str(e)}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise APIError(f"Ensemble prediction failed: {str(e)}", status_code=500)
    
    async def _get_model_prediction(self, 
                                  model_id: str, 
                                  input_data: Dict[str, Any],
                                  task_type: str) -> Tuple[Any, float]:
        """Get prediction from a specific model"""
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # This would call the actual model's prediction method
        # For now, simulate prediction
        prediction = {"result": f"prediction_from_{model_id}"}
        confidence = np.random.uniform(0.7, 0.95)
        
        return prediction, confidence
    
    async def _combine_predictions(self,
                                 predictions: Dict[str, Any],
                                 confidences: Dict[str, float],
                                 contributions: Dict[str, float],
                                 strategy: EnsembleStrategy) -> Dict[str, Any]:
        """Combine predictions using specified strategy"""
        
        if strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_combination(
                predictions, confidences, contributions
            )
        elif strategy == EnsembleStrategy.VOTING:
            return await self._voting_combination(predictions, confidences)
        elif strategy == EnsembleStrategy.CONSENSUS:
            return await self._consensus_combination(predictions, confidences)
        else:
            # Default to weighted average
            return await self._weighted_average_combination(
                predictions, confidences, contributions
            )
    
    async def _weighted_average_combination(self,
                                          predictions: Dict[str, Any],
                                          confidences: Dict[str, float],
                                          contributions: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions using weighted average"""
        
        # Normalize contributions
        total_contribution = sum(contributions.values())
        if total_contribution > 0:
            normalized_contributions = {
                k: v / total_contribution for k, v in contributions.items()
            }
        else:
            normalized_contributions = {
                k: 1.0 / len(contributions) for k in contributions.keys()
            }
        
        # For numeric predictions, compute weighted average
        # For categorical predictions, use voting
        # This is a simplified implementation
        
        combined_result = {
            'ensemble_prediction': 'weighted_combination',
            'individual_predictions': predictions,
            'weights': normalized_contributions,
            'confidence': np.average(list(confidences.values()), 
                                   weights=list(normalized_contributions.values()))
        }
        
        return combined_result
    
    async def _voting_combination(self,
                                predictions: Dict[str, Any],
                                confidences: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions using majority voting"""
        
        # Simple voting implementation
        votes = {}
        for model_id, prediction in predictions.items():
            pred_str = str(prediction)
            if pred_str not in votes:
                votes[pred_str] = 0
            votes[pred_str] += confidences[model_id]
        
        # Select prediction with highest weighted votes
        winning_prediction = max(votes.items(), key=lambda x: x[1])
        
        combined_result = {
            'ensemble_prediction': winning_prediction[0],
            'vote_weights': votes,
            'individual_predictions': predictions,
            'confidence': winning_prediction[1] / sum(votes.values())
        }
        
        return combined_result
    
    async def _consensus_combination(self,
                                   predictions: Dict[str, Any],
                                   confidences: Dict[str, float]) -> Dict[str, Any]:
        """Combine predictions requiring consensus"""
        
        # Check if all predictions agree (simplified)
        prediction_strings = [str(pred) for pred in predictions.values()]
        
        if len(set(prediction_strings)) == 1:
            # Full consensus
            consensus_confidence = np.mean(list(confidences.values()))
            result = {
                'ensemble_prediction': list(predictions.values())[0],
                'consensus': True,
                'confidence': consensus_confidence,
                'individual_predictions': predictions
            }
        else:
            # No consensus - use weighted average fallback
            result = await self._weighted_average_combination(
                predictions, confidences, confidences
            )
            result['consensus'] = False
        
        return result
    
    async def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        
        if not self.prediction_history:
            return {'message': 'No prediction history available'}
        
        recent_predictions = list(self.prediction_history)[-100:]
        
        stats = {
            'total_predictions': len(self.prediction_history),
            'recent_predictions': len(recent_predictions),
            'average_confidence': np.mean([
                p.ensemble_confidence for p in recent_predictions
            ]),
            'average_processing_time_ms': np.mean([
                p.processing_time_ms for p in recent_predictions
            ]),
            'strategy_distribution': {},
            'model_usage': defaultdict(int),
            'performance_trends': {}
        }
        
        # Strategy distribution
        for prediction in recent_predictions:
            strategy = prediction.strategy_used.value
            if strategy not in stats['strategy_distribution']:
                stats['strategy_distribution'][strategy] = 0
            stats['strategy_distribution'][strategy] += 1
        
        # Model usage statistics
        for prediction in recent_predictions:
            for model_id in prediction.models_used:
                stats['model_usage'][model_id] += 1
        
        stats['model_usage'] = dict(stats['model_usage'])
        
        # Model health summary
        stats['model_health'] = {}
        for model_id in self.models.keys():
            status = self.health_monitor.get_model_status(model_id)
            stats['model_health'][model_id] = status.value
        
        return stats
    
    async def optimize_ensemble_configuration(self, 
                                            validation_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize ensemble configuration based on validation data"""
        
        if validation_data is None:
            # Use synthetic validation data for demonstration
            validation_data = [
                {'input': f'sample_{i}', 'expected': f'output_{i}'} 
                for i in range(100)
            ]
        
        optimization_result = await self.optimizer.optimize_ensemble(
            validation_data=validation_data,
            optimization_metric='f1_score',
            n_trials=50
        )
        
        # Apply optimized configuration
        if self.optimizer.best_params:
            await self._apply_optimized_config(self.optimizer.best_params)
        
        return optimization_result
    
    async def _apply_optimized_config(self, optimized_params: Dict[str, Any]):
        """Apply optimized configuration to ensemble"""
        
        # Update model weights
        for model_id, config in self.model_configs.items():
            weight_key = f'weight_{config.model_type.value}'
            if weight_key in optimized_params:
                config.weight = optimized_params[weight_key]
        
        # Update other configuration parameters
        if 'confidence_threshold' in optimized_params:
            for config in self.model_configs.values():
                config.min_confidence = optimized_params['confidence_threshold']
        
        logger.info("Applied optimized ensemble configuration")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.health_monitor.stop_monitoring()
        
        # Clear caches
        self.ensemble_cache.clear()
        self.prediction_history.clear()
        
        logger.info("Advanced model ensemble cleanup completed")


# Global instance
advanced_model_ensemble = AdvancedModelEnsemble()