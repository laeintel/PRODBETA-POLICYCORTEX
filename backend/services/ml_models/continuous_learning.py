"""
Patent #4: Predictive Policy Compliance Engine
Continuous Learning Pipeline

This module implements the continuous learning subsystem as specified in Patent #4,
including human feedback collection, online learning, automated retraining, and
A/B testing for model improvement.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json
import pickle
import hashlib
from enum import Enum
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy import stats
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    ACCURACY = "accuracy"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    EXPERT_ANNOTATION = "expert_annotation"
    OUTCOME_CONFIRMATION = "outcome_confirmation"
    REMEDIATION_EFFECTIVENESS = "remediation_effectiveness"
    FEATURE_RELEVANCE = "feature_relevance"
    PREDICTION_CONFIDENCE = "prediction_confidence"


class ModelPerformanceMetric(Enum):
    """Performance metrics for model evaluation."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    FALSE_POSITIVE_RATE = "false_positive_rate"
    LATENCY = "latency"
    THROUGHPUT = "throughput"


@dataclass
class FeedbackEntry:
    """Individual feedback entry."""
    feedback_id: str
    prediction_id: str
    feedback_type: FeedbackType
    value: Any
    confidence: float
    user_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False
    model_version: Optional[str] = None


@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning pipeline."""
    # Feedback collection
    min_feedback_for_retrain: int = 100
    feedback_buffer_size: int = 10000
    feedback_validation_enabled: bool = True
    feedback_quality_threshold: float = 0.7
    
    # Online learning
    online_learning_rate: float = 0.001
    online_batch_size: int = 32
    online_update_frequency: int = 100  # Updates per N feedback items
    
    # Concept drift detection
    drift_detection_method: str = "adwin"  # adwin, page_hinkley, kswin
    drift_threshold: float = 0.05
    drift_window_size: int = 1000
    
    # Automated retraining
    retrain_schedule: str = "adaptive"  # fixed, adaptive, triggered
    retrain_interval_hours: int = 168  # Weekly default
    performance_degradation_threshold: float = 0.05
    min_performance_samples: int = 100
    
    # A/B testing
    ab_test_enabled: bool = True
    ab_test_traffic_split: float = 0.1  # Percentage for challenger model
    ab_test_min_samples: int = 1000
    ab_test_confidence_level: float = 0.95
    
    # Model versioning
    max_model_versions: int = 10
    model_rollback_enabled: bool = True
    
    # Resource limits
    max_concurrent_training: int = 2
    training_timeout_seconds: int = 3600


class FeedbackCollector:
    """
    Collects and validates feedback for model improvement.
    Implements the feedback collection system from Patent #4.
    """
    
    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.feedback_buffer = deque(maxlen=config.feedback_buffer_size)
        self.feedback_stats = defaultdict(lambda: {"count": 0, "quality": []})
        self.lock = threading.Lock()
        
    def collect_feedback(self, feedback: FeedbackEntry) -> bool:
        """
        Collect and validate feedback.
        
        Args:
            feedback: Feedback entry to collect
            
        Returns:
            True if feedback was accepted, False otherwise
        """
        # Validate feedback if enabled
        if self.config.feedback_validation_enabled:
            if not self._validate_feedback(feedback):
                logger.warning(f"Invalid feedback rejected: {feedback.feedback_id}")
                return False
                
        # Add to buffer
        with self.lock:
            self.feedback_buffer.append(feedback)
            
            # Update statistics
            self.feedback_stats[feedback.feedback_type]["count"] += 1
            self.feedback_stats[feedback.feedback_type]["quality"].append(feedback.confidence)
            
        logger.info(f"Feedback collected: {feedback.feedback_id} ({feedback.feedback_type.value})")
        return True
        
    def _validate_feedback(self, feedback: FeedbackEntry) -> bool:
        """
        Validate feedback quality and consistency.
        
        Args:
            feedback: Feedback to validate
            
        Returns:
            True if feedback is valid
        """
        # Check confidence threshold
        if feedback.confidence < self.config.feedback_quality_threshold:
            return False
            
        # Check feedback type-specific validation
        if feedback.feedback_type == FeedbackType.ACCURACY:
            if feedback.value not in [0, 1, True, False]:
                return False
                
        elif feedback.feedback_type in [FeedbackType.FALSE_POSITIVE, FeedbackType.FALSE_NEGATIVE]:
            if not isinstance(feedback.value, bool):
                return False
                
        elif feedback.feedback_type == FeedbackType.REMEDIATION_EFFECTIVENESS:
            if not 0 <= feedback.value <= 1:
                return False
                
        # Check for duplicate feedback
        recent_feedback = list(self.feedback_buffer)[-100:]
        for recent in recent_feedback:
            if (recent.prediction_id == feedback.prediction_id and 
                recent.user_id == feedback.user_id and
                recent.feedback_type == feedback.feedback_type):
                return False  # Duplicate feedback
                
        return True
        
    def get_feedback_batch(self, batch_size: int, 
                          feedback_type: Optional[FeedbackType] = None) -> List[FeedbackEntry]:
        """
        Get batch of feedback for processing.
        
        Args:
            batch_size: Number of feedback entries to retrieve
            feedback_type: Optional filter by feedback type
            
        Returns:
            List of feedback entries
        """
        with self.lock:
            if feedback_type:
                filtered = [f for f in self.feedback_buffer 
                          if f.feedback_type == feedback_type and not f.processed]
            else:
                filtered = [f for f in self.feedback_buffer if not f.processed]
                
            batch = filtered[:batch_size]
            
            # Mark as processed
            for feedback in batch:
                feedback.processed = True
                
        return batch
        
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """
        Get feedback collection statistics.
        
        Returns:
            Dictionary of feedback statistics
        """
        with self.lock:
            stats = {
                "total_feedback": len(self.feedback_buffer),
                "unprocessed": sum(1 for f in self.feedback_buffer if not f.processed),
                "by_type": {}
            }
            
            for feedback_type, type_stats in self.feedback_stats.items():
                avg_quality = np.mean(type_stats["quality"]) if type_stats["quality"] else 0
                stats["by_type"][feedback_type.value] = {
                    "count": type_stats["count"],
                    "average_quality": avg_quality
                }
                
        return stats
        
    def detect_feedback_bias(self) -> Dict[str, float]:
        """
        Detect potential bias in feedback.
        
        Returns:
            Bias detection results
        """
        bias_indicators = {}
        
        with self.lock:
            # Check user concentration
            user_feedback_counts = defaultdict(int)
            for feedback in self.feedback_buffer:
                user_feedback_counts[feedback.user_id] += 1
                
            if user_feedback_counts:
                max_user_feedback = max(user_feedback_counts.values())
                total_feedback = len(self.feedback_buffer)
                bias_indicators["user_concentration"] = max_user_feedback / total_feedback
                
            # Check temporal clustering
            if self.feedback_buffer:
                timestamps = [f.timestamp for f in self.feedback_buffer]
                time_diffs = np.diff(sorted(timestamps))
                if len(time_diffs) > 0:
                    bias_indicators["temporal_clustering"] = np.std(time_diffs.astype(float))
                    
            # Check feedback type imbalance
            type_counts = defaultdict(int)
            for feedback in self.feedback_buffer:
                type_counts[feedback.feedback_type] += 1
                
            if type_counts:
                counts = list(type_counts.values())
                bias_indicators["type_imbalance"] = np.std(counts) / np.mean(counts)
                
        return bias_indicators


class OnlineLearner:
    """
    Implements online learning algorithms for continuous model updates.
    Supports incremental learning without full retraining.
    """
    
    def __init__(self, model: nn.Module, config: ContinuousLearningConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.online_learning_rate)
        self.update_count = 0
        self.performance_history = deque(maxlen=1000)
        
    def incremental_update(self, features: torch.Tensor, 
                          targets: torch.Tensor, 
                          sample_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Perform incremental model update.
        
        Args:
            features: Input features
            targets: Target labels
            sample_weights: Optional sample weights
            
        Returns:
            Update metrics
        """
        self.model.train()
        
        # Mini-batch gradient descent
        batch_size = min(self.config.online_batch_size, len(features))
        n_batches = len(features) // batch_size
        
        total_loss = 0
        predictions = []
        
        for i in range(n_batches):
            batch_features = features[i*batch_size:(i+1)*batch_size]
            batch_targets = targets[i*batch_size:(i+1)*batch_size]
            
            if sample_weights is not None:
                batch_weights = sample_weights[i*batch_size:(i+1)*batch_size]
            else:
                batch_weights = torch.ones_like(batch_targets)
                
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            
            # Handle dictionary output
            if isinstance(outputs, dict):
                logits = outputs['predictions']
            else:
                logits = outputs
                
            # Calculate weighted loss
            loss = nn.CrossEntropyLoss(reduction='none')(logits, batch_targets)
            weighted_loss = (loss * batch_weights).mean()
            
            # Backward pass
            weighted_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update
            self.optimizer.step()
            
            total_loss += weighted_loss.item()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
        # Calculate metrics
        accuracy = accuracy_score(targets.cpu().numpy()[:len(predictions)], predictions)
        
        self.update_count += 1
        self.performance_history.append(accuracy)
        
        return {
            "loss": total_loss / n_batches if n_batches > 0 else 0,
            "accuracy": accuracy,
            "update_count": self.update_count
        }
        
    def adapt_learning_rate(self, performance_trend: List[float]):
        """
        Adapt learning rate based on performance trend.
        
        Args:
            performance_trend: Recent performance metrics
        """
        if len(performance_trend) < 10:
            return
            
        # Calculate trend
        x = np.arange(len(performance_trend))
        slope, _, _, _, _ = stats.linregress(x, performance_trend)
        
        # Adjust learning rate
        if slope < -0.001:  # Performance degrading
            # Reduce learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
            logger.info(f"Reduced learning rate to {param_group['lr']}")
            
        elif slope > 0.001 and len(self.performance_history) > 100:  # Stable improvement
            # Slightly increase learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 1.05
                param_group['lr'] = min(param_group['lr'], 0.01)  # Cap at 0.01
            logger.info(f"Increased learning rate to {param_group['lr']}")


class ConceptDriftDetector:
    """
    Detects concept drift in data distribution and model performance.
    Implements multiple drift detection algorithms.
    """
    
    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.method = config.drift_detection_method
        
        # Initialize drift detectors
        self.adwin = ADWINDriftDetector() if self.method == "adwin" else None
        self.page_hinkley = PageHinkleyDetector() if self.method == "page_hinkley" else None
        self.kswin = KSWINDetector(window_size=config.drift_window_size) if self.method == "kswin" else None
        
        self.drift_history = deque(maxlen=1000)
        self.current_distribution = None
        
    def detect_drift(self, value: float) -> bool:
        """
        Detect if drift has occurred.
        
        Args:
            value: New value to check for drift
            
        Returns:
            True if drift detected
        """
        drift_detected = False
        
        if self.method == "adwin" and self.adwin:
            drift_detected = self.adwin.update(value)
            
        elif self.method == "page_hinkley" and self.page_hinkley:
            drift_detected = self.page_hinkley.update(value)
            
        elif self.method == "kswin" and self.kswin:
            drift_detected = self.kswin.update(value)
            
        if drift_detected:
            self.drift_history.append({
                "timestamp": datetime.now(),
                "value": value,
                "method": self.method
            })
            logger.warning(f"Concept drift detected using {self.method}")
            
        return drift_detected
        
    def detect_data_drift(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift in data distribution.
        
        Args:
            new_data: New data batch
            
        Returns:
            Drift detection results
        """
        if self.current_distribution is None:
            self.current_distribution = new_data
            return {"drift_detected": False, "drift_score": 0.0}
            
        # Kolmogorov-Smirnov test
        drift_scores = []
        for i in range(new_data.shape[1]):
            statistic, p_value = stats.ks_2samp(
                self.current_distribution[:, i],
                new_data[:, i]
            )
            drift_scores.append(1 - p_value)
            
        max_drift = max(drift_scores)
        drift_detected = max_drift > (1 - self.config.drift_threshold)
        
        if drift_detected:
            # Update distribution
            self.current_distribution = new_data
            
        return {
            "drift_detected": drift_detected,
            "drift_score": max_drift,
            "feature_drift_scores": drift_scores
        }


class ADWINDriftDetector:
    """ADWIN (Adaptive Windowing) drift detector."""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.total = 0
        self.variance = 0
        self.width = 0
        
    def update(self, value: float) -> bool:
        """Update with new value and check for drift."""
        self.window.append(value)
        self.total += value
        self.width += 1
        
        if self.width < 2:
            return False
            
        # Check for drift
        drift_detected = False
        for split_point in range(1, self.width):
            n1 = split_point
            n2 = self.width - split_point
            
            sum1 = sum(self.window[:split_point])
            sum2 = self.total - sum1
            
            mean1 = sum1 / n1
            mean2 = sum2 / n2
            
            # ADWIN bound
            dd = np.log(2 * self.width / self.delta)
            eps = np.sqrt(2 * dd / self.width) + (2 * dd) / (3 * self.width)
            
            if abs(mean1 - mean2) > eps:
                # Drift detected, remove old data
                self.window = self.window[split_point:]
                self.total = sum2
                self.width = n2
                drift_detected = True
                break
                
        return drift_detected


class PageHinkleyDetector:
    """Page-Hinkley drift detector."""
    
    def __init__(self, threshold: float = 50, alpha: float = 0.9999):
        self.threshold = threshold
        self.alpha = alpha
        self.mean = 0
        self.sum = 0
        self.n = 0
        
    def update(self, value: float) -> bool:
        """Update with new value and check for drift."""
        self.n += 1
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        self.sum += value - self.mean
        
        return abs(self.sum) > self.threshold


class KSWINDetector:
    """KSWIN (Kolmogorov-Smirnov Windowing) drift detector."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.reference = None
        
    def update(self, value: float) -> bool:
        """Update with new value and check for drift."""
        self.window.append(value)
        
        if len(self.window) < self.window_size:
            return False
            
        if self.reference is None:
            self.reference = list(self.window)
            return False
            
        # KS test
        statistic, p_value = stats.ks_2samp(self.reference, list(self.window))
        
        if p_value < self.threshold:
            # Update reference window
            self.reference = list(self.window)
            return True
            
        return False


class AutomatedRetrainingOrchestrator:
    """
    Orchestrates automated model retraining based on performance monitoring.
    Implements the automated retraining pipeline from Patent #4.
    """
    
    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        self.training_queue = deque()
        self.active_trainings = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_training)
        self.last_retrain_time = datetime.now()
        
    def should_retrain(self, feedback_count: int, 
                       performance_metrics: Dict[str, float]) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            feedback_count: Number of new feedback entries
            performance_metrics: Current performance metrics
            
        Returns:
            True if retraining should be triggered
        """
        # Check feedback threshold
        if feedback_count >= self.config.min_feedback_for_retrain:
            logger.info(f"Retraining triggered: feedback threshold reached ({feedback_count})")
            return True
            
        # Check performance degradation
        if self.performance_monitor.has_degraded(
            performance_metrics, 
            self.config.performance_degradation_threshold
        ):
            logger.info("Retraining triggered: performance degradation detected")
            return True
            
        # Check scheduled retraining
        if self.config.retrain_schedule == "fixed":
            time_since_last = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if time_since_last >= self.config.retrain_interval_hours:
                logger.info("Retraining triggered: scheduled interval reached")
                return True
                
        return False
        
    async def trigger_retraining(self, 
                                 training_data: Tuple[np.ndarray, np.ndarray],
                                 model_class: type,
                                 model_config: Dict[str, Any]) -> str:
        """
        Trigger asynchronous model retraining.
        
        Args:
            training_data: Tuple of (features, labels)
            model_class: Model class to instantiate
            model_config: Model configuration
            
        Returns:
            Training job ID
        """
        job_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        
        # Add to queue
        self.training_queue.append({
            "job_id": job_id,
            "data": training_data,
            "model_class": model_class,
            "config": model_config,
            "status": "queued",
            "created_at": datetime.now()
        })
        
        # Submit training job
        future = self.executor.submit(
            self._train_model,
            job_id,
            training_data,
            model_class,
            model_config
        )
        
        self.active_trainings[job_id] = future
        
        logger.info(f"Retraining job {job_id} submitted")
        return job_id
        
    def _train_model(self, job_id: str, 
                    training_data: Tuple[np.ndarray, np.ndarray],
                    model_class: type,
                    model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model (runs in separate thread).
        
        Args:
            job_id: Training job ID
            training_data: Training data
            model_class: Model class
            model_config: Model configuration
            
        Returns:
            Training results
        """
        try:
            logger.info(f"Starting training job {job_id}")
            
            # Create model
            model = model_class(**model_config)
            
            # Convert to tensors
            X_train, y_train = training_data
            X_tensor = torch.FloatTensor(X_train)
            y_tensor = torch.LongTensor(y_train)
            
            # Training loop (simplified)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            model.train()
            for epoch in range(100):  # Fixed epochs for demo
                optimizer.zero_grad()
                outputs = model(X_tensor)
                
                if isinstance(outputs, dict):
                    loss = criterion(outputs['predictions'], y_tensor)
                else:
                    loss = criterion(outputs, y_tensor)
                    
                loss.backward()
                optimizer.step()
                
            # Evaluate model
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                if isinstance(outputs, dict):
                    predictions = torch.argmax(outputs['predictions'], dim=1)
                else:
                    predictions = torch.argmax(outputs, dim=1)
                    
                accuracy = accuracy_score(y_tensor.numpy(), predictions.numpy())
                
            self.last_retrain_time = datetime.now()
            
            return {
                "job_id": job_id,
                "status": "completed",
                "accuracy": accuracy,
                "model": model,
                "completed_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {str(e)}")
            return {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            }
            
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of training job.
        
        Args:
            job_id: Training job ID
            
        Returns:
            Training status
        """
        if job_id in self.active_trainings:
            future = self.active_trainings[job_id]
            if future.done():
                return future.result()
            else:
                return {"job_id": job_id, "status": "running"}
        else:
            return {"job_id": job_id, "status": "not_found"}


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.baseline_metrics = {}
        
    def update_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics."""
        for metric_name, value in metrics.items():
            self.metrics_history[metric_name].append({
                "value": value,
                "timestamp": datetime.now()
            })
            
    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline performance metrics."""
        self.baseline_metrics = metrics.copy()
        
    def has_degraded(self, current_metrics: Dict[str, float], 
                     threshold: float) -> bool:
        """
        Check if performance has degraded.
        
        Args:
            current_metrics: Current performance metrics
            threshold: Degradation threshold
            
        Returns:
            True if performance degraded
        """
        if not self.baseline_metrics:
            return False
            
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                
                # Check degradation based on metric type
                if metric_name in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
                    # Higher is better
                    if baseline_value - current_value > threshold:
                        return True
                elif metric_name in ["false_positive_rate", "latency"]:
                    # Lower is better
                    if current_value - baseline_value > threshold:
                        return True
                        
        return False
        
    def get_trend(self, metric_name: str, window: int = 100) -> Optional[float]:
        """
        Get trend for a metric.
        
        Args:
            metric_name: Name of metric
            window: Window size for trend calculation
            
        Returns:
            Trend slope or None
        """
        if metric_name not in self.metrics_history:
            return None
            
        history = list(self.metrics_history[metric_name])[-window:]
        
        if len(history) < 10:
            return None
            
        values = [h["value"] for h in history]
        x = np.arange(len(values))
        
        slope, _, _, _, _ = stats.linregress(x, values)
        
        return slope


class ABTestingFramework:
    """
    A/B testing framework for model comparison.
    Implements statistical testing for model performance comparison.
    """
    
    def __init__(self, config: ContinuousLearningConfig):
        self.config = config
        self.test_results = defaultdict(lambda: {"control": [], "treatment": []})
        self.active_tests = {}
        
    def create_test(self, test_id: str, 
                    control_model: Any, 
                    treatment_model: Any) -> Dict[str, Any]:
        """
        Create new A/B test.
        
        Args:
            test_id: Test identifier
            control_model: Control (current) model
            treatment_model: Treatment (new) model
            
        Returns:
            Test configuration
        """
        self.active_tests[test_id] = {
            "control_model": control_model,
            "treatment_model": treatment_model,
            "traffic_split": self.config.ab_test_traffic_split,
            "created_at": datetime.now(),
            "status": "active"
        }
        
        logger.info(f"Created A/B test {test_id}")
        
        return self.active_tests[test_id]
        
    def route_request(self, test_id: str) -> str:
        """
        Route request to control or treatment.
        
        Args:
            test_id: Test identifier
            
        Returns:
            "control" or "treatment"
        """
        if test_id not in self.active_tests:
            return "control"
            
        if self.active_tests[test_id]["status"] != "active":
            return "control"
            
        # Random routing based on traffic split
        if np.random.random() < self.config.ab_test_traffic_split:
            return "treatment"
        else:
            return "control"
            
    def record_result(self, test_id: str, 
                      variant: str, 
                      metric_name: str, 
                      value: float):
        """
        Record test result.
        
        Args:
            test_id: Test identifier
            variant: "control" or "treatment"
            metric_name: Metric name
            value: Metric value
        """
        key = f"{test_id}_{metric_name}"
        self.test_results[key][variant].append(value)
        
    def analyze_test(self, test_id: str, 
                     metric_name: str) -> Dict[str, Any]:
        """
        Analyze A/B test results.
        
        Args:
            test_id: Test identifier
            metric_name: Metric to analyze
            
        Returns:
            Statistical analysis results
        """
        key = f"{test_id}_{metric_name}"
        
        if key not in self.test_results:
            return {"error": "No results for test"}
            
        control_data = np.array(self.test_results[key]["control"])
        treatment_data = np.array(self.test_results[key]["treatment"])
        
        if len(control_data) < self.config.ab_test_min_samples or \
           len(treatment_data) < self.config.ab_test_min_samples:
            return {
                "status": "insufficient_data",
                "control_samples": len(control_data),
                "treatment_samples": len(treatment_data),
                "required_samples": self.config.ab_test_min_samples
            }
            
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(control_data, treatment_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((control_data.std()**2 + treatment_data.std()**2) / 2)
        effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
        
        # Determine winner
        significant = p_value < (1 - self.config.ab_test_confidence_level)
        
        if significant:
            if treatment_data.mean() > control_data.mean():
                winner = "treatment"
            else:
                winner = "control"
        else:
            winner = "no_difference"
            
        return {
            "status": "complete",
            "control_mean": float(control_data.mean()),
            "control_std": float(control_data.std()),
            "treatment_mean": float(treatment_data.mean()),
            "treatment_std": float(treatment_data.std()),
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "significant": significant,
            "winner": winner,
            "confidence_level": self.config.ab_test_confidence_level
        }
        
    def conclude_test(self, test_id: str) -> Dict[str, Any]:
        """
        Conclude A/B test and recommend model.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test conclusion and recommendation
        """
        if test_id not in self.active_tests:
            return {"error": "Test not found"}
            
        # Analyze all metrics
        results = {}
        for key in self.test_results:
            if key.startswith(test_id):
                metric_name = key.replace(f"{test_id}_", "")
                results[metric_name] = self.analyze_test(test_id, metric_name)
                
        # Determine overall winner
        treatment_wins = 0
        control_wins = 0
        
        for metric_result in results.values():
            if metric_result.get("winner") == "treatment":
                treatment_wins += 1
            elif metric_result.get("winner") == "control":
                control_wins += 1
                
        if treatment_wins > control_wins:
            recommendation = "deploy_treatment"
        else:
            recommendation = "keep_control"
            
        # Mark test as concluded
        self.active_tests[test_id]["status"] = "concluded"
        self.active_tests[test_id]["concluded_at"] = datetime.now()
        
        return {
            "test_id": test_id,
            "results": results,
            "treatment_wins": treatment_wins,
            "control_wins": control_wins,
            "recommendation": recommendation
        }


class IntegratedContinuousLearningPipeline:
    """
    Integrated continuous learning pipeline combining all components.
    Implements the complete continuous learning subsystem from Patent #4.
    """
    
    def __init__(self, model: nn.Module, 
                 config: Optional[ContinuousLearningConfig] = None):
        self.config = config or ContinuousLearningConfig()
        self.model = model
        
        # Initialize components
        self.feedback_collector = FeedbackCollector(self.config)
        self.online_learner = OnlineLearner(model, self.config)
        self.drift_detector = ConceptDriftDetector(self.config)
        self.retraining_orchestrator = AutomatedRetrainingOrchestrator(self.config)
        self.ab_testing = ABTestingFramework(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Model versioning
        self.model_versions = deque(maxlen=self.config.max_model_versions)
        self.current_version = "v1.0.0"
        
        logger.info("Initialized integrated continuous learning pipeline")
        
    def process_feedback(self, feedback: FeedbackEntry) -> bool:
        """
        Process incoming feedback.
        
        Args:
            feedback: Feedback entry
            
        Returns:
            True if feedback processed successfully
        """
        # Collect feedback
        if not self.feedback_collector.collect_feedback(feedback):
            return False
            
        # Check if online update needed
        feedback_count = self.feedback_collector.get_feedback_statistics()["total_feedback"]
        
        if feedback_count % self.config.online_update_frequency == 0:
            self._perform_online_update()
            
        # Check if retraining needed
        current_metrics = self._evaluate_current_model()
        
        if self.retraining_orchestrator.should_retrain(feedback_count, current_metrics):
            asyncio.create_task(self._trigger_full_retrain())
            
        return True
        
    def _perform_online_update(self):
        """Perform online model update with recent feedback."""
        # Get feedback batch
        feedback_batch = self.feedback_collector.get_feedback_batch(
            self.config.online_batch_size
        )
        
        if not feedback_batch:
            return
            
        # Convert feedback to training data
        features = []
        targets = []
        weights = []
        
        for feedback in feedback_batch:
            if feedback.metadata.get("features") is not None:
                features.append(feedback.metadata["features"])
                
                if feedback.feedback_type == FeedbackType.ACCURACY:
                    targets.append(int(feedback.value))
                    weights.append(feedback.confidence)
                    
        if features:
            features_tensor = torch.FloatTensor(features)
            targets_tensor = torch.LongTensor(targets)
            weights_tensor = torch.FloatTensor(weights)
            
            # Perform update
            update_metrics = self.online_learner.incremental_update(
                features_tensor, targets_tensor, weights_tensor
            )
            
            logger.info(f"Online update completed: {update_metrics}")
            
            # Check for drift
            if self.drift_detector.detect_drift(update_metrics["accuracy"]):
                logger.warning("Concept drift detected during online update")
                
    async def _trigger_full_retrain(self):
        """Trigger full model retraining."""
        # Prepare training data from feedback
        all_feedback = list(self.feedback_collector.feedback_buffer)
        
        features = []
        labels = []
        
        for feedback in all_feedback:
            if feedback.metadata.get("features") is not None and \
               feedback.feedback_type == FeedbackType.ACCURACY:
                features.append(feedback.metadata["features"])
                labels.append(int(feedback.value))
                
        if len(features) < self.config.min_feedback_for_retrain:
            logger.warning("Insufficient data for retraining")
            return
            
        training_data = (np.array(features), np.array(labels))
        
        # Trigger retraining
        job_id = await self.retraining_orchestrator.trigger_retraining(
            training_data,
            type(self.model),
            {"input_size": features[0].shape[0]}
        )
        
        logger.info(f"Full retraining triggered: job {job_id}")
        
    def _evaluate_current_model(self) -> Dict[str, float]:
        """Evaluate current model performance."""
        # This would typically evaluate on a validation set
        # For demo, return mock metrics
        return {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.91,
            "false_positive_rate": 0.11
        }
        
    def deploy_model_version(self, model: nn.Module, version: str):
        """
        Deploy new model version.
        
        Args:
            model: New model
            version: Version identifier
        """
        # Save current version
        self.model_versions.append({
            "version": self.current_version,
            "model": self.model,
            "deployed_at": datetime.now()
        })
        
        # Deploy new version
        self.model = model
        self.current_version = version
        self.online_learner.model = model
        
        # Set up A/B test if configured
        if self.config.ab_test_enabled and self.model_versions:
            previous_model = self.model_versions[-1]["model"]
            test_id = f"test_{version}"
            self.ab_testing.create_test(test_id, previous_model, model)
            
        logger.info(f"Deployed model version {version}")
        
    def rollback_model(self) -> bool:
        """
        Rollback to previous model version.
        
        Returns:
            True if rollback successful
        """
        if not self.config.model_rollback_enabled or not self.model_versions:
            return False
            
        previous = self.model_versions.pop()
        self.model = previous["model"]
        self.current_version = previous["version"]
        self.online_learner.model = previous["model"]
        
        logger.info(f"Rolled back to model version {self.current_version}")
        return True
        
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Pipeline status information
        """
        return {
            "current_model_version": self.current_version,
            "feedback_statistics": self.feedback_collector.get_feedback_statistics(),
            "feedback_bias": self.feedback_collector.detect_feedback_bias(),
            "online_updates": self.online_learner.update_count,
            "drift_detected": len(self.drift_detector.drift_history) > 0,
            "active_ab_tests": len(self.ab_testing.active_tests),
            "model_versions_stored": len(self.model_versions),
            "performance_trend": self.performance_monitor.get_trend("accuracy")
        }


if __name__ == "__main__":
    # Test the continuous learning pipeline
    import sys
    sys.path.append('..')
    from policy_compliance_predictor import create_policy_compliance_predictor
    
    # Create model
    config = {'input_size': 256, 'num_classes': 2}
    model = create_policy_compliance_predictor(config)
    
    # Create continuous learning pipeline
    cl_config = ContinuousLearningConfig(
        min_feedback_for_retrain=50,
        online_update_frequency=10,
        ab_test_enabled=True
    )
    
    pipeline = IntegratedContinuousLearningPipeline(model, cl_config)
    
    print("Testing Continuous Learning Pipeline...")
    
    # Simulate feedback collection
    for i in range(100):
        feedback = FeedbackEntry(
            feedback_id=f"fb_{i}",
            prediction_id=f"pred_{i}",
            feedback_type=FeedbackType.ACCURACY,
            value=np.random.choice([0, 1]),
            confidence=np.random.uniform(0.7, 1.0),
            user_id=f"user_{np.random.randint(1, 10)}",
            timestamp=datetime.now(),
            metadata={
                "features": np.random.randn(256).tolist()
            }
        )
        
        pipeline.process_feedback(feedback)
        
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    
    print(f"\nPipeline Status:")
    print(f"  Current model version: {status['current_model_version']}")
    print(f"  Total feedback: {status['feedback_statistics']['total_feedback']}")
    print(f"  Online updates: {status['online_updates']}")
    print(f"  Drift detected: {status['drift_detected']}")
    
    # Test A/B testing
    test_model = create_policy_compliance_predictor(config)
    pipeline.deploy_model_version(test_model, "v2.0.0")
    
    # Simulate A/B test results
    for i in range(200):
        variant = pipeline.ab_testing.route_request("test_v2.0.0")
        metric_value = np.random.normal(0.9, 0.05) if variant == "treatment" else np.random.normal(0.88, 0.05)
        pipeline.ab_testing.record_result("test_v2.0.0", variant, "accuracy", metric_value)
        
    # Analyze A/B test
    ab_results = pipeline.ab_testing.analyze_test("test_v2.0.0", "accuracy")
    
    print(f"\nA/B Test Results:")
    print(f"  Control mean: {ab_results.get('control_mean', 0):.4f}")
    print(f"  Treatment mean: {ab_results.get('treatment_mean', 0):.4f}")
    print(f"  P-value: {ab_results.get('p_value', 1):.4f}")
    print(f"  Winner: {ab_results.get('winner', 'unknown')}")
    
    print("\nContinuous learning pipeline test completed successfully!")