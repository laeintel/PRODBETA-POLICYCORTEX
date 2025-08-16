# File: ml/ab_testing.py
# A/B Testing Framework for PolicyCortex Models

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import hashlib
from enum import Enum
from scipy import stats
import json
import logging

logger = logging.getLogger(__name__)

class TestStatus(Enum):
    """Status of A/B test"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_name: str
    model_path: str
    created_at: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestConfig:
    """A/B test configuration"""
    test_id: str
    test_name: str
    model_a: ModelVersion
    model_b: ModelVersion
    traffic_split: float  # Percentage to model B
    min_sample_size: int
    confidence_level: float
    metrics_to_track: List[str]
    success_criteria: Dict[str, float]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_duration_days: int = 30

@dataclass
class TestResult:
    """Results of A/B test"""
    test_id: str
    winner: str
    confidence: float
    lift: Dict[str, float]
    sample_sizes: Dict[str, int]
    metrics_a: Dict[str, float]
    metrics_b: Dict[str, float]
    p_values: Dict[str, float]
    recommendations: List[str]

class ModelABTester:
    """A/B testing framework for model comparison"""
    
    def __init__(self):
        self.tests: Dict[str, TestConfig] = {}
        self.test_results: Dict[str, List[Dict]] = {}
        self.active_test: Optional[str] = None
        self.model_registry = {}
        self.default_confidence = 0.95
        
    def create_test(
        self, 
        test_name: str,
        model_a: ModelVersion,
        model_b: ModelVersion,
        traffic_split: float = 0.1,
        min_sample_size: int = 1000,
        confidence_level: float = 0.95,
        metrics_to_track: Optional[List[str]] = None,
        success_criteria: Optional[Dict[str, float]] = None
    ) -> str:
        """Create a new A/B test"""
        
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if metrics_to_track is None:
            metrics_to_track = ['accuracy', 'latency', 'error_rate']
        
        if success_criteria is None:
            success_criteria = {
                'accuracy': 0.05,  # 5% improvement
                'latency': -0.10,  # 10% reduction
                'error_rate': -0.20  # 20% reduction
            }
        
        test_config = TestConfig(
            test_id=test_id,
            test_name=test_name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            min_sample_size=min_sample_size,
            confidence_level=confidence_level,
            metrics_to_track=metrics_to_track,
            success_criteria=success_criteria
        )
        
        self.tests[test_id] = test_config
        self.test_results[test_id] = []
        
        logger.info(f"Created A/B test {test_id}: {test_name}")
        
        return test_id
    
    def start_test(self, test_id: str):
        """Start an A/B test"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        test.start_time = datetime.now()
        self.active_test = test_id
        
        # Register models
        self.model_registry[test.model_a.version_id] = self._load_model(test.model_a.model_path)
        self.model_registry[test.model_b.version_id] = self._load_model(test.model_b.model_path)
        
        logger.info(f"Started A/B test {test_id}")
    
    def predict_with_ab_test(
        self, 
        features: np.ndarray,
        user_id: Optional[str] = None
    ) -> Tuple[Any, str, str]:
        """Make prediction with A/B testing"""
        
        if self.active_test is None:
            raise ValueError("No active A/B test")
        
        test = self.tests[self.active_test]
        
        # Determine which model to use
        if user_id:
            # Consistent assignment based on user ID
            model_version = self._get_consistent_assignment(user_id, test.traffic_split)
        else:
            # Random assignment
            model_version = 'B' if random.random() < test.traffic_split else 'A'
        
        # Get the appropriate model
        if model_version == 'A':
            model = self.model_registry.get(test.model_a.version_id)
            model_id = test.model_a.version_id
        else:
            model = self.model_registry.get(test.model_b.version_id)
            model_id = test.model_b.version_id
        
        # Make prediction
        start_time = datetime.now()
        prediction = self._predict_with_model(model, features)
        latency = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        # Log prediction for analysis
        self._log_prediction(
            test_id=self.active_test,
            model_version=model_version,
            prediction=prediction,
            latency=latency,
            features=features
        )
        
        return prediction, model_version, model_id
    
    def _get_consistent_assignment(self, user_id: str, traffic_split: float) -> str:
        """Get consistent model assignment for a user"""
        # Hash user ID to get consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0
        
        return 'B' if assignment_value < traffic_split else 'A'
    
    def _predict_with_model(self, model: Any, features: np.ndarray) -> Any:
        """Make prediction with a specific model"""
        # In production, this would call the actual model
        # For demo, simulate prediction
        if model is None:
            return np.random.choice([0, 1])
        
        try:
            if hasattr(model, 'predict'):
                return model.predict(features)
            else:
                return np.random.choice([0, 1])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return np.random.choice([0, 1])
    
    def _log_prediction(
        self,
        test_id: str,
        model_version: str,
        prediction: Any,
        latency: float,
        features: np.ndarray
    ):
        """Log prediction for later analysis"""
        log_entry = {
            'timestamp': datetime.now(),
            'model_version': model_version,
            'prediction': prediction,
            'latency': latency,
            'features_hash': hashlib.md5(features.tobytes()).hexdigest()
        }
        
        self.test_results[test_id].append(log_entry)
    
    def record_feedback(
        self,
        test_id: str,
        prediction_id: str,
        actual_outcome: Any,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Record feedback for a prediction"""
        if test_id not in self.test_results:
            return
        
        # Find the prediction entry
        for entry in self.test_results[test_id]:
            if entry.get('prediction_id') == prediction_id:
                entry['actual_outcome'] = actual_outcome
                entry['correct'] = entry['prediction'] == actual_outcome
                
                if additional_metrics:
                    entry['metrics'] = additional_metrics
                
                break
    
    def analyze_test(self, test_id: str) -> TestResult:
        """Analyze A/B test results"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        results = self.test_results[test_id]
        
        if not results:
            return TestResult(
                test_id=test_id,
                winner="insufficient_data",
                confidence=0,
                lift={},
                sample_sizes={'A': 0, 'B': 0},
                metrics_a={},
                metrics_b={},
                p_values={},
                recommendations=["Need more data for analysis"]
            )
        
        # Separate results by model version
        results_a = [r for r in results if r['model_version'] == 'A']
        results_b = [r for r in results if r['model_version'] == 'B']
        
        # Calculate metrics
        metrics_a = self._calculate_metrics(results_a)
        metrics_b = self._calculate_metrics(results_b)
        
        # Perform statistical tests
        p_values = {}
        lift = {}
        
        for metric in test.metrics_to_track:
            if metric in metrics_a and metric in metrics_b:
                # Calculate p-value
                p_value = self._calculate_p_value(
                    results_a, results_b, metric
                )
                p_values[metric] = p_value
                
                # Calculate lift
                if metrics_a[metric] != 0:
                    lift[metric] = (metrics_b[metric] - metrics_a[metric]) / metrics_a[metric]
                else:
                    lift[metric] = 0
        
        # Determine winner
        winner, confidence = self._determine_winner(
            metrics_a, metrics_b, p_values, test.success_criteria
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            winner, confidence, lift, len(results_a), len(results_b)
        )
        
        return TestResult(
            test_id=test_id,
            winner=winner,
            confidence=confidence,
            lift=lift,
            sample_sizes={'A': len(results_a), 'B': len(results_b)},
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            p_values=p_values,
            recommendations=recommendations
        )
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics for a set of results"""
        if not results:
            return {}
        
        metrics = {}
        
        # Accuracy
        correct_predictions = [r for r in results if r.get('correct', False)]
        metrics['accuracy'] = len(correct_predictions) / len(results) if results else 0
        
        # Latency
        latencies = [r['latency'] for r in results if 'latency' in r]
        metrics['latency'] = np.mean(latencies) if latencies else 0
        
        # Error rate
        errors = [r for r in results if r.get('error', False)]
        metrics['error_rate'] = len(errors) / len(results) if results else 0
        
        # Custom metrics
        for result in results:
            if 'metrics' in result:
                for key, value in result['metrics'].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
        
        # Average custom metrics
        for key in list(metrics.keys()):
            if isinstance(metrics[key], list):
                metrics[key] = np.mean(metrics[key])
        
        return metrics
    
    def _calculate_p_value(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        metric: str
    ) -> float:
        """Calculate p-value for a metric"""
        
        # Extract metric values
        values_a = self._extract_metric_values(results_a, metric)
        values_b = self._extract_metric_values(results_b, metric)
        
        if not values_a or not values_b:
            return 1.0  # No significant difference
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            return p_value
        except:
            return 1.0
    
    def _extract_metric_values(self, results: List[Dict], metric: str) -> List[float]:
        """Extract values for a specific metric"""
        values = []
        
        for result in results:
            if metric == 'accuracy':
                values.append(1.0 if result.get('correct', False) else 0.0)
            elif metric == 'latency':
                if 'latency' in result:
                    values.append(result['latency'])
            elif metric == 'error_rate':
                values.append(1.0 if result.get('error', False) else 0.0)
            elif 'metrics' in result and metric in result['metrics']:
                values.append(result['metrics'][metric])
        
        return values
    
    def _determine_winner(
        self,
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        p_values: Dict[str, float],
        success_criteria: Dict[str, float]
    ) -> Tuple[str, float]:
        """Determine the winning model"""
        
        # Count significant improvements
        improvements = 0
        regressions = 0
        
        for metric, threshold in success_criteria.items():
            if metric in metrics_a and metric in metrics_b and metric in p_values:
                # Check if statistically significant
                if p_values[metric] < 0.05:
                    # Calculate relative change
                    if metrics_a[metric] != 0:
                        change = (metrics_b[metric] - metrics_a[metric]) / metrics_a[metric]
                    else:
                        change = 0
                    
                    # Check against success criteria
                    if threshold > 0:  # Expecting improvement
                        if change >= threshold:
                            improvements += 1
                        elif change < -threshold:
                            regressions += 1
                    else:  # Expecting reduction
                        if change <= threshold:
                            improvements += 1
                        elif change > -threshold:
                            regressions += 1
        
        # Determine winner
        if improvements > regressions:
            winner = 'model_b'
            confidence = min(0.95, 0.6 + improvements * 0.1)
        elif regressions > improvements:
            winner = 'model_a'
            confidence = min(0.95, 0.6 + regressions * 0.1)
        else:
            winner = 'no_winner'
            confidence = 0.5
        
        return winner, confidence
    
    def _generate_recommendations(
        self,
        winner: str,
        confidence: float,
        lift: Dict[str, float],
        sample_size_a: int,
        sample_size_b: int
    ) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check sample size
        min_sample = 100  # Minimum for reliable results
        if sample_size_a < min_sample or sample_size_b < min_sample:
            recommendations.append(
                f"Continue test - need at least {min_sample} samples per variant"
            )
        
        # Check winner
        if winner == 'model_b' and confidence > 0.8:
            recommendations.append(
                f"Strong evidence favors Model B - consider promoting to production"
            )
            
            # Specific improvements
            for metric, improvement in lift.items():
                if improvement > 0.1:
                    recommendations.append(
                        f"Model B shows {improvement*100:.1f}% improvement in {metric}"
                    )
        elif winner == 'model_a' and confidence > 0.8:
            recommendations.append(
                f"Model A performs better - keep current model"
            )
        elif winner == 'no_winner':
            recommendations.append(
                "No significant difference detected - consider extending test duration"
            )
        
        # Check for specific concerns
        if 'error_rate' in lift and lift['error_rate'] > 0.1:
            recommendations.append(
                "⚠️ Model B has higher error rate - investigate before deployment"
            )
        
        if 'latency' in lift and lift['latency'] > 0.2:
            recommendations.append(
                "⚠️ Model B has higher latency - may impact user experience"
            )
        
        return recommendations
    
    def stop_test(self, test_id: str) -> TestResult:
        """Stop an A/B test and return final results"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        test.end_time = datetime.now()
        
        if self.active_test == test_id:
            self.active_test = None
        
        # Analyze final results
        results = self.analyze_test(test_id)
        
        logger.info(f"Stopped A/B test {test_id}. Winner: {results.winner}")
        
        return results
    
    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of a test"""
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.tests[test_id]
        results = self.test_results[test_id]
        
        status = {
            'test_id': test_id,
            'test_name': test.test_name,
            'status': self._get_test_status(test),
            'start_time': test.start_time,
            'sample_count': len(results),
            'traffic_split': test.traffic_split,
            'duration_days': (datetime.now() - test.start_time).days if test.start_time else 0
        }
        
        # Add current metrics if running
        if self.active_test == test_id:
            current_results = self.analyze_test(test_id)
            status['current_metrics'] = {
                'model_a': current_results.metrics_a,
                'model_b': current_results.metrics_b,
                'p_values': current_results.p_values,
                'current_winner': current_results.winner
            }
        
        return status
    
    def _get_test_status(self, test: TestConfig) -> str:
        """Determine test status"""
        if test.end_time:
            return TestStatus.COMPLETED.value
        elif test.start_time:
            if self.active_test == test.test_id:
                return TestStatus.RUNNING.value
            else:
                return TestStatus.PAUSED.value
        else:
            return TestStatus.DRAFT.value
    
    def _load_model(self, model_path: str) -> Any:
        """Load a model from path"""
        # In production, would actually load the model
        # For demo, return None
        logger.info(f"Loading model from {model_path}")
        return None

class MultiArmedBandit:
    """Multi-armed bandit for dynamic model selection"""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.arms = {}
        self.rewards = {}
        self.counts = {}
        
    def add_arm(self, arm_id: str, model: Any):
        """Add a new arm (model) to the bandit"""
        self.arms[arm_id] = model
        self.rewards[arm_id] = []
        self.counts[arm_id] = 0
    
    def select_arm(self) -> str:
        """Select an arm using epsilon-greedy strategy"""
        if random.random() < self.epsilon:
            # Exploration: random selection
            return random.choice(list(self.arms.keys()))
        else:
            # Exploitation: select best performing
            if not any(self.counts.values()):
                return random.choice(list(self.arms.keys()))
            
            avg_rewards = {
                arm: np.mean(rewards) if rewards else 0
                for arm, rewards in self.rewards.items()
            }
            
            return max(avg_rewards, key=avg_rewards.get)
    
    def update_reward(self, arm_id: str, reward: float):
        """Update reward for an arm"""
        if arm_id in self.arms:
            self.rewards[arm_id].append(reward)
            self.counts[arm_id] += 1
            
            # Keep only last 1000 rewards
            if len(self.rewards[arm_id]) > 1000:
                self.rewards[arm_id] = self.rewards[arm_id][-1000:]

# Export main components
__all__ = [
    'ModelABTester',
    'MultiArmedBandit',
    'TestConfig',
    'TestResult',
    'ModelVersion',
    'TestStatus'
]