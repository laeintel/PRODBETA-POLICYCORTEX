"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/anomaly_detection.py
# Anomaly Detection System for PolicyCortex

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class Anomaly:
    """Detected anomaly information"""
    resource_id: str
    anomaly_type: str
    severity: str
    confidence: float
    timestamp: datetime
    metrics: Dict[str, float]
    explanation: str
    recommended_action: str
    related_anomalies: List[str]

@dataclass
class AnomalyPattern:
    """Pattern of related anomalies"""
    pattern_id: str
    anomalies: List[Anomaly]
    root_cause: str
    impact_scope: List[str]
    remediation_steps: List[str]

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoding(self, x):
        """Get the encoded representation"""
        return self.encoder(x)

class AnomalyDetector:
    """Multi-method anomaly detection system"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.threshold = 0.95
        self.history_window = deque(maxlen=1000)
        self.anomaly_patterns = {}
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize anomaly detection models"""
        # Initialize autoencoder (will be trained on data)
        self.autoencoder = Autoencoder(input_dim=20)
        
        # Initialize DBSCAN for clustering anomalies
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        
        # Initialize PCA for dimensionality reduction
        self.pca = PCA(n_components=10)
    
    def detect_anomalies(self, resource_metrics: Dict[str, Any]) -> List[Anomaly]:
        """Detect anomalies in resource metrics"""
        anomalies = []
        
        # Convert metrics to feature array
        features = self._extract_features(resource_metrics)
        
        # Method 1: Isolation Forest
        isolation_anomalies = self._detect_isolation_forest(features, resource_metrics)
        anomalies.extend(isolation_anomalies)
        
        # Method 2: Autoencoder reconstruction error
        autoencoder_anomalies = self._detect_autoencoder(features, resource_metrics)
        anomalies.extend(autoencoder_anomalies)
        
        # Method 3: Statistical anomalies
        statistical_anomalies = self._detect_statistical(features, resource_metrics)
        anomalies.extend(statistical_anomalies)
        
        # Method 4: Pattern-based anomalies
        pattern_anomalies = self._detect_pattern_based(resource_metrics)
        anomalies.extend(pattern_anomalies)
        
        # Combine and deduplicate anomalies
        anomalies = self._combine_anomalies(anomalies)
        
        # Cluster related anomalies
        if anomalies:
            self._cluster_anomalies(anomalies)
        
        # Update history
        self.history_window.append({
            'timestamp': datetime.now(),
            'metrics': resource_metrics,
            'anomalies': len(anomalies)
        })
        
        return anomalies
    
    def _extract_features(self, metrics: Dict[str, Any]) -> np.ndarray:
        """Extract numeric features from metrics"""
        features = []
        
        # CPU and Memory metrics
        features.append(metrics.get('cpu_utilization', 0))
        features.append(metrics.get('memory_utilization', 0))
        features.append(metrics.get('disk_utilization', 0))
        
        # Network metrics
        features.append(metrics.get('network_in_bytes', 0))
        features.append(metrics.get('network_out_bytes', 0))
        features.append(metrics.get('packet_loss_rate', 0))
        features.append(metrics.get('latency_ms', 0))
        
        # Error and request metrics
        features.append(metrics.get('error_rate', 0))
        features.append(metrics.get('request_rate', 0))
        features.append(metrics.get('response_time_ms', 0))
        
        # Resource-specific metrics
        features.append(metrics.get('active_connections', 0))
        features.append(metrics.get('queue_depth', 0))
        features.append(metrics.get('cache_hit_rate', 0))
        
        # Cost and compliance metrics
        features.append(metrics.get('hourly_cost', 0))
        features.append(metrics.get('compliance_score', 100))
        features.append(metrics.get('security_score', 100))
        
        # Time-based features
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        features.append(hour)
        features.append(day_of_week)
        
        # Derived features
        features.append(metrics.get('cpu_memory_ratio', features[0] / (features[1] + 1)))
        features.append(metrics.get('io_ratio', features[3] / (features[4] + 1)))
        
        return np.array(features).reshape(1, -1)
    
    def _detect_isolation_forest(
        self, features: np.ndarray, metrics: Dict[str, Any]
    ) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest"""
        anomalies = []
        
        try:
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit and predict (in production, would use pre-trained model)
            self.isolation_forest.fit(scaled_features)
            outlier_scores = self.isolation_forest.decision_function(scaled_features)
            predictions = self.isolation_forest.predict(scaled_features)
            
            if predictions[0] == -1:  # Anomaly detected
                anomaly_score = abs(outlier_scores[0])
                severity = self._determine_severity(anomaly_score)
                
                anomalies.append(Anomaly(
                    resource_id=metrics.get('resource_id', 'unknown'),
                    anomaly_type='isolation_forest',
                    severity=severity,
                    confidence=min(anomaly_score, 1.0),
                    timestamp=datetime.now(),
                    metrics=self._get_anomalous_metrics(features[0], metrics),
                    explanation=f"Resource behavior deviates significantly from normal patterns (score: {anomaly_score:.2f})",
                    recommended_action="Investigate resource configuration and recent changes",
                    related_anomalies=[]
                ))
        except Exception as e:
            logger.error(f"Isolation Forest detection failed: {e}")
        
        return anomalies
    
    def _detect_autoencoder(
        self, features: np.ndarray, metrics: Dict[str, Any]
    ) -> List[Anomaly]:
        """Detect anomalies using autoencoder reconstruction error"""
        anomalies = []
        
        if self.autoencoder is None:
            return anomalies
        
        try:
            # Convert to tensor
            features_tensor = torch.FloatTensor(features)
            
            # Get reconstruction
            self.autoencoder.eval()
            with torch.no_grad():
                reconstructed = self.autoencoder(features_tensor)
            
            # Calculate reconstruction error
            reconstruction_error = F.mse_loss(reconstructed, features_tensor).item()
            
            # Determine if anomaly based on error threshold
            if reconstruction_error > self.threshold:
                severity = self._determine_severity(reconstruction_error / self.threshold)
                
                anomalies.append(Anomaly(
                    resource_id=metrics.get('resource_id', 'unknown'),
                    anomaly_type='autoencoder',
                    severity=severity,
                    confidence=min((reconstruction_error / self.threshold) * 0.8, 1.0),
                    timestamp=datetime.now(),
                    metrics=self._get_reconstruction_diff(features[0], reconstructed.numpy()[0], metrics),
                    explanation=f"Unusual pattern detected with reconstruction error: {reconstruction_error:.3f}",
                    recommended_action="Review resource metrics for unusual combinations",
                    related_anomalies=[]
                ))
        except Exception as e:
            logger.error(f"Autoencoder detection failed: {e}")
        
        return anomalies
    
    def _detect_statistical(
        self, features: np.ndarray, metrics: Dict[str, Any]
    ) -> List[Anomaly]:
        """Detect statistical anomalies using z-scores and thresholds"""
        anomalies = []
        
        # Check each metric against statistical thresholds
        critical_metrics = {
            'cpu_utilization': {'high': 90, 'low': 5},
            'memory_utilization': {'high': 85, 'low': 10},
            'error_rate': {'high': 5, 'low': -1},
            'response_time_ms': {'high': 1000, 'low': -1},
            'packet_loss_rate': {'high': 1, 'low': -1}
        }
        
        for metric_name, thresholds in critical_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Check high threshold
                if thresholds['high'] > 0 and value > thresholds['high']:
                    anomalies.append(Anomaly(
                        resource_id=metrics.get('resource_id', 'unknown'),
                        anomaly_type='statistical_threshold',
                        severity='high' if value > thresholds['high'] * 1.2 else 'medium',
                        confidence=0.95,
                        timestamp=datetime.now(),
                        metrics={metric_name: value},
                        explanation=f"{metric_name} exceeded threshold: {value:.2f} > {thresholds['high']}",
                        recommended_action=self._get_metric_recommendation(metric_name, 'high'),
                        related_anomalies=[]
                    ))
                
                # Check low threshold
                if thresholds['low'] >= 0 and value < thresholds['low']:
                    anomalies.append(Anomaly(
                        resource_id=metrics.get('resource_id', 'unknown'),
                        anomaly_type='statistical_threshold',
                        severity='medium',
                        confidence=0.90,
                        timestamp=datetime.now(),
                        metrics={metric_name: value},
                        explanation=f"{metric_name} below threshold: {value:.2f} < {thresholds['low']}",
                        recommended_action=self._get_metric_recommendation(metric_name, 'low'),
                        related_anomalies=[]
                    ))
        
        # Check for sudden changes (using history)
        if len(self.history_window) > 10:
            recent_history = list(self.history_window)[-10:]
            for metric_name in ['cpu_utilization', 'memory_utilization', 'error_rate']:
                if metric_name in metrics:
                    historical_values = [h['metrics'].get(metric_name, 0) for h in recent_history]
                    current_value = metrics[metric_name]
                    
                    if historical_values:
                        mean_val = np.mean(historical_values)
                        std_val = np.std(historical_values)
                        
                        if std_val > 0:
                            z_score = abs((current_value - mean_val) / std_val)
                            
                            if z_score > 3:
                                anomalies.append(Anomaly(
                                    resource_id=metrics.get('resource_id', 'unknown'),
                                    anomaly_type='statistical_spike',
                                    severity='high' if z_score > 4 else 'medium',
                                    confidence=min(0.7 + z_score * 0.1, 1.0),
                                    timestamp=datetime.now(),
                                    metrics={metric_name: current_value, 'z_score': z_score},
                                    explanation=f"Sudden change in {metric_name}: z-score = {z_score:.2f}",
                                    recommended_action="Investigate recent changes or events",
                                    related_anomalies=[]
                                ))
        
        return anomalies
    
    def _detect_pattern_based(self, metrics: Dict[str, Any]) -> List[Anomaly]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        # Pattern 1: High CPU with low memory (possible CPU bottleneck)
        if metrics.get('cpu_utilization', 0) > 80 and metrics.get('memory_utilization', 100) < 30:
            anomalies.append(Anomaly(
                resource_id=metrics.get('resource_id', 'unknown'),
                anomaly_type='pattern_cpu_bottleneck',
                severity='medium',
                confidence=0.85,
                timestamp=datetime.now(),
                metrics={'cpu': metrics['cpu_utilization'], 'memory': metrics.get('memory_utilization')},
                explanation="CPU bottleneck detected - high CPU with low memory usage",
                recommended_action="Consider CPU optimization or scaling",
                related_anomalies=[]
            ))
        
        # Pattern 2: High error rate with normal load (possible application issue)
        if (metrics.get('error_rate', 0) > 5 and 
            metrics.get('cpu_utilization', 0) < 50 and 
            metrics.get('request_rate', 0) < metrics.get('avg_request_rate', 100)):
            anomalies.append(Anomaly(
                resource_id=metrics.get('resource_id', 'unknown'),
                anomaly_type='pattern_application_error',
                severity='high',
                confidence=0.90,
                timestamp=datetime.now(),
                metrics={'error_rate': metrics['error_rate'], 'load': metrics.get('cpu_utilization')},
                explanation="Application errors detected without corresponding load increase",
                recommended_action="Check application logs and recent deployments",
                related_anomalies=[]
            ))
        
        # Pattern 3: Network anomaly (high packet loss with normal bandwidth)
        if (metrics.get('packet_loss_rate', 0) > 1 and 
            metrics.get('network_in_bytes', 0) < metrics.get('network_capacity', float('inf')) * 0.5):
            anomalies.append(Anomaly(
                resource_id=metrics.get('resource_id', 'unknown'),
                anomaly_type='pattern_network_issue',
                severity='high',
                confidence=0.88,
                timestamp=datetime.now(),
                metrics={'packet_loss': metrics['packet_loss_rate'], 'bandwidth_usage': metrics.get('network_in_bytes')},
                explanation="Network quality issues detected - packet loss without congestion",
                recommended_action="Check network configuration and connectivity",
                related_anomalies=[]
            ))
        
        # Pattern 4: Cost anomaly (sudden cost increase without resource changes)
        if (metrics.get('hourly_cost', 0) > metrics.get('avg_hourly_cost', 0) * 1.5 and
            metrics.get('instance_count', 1) == metrics.get('prev_instance_count', 1)):
            anomalies.append(Anomaly(
                resource_id=metrics.get('resource_id', 'unknown'),
                anomaly_type='pattern_cost_spike',
                severity='medium',
                confidence=0.80,
                timestamp=datetime.now(),
                metrics={'current_cost': metrics['hourly_cost'], 'expected_cost': metrics.get('avg_hourly_cost')},
                explanation="Unexpected cost increase without resource scaling",
                recommended_action="Review resource configuration and usage patterns",
                related_anomalies=[]
            ))
        
        return anomalies
    
    def _combine_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Combine and deduplicate anomalies"""
        combined = {}
        
        for anomaly in anomalies:
            key = f"{anomaly.resource_id}_{anomaly.anomaly_type}"
            
            if key not in combined:
                combined[key] = anomaly
            else:
                # Merge anomalies - keep higher confidence
                if anomaly.confidence > combined[key].confidence:
                    combined[key] = anomaly
                    
                # Combine related anomalies
                combined[key].related_anomalies.extend(anomaly.related_anomalies)
        
        return list(combined.values())
    
    def _cluster_anomalies(self, anomalies: List[Anomaly]):
        """Cluster related anomalies to identify patterns"""
        if len(anomalies) < 2:
            return
        
        # Extract features for clustering
        anomaly_features = []
        for anomaly in anomalies:
            features = [
                hash(anomaly.anomaly_type) % 100,
                anomaly.confidence,
                self._encode_severity(anomaly.severity),
                len(anomaly.metrics)
            ]
            anomaly_features.append(features)
        
        # Perform clustering
        try:
            clusters = self.dbscan.fit_predict(anomaly_features)
            
            # Group anomalies by cluster
            for i, cluster_id in enumerate(clusters):
                if cluster_id != -1:  # Not noise
                    # Find other anomalies in same cluster
                    related = [
                        anomalies[j].resource_id 
                        for j, c in enumerate(clusters) 
                        if c == cluster_id and i != j
                    ]
                    anomalies[i].related_anomalies.extend(related)
        except Exception as e:
            logger.error(f"Anomaly clustering failed: {e}")
    
    def _determine_severity(self, score: float) -> str:
        """Determine anomaly severity based on score"""
        if score > 2.0:
            return 'critical'
        elif score > 1.5:
            return 'high'
        elif score > 1.0:
            return 'medium'
        else:
            return 'low'
    
    def _encode_severity(self, severity: str) -> float:
        """Encode severity as numeric value"""
        severity_map = {
            'critical': 4.0,
            'high': 3.0,
            'medium': 2.0,
            'low': 1.0
        }
        return severity_map.get(severity, 1.0)
    
    def _get_anomalous_metrics(
        self, features: np.ndarray, metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract the most anomalous metrics"""
        anomalous = {}
        
        # Find metrics that deviate most from normal
        feature_names = ['cpu', 'memory', 'disk', 'network_in', 'network_out', 
                        'packet_loss', 'latency', 'error_rate', 'requests', 'response_time']
        
        for i, name in enumerate(feature_names[:len(features)]):
            if i < len(features):
                value = features[i]
                # Simple threshold check
                if name in ['cpu', 'memory', 'disk'] and value > 80:
                    anomalous[name] = value
                elif name in ['error_rate', 'packet_loss'] and value > 1:
                    anomalous[name] = value
                elif name in ['latency', 'response_time'] and value > 500:
                    anomalous[name] = value
        
        return anomalous
    
    def _get_reconstruction_diff(
        self, original: np.ndarray, reconstructed: np.ndarray, metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get metrics with largest reconstruction differences"""
        diff_metrics = {}
        
        feature_names = ['cpu', 'memory', 'disk', 'network_in', 'network_out']
        
        for i, name in enumerate(feature_names):
            if i < len(original) and i < len(reconstructed):
                diff = abs(original[i] - reconstructed[i])
                if diff > 10:  # Significant difference
                    diff_metrics[name] = {
                        'original': original[i],
                        'reconstructed': reconstructed[i],
                        'difference': diff
                    }
        
        return diff_metrics
    
    def _get_metric_recommendation(self, metric_name: str, condition: str) -> str:
        """Get recommendation for specific metric condition"""
        recommendations = {
            'cpu_utilization': {
                'high': "Scale up compute resources or optimize CPU-intensive operations",
                'low': "Consider scaling down to reduce costs"
            },
            'memory_utilization': {
                'high': "Increase memory allocation or optimize memory usage",
                'low': "Review memory allocation for cost optimization"
            },
            'error_rate': {
                'high': "Investigate application errors and recent deployments",
                'low': "System operating normally"
            },
            'response_time_ms': {
                'high': "Optimize application performance or scale resources",
                'low': "System performing well"
            },
            'packet_loss_rate': {
                'high': "Check network configuration and connectivity",
                'low': "Network operating normally"
            }
        }
        
        return recommendations.get(metric_name, {}).get(condition, "Review resource configuration")
    
    def train_autoencoder(self, training_data: np.ndarray, epochs: int = 100):
        """Train the autoencoder on normal data"""
        if self.autoencoder is None:
            return
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(training_data)
        
        # Training setup
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.info(f"Autoencoder training - Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Calculate threshold based on training data
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(data_tensor)
            errors = F.mse_loss(reconstructed, data_tensor, reduction='none').mean(dim=1)
            self.threshold = errors.mean().item() + 2 * errors.std().item()
        
        logger.info(f"Autoencoder trained. Threshold set to: {self.threshold:.4f}")

class AnomalyCorrelator:
    """Correlate anomalies across resources to identify root causes"""
    
    def __init__(self):
        self.correlation_window = timedelta(minutes=5)
        self.min_correlation_score = 0.7
        
    def correlate_anomalies(self, anomalies: List[Anomaly]) -> List[AnomalyPattern]:
        """Correlate anomalies to identify patterns and root causes"""
        patterns = []
        
        # Group anomalies by time window
        time_groups = self._group_by_time(anomalies)
        
        for timestamp, group in time_groups.items():
            if len(group) > 1:
                # Analyze the group for patterns
                pattern = self._analyze_anomaly_group(group)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _group_by_time(self, anomalies: List[Anomaly]) -> Dict[datetime, List[Anomaly]]:
        """Group anomalies that occurred within the correlation window"""
        groups = {}
        
        for anomaly in anomalies:
            # Find or create group
            group_found = False
            for group_time in groups:
                if abs((anomaly.timestamp - group_time).total_seconds()) < self.correlation_window.total_seconds():
                    groups[group_time].append(anomaly)
                    group_found = True
                    break
            
            if not group_found:
                groups[anomaly.timestamp] = [anomaly]
        
        return groups
    
    def _analyze_anomaly_group(self, anomalies: List[Anomaly]) -> Optional[AnomalyPattern]:
        """Analyze a group of anomalies to identify patterns"""
        if len(anomalies) < 2:
            return None
        
        # Identify common characteristics
        anomaly_types = [a.anomaly_type for a in anomalies]
        severities = [a.severity for a in anomalies]
        
        # Determine pattern type and root cause
        root_cause = self._identify_root_cause(anomalies)
        impact_scope = list(set([a.resource_id for a in anomalies]))
        
        # Generate remediation steps
        remediation_steps = self._generate_remediation_steps(anomalies, root_cause)
        
        return AnomalyPattern(
            pattern_id=f"PAT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            anomalies=anomalies,
            root_cause=root_cause,
            impact_scope=impact_scope,
            remediation_steps=remediation_steps
        )
    
    def _identify_root_cause(self, anomalies: List[Anomaly]) -> str:
        """Identify the likely root cause of correlated anomalies"""
        # Simple heuristic-based root cause analysis
        
        # Check for cascade patterns
        if any('cpu' in a.anomaly_type for a in anomalies) and any('memory' in a.anomaly_type for a in anomalies):
            return "Resource exhaustion - CPU and memory constraints detected"
        
        if any('network' in a.anomaly_type for a in anomalies) and len(anomalies) > 3:
            return "Network connectivity issues affecting multiple resources"
        
        if any('application_error' in a.anomaly_type for a in anomalies):
            return "Application failure causing downstream impacts"
        
        if all(a.severity in ['high', 'critical'] for a in anomalies):
            return "Critical system-wide issue requiring immediate attention"
        
        return "Multiple correlated anomalies detected - investigation required"
    
    def _generate_remediation_steps(self, anomalies: List[Anomaly], root_cause: str) -> List[str]:
        """Generate remediation steps based on anomalies and root cause"""
        steps = []
        
        # Add steps based on root cause
        if "resource exhaustion" in root_cause.lower():
            steps.extend([
                "1. Identify resource-intensive processes",
                "2. Scale up affected resources or optimize usage",
                "3. Implement auto-scaling policies",
                "4. Review and optimize application code"
            ])
        elif "network" in root_cause.lower():
            steps.extend([
                "1. Check network connectivity and routing",
                "2. Review firewall and security group rules",
                "3. Verify DNS resolution",
                "4. Check for DDoS or unusual traffic patterns"
            ])
        elif "application" in root_cause.lower():
            steps.extend([
                "1. Check application logs for errors",
                "2. Review recent deployments or configuration changes",
                "3. Rollback if necessary",
                "4. Scale application instances if needed"
            ])
        else:
            steps.extend([
                "1. Review all anomaly details",
                "2. Check system logs and metrics",
                "3. Identify common factors",
                "4. Apply targeted remediation"
            ])
        
        return steps

# Export main components
__all__ = [
    'AnomalyDetector',
    'AnomalyCorrelator',
    'Anomaly',
    'AnomalyPattern',
    'Autoencoder'
]