"""
Patent #4: Predictive Policy Compliance Engine
Feature Engineering Pipeline

This module implements the multi-modal feature extraction subsystem as specified in Patent #4,
processing configuration telemetry, policy definitions, temporal patterns, and contextual metadata.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
import hashlib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""
    # Temporal features
    temporal_window_sizes: List[int] = field(default_factory=lambda: [1, 7, 30, 90])
    lag_features: int = 10
    rolling_statistics: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max', 'median'])
    
    # Configuration features
    config_embedding_dim: int = 256
    config_hash_features: bool = True
    config_categorical_encoding: str = 'onehot'  # 'onehot', 'target', 'ordinal'
    
    # Policy features
    policy_embedding_dim: int = 128
    policy_complexity_metrics: bool = True
    policy_graph_features: bool = True
    
    # Context features
    business_criticality_levels: int = 5
    dependency_graph_features: bool = True
    organizational_hierarchy_depth: int = 3
    
    # Feature reduction
    use_pca: bool = False
    pca_components: int = 100
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'chi2', 'anova'
    
    # Normalization
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Caching
    cache_features: bool = True
    cache_ttl_seconds: int = 3600


class ConfigurationFeatureExtractor:
    """
    Extract features from security configuration parameters.
    Processes encryption settings, access controls, network configs, and resource metadata.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.scaler = self._create_scaler()
        self.encoder_cache = {}
        
    def _create_scaler(self):
        """Create appropriate scaler based on configuration."""
        if self.config.scaler_type == 'standard':
            return StandardScaler()
        elif self.config.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.config.scaler_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
            
    def extract_features(self, configuration: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from configuration dictionary.
        
        Args:
            configuration: Resource configuration dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Encryption features
        encryption_features = self._extract_encryption_features(
            configuration.get('encryption', {})
        )
        features.extend(encryption_features)
        
        # Access control features
        access_features = self._extract_access_control_features(
            configuration.get('access_control', {})
        )
        features.extend(access_features)
        
        # Network configuration features
        network_features = self._extract_network_features(
            configuration.get('network', {})
        )
        features.extend(network_features)
        
        # Resource metadata features
        metadata_features = self._extract_metadata_features(
            configuration.get('metadata', {})
        )
        features.extend(metadata_features)
        
        # Configuration complexity metrics
        complexity_features = self._calculate_complexity_metrics(configuration)
        features.extend(complexity_features)
        
        # Hash-based features for detecting changes
        if self.config.config_hash_features:
            hash_features = self._extract_hash_features(configuration)
            features.extend(hash_features)
            
        return np.array(features, dtype=np.float32)
        
    def _extract_encryption_features(self, encryption_config: Dict) -> List[float]:
        """Extract encryption-related features."""
        features = []
        
        # Encryption enabled
        features.append(float(encryption_config.get('enabled', False)))
        
        # Encryption algorithm strength (mapped to numeric)
        algorithm_strength = {
            'AES-256': 1.0,
            'AES-128': 0.7,
            'RSA-2048': 0.8,
            'RSA-4096': 1.0,
            'None': 0.0
        }
        algorithm = encryption_config.get('algorithm', 'None')
        features.append(algorithm_strength.get(algorithm, 0.5))
        
        # Key rotation
        features.append(float(encryption_config.get('key_rotation_enabled', False)))
        rotation_days = encryption_config.get('key_rotation_days', 365)
        features.append(1.0 / (1.0 + rotation_days))  # Normalize
        
        # Encryption at rest and in transit
        features.append(float(encryption_config.get('at_rest', False)))
        features.append(float(encryption_config.get('in_transit', False)))
        
        # Certificate validation
        features.append(float(encryption_config.get('certificate_validation', False)))
        
        # TLS version (higher is better)
        tls_versions = {'TLS1.0': 0.2, 'TLS1.1': 0.4, 'TLS1.2': 0.7, 'TLS1.3': 1.0}
        tls_version = encryption_config.get('tls_version', 'TLS1.0')
        features.append(tls_versions.get(tls_version, 0.5))
        
        return features
        
    def _extract_access_control_features(self, access_config: Dict) -> List[float]:
        """Extract access control features."""
        features = []
        
        # Authentication type
        auth_types = {'none': 0.0, 'basic': 0.3, 'oauth': 0.7, 'mfa': 1.0}
        auth_type = access_config.get('authentication_type', 'none')
        features.append(auth_types.get(auth_type, 0.5))
        
        # MFA enabled
        features.append(float(access_config.get('mfa_enabled', False)))
        
        # Number of roles
        n_roles = len(access_config.get('roles', []))
        features.append(n_roles / 100.0)  # Normalize
        
        # Privilege level distribution
        privileges = access_config.get('privilege_distribution', {})
        features.append(privileges.get('admin', 0) / 100.0)
        features.append(privileges.get('write', 0) / 100.0)
        features.append(privileges.get('read', 0) / 100.0)
        
        # Conditional access policies
        features.append(len(access_config.get('conditional_policies', [])) / 10.0)
        
        # Session timeout
        timeout_minutes = access_config.get('session_timeout_minutes', 60)
        features.append(1.0 / (1.0 + timeout_minutes / 60.0))
        
        # IP restrictions
        features.append(float(access_config.get('ip_restrictions_enabled', False)))
        
        return features
        
    def _extract_network_features(self, network_config: Dict) -> List[float]:
        """Extract network configuration features."""
        features = []
        
        # Public/Private exposure
        features.append(float(network_config.get('public_access', False)))
        features.append(float(network_config.get('private_endpoint', False)))
        
        # Firewall rules
        n_inbound_rules = len(network_config.get('inbound_rules', []))
        n_outbound_rules = len(network_config.get('outbound_rules', []))
        features.append(n_inbound_rules / 100.0)
        features.append(n_outbound_rules / 100.0)
        
        # Port exposure
        exposed_ports = network_config.get('exposed_ports', [])
        high_risk_ports = [22, 23, 3389, 445, 135, 139]  # SSH, Telnet, RDP, SMB, etc.
        n_high_risk = sum(1 for port in exposed_ports if port in high_risk_ports)
        features.append(n_high_risk / 10.0)
        features.append(len(exposed_ports) / 100.0)
        
        # Network segmentation
        features.append(float(network_config.get('network_segmentation', False)))
        n_subnets = network_config.get('n_subnets', 1)
        features.append(n_subnets / 10.0)
        
        # DDoS protection
        features.append(float(network_config.get('ddos_protection', False)))
        
        # VPN configuration
        features.append(float(network_config.get('vpn_enabled', False)))
        
        return features
        
    def _extract_metadata_features(self, metadata: Dict) -> List[float]:
        """Extract resource metadata features."""
        features = []
        
        # Resource type encoding
        resource_types = {
            'compute': 0.2, 'storage': 0.4, 'network': 0.6, 
            'database': 0.8, 'container': 1.0
        }
        resource_type = metadata.get('resource_type', 'unknown')
        features.append(resource_types.get(resource_type, 0.5))
        
        # Environment
        environments = {'dev': 0.2, 'test': 0.4, 'staging': 0.6, 'prod': 1.0}
        environment = metadata.get('environment', 'dev')
        features.append(environments.get(environment, 0.3))
        
        # Age of resource (days)
        created_date = metadata.get('created_date')
        if created_date:
            age_days = (datetime.now() - pd.to_datetime(created_date)).days
            features.append(age_days / 365.0)  # Normalize to years
        else:
            features.append(0.0)
            
        # Last modified (days ago)
        modified_date = metadata.get('last_modified')
        if modified_date:
            days_since_modified = (datetime.now() - pd.to_datetime(modified_date)).days
            features.append(1.0 / (1.0 + days_since_modified))
        else:
            features.append(0.0)
            
        # Tags count
        n_tags = len(metadata.get('tags', {}))
        features.append(n_tags / 50.0)
        
        # Compliance tags
        compliance_tags = ['compliant', 'gdpr', 'hipaa', 'pci', 'sox']
        tags = metadata.get('tags', {})
        for tag in compliance_tags:
            features.append(float(tag in tags))
            
        return features
        
    def _calculate_complexity_metrics(self, configuration: Dict) -> List[float]:
        """Calculate configuration complexity metrics."""
        features = []
        
        # Configuration depth (nested levels)
        depth = self._calculate_dict_depth(configuration)
        features.append(depth / 10.0)
        
        # Number of configuration keys
        n_keys = self._count_keys(configuration)
        features.append(n_keys / 1000.0)
        
        # Configuration entropy (diversity of values)
        entropy = self._calculate_configuration_entropy(configuration)
        features.append(entropy / 10.0)
        
        # Number of unique values
        unique_values = len(set(str(v) for v in self._flatten_dict(configuration).values()))
        features.append(unique_values / 100.0)
        
        return features
        
    def _extract_hash_features(self, configuration: Dict) -> List[float]:
        """Extract hash-based features for change detection."""
        features = []
        
        # Full configuration hash (converted to numeric features)
        config_str = json.dumps(configuration, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Use first 8 bytes of hash as features
        for i in range(0, 16, 2):
            byte_val = int(config_hash[i:i+2], 16)
            features.append(byte_val / 255.0)
            
        # Section-wise hashes
        for section in ['encryption', 'access_control', 'network', 'metadata']:
            if section in configuration:
                section_str = json.dumps(configuration[section], sort_keys=True)
                section_hash = hashlib.md5(section_str.encode()).hexdigest()
                # Use first 2 bytes
                features.append(int(section_hash[:4], 16) / 65535.0)
            else:
                features.append(0.0)
                
        return features
        
    def _calculate_dict_depth(self, d: Dict, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary."""
        if not isinstance(d, dict):
            return current_depth
        if not d:
            return current_depth
        return max(self._calculate_dict_depth(v, current_depth + 1) 
                  for v in d.values())
        
    def _count_keys(self, d: Dict) -> int:
        """Count total number of keys in nested dictionary."""
        count = len(d)
        for value in d.values():
            if isinstance(value, dict):
                count += self._count_keys(value)
        return count
        
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _calculate_configuration_entropy(self, configuration: Dict) -> float:
        """Calculate Shannon entropy of configuration values."""
        flat_config = self._flatten_dict(configuration)
        values = [str(v) for v in flat_config.values()]
        
        if not values:
            return 0.0
            
        # Count occurrences
        value_counts = defaultdict(int)
        for v in values:
            value_counts[v] += 1
            
        # Calculate entropy
        total = len(values)
        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
                
        return entropy


class TemporalFeatureExtractor:
    """
    Extract temporal features from configuration and compliance time series.
    Captures change velocity, frequency patterns, seasonal trends, and temporal correlations.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.seasonal_decomposer = None
        
    def extract_features(self, time_series: pd.DataFrame) -> np.ndarray:
        """
        Extract temporal features from time series data.
        
        Args:
            time_series: DataFrame with timestamp index and feature columns
            
        Returns:
            Temporal feature vector
        """
        features = []
        
        # Change velocity features
        velocity_features = self._extract_change_velocity(time_series)
        features.extend(velocity_features)
        
        # Frequency pattern features
        frequency_features = self._extract_frequency_patterns(time_series)
        features.extend(frequency_features)
        
        # Seasonal trend features
        seasonal_features = self._extract_seasonal_trends(time_series)
        features.extend(seasonal_features)
        
        # Temporal correlation features
        correlation_features = self._extract_temporal_correlations(time_series)
        features.extend(correlation_features)
        
        # Lag features
        lag_features = self._extract_lag_features(time_series)
        features.extend(lag_features)
        
        # Rolling statistics
        rolling_features = self._extract_rolling_statistics(time_series)
        features.extend(rolling_features)
        
        return np.array(features, dtype=np.float32)
        
    def _extract_change_velocity(self, time_series: pd.DataFrame) -> List[float]:
        """Extract rate of change features."""
        features = []
        
        for col in time_series.columns:
            if pd.api.types.is_numeric_dtype(time_series[col]):
                # First derivative (velocity)
                velocity = time_series[col].diff()
                features.append(velocity.mean())
                features.append(velocity.std())
                features.append(velocity.abs().mean())
                
                # Second derivative (acceleration)
                acceleration = velocity.diff()
                features.append(acceleration.mean())
                features.append(acceleration.std())
                
                # Change frequency
                changes = (velocity != 0).sum()
                features.append(changes / len(time_series))
                
        return features
        
    def _extract_frequency_patterns(self, time_series: pd.DataFrame) -> List[float]:
        """Extract frequency domain features using FFT."""
        features = []
        
        for col in time_series.columns:
            if pd.api.types.is_numeric_dtype(time_series[col]):
                values = time_series[col].fillna(0).values
                
                if len(values) > 10:
                    # FFT
                    fft_values = np.fft.fft(values)
                    fft_abs = np.abs(fft_values)
                    
                    # Dominant frequencies
                    top_freqs = np.argsort(fft_abs)[-5:]
                    features.extend(top_freqs / len(values))
                    
                    # Power spectrum statistics
                    power_spectrum = fft_abs ** 2
                    features.append(power_spectrum.mean())
                    features.append(power_spectrum.std())
                    features.append(power_spectrum.max())
                else:
                    features.extend([0.0] * 8)
                    
        return features
        
    def _extract_seasonal_trends(self, time_series: pd.DataFrame) -> List[float]:
        """Extract seasonal decomposition features."""
        features = []
        
        for col in time_series.columns:
            if pd.api.types.is_numeric_dtype(time_series[col]):
                values = time_series[col].fillna(method='ffill').fillna(0)
                
                if len(values) > 30:
                    # Simple seasonal decomposition
                    # Weekly pattern
                    if len(values) >= 7:
                        weekly_avg = values.rolling(window=7).mean()
                        weekly_std = values.rolling(window=7).std()
                        features.append(weekly_avg.mean())
                        features.append(weekly_std.mean())
                    else:
                        features.extend([0.0, 0.0])
                        
                    # Monthly pattern
                    if len(values) >= 30:
                        monthly_avg = values.rolling(window=30).mean()
                        monthly_std = values.rolling(window=30).std()
                        features.append(monthly_avg.mean())
                        features.append(monthly_std.mean())
                    else:
                        features.extend([0.0, 0.0])
                        
                    # Trend strength
                    if len(values) > 2:
                        x = np.arange(len(values))
                        slope, intercept = np.polyfit(x, values, 1)
                        features.append(slope)
                        features.append(intercept)
                    else:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0] * 6)
                    
        return features
        
    def _extract_temporal_correlations(self, time_series: pd.DataFrame) -> List[float]:
        """Extract autocorrelation and cross-correlation features."""
        features = []
        
        # Autocorrelation for each column
        for col in time_series.columns:
            if pd.api.types.is_numeric_dtype(time_series[col]):
                values = time_series[col].fillna(0)
                
                # Autocorrelation at different lags
                for lag in [1, 7, 30]:
                    if len(values) > lag:
                        autocorr = values.autocorr(lag=lag)
                        features.append(autocorr if not np.isnan(autocorr) else 0.0)
                    else:
                        features.append(0.0)
                        
        # Cross-correlation between columns (limit to first 3 columns)
        numeric_cols = [col for col in time_series.columns[:3] 
                       if pd.api.types.is_numeric_dtype(time_series[col])]
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = time_series[numeric_cols[i]].corr(time_series[numeric_cols[j]])
                features.append(corr if not np.isnan(corr) else 0.0)
                
        # Pad if needed
        while len(features) < 12:
            features.append(0.0)
            
        return features[:12]  # Limit to consistent size
        
    def _extract_lag_features(self, time_series: pd.DataFrame) -> List[float]:
        """Extract lagged feature values."""
        features = []
        
        for col in time_series.columns[:3]:  # Limit columns
            if pd.api.types.is_numeric_dtype(time_series[col]):
                values = time_series[col].fillna(0)
                
                # Get last N lag values
                for lag in range(1, min(self.config.lag_features + 1, len(values))):
                    if len(values) > lag:
                        features.append(values.iloc[-lag])
                    else:
                        features.append(0.0)
                        
                # Pad if needed
                while len(features) % self.config.lag_features != 0:
                    features.append(0.0)
                    
        return features[:self.config.lag_features * 3]  # Consistent size
        
    def _extract_rolling_statistics(self, time_series: pd.DataFrame) -> List[float]:
        """Extract rolling window statistics."""
        features = []
        
        for window_size in self.config.temporal_window_sizes:
            for col in time_series.columns[:2]:  # Limit columns
                if pd.api.types.is_numeric_dtype(time_series[col]):
                    values = time_series[col].fillna(0)
                    
                    if len(values) >= window_size:
                        rolling = values.rolling(window=window_size)
                        
                        for stat in self.config.rolling_statistics:
                            if stat == 'mean':
                                features.append(rolling.mean().iloc[-1])
                            elif stat == 'std':
                                features.append(rolling.std().iloc[-1])
                            elif stat == 'min':
                                features.append(rolling.min().iloc[-1])
                            elif stat == 'max':
                                features.append(rolling.max().iloc[-1])
                            elif stat == 'median':
                                features.append(rolling.median().iloc[-1])
                    else:
                        features.extend([0.0] * len(self.config.rolling_statistics))
                        
        return features


class PolicyFeatureExtractor:
    """
    Extract features from policy definitions and attachment patterns.
    Analyzes policy complexity, inheritance, and conflict detection.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50)
        self.policy_graph = nx.DiGraph()
        
    def extract_features(self, policy: Dict[str, Any], 
                        policy_graph: Optional[nx.DiGraph] = None) -> np.ndarray:
        """
        Extract features from policy definition.
        
        Args:
            policy: Policy definition dictionary
            policy_graph: Optional policy inheritance graph
            
        Returns:
            Policy feature vector
        """
        features = []
        
        # Policy complexity metrics
        complexity_features = self._extract_complexity_metrics(policy)
        features.extend(complexity_features)
        
        # Rule-based features
        rule_features = self._extract_rule_features(policy.get('rules', []))
        features.extend(rule_features)
        
        # Inheritance features
        if policy_graph:
            inheritance_features = self._extract_inheritance_features(
                policy.get('id'), policy_graph
            )
            features.extend(inheritance_features)
        else:
            features.extend([0.0] * 5)
            
        # Attachment pattern features
        attachment_features = self._extract_attachment_patterns(policy)
        features.extend(attachment_features)
        
        # Text-based features from policy description
        text_features = self._extract_text_features(policy.get('description', ''))
        features.extend(text_features)
        
        # Conflict detection features
        conflict_features = self._extract_conflict_features(policy)
        features.extend(conflict_features)
        
        return np.array(features, dtype=np.float32)
        
    def _extract_complexity_metrics(self, policy: Dict) -> List[float]:
        """Calculate policy complexity metrics."""
        features = []
        
        # Number of rules
        n_rules = len(policy.get('rules', []))
        features.append(n_rules / 100.0)
        
        # Rule depth (nested conditions)
        max_depth = self._calculate_rule_depth(policy.get('rules', []))
        features.append(max_depth / 10.0)
        
        # Number of conditions
        n_conditions = self._count_conditions(policy.get('rules', []))
        features.append(n_conditions / 100.0)
        
        # Number of actions
        n_actions = self._count_actions(policy.get('rules', []))
        features.append(n_actions / 50.0)
        
        # Policy scope (resources affected)
        scope = policy.get('scope', {})
        n_resource_types = len(scope.get('resource_types', []))
        features.append(n_resource_types / 20.0)
        
        # Policy severity
        severity_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        severity = policy.get('severity', 'medium')
        features.append(severity_map.get(severity, 0.5))
        
        return features
        
    def _extract_rule_features(self, rules: List[Dict]) -> List[float]:
        """Extract features from policy rules."""
        features = []
        
        if not rules:
            return [0.0] * 10
            
        # Rule type distribution
        rule_types = defaultdict(int)
        for rule in rules:
            rule_types[rule.get('type', 'unknown')] += 1
            
        # Common rule types
        for rule_type in ['allow', 'deny', 'audit', 'enforce']:
            features.append(rule_types[rule_type] / len(rules))
            
        # Condition operators
        operators = defaultdict(int)
        for rule in rules:
            for condition in rule.get('conditions', []):
                operators[condition.get('operator', 'eq')] += 1
                
        # Common operators
        for op in ['eq', 'neq', 'gt', 'lt', 'contains', 'regex']:
            total_conditions = sum(operators.values())
            if total_conditions > 0:
                features.append(operators[op] / total_conditions)
            else:
                features.append(0.0)
                
        return features[:10]  # Ensure consistent size
        
    def _extract_inheritance_features(self, policy_id: str, 
                                     policy_graph: nx.DiGraph) -> List[float]:
        """Extract policy inheritance graph features."""
        features = []
        
        if policy_id not in policy_graph:
            return [0.0] * 5
            
        # Inheritance depth
        try:
            ancestors = nx.ancestors(policy_graph, policy_id)
            inheritance_depth = len(ancestors)
            features.append(inheritance_depth / 10.0)
        except:
            features.append(0.0)
            
        # Number of children
        try:
            descendants = nx.descendants(policy_graph, policy_id)
            n_children = len(descendants)
            features.append(n_children / 20.0)
        except:
            features.append(0.0)
            
        # Centrality measures
        try:
            degree_centrality = nx.degree_centrality(policy_graph).get(policy_id, 0)
            features.append(degree_centrality)
            
            betweenness = nx.betweenness_centrality(policy_graph).get(policy_id, 0)
            features.append(betweenness)
            
            if policy_graph.number_of_nodes() > 1:
                closeness = nx.closeness_centrality(policy_graph).get(policy_id, 0)
                features.append(closeness)
            else:
                features.append(0.0)
        except:
            features.extend([0.0] * 3)
            
        return features[:5]
        
    def _extract_attachment_patterns(self, policy: Dict) -> List[float]:
        """Extract policy attachment pattern features."""
        features = []
        
        attachments = policy.get('attachments', [])
        
        # Number of attachments
        features.append(len(attachments) / 50.0)
        
        # Attachment types
        attachment_types = defaultdict(int)
        for attachment in attachments:
            attachment_types[attachment.get('type', 'resource')] += 1
            
        # Common attachment types
        for attach_type in ['resource', 'user', 'group', 'role']:
            if attachments:
                features.append(attachment_types[attach_type] / len(attachments))
            else:
                features.append(0.0)
                
        # Attachment scope
        unique_resources = set()
        for attachment in attachments:
            unique_resources.add(attachment.get('resource_id'))
        features.append(len(unique_resources) / 100.0)
        
        return features
        
    def _extract_text_features(self, description: str) -> List[float]:
        """Extract features from policy description text."""
        if not description:
            return [0.0] * 10
            
        # Simple text statistics
        features = []
        
        # Length features
        features.append(len(description) / 1000.0)
        features.append(len(description.split()) / 100.0)
        
        # Keyword presence
        keywords = ['compliance', 'security', 'encryption', 'access', 'audit',
                   'monitor', 'enforce', 'restrict', 'allow', 'deny']
        
        description_lower = description.lower()
        for keyword in keywords[:8]:
            features.append(float(keyword in description_lower))
            
        return features[:10]
        
    def _extract_conflict_features(self, policy: Dict) -> List[float]:
        """Detect potential policy conflicts."""
        features = []
        
        rules = policy.get('rules', [])
        
        # Check for conflicting rules
        conflicts = 0
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                if self._rules_conflict(rule1, rule2):
                    conflicts += 1
                    
        features.append(conflicts / max(len(rules), 1))
        
        # Check for overlapping scopes
        overlaps = 0
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i+1:]:
                if self._scopes_overlap(rule1.get('scope', {}), rule2.get('scope', {})):
                    overlaps += 1
                    
        features.append(overlaps / max(len(rules), 1))
        
        # Ambiguity score (based on condition complexity)
        ambiguity = 0
        for rule in rules:
            conditions = rule.get('conditions', [])
            if len(conditions) > 3:
                ambiguity += 1
                
        features.append(ambiguity / max(len(rules), 1))
        
        return features
        
    def _calculate_rule_depth(self, rules: List[Dict], current_depth: int = 0) -> int:
        """Calculate maximum depth of rule conditions."""
        max_depth = current_depth
        
        for rule in rules:
            conditions = rule.get('conditions', [])
            for condition in conditions:
                if 'nested' in condition:
                    nested_depth = self._calculate_rule_depth(
                        condition['nested'], current_depth + 1
                    )
                    max_depth = max(max_depth, nested_depth)
                    
        return max_depth
        
    def _count_conditions(self, rules: List[Dict]) -> int:
        """Count total number of conditions in rules."""
        count = 0
        for rule in rules:
            count += len(rule.get('conditions', []))
        return count
        
    def _count_actions(self, rules: List[Dict]) -> int:
        """Count total number of actions in rules."""
        count = 0
        for rule in rules:
            count += len(rule.get('actions', []))
        return count
        
    def _rules_conflict(self, rule1: Dict, rule2: Dict) -> bool:
        """Check if two rules conflict."""
        # Simple conflict detection: allow vs deny on same resource
        if rule1.get('type') == 'allow' and rule2.get('type') == 'deny':
            if rule1.get('resource') == rule2.get('resource'):
                return True
        return False
        
    def _scopes_overlap(self, scope1: Dict, scope2: Dict) -> bool:
        """Check if two scopes overlap."""
        resources1 = set(scope1.get('resources', []))
        resources2 = set(scope2.get('resources', []))
        return bool(resources1.intersection(resources2))


class ContextualFeatureExtractor:
    """
    Extract contextual features including resource dependencies,
    business criticality, and organizational metadata.
    """
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.dependency_graph = nx.DiGraph()
        
    def extract_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract contextual features.
        
        Args:
            context: Context information dictionary
            
        Returns:
            Context feature vector
        """
        features = []
        
        # Resource dependency features
        dependency_features = self._extract_dependency_features(
            context.get('resource_id'),
            context.get('dependencies', [])
        )
        features.extend(dependency_features)
        
        # Business criticality features
        criticality_features = self._extract_criticality_features(
            context.get('business_context', {})
        )
        features.extend(criticality_features)
        
        # Organizational features
        org_features = self._extract_organizational_features(
            context.get('organization', {})
        )
        features.extend(org_features)
        
        # Regulatory context features
        regulatory_features = self._extract_regulatory_features(
            context.get('regulatory', {})
        )
        features.extend(regulatory_features)
        
        return np.array(features, dtype=np.float32)
        
    def _extract_dependency_features(self, resource_id: str, 
                                    dependencies: List[str]) -> List[float]:
        """Extract resource dependency graph features."""
        features = []
        
        # Number of direct dependencies
        features.append(len(dependencies) / 50.0)
        
        # Build dependency graph
        if resource_id:
            for dep in dependencies:
                self.dependency_graph.add_edge(resource_id, dep)
                
            # Graph metrics
            if resource_id in self.dependency_graph:
                # In-degree (resources depending on this)
                in_degree = self.dependency_graph.in_degree(resource_id)
                features.append(in_degree / 20.0)
                
                # Out-degree (dependencies of this resource)
                out_degree = self.dependency_graph.out_degree(resource_id)
                features.append(out_degree / 20.0)
                
                # Transitive dependencies
                try:
                    descendants = nx.descendants(self.dependency_graph, resource_id)
                    features.append(len(descendants) / 100.0)
                except:
                    features.append(0.0)
                    
                # Dependency chain length
                try:
                    paths = nx.single_source_shortest_path_length(
                        self.dependency_graph, resource_id
                    )
                    max_path = max(paths.values()) if paths else 0
                    features.append(max_path / 10.0)
                except:
                    features.append(0.0)
            else:
                features.extend([0.0] * 4)
        else:
            features.extend([0.0] * 5)
            
        return features
        
    def _extract_criticality_features(self, business_context: Dict) -> List[float]:
        """Extract business criticality features."""
        features = []
        
        # Criticality level
        criticality_map = {
            'low': 0.2, 'medium': 0.4, 'high': 0.6, 
            'critical': 0.8, 'mission-critical': 1.0
        }
        criticality = business_context.get('criticality', 'medium')
        features.append(criticality_map.get(criticality, 0.5))
        
        # Data sensitivity
        sensitivity_map = {
            'public': 0.0, 'internal': 0.3, 'confidential': 0.6, 
            'restricted': 0.8, 'top-secret': 1.0
        }
        sensitivity = business_context.get('data_sensitivity', 'internal')
        features.append(sensitivity_map.get(sensitivity, 0.3))
        
        # Business impact scores
        features.append(business_context.get('availability_impact', 0) / 10.0)
        features.append(business_context.get('integrity_impact', 0) / 10.0)
        features.append(business_context.get('confidentiality_impact', 0) / 10.0)
        
        # Recovery metrics
        rto_hours = business_context.get('rto_hours', 24)
        features.append(1.0 / (1.0 + rto_hours / 24.0))
        
        rpo_hours = business_context.get('rpo_hours', 4)
        features.append(1.0 / (1.0 + rpo_hours / 4.0))
        
        # Service tier
        tier_map = {'bronze': 0.25, 'silver': 0.5, 'gold': 0.75, 'platinum': 1.0}
        tier = business_context.get('service_tier', 'silver')
        features.append(tier_map.get(tier, 0.5))
        
        return features
        
    def _extract_organizational_features(self, organization: Dict) -> List[float]:
        """Extract organizational metadata features."""
        features = []
        
        # Department type
        dept_map = {
            'it': 0.2, 'finance': 0.4, 'hr': 0.3, 
            'operations': 0.5, 'security': 0.8, 'executive': 1.0
        }
        department = organization.get('department', 'it')
        features.append(dept_map.get(department, 0.5))
        
        # Team size
        team_size = organization.get('team_size', 10)
        features.append(team_size / 100.0)
        
        # Organizational level
        org_level = organization.get('hierarchy_level', 3)
        features.append(org_level / self.config.organizational_hierarchy_depth)
        
        # Geographic distribution
        locations = organization.get('locations', [])
        features.append(len(locations) / 10.0)
        
        # Compliance requirements count
        compliance_reqs = organization.get('compliance_requirements', [])
        features.append(len(compliance_reqs) / 20.0)
        
        return features
        
    def _extract_regulatory_features(self, regulatory: Dict) -> List[float]:
        """Extract regulatory context features."""
        features = []
        
        # Compliance frameworks
        frameworks = ['gdpr', 'hipaa', 'pci-dss', 'sox', 'iso27001', 'nist']
        for framework in frameworks:
            features.append(float(framework in regulatory.get('frameworks', [])))
            
        # Audit frequency
        audit_frequency = regulatory.get('audit_frequency_days', 365)
        features.append(1.0 / (1.0 + audit_frequency / 365.0))
        
        # Compliance score
        compliance_score = regulatory.get('compliance_score', 0.8)
        features.append(compliance_score)
        
        # Number of controls
        n_controls = regulatory.get('n_controls', 0)
        features.append(n_controls / 500.0)
        
        # Last audit days ago
        last_audit_days = regulatory.get('last_audit_days_ago', 180)
        features.append(1.0 / (1.0 + last_audit_days / 180.0))
        
        return features


class IntegratedFeatureEngineeringPipeline:
    """
    Integrated feature engineering pipeline combining all extractors.
    Implements the complete feature engineering subsystem from Patent #4.
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        
        # Initialize extractors
        self.config_extractor = ConfigurationFeatureExtractor(self.config)
        self.temporal_extractor = TemporalFeatureExtractor(self.config)
        self.policy_extractor = PolicyFeatureExtractor(self.config)
        self.context_extractor = ContextualFeatureExtractor(self.config)
        
        # Feature cache
        self.feature_cache = {}
        
        # Dimensionality reduction
        self.pca = None
        if self.config.use_pca:
            self.pca = PCA(n_components=self.config.pca_components)
            
        logger.info("Initialized integrated feature engineering pipeline")
        
    def extract_features(self, 
                        configuration: Dict[str, Any],
                        time_series: Optional[pd.DataFrame] = None,
                        policy: Optional[Dict[str, Any]] = None,
                        context: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Extract all features from multi-modal inputs.
        
        Args:
            configuration: Resource configuration
            time_series: Optional temporal data
            policy: Optional policy definition
            context: Optional contextual information
            
        Returns:
            Complete feature vector
        """
        # Check cache
        cache_key = self._generate_cache_key(configuration, policy, context)
        if self.config.cache_features and cache_key in self.feature_cache:
            cached_entry = self.feature_cache[cache_key]
            if (datetime.now() - cached_entry['timestamp']).seconds < self.config.cache_ttl_seconds:
                return cached_entry['features']
                
        all_features = []
        
        # Configuration features
        config_features = self.config_extractor.extract_features(configuration)
        all_features.append(config_features)
        
        # Temporal features
        if time_series is not None and not time_series.empty:
            temporal_features = self.temporal_extractor.extract_features(time_series)
            all_features.append(temporal_features)
            
        # Policy features
        if policy is not None:
            policy_features = self.policy_extractor.extract_features(policy)
            all_features.append(policy_features)
            
        # Context features
        if context is not None:
            context_features = self.context_extractor.extract_features(context)
            all_features.append(context_features)
            
        # Concatenate all features
        combined_features = np.concatenate(all_features)
        
        # Apply dimensionality reduction if configured
        if self.config.use_pca and self.pca is not None:
            if not hasattr(self.pca, 'components_'):
                # Fit PCA if not already fitted
                self.pca.fit(combined_features.reshape(1, -1))
            combined_features = self.pca.transform(combined_features.reshape(1, -1))[0]
            
        # Cache features
        if self.config.cache_features:
            self.feature_cache[cache_key] = {
                'features': combined_features,
                'timestamp': datetime.now()
            }
            
        return combined_features
        
    def _generate_cache_key(self, configuration: Dict, 
                           policy: Optional[Dict], 
                           context: Optional[Dict]) -> str:
        """Generate cache key for feature vector."""
        key_parts = [
            hashlib.md5(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
        ]
        
        if policy:
            key_parts.append(
                hashlib.md5(json.dumps(policy, sort_keys=True).encode()).hexdigest()
            )
            
        if context:
            key_parts.append(
                hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
            )
            
        return '_'.join(key_parts)
        
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        names = []
        
        # Configuration features
        names.extend([f"config_{i}" for i in range(50)])  # Approximate
        
        # Temporal features
        names.extend([f"temporal_{i}" for i in range(100)])  # Approximate
        
        # Policy features
        names.extend([f"policy_{i}" for i in range(50)])  # Approximate
        
        # Context features
        names.extend([f"context_{i}" for i in range(50)])  # Approximate
        
        if self.config.use_pca:
            return [f"pca_{i}" for i in range(self.config.pca_components)]
            
        return names
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if PCA was used."""
        if self.pca and hasattr(self.pca, 'explained_variance_ratio_'):
            importance = {}
            for i, variance in enumerate(self.pca.explained_variance_ratio_):
                importance[f"pca_{i}"] = variance
            return importance
        return {}


if __name__ == "__main__":
    # Test the feature engineering pipeline
    
    # Create test data
    configuration = {
        'encryption': {
            'enabled': True,
            'algorithm': 'AES-256',
            'key_rotation_enabled': True,
            'key_rotation_days': 90
        },
        'access_control': {
            'authentication_type': 'mfa',
            'mfa_enabled': True,
            'roles': ['admin', 'user', 'viewer'],
            'session_timeout_minutes': 30
        },
        'network': {
            'public_access': False,
            'private_endpoint': True,
            'exposed_ports': [443, 8080],
            'ddos_protection': True
        },
        'metadata': {
            'resource_type': 'database',
            'environment': 'prod',
            'created_date': '2024-01-01',
            'tags': {'compliant': 'true', 'gdpr': 'true'}
        }
    }
    
    # Create time series data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    time_series = pd.DataFrame({
        'cpu_usage': np.random.randn(100) * 10 + 50,
        'memory_usage': np.random.randn(100) * 5 + 70,
        'network_traffic': np.random.randn(100) * 100 + 1000
    }, index=dates)
    
    # Create policy
    policy = {
        'id': 'policy_123',
        'severity': 'high',
        'rules': [
            {
                'type': 'deny',
                'conditions': [
                    {'field': 'encryption.enabled', 'operator': 'eq', 'value': False}
                ],
                'actions': ['alert', 'block']
            }
        ],
        'description': 'Enforce encryption for all production databases'
    }
    
    # Create context
    context = {
        'resource_id': 'res_456',
        'dependencies': ['res_789', 'res_012'],
        'business_context': {
            'criticality': 'high',
            'data_sensitivity': 'confidential',
            'rto_hours': 4,
            'rpo_hours': 1
        },
        'organization': {
            'department': 'security',
            'team_size': 25,
            'hierarchy_level': 2
        },
        'regulatory': {
            'frameworks': ['gdpr', 'hipaa'],
            'compliance_score': 0.92,
            'n_controls': 127
        }
    }
    
    # Create pipeline
    config = FeatureEngineeringConfig(
        use_pca=False,
        cache_features=True
    )
    
    pipeline = IntegratedFeatureEngineeringPipeline(config)
    
    # Extract features
    print("Extracting features...")
    features = pipeline.extract_features(
        configuration=configuration,
        time_series=time_series,
        policy=policy,
        context=context
    )
    
    print(f"\nFeature Engineering Results:")
    print(f"  Total features extracted: {len(features)}")
    print(f"  Feature vector shape: {features.shape}")
    print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Feature mean: {features.mean():.4f}")
    print(f"  Feature std: {features.std():.4f}")
    print(f"  Non-zero features: {(features != 0).sum()}/{len(features)}")
    
    # Test individual extractors
    print(f"\nIndividual Extractor Tests:")
    
    config_features = pipeline.config_extractor.extract_features(configuration)
    print(f"  Configuration features: {len(config_features)}")
    
    temporal_features = pipeline.temporal_extractor.extract_features(time_series)
    print(f"  Temporal features: {len(temporal_features)}")
    
    policy_features = pipeline.policy_extractor.extract_features(policy)
    print(f"  Policy features: {len(policy_features)}")
    
    context_features = pipeline.context_extractor.extract_features(context)
    print(f"  Context features: {len(context_features)}")
    
    print("\nFeature engineering pipeline test completed successfully!")