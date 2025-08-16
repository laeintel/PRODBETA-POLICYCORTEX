"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/feature_store.py
# Feature Store for PolicyCortex ML Pipeline

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis
import pickle
import json
import hashlib
from collections import defaultdict
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class Feature:
    """Feature definition"""
    name: str
    type: str  # 'numeric', 'categorical', 'boolean', 'text', 'embedding'
    description: str
    source: str
    update_frequency: str  # 'realtime', 'hourly', 'daily', 'static'
    aggregations: List[str] = field(default_factory=list)  # ['mean', 'max', 'min', 'count']
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureSet:
    """Collection of related features"""
    name: str
    version: str
    features: List[Feature]
    entity_type: str  # 'resource', 'user', 'policy', 'violation'
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureVector:
    """Feature vector for ML consumption"""
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class HistoricalFeatureStore:
    """Store for historical feature aggregates"""
    
    def __init__(self):
        self.storage = {}  # In production, use database
        self.aggregation_windows = ['1h', '6h', '1d', '7d', '30d']
        
    async def get_aggregates(
        self,
        entity_id: str,
        feature_names: List[str],
        windows: List[str]
    ) -> Dict[str, Any]:
        """Get historical aggregates for features"""
        aggregates = {}
        
        for feature_name in feature_names:
            feature_key = f"{entity_id}:{feature_name}"
            
            if feature_key in self.storage:
                feature_data = self.storage[feature_key]
                
                for window in windows:
                    if window in self.aggregation_windows:
                        window_key = f"{feature_name}_{window}"
                        aggregates[window_key] = self._calculate_window_aggregate(
                            feature_data, window
                        )
        
        return aggregates
    
    def _calculate_window_aggregate(
        self,
        data: List[Tuple[datetime, float]],
        window: str
    ) -> Dict[str, float]:
        """Calculate aggregate statistics for a time window"""
        # Parse window duration
        window_duration = self._parse_window(window)
        cutoff_time = datetime.now() - window_duration
        
        # Filter data within window
        window_data = [value for timestamp, value in data if timestamp > cutoff_time]
        
        if not window_data:
            return {'mean': 0, 'max': 0, 'min': 0, 'count': 0}
        
        return {
            'mean': np.mean(window_data),
            'max': np.max(window_data),
            'min': np.min(window_data),
            'std': np.std(window_data),
            'count': len(window_data)
        }
    
    def _parse_window(self, window: str) -> timedelta:
        """Parse window string to timedelta"""
        if window.endswith('h'):
            hours = int(window[:-1])
            return timedelta(hours=hours)
        elif window.endswith('d'):
            days = int(window[:-1])
            return timedelta(days=days)
        else:
            return timedelta(hours=1)
    
    async def store_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: Optional[datetime] = None
    ):
        """Store a feature value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        feature_key = f"{entity_id}:{feature_name}"
        
        if feature_key not in self.storage:
            self.storage[feature_key] = []
        
        self.storage[feature_key].append((timestamp, value))
        
        # Keep only last 30 days of data
        cutoff = datetime.now() - timedelta(days=30)
        self.storage[feature_key] = [
            (ts, val) for ts, val in self.storage[feature_key]
            if ts > cutoff
        ]

class FeatureStore:
    """Central feature store for ML pipeline"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=False
            )
            self.redis_client.ping()
        except:
            logger.warning("Redis not available, using in-memory cache")
            self.redis_client = None
            self.memory_cache = {}
        
        self.historical_store = HistoricalFeatureStore()
        self.feature_registry = {}
        self.feature_sets = {}
        self.initialize_default_features()
    
    def initialize_default_features(self):
        """Initialize default feature definitions"""
        
        # Resource features
        resource_features = [
            Feature(
                name='cpu_utilization',
                type='numeric',
                description='CPU utilization percentage',
                source='metrics',
                update_frequency='realtime',
                aggregations=['mean', 'max', 'p95']
            ),
            Feature(
                name='memory_utilization',
                type='numeric',
                description='Memory utilization percentage',
                source='metrics',
                update_frequency='realtime',
                aggregations=['mean', 'max', 'p95']
            ),
            Feature(
                name='compliance_score',
                type='numeric',
                description='Compliance score (0-100)',
                source='compliance_engine',
                update_frequency='hourly',
                aggregations=['mean', 'min']
            ),
            Feature(
                name='cost_per_hour',
                type='numeric',
                description='Hourly cost in USD',
                source='billing',
                update_frequency='hourly',
                aggregations=['sum', 'mean']
            ),
            Feature(
                name='policy_violations',
                type='numeric',
                description='Number of policy violations',
                source='policy_engine',
                update_frequency='hourly',
                aggregations=['count', 'sum']
            ),
            Feature(
                name='resource_age_days',
                type='numeric',
                description='Age of resource in days',
                source='resource_graph',
                update_frequency='daily',
                aggregations=['max']
            ),
            Feature(
                name='is_production',
                type='boolean',
                description='Whether resource is in production',
                source='tags',
                update_frequency='static'
            ),
            Feature(
                name='resource_type',
                type='categorical',
                description='Type of resource',
                source='resource_graph',
                update_frequency='static'
            )
        ]
        
        # Create resource feature set
        self.feature_sets['resource_features'] = FeatureSet(
            name='resource_features',
            version='1.0.0',
            features=resource_features,
            entity_type='resource',
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Register features
        for feature in resource_features:
            self.feature_registry[feature.name] = feature
    
    async def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        include_historical: bool = True
    ) -> Dict[str, Any]:
        """Get features for an entity"""
        features = {}
        
        # Get real-time features
        real_time = await self.get_real_time_features(entity_id, feature_names)
        features.update(real_time)
        
        # Get historical aggregates if requested
        if include_historical:
            historical = await self.historical_store.get_aggregates(
                entity_id,
                feature_names,
                windows=['1h', '1d', '7d', '30d']
            )
            features.update(historical)
        
        # Get derived features
        derived = await self.get_derived_features(entity_id, features)
        features.update(derived)
        
        return features
    
    async def get_real_time_features(
        self,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Get real-time features from cache"""
        features = {}
        
        for feature_name in feature_names:
            key = f"feature:{entity_id}:{feature_name}"
            
            # Try Redis first
            if self.redis_client:
                try:
                    value = self.redis_client.get(key)
                    if value:
                        features[feature_name] = pickle.loads(value)
                except Exception as e:
                    logger.error(f"Redis error: {e}")
            
            # Fallback to memory cache
            if feature_name not in features and hasattr(self, 'memory_cache'):
                if key in self.memory_cache:
                    features[feature_name] = self.memory_cache[key]
        
        return features
    
    async def get_derived_features(
        self,
        entity_id: str,
        base_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate derived features from base features"""
        derived = {}
        
        # CPU/Memory ratio
        if 'cpu_utilization' in base_features and 'memory_utilization' in base_features:
            cpu = base_features['cpu_utilization']
            memory = base_features['memory_utilization']
            if memory > 0:
                derived['cpu_memory_ratio'] = cpu / memory
        
        # Cost efficiency score
        if 'cost_per_hour' in base_features and 'cpu_utilization' in base_features:
            cost = base_features['cost_per_hour']
            cpu = base_features['cpu_utilization']
            if cost > 0:
                derived['cost_efficiency'] = cpu / cost
        
        # Risk score
        if 'policy_violations' in base_features and 'compliance_score' in base_features:
            violations = base_features['policy_violations']
            compliance = base_features['compliance_score']
            derived['risk_score'] = (violations * 10) + (100 - compliance)
        
        # Time-based features
        now = datetime.now()
        derived['hour_of_day'] = now.hour
        derived['day_of_week'] = now.weekday()
        derived['is_weekend'] = now.weekday() >= 5
        
        return derived
    
    async def update_feature(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        ttl: int = 3600
    ):
        """Update a feature value"""
        key = f"feature:{entity_id}:{feature_name}"
        
        # Store in Redis
        if self.redis_client:
            try:
                self.redis_client.set(key, pickle.dumps(value), ex=ttl)
            except Exception as e:
                logger.error(f"Redis update error: {e}")
        
        # Store in memory cache
        if hasattr(self, 'memory_cache'):
            self.memory_cache[key] = value
        
        # Store in historical store
        await self.historical_store.store_feature(entity_id, feature_name, value)
    
    async def batch_update_features(
        self,
        updates: List[Tuple[str, str, Any]],
        ttl: int = 3600
    ):
        """Batch update multiple features"""
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            
            for entity_id, feature_name, value in updates:
                key = f"feature:{entity_id}:{feature_name}"
                pipe.set(key, pickle.dumps(value), ex=ttl)
            
            try:
                pipe.execute()
            except Exception as e:
                logger.error(f"Redis batch update error: {e}")
        
        # Update memory cache
        if hasattr(self, 'memory_cache'):
            for entity_id, feature_name, value in updates:
                key = f"feature:{entity_id}:{feature_name}"
                self.memory_cache[key] = value
        
        # Update historical store
        for entity_id, feature_name, value in updates:
            await self.historical_store.store_feature(entity_id, feature_name, value)
    
    def create_feature_set(
        self,
        name: str,
        features: List[Feature],
        entity_type: str,
        version: str = '1.0.0'
    ) -> FeatureSet:
        """Create a new feature set"""
        feature_set = FeatureSet(
            name=name,
            version=version,
            features=features,
            entity_type=entity_type,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.feature_sets[name] = feature_set
        
        # Register features
        for feature in features:
            self.feature_registry[f"{name}.{feature.name}"] = feature
        
        return feature_set
    
    async def get_feature_vector(
        self,
        entity_id: str,
        feature_set_name: str
    ) -> FeatureVector:
        """Get feature vector for ML model consumption"""
        if feature_set_name not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set_name} not found")
        
        feature_set = self.feature_sets[feature_set_name]
        feature_names = [f.name for f in feature_set.features]
        
        # Get all features
        features = await self.get_features(entity_id, feature_names)
        
        # Create feature vector
        vector = FeatureVector(
            entity_id=entity_id,
            features=features,
            timestamp=datetime.now(),
            version=feature_set.version,
            metadata={'feature_set': feature_set_name}
        )
        
        return vector
    
    async def get_training_data(
        self,
        entity_ids: List[str],
        feature_set_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get training data for multiple entities"""
        if feature_set_name not in self.feature_sets:
            raise ValueError(f"Feature set {feature_set_name} not found")
        
        feature_set = self.feature_sets[feature_set_name]
        
        data = []
        for entity_id in entity_ids:
            try:
                vector = await self.get_feature_vector(entity_id, feature_set_name)
                row = {'entity_id': entity_id}
                row.update(vector.features)
                data.append(row)
            except Exception as e:
                logger.error(f"Error getting features for {entity_id}: {e}")
        
        df = pd.DataFrame(data)
        
        # Filter by time if specified
        if start_time and 'timestamp' in df.columns:
            df = df[df['timestamp'] >= start_time]
        if end_time and 'timestamp' in df.columns:
            df = df[df['timestamp'] <= end_time]
        
        return df
    
    def register_feature(self, feature: Feature):
        """Register a new feature"""
        self.feature_registry[feature.name] = feature
    
    def get_feature_info(self, feature_name: str) -> Optional[Feature]:
        """Get feature information"""
        return self.feature_registry.get(feature_name)
    
    def list_features(self, entity_type: Optional[str] = None) -> List[Feature]:
        """List all registered features"""
        features = list(self.feature_registry.values())
        
        if entity_type:
            # Filter by entity type
            features = [
                f for f in features
                if any(fs.entity_type == entity_type 
                      for fs in self.feature_sets.values()
                      if f in fs.features)
            ]
        
        return features
    
    async def compute_feature_importance(
        self,
        feature_set_name: str,
        target_variable: str,
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """Compute feature importance for a feature set"""
        # In production, would use actual ML model
        # For demo, return mock importance scores
        
        if feature_set_name not in self.feature_sets:
            return {}
        
        feature_set = self.feature_sets[feature_set_name]
        importance = {}
        
        for feature in feature_set.features:
            # Simulate importance based on feature type
            if feature.name == target_variable:
                importance[feature.name] = 1.0
            elif 'compliance' in feature.name or 'violation' in feature.name:
                importance[feature.name] = np.random.uniform(0.7, 0.9)
            elif 'cost' in feature.name:
                importance[feature.name] = np.random.uniform(0.5, 0.7)
            else:
                importance[feature.name] = np.random.uniform(0.1, 0.5)
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance

class FeatureEngineering:
    """Feature engineering utilities"""
    
    @staticmethod
    def encode_categorical(value: str, categories: List[str]) -> np.ndarray:
        """One-hot encode categorical variable"""
        encoding = np.zeros(len(categories))
        if value in categories:
            encoding[categories.index(value)] = 1
        return encoding
    
    @staticmethod
    def normalize_numeric(value: float, min_val: float, max_val: float) -> float:
        """Normalize numeric value to [0, 1]"""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def create_time_features(timestamp: datetime) -> Dict[str, float]:
        """Create time-based features"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': 1 if timestamp.weekday() >= 5 else 0,
            'is_business_hours': 1 if 9 <= timestamp.hour < 17 else 0
        }
    
    @staticmethod
    def create_interaction_features(
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """Create interaction features"""
        interactions = {}
        
        # Create selected interactions
        if 'cpu_utilization' in features and 'memory_utilization' in features:
            interactions['cpu_memory_product'] = (
                features['cpu_utilization'] * features['memory_utilization']
            )
        
        if 'cost_per_hour' in features and 'compliance_score' in features:
            interactions['cost_compliance_ratio'] = (
                features['cost_per_hour'] / (features['compliance_score'] + 1)
            )
        
        return interactions

# Export main components
__all__ = [
    'FeatureStore',
    'Feature',
    'FeatureSet',
    'FeatureVector',
    'HistoricalFeatureStore',
    'FeatureEngineering'
]