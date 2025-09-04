"""
Feature Store implementation using Feast
Manages feature engineering, versioning, and serving for ML models
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
from feast import (
    FeatureStore,
    Entity,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    FeatureService,
    ValueType
)
from feast.types import Float32, Float64, Int64, String, Bool, Timestamp
from feast.data_source import DataSource
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import on_demand_feature_view
import hashlib
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    name: str
    description: str
    dtype: str
    source: str
    transformation: Optional[str] = None
    aggregation: Optional[str] = None
    window: Optional[str] = None
    is_derived: bool = False

class GovernanceFeatureStore:
    """
    Feature store for PolicyCortex governance ML models
    Implements feature engineering, versioning, and serving
    """
    
    def __init__(self, repo_path: str = "./feast_repo"):
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(exist_ok=True)
        
        # Initialize feature definitions
        self.feature_definitions = self._init_feature_definitions()
        
        # Create feature store configuration
        self._create_feature_store_config()
        
        # Initialize Feast
        try:
            self.fs = FeatureStore(repo_path=str(self.repo_path))
        except:
            logger.warning("Feast not initialized, using mock store")
            self.fs = None
    
    def _init_feature_definitions(self) -> Dict[str, FeatureConfig]:
        """Initialize feature definitions for governance models"""
        return {
            # Resource features
            'resource_cpu_utilization': FeatureConfig(
                name='resource_cpu_utilization',
                description='Average CPU utilization over time window',
                dtype='float',
                source='metrics',
                aggregation='mean',
                window='1h'
            ),
            'resource_memory_utilization': FeatureConfig(
                name='resource_memory_utilization',
                description='Average memory utilization',
                dtype='float',
                source='metrics',
                aggregation='mean',
                window='1h'
            ),
            'resource_cost_daily': FeatureConfig(
                name='resource_cost_daily',
                description='Daily resource cost',
                dtype='float',
                source='billing',
                aggregation='sum',
                window='24h'
            ),
            'resource_age_days': FeatureConfig(
                name='resource_age_days',
                description='Age of resource in days',
                dtype='int',
                source='inventory',
                transformation='date_diff'
            ),
            
            # Compliance features
            'policy_violations_count': FeatureConfig(
                name='policy_violations_count',
                description='Number of policy violations',
                dtype='int',
                source='compliance',
                aggregation='count',
                window='7d'
            ),
            'compliance_score': FeatureConfig(
                name='compliance_score',
                description='Overall compliance score',
                dtype='float',
                source='compliance',
                transformation='calculate_score'
            ),
            'last_audit_days': FeatureConfig(
                name='last_audit_days',
                description='Days since last audit',
                dtype='int',
                source='audit',
                transformation='date_diff'
            ),
            
            # Security features
            'security_vulnerabilities': FeatureConfig(
                name='security_vulnerabilities',
                description='Number of security vulnerabilities',
                dtype='int',
                source='security',
                aggregation='sum'
            ),
            'risk_score': FeatureConfig(
                name='risk_score',
                description='Security risk score',
                dtype='float',
                source='security',
                transformation='calculate_risk'
            ),
            'unusual_activity_flag': FeatureConfig(
                name='unusual_activity_flag',
                description='Flag for unusual activity detected',
                dtype='bool',
                source='security',
                transformation='detect_anomaly'
            ),
            
            # Network features
            'network_ingress_bytes': FeatureConfig(
                name='network_ingress_bytes',
                description='Network ingress traffic',
                dtype='float',
                source='network',
                aggregation='sum',
                window='1h'
            ),
            'network_egress_bytes': FeatureConfig(
                name='network_egress_bytes',
                description='Network egress traffic',
                dtype='float',
                source='network',
                aggregation='sum',
                window='1h'
            ),
            
            # Derived features
            'cost_anomaly_score': FeatureConfig(
                name='cost_anomaly_score',
                description='Cost anomaly detection score',
                dtype='float',
                source='derived',
                transformation='isolation_forest',
                is_derived=True
            ),
            'resource_efficiency_ratio': FeatureConfig(
                name='resource_efficiency_ratio',
                description='Resource utilization efficiency',
                dtype='float',
                source='derived',
                transformation='calculate_efficiency',
                is_derived=True
            ),
            'compliance_trend': FeatureConfig(
                name='compliance_trend',
                description='Compliance score trend',
                dtype='float',
                source='derived',
                transformation='linear_trend',
                window='30d',
                is_derived=True
            )
        }
    
    def _create_feature_store_config(self):
        """Create Feast feature store configuration"""
        config = f"""
project: policycortex_governance
registry: {self.repo_path}/registry.db
provider: local
online_store:
    type: redis
    redis_type: redis
    connection_string: "localhost:6379"
offline_store:
    type: file
entity_key_serialization_version: 2
"""
        config_path = self.repo_path / "feature_store.yaml"
        config_path.write_text(config)
    
    def create_feature_views(self):
        """Create Feast feature views for governance features"""
        
        # Define entities
        resource_entity = Entity(
            name="resource_id",
            description="Cloud resource identifier"
        )
        
        subscription_entity = Entity(
            name="subscription_id",
            description="Azure subscription identifier"
        )
        
        # Create resource features view
        resource_features = FeatureView(
            name="resource_features",
            entities=[resource_entity],
            ttl=timedelta(days=7),
            schema=[
                Field(name="cpu_utilization", dtype=Float32),
                Field(name="memory_utilization", dtype=Float32),
                Field(name="cost_daily", dtype=Float32),
                Field(name="age_days", dtype=Int64),
            ],
            source=FileSource(
                path=f"{self.repo_path}/data/resource_features.parquet",
                timestamp_field="timestamp"
            ),
            tags={"team": "ml_platform", "model": "governance"}
        )
        
        # Create compliance features view
        compliance_features = FeatureView(
            name="compliance_features",
            entities=[resource_entity, subscription_entity],
            ttl=timedelta(days=30),
            schema=[
                Field(name="violations_count", dtype=Int64),
                Field(name="compliance_score", dtype=Float32),
                Field(name="last_audit_days", dtype=Int64),
            ],
            source=FileSource(
                path=f"{self.repo_path}/data/compliance_features.parquet",
                timestamp_field="timestamp"
            ),
            tags={"team": "ml_platform", "model": "compliance"}
        )
        
        # Create security features view
        security_features = FeatureView(
            name="security_features",
            entities=[resource_entity],
            ttl=timedelta(days=7),
            schema=[
                Field(name="vulnerabilities", dtype=Int64),
                Field(name="risk_score", dtype=Float32),
                Field(name="unusual_activity", dtype=Bool),
            ],
            source=FileSource(
                path=f"{self.repo_path}/data/security_features.parquet",
                timestamp_field="timestamp"
            ),
            tags={"team": "ml_platform", "model": "security"}
        )
        
        # Create on-demand feature view for derived features
        @on_demand_feature_view(
            sources=[resource_features, compliance_features],
            schema=[
                Field(name="efficiency_ratio", dtype=Float32),
                Field(name="combined_risk_score", dtype=Float32),
            ]
        )
        def derived_features(inputs: pd.DataFrame) -> pd.DataFrame:
            """Calculate derived features on-demand"""
            df = pd.DataFrame()
            
            # Calculate efficiency ratio
            df['efficiency_ratio'] = (
                inputs['cpu_utilization'] * inputs['memory_utilization'] /
                (inputs['cost_daily'] + 1)
            )
            
            # Calculate combined risk score
            df['combined_risk_score'] = (
                (100 - inputs['compliance_score']) * 0.5 +
                inputs['violations_count'] * 10
            )
            
            return df
        
        # Create feature service
        governance_service = FeatureService(
            name="governance_model_features",
            features=[
                resource_features,
                compliance_features,
                security_features,
                derived_features
            ],
            tags={"model": "predictive_compliance", "version": "1.0"}
        )
        
        # Apply to feature store
        if self.fs:
            try:
                self.fs.apply([
                    resource_entity,
                    subscription_entity,
                    resource_features,
                    compliance_features,
                    security_features,
                    derived_features,
                    governance_service
                ])
                logger.info("Feature views created successfully")
            except Exception as e:
                logger.error(f"Failed to create feature views: {e}")
    
    def materialize_features(
        self,
        start_date: datetime,
        end_date: datetime
    ):
        """
        Materialize features to online store for serving
        
        Args:
            start_date: Start date for materialization
            end_date: End date for materialization
        """
        if not self.fs:
            logger.warning("Feature store not initialized")
            return
        
        try:
            self.fs.materialize(start_date, end_date)
            logger.info(f"Materialized features from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Materialization failed: {e}")
    
    def get_online_features(
        self,
        entity_dict: Dict[str, List[Any]],
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get features from online store for real-time serving
        
        Args:
            entity_dict: Dictionary of entity keys and values
            features: Optional list of specific features to retrieve
            
        Returns:
            DataFrame with requested features
        """
        if not self.fs:
            # Return mock features
            return self._generate_mock_features(entity_dict)
        
        try:
            if features is None:
                features = [
                    "resource_features:cpu_utilization",
                    "resource_features:memory_utilization",
                    "resource_features:cost_daily",
                    "compliance_features:compliance_score",
                    "security_features:risk_score"
                ]
            
            feature_vector = self.fs.get_online_features(
                features=features,
                entity_rows=entity_dict
            )
            
            return feature_vector.to_df()
            
        except Exception as e:
            logger.error(f"Failed to get online features: {e}")
            return self._generate_mock_features(entity_dict)
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get historical features for training
        
        Args:
            entity_df: DataFrame with entity keys and timestamps
            features: Optional list of specific features
            
        Returns:
            DataFrame with historical features
        """
        if not self.fs:
            return self._generate_mock_historical_features(entity_df)
        
        try:
            if features is None:
                features = self.fs.get_feature_service("governance_model_features")
            
            training_df = self.fs.get_historical_features(
                entity_df=entity_df,
                features=features
            )
            
            return training_df.to_df()
            
        except Exception as e:
            logger.error(f"Failed to get historical features: {e}")
            return self._generate_mock_historical_features(entity_df)
    
    def register_feature_pipeline(
        self,
        pipeline_name: str,
        transformations: List[Dict[str, Any]],
        schedule: str = "0 * * * *"  # Hourly by default
    ):
        """
        Register a feature engineering pipeline
        
        Args:
            pipeline_name: Name of the pipeline
            transformations: List of transformation steps
            schedule: Cron schedule for pipeline execution
        """
        pipeline_config = {
            'name': pipeline_name,
            'transformations': transformations,
            'schedule': schedule,
            'created_at': datetime.utcnow().isoformat(),
            'version': self._generate_version_hash(transformations)
        }
        
        # Save pipeline configuration
        pipeline_path = self.repo_path / f"pipelines/{pipeline_name}.json"
        pipeline_path.parent.mkdir(exist_ok=True)
        
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline_config, f, indent=2)
        
        logger.info(f"Registered feature pipeline: {pipeline_name}")
    
    def compute_feature_statistics(
        self,
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute statistics for feature monitoring
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary of feature statistics
        """
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'row_count': len(features_df),
            'features': {}
        }
        
        for column in features_df.columns:
            if features_df[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                stats['features'][column] = {
                    'mean': float(features_df[column].mean()),
                    'std': float(features_df[column].std()),
                    'min': float(features_df[column].min()),
                    'max': float(features_df[column].max()),
                    'median': float(features_df[column].median()),
                    'missing_count': int(features_df[column].isnull().sum()),
                    'unique_count': int(features_df[column].nunique())
                }
            else:
                stats['features'][column] = {
                    'unique_count': int(features_df[column].nunique()),
                    'missing_count': int(features_df[column].isnull().sum()),
                    'most_common': str(features_df[column].mode().iloc[0]) if len(features_df[column].mode()) > 0 else None
                }
        
        return stats
    
    def validate_features(
        self,
        features_df: pd.DataFrame,
        validation_rules: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate features against defined rules
        
        Args:
            features_df: DataFrame with features
            validation_rules: Dictionary of validation rules
            
        Returns:
            Tuple of (is_valid, violations)
        """
        violations = []
        
        for feature, rules in validation_rules.items():
            if feature not in features_df.columns:
                violations.append(f"Missing feature: {feature}")
                continue
            
            values = features_df[feature]
            
            # Check data type
            if 'dtype' in rules:
                expected_dtype = rules['dtype']
                if not pd.api.types.is_dtype_equal(values.dtype, expected_dtype):
                    violations.append(f"{feature}: Wrong dtype {values.dtype} (expected {expected_dtype})")
            
            # Check range
            if 'min' in rules and values.min() < rules['min']:
                violations.append(f"{feature}: Value {values.min()} below minimum {rules['min']}")
            
            if 'max' in rules and values.max() > rules['max']:
                violations.append(f"{feature}: Value {values.max()} exceeds maximum {rules['max']}")
            
            # Check missing values
            if 'max_missing_pct' in rules:
                missing_pct = values.isnull().mean() * 100
                if missing_pct > rules['max_missing_pct']:
                    violations.append(f"{feature}: Missing {missing_pct:.1f}% exceeds threshold")
            
            # Check uniqueness
            if 'min_unique' in rules and values.nunique() < rules['min_unique']:
                violations.append(f"{feature}: Only {values.nunique()} unique values")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _generate_mock_features(self, entity_dict: Dict[str, List[Any]]) -> pd.DataFrame:
        """Generate mock features for testing"""
        num_entities = len(list(entity_dict.values())[0])
        
        return pd.DataFrame({
            'resource_id': entity_dict.get('resource_id', [f'res_{i}' for i in range(num_entities)]),
            'cpu_utilization': np.random.uniform(0, 100, num_entities),
            'memory_utilization': np.random.uniform(0, 100, num_entities),
            'cost_daily': np.random.uniform(10, 1000, num_entities),
            'compliance_score': np.random.uniform(60, 100, num_entities),
            'risk_score': np.random.uniform(0, 1, num_entities),
            'violations_count': np.random.poisson(2, num_entities),
            'age_days': np.random.randint(1, 365, num_entities)
        })
    
    def _generate_mock_historical_features(self, entity_df: pd.DataFrame) -> pd.DataFrame:
        """Generate mock historical features"""
        features_df = entity_df.copy()
        num_rows = len(features_df)
        
        features_df['cpu_utilization'] = np.random.uniform(0, 100, num_rows)
        features_df['memory_utilization'] = np.random.uniform(0, 100, num_rows)
        features_df['cost_daily'] = np.random.uniform(10, 1000, num_rows)
        features_df['compliance_score'] = np.random.uniform(60, 100, num_rows)
        features_df['risk_score'] = np.random.uniform(0, 1, num_rows)
        
        return features_df
    
    def _generate_version_hash(self, transformations: List[Dict]) -> str:
        """Generate version hash for transformations"""
        transform_str = json.dumps(transformations, sort_keys=True)
        return hashlib.md5(transform_str.encode()).hexdigest()[:8]

class FeatureEngineering:
    """
    Feature engineering transformations for governance models
    """
    
    @staticmethod
    def create_time_based_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        
        return df
    
    @staticmethod
    def create_aggregation_features(
        df: pd.DataFrame,
        group_cols: List[str],
        agg_cols: List[str],
        windows: List[str] = ['1h', '1d', '7d']
    ) -> pd.DataFrame:
        """Create aggregation features over time windows"""
        df = df.copy()
        
        for window in windows:
            for col in agg_cols:
                # Rolling statistics
                df[f'{col}_mean_{window}'] = df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_std_{window}'] = df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                df[f'{col}_max_{window}'] = df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )
                df[f'{col}_min_{window}'] = df.groupby(group_cols)[col].transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )
        
        return df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between feature pairs"""
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        return df
    
    @staticmethod
    def create_lag_features(
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3, 7, 14]
    ) -> pd.DataFrame:
        """Create lag features"""
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                df[f'{col}_diff_{lag}'] = df[col] - df[col].shift(lag)
        
        return df

# Example usage
if __name__ == "__main__":
    # Initialize feature store
    store = GovernanceFeatureStore()
    
    # Create feature views
    store.create_feature_views()
    
    # Get online features for serving
    entities = {
        'resource_id': ['vm_001', 'vm_002', 'vm_003']
    }
    online_features = store.get_online_features(entities)
    print("Online Features:")
    print(online_features)
    
    # Compute feature statistics
    stats = store.compute_feature_statistics(online_features)
    print("\nFeature Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Validate features
    validation_rules = {
        'cpu_utilization': {'min': 0, 'max': 100, 'max_missing_pct': 5},
        'compliance_score': {'min': 0, 'max': 100},
        'cost_daily': {'min': 0}
    }
    
    is_valid, violations = store.validate_features(online_features, validation_rules)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if violations:
        print("Violations:", violations)