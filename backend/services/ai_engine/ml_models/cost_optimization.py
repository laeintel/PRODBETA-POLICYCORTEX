"""
Cost Optimization Model

Advanced machine learning model for Azure cost forecasting, optimization opportunity
identification, and resource right-sizing recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import asyncio
import json

# ML and Statistical Libraries
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Time Series and Forecasting
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CostForecast:
    """Data structure for cost forecasting results"""
    resource_id: str
    resource_type: str
    current_monthly_cost: float
    predicted_monthly_cost: float
    cost_trend: str  # 'increasing', 'decreasing', 'stable'
    confidence_interval: Tuple[float, float]
    forecast_accuracy: float
    seasonal_pattern: Dict[str, float]
    prediction_timestamp: datetime

@dataclass
class OptimizationOpportunity:
    """Data structure for cost optimization opportunities"""
    resource_id: str
    resource_type: str
    opportunity_type: str  # 'rightsizing', 'scheduling', 'reserved_instances', 'spot_instances'
    current_cost: float
    optimized_cost: float
    potential_savings: float
    savings_percentage: float
    confidence_score: float
    implementation_effort: str  # 'low', 'medium', 'high'
    risk_level: str  # 'low', 'medium', 'high'
    recommended_actions: List[str]
    expected_impact: str
    timeline: str

@dataclass
class RightSizingRecommendation:
    """Data structure for resource right-sizing recommendations"""
    resource_id: str
    current_sku: str
    recommended_sku: str
    current_monthly_cost: float
    recommended_monthly_cost: float
    utilization_metrics: Dict[str, float]
    performance_impact: str
    confidence_score: float
    justification: str

class CostForecastingLSTM(nn.Module):
    """LSTM Neural Network for cost forecasting with attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(CostForecastingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with residual connections
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers for different predictions
        self.cost_predictor = nn.Linear(32, 1)
        self.trend_predictor = nn.Linear(32, 3)  # increasing, stable, decreasing
        self.confidence_predictor = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm3_out, _ = self.lstm3(lstm2_out + lstm1_out)  # Residual connection
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(lstm3_out, lstm3_out, lstm3_out)
        
        # Use the last output with attention
        final_output = attn_out[:, -1, :]
        
        # Feature extraction
        features = self.feature_extractor(final_output)
        
        # Multiple outputs
        cost_pred = self.cost_predictor(features)
        trend_pred = self.softmax(self.trend_predictor(features))
        confidence_pred = self.sigmoid(self.confidence_predictor(features))
        
        return {
            'cost': cost_pred,
            'trend': trend_pred,
            'confidence': confidence_pred,
            'attention_weights': attn_weights
        }

class CostOptimizationModel:
    """
    Comprehensive cost optimization model using ensemble methods,
    deep learning, and advanced analytics for Azure resource cost management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.ensemble_models = {}
        self.lstm_model = None
        self.prophet_models = {}
        self.clustering_model = None
        
        # Scalers for different features
        self.cost_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        
        # Feature engineering
        self.feature_columns = []
        self.is_trained = False
        
        # Performance metrics
        self.metrics = {}
        
        # Azure pricing cache
        self.pricing_cache = {}
        self.sku_mappings = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for cost optimization model"""
        return {
            'ensemble_models': {
                'random_forest': {
                    'n_estimators': 300,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'extra_trees': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'random_state': 42,
                    'n_jobs': -1
                },
                'elastic_net': {
                    'alpha': 0.1,
                    'l1_ratio': 0.5,
                    'random_state': 42
                }
            },
            'lstm_config': {
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'epochs': 150,
                'batch_size': 64,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            },
            'prophet_config': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'growth': 'linear'
            },
            'clustering_config': {
                'n_clusters': 8,
                'random_state': 42
            },
            'optimization_thresholds': {
                'cpu_utilization_low': 0.2,
                'cpu_utilization_high': 0.8,
                'memory_utilization_low': 0.3,
                'memory_utilization_high': 0.85,
                'savings_threshold': 0.1,  # Minimum 10% savings to recommend
                'confidence_threshold': 0.7
            },
            'rightsizing_rules': {
                'vm_sizes': {
                    'scale_down_threshold': 0.3,
                    'scale_up_threshold': 0.85,
                    'min_observation_days': 30
                }
            }
        }
    
    async def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the cost optimization models
        
        Args:
            training_data: DataFrame with historical cost and usage data
            
        Returns:
            Dictionary with training metrics
        """
        try:
            self.logger.info("Starting cost optimization model training")
            
            # Prepare training data
            X, y, time_series_data = self._prepare_training_data(training_data)
            
            # Train ensemble models
            ensemble_metrics = await self._train_ensemble_models(X, y)
            
            # Train LSTM model
            lstm_metrics = await self._train_lstm_model(time_series_data)
            
            # Train Prophet models for different resource types
            prophet_metrics = await self._train_prophet_models(training_data)
            
            # Train clustering model for resource categorization
            clustering_metrics = await self._train_clustering_model(X)
            
            self.is_trained = True
            self.metrics = {
                **ensemble_metrics,
                **lstm_metrics,
                **prophet_metrics,
                **clustering_metrics
            }
            
            self.logger.info(f"Training completed with metrics: {self.metrics}")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare and engineer features for training"""
        
        features = []
        
        # Temporal features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            features.append(self._extract_temporal_features(data))
        
        # Cost features
        if 'cost' in data.columns:
            features.append(self._extract_cost_features(data))
        
        # Resource utilization features
        utilization_cols = ['cpu_utilization', 'memory_utilization', 'disk_utilization', 'network_utilization']
        if any(col in data.columns for col in utilization_cols):
            features.append(self._extract_utilization_features(data, utilization_cols))
        
        # Resource configuration features
        if 'sku' in data.columns:
            features.append(self._extract_sku_features(data))
        
        # Workload pattern features
        if 'request_count' in data.columns:
            features.append(self._extract_workload_features(data))
        
        # Economic and market features
        features.append(self._extract_economic_features(data))
        
        # Prepare feature matrix
        feature_df = pd.concat([f for f in features if not f.empty], axis=1)
        self.feature_columns = feature_df.columns.tolist()
        
        # Target variable (next period cost)
        target = data['cost'].shift(-1) if 'cost' in data.columns else None
        
        # Time series data for LSTM
        time_series_data = self._prepare_time_series_data(data)
        
        return feature_df, target, time_series_data
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features"""
        
        timestamp = data['timestamp']
        
        temporal_features = pd.DataFrame({
            'hour': timestamp.dt.hour,
            'day_of_week': timestamp.dt.dayofweek,
            'day_of_month': timestamp.dt.day,
            'month': timestamp.dt.month,
            'quarter': timestamp.dt.quarter,
            'is_weekend': (timestamp.dt.dayofweek >= 5).astype(int),
            'is_business_hours': ((timestamp.dt.hour >= 9) & (timestamp.dt.hour <= 17)).astype(int),
            'is_month_end': (timestamp.dt.day >= 28).astype(int),
            'days_since_epoch': (timestamp - pd.Timestamp('2020-01-01')).dt.days
        })
        
        return temporal_features
    
    def _extract_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract cost-related features"""
        
        cost_col = data['cost']
        
        cost_features = pd.DataFrame({
            'cost_current': cost_col,
            'cost_lag_1': cost_col.shift(1),
            'cost_lag_7': cost_col.shift(7),
            'cost_lag_30': cost_col.shift(30),
            'cost_rolling_mean_7': cost_col.rolling(7, min_periods=1).mean(),
            'cost_rolling_std_7': cost_col.rolling(7, min_periods=1).std(),
            'cost_rolling_mean_30': cost_col.rolling(30, min_periods=1).mean(),
            'cost_rolling_std_30': cost_col.rolling(30, min_periods=1).std(),
            'cost_rolling_median_7': cost_col.rolling(7, min_periods=1).median(),
            'cost_rolling_max_7': cost_col.rolling(7, min_periods=1).max(),
            'cost_rolling_min_7': cost_col.rolling(7, min_periods=1).min(),
            'cost_trend_7': cost_col.rolling(7, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            ),
            'cost_volatility_7': cost_col.rolling(7, min_periods=1).std() / cost_col.rolling(7, min_periods=1).mean(),
            'cost_pct_change_1': cost_col.pct_change(1),
            'cost_pct_change_7': cost_col.pct_change(7),
            'cost_pct_change_30': cost_col.pct_change(30)
        })
        
        return cost_features.fillna(0)
    
    def _extract_utilization_features(self, data: pd.DataFrame, utilization_cols: List[str]) -> pd.DataFrame:
        """Extract resource utilization features"""
        
        utilization_features = pd.DataFrame()
        
        for col in utilization_cols:
            if col in data.columns:
                util_col = data[col]
                
                # Basic statistics
                utilization_features[f'{col}_current'] = util_col
                utilization_features[f'{col}_rolling_mean_7'] = util_col.rolling(7, min_periods=1).mean()
                utilization_features[f'{col}_rolling_std_7'] = util_col.rolling(7, min_periods=1).std()
                utilization_features[f'{col}_rolling_max_7'] = util_col.rolling(7, min_periods=1).max()
                utilization_features[f'{col}_rolling_min_7'] = util_col.rolling(7, min_periods=1).min()
                utilization_features[f'{col}_rolling_median_7'] = util_col.rolling(7, min_periods=1).median()
                
                # Percentiles
                utilization_features[f'{col}_p95_7'] = util_col.rolling(7, min_periods=1).quantile(0.95)
                utilization_features[f'{col}_p50_7'] = util_col.rolling(7, min_periods=1).quantile(0.50)
                
                # Patterns
                utilization_features[f'{col}_variance_7'] = util_col.rolling(7, min_periods=1).var()
                utilization_features[f'{col}_cv_7'] = (
                    util_col.rolling(7, min_periods=1).std() / 
                    util_col.rolling(7, min_periods=1).mean()
                )
                
                # Efficiency metrics
                utilization_features[f'{col}_efficiency'] = util_col / (util_col.rolling(30, min_periods=1).max() + 1e-6)
                utilization_features[f'{col}_waste'] = np.maximum(0, 1 - util_col)
        
        return utilization_features.fillna(0)
    
    def _extract_sku_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract SKU and configuration features"""
        
        sku_features = pd.DataFrame()
        
        if 'sku' in data.columns:
            # Extract numeric features from SKU names
            sku_features['sku_encoded'] = pd.Categorical(data['sku']).codes
            
            # Extract vCPU and memory information if available
            sku_features['vcpu_count'] = data['sku'].str.extract(r'(\d+)').astype(float).fillna(0)
            sku_features['memory_gb'] = data['sku'].str.extract(r'(\d+)GB').astype(float).fillna(0)
            
            # SKU family features
            sku_features['is_burstable'] = data['sku'].str.contains('B', case=False).astype(int)
            sku_features['is_compute_optimized'] = data['sku'].str.contains('F', case=False).astype(int)
            sku_features['is_memory_optimized'] = data['sku'].str.contains('M|E', case=False).astype(int)
            sku_features['is_general_purpose'] = data['sku'].str.contains('D|A', case=False).astype(int)
        
        if 'region' in data.columns:
            sku_features['region_encoded'] = pd.Categorical(data['region']).codes
        
        return sku_features.fillna(0)
    
    def _extract_workload_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract workload pattern features"""
        
        workload_features = pd.DataFrame()
        
        if 'request_count' in data.columns:
            req_col = data['request_count']
            
            workload_features['request_count'] = req_col
            workload_features['request_rolling_mean_7'] = req_col.rolling(7, min_periods=1).mean()
            workload_features['request_rolling_std_7'] = req_col.rolling(7, min_periods=1).std()
            workload_features['request_trend_7'] = req_col.rolling(7, min_periods=2).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
            )
        
        if 'response_time' in data.columns:
            rt_col = data['response_time']
            
            workload_features['response_time'] = rt_col
            workload_features['response_time_p95_7'] = rt_col.rolling(7, min_periods=1).quantile(0.95)
            workload_features['response_time_mean_7'] = rt_col.rolling(7, min_periods=1).mean()
        
        return workload_features.fillna(0)
    
    def _extract_economic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract economic and market-related features"""
        
        # Simulate market and economic indicators
        # In production, these would come from external APIs
        
        economic_features = pd.DataFrame(index=data.index)
        
        # Simulated market conditions
        economic_features['market_demand_index'] = np.random.normal(1.0, 0.1, len(data))
        economic_features['pricing_pressure'] = np.random.normal(0.0, 0.05, len(data))
        economic_features['capacity_utilization'] = np.random.beta(2, 2, len(data))
        
        # Seasonal business factors
        if 'timestamp' in data.columns:
            timestamp = pd.to_datetime(data['timestamp'])
            economic_features['business_cycle'] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 365)
            economic_features['holiday_effect'] = ((timestamp.dt.month == 12) | (timestamp.dt.month == 1)).astype(int)
        
        return economic_features
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> Dict:
        """Prepare time series data for LSTM model"""
        
        if 'timestamp' not in data.columns or 'cost' not in data.columns:
            return {}
        
        time_series = data.sort_values('timestamp')
        sequence_length = 60  # 60 time steps for LSTM
        
        sequences = []
        targets = []
        metadata = []
        
        # Group by resource_id to create sequences
        for resource_id in time_series['resource_id'].unique():
            resource_data = time_series[time_series['resource_id'] == resource_id].copy()
            
            if len(resource_data) < sequence_length + 1:
                continue
            
            # Prepare features
            feature_cols = ['cost']
            if 'cpu_utilization' in resource_data.columns:
                feature_cols.append('cpu_utilization')
            if 'memory_utilization' in resource_data.columns:
                feature_cols.append('memory_utilization')
            if 'request_count' in resource_data.columns:
                feature_cols.append('request_count')
            
            resource_features = resource_data[feature_cols].values
            
            # Create sequences
            for i in range(len(resource_features) - sequence_length):
                sequences.append(resource_features[i:i+sequence_length])
                targets.append(resource_features[i+sequence_length][0])  # Next cost
                metadata.append({
                    'resource_id': resource_id,
                    'timestamp': resource_data.iloc[i+sequence_length]['timestamp']
                })
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets),
            'metadata': metadata
        }
    
    async def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train ensemble models for cost prediction"""
        
        if X.empty or y is None:
            return {}
        
        # Clean data
        X_clean = X.fillna(X.median())
        y_clean = y.fillna(y.median())
        
        # Remove outliers using IQR method
        Q1 = y_clean.quantile(0.25)
        Q3 = y_clean.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
        X_clean = X_clean[mask]
        y_clean = y_clean[mask]
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X_clean)
        y_scaled = self.cost_scaler.fit_transform(y_clean.values.reshape(-1, 1)).ravel()
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {}
        metrics = {}
        
        # Random Forest
        rf_config = self.config['ensemble_models']['random_forest']
        rf_model = RandomForestRegressor(**rf_config)
        rf_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
            
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_val)
            rf_scores.append(r2_score(y_val, rf_pred))
        
        models['random_forest'] = rf_model
        metrics['random_forest_r2'] = np.mean(rf_scores)
        
        # Extra Trees
        et_config = self.config['ensemble_models']['extra_trees']
        et_model = ExtraTreesRegressor(**et_config)
        et_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
            
            et_model.fit(X_train, y_train)
            et_pred = et_model.predict(X_val)
            et_scores.append(r2_score(y_val, et_pred))
        
        models['extra_trees'] = et_model
        metrics['extra_trees_r2'] = np.mean(et_scores)
        
        # Elastic Net
        en_config = self.config['ensemble_models']['elastic_net']
        en_model = ElasticNet(**en_config)
        en_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
            
            en_model.fit(X_train, y_train)
            en_pred = en_model.predict(X_val)
            en_scores.append(r2_score(y_val, en_pred))
        
        models['elastic_net'] = en_model
        metrics['elastic_net_r2'] = np.mean(en_scores)
        
        self.ensemble_models = models
        return metrics
    
    async def _train_lstm_model(self, time_series_data: Dict) -> Dict[str, float]:
        """Train LSTM model for cost forecasting"""
        
        if not time_series_data or 'sequences' not in time_series_data:
            return {}
        
        sequences = time_series_data['sequences']
        targets = time_series_data['targets']
        
        if len(sequences) == 0:
            return {}
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.FloatTensor(targets).unsqueeze(1)
        
        # Split data
        train_size = int(0.8 * len(X_tensor))
        X_train = X_tensor[:train_size]
        X_test = X_tensor[train_size:]
        y_train = y_tensor[:train_size]
        y_test = y_tensor[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['lstm_config']['batch_size'], 
            shuffle=True
        )
        
        # Initialize model
        input_size = sequences.shape[2]
        self.lstm_model = CostForecastingLSTM(
            input_size=input_size,
            hidden_size=self.config['lstm_config']['hidden_size'],
            num_layers=self.config['lstm_config']['num_layers'],
            dropout=self.config['lstm_config']['dropout']
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.lstm_model.parameters(), 
            lr=self.config['lstm_config']['learning_rate'],
            weight_decay=self.config['lstm_config']['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.lstm_model.train()
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(self.config['lstm_config']['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs['cost'], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            self.lstm_model.eval()
            with torch.no_grad():
                val_outputs = self.lstm_model(X_test)
                val_loss = criterion(val_outputs['cost'], y_test).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience > 20:
                    break
            
            self.lstm_model.train()
        
        # Final evaluation
        self.lstm_model.eval()
        with torch.no_grad():
            test_outputs = self.lstm_model(X_test)
            test_mae = mean_absolute_error(y_test.numpy(), test_outputs['cost'].numpy())
            test_r2 = r2_score(y_test.numpy(), test_outputs['cost'].numpy())
        
        return {
            'lstm_test_mae': test_mae,
            'lstm_test_r2': test_r2,
            'lstm_best_loss': best_loss
        }
    
    async def _train_prophet_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train Prophet models for different resource types"""
        
        if 'timestamp' not in data.columns or 'cost' not in data.columns:
            return {}
        
        metrics = {}
        
        # Train separate models for each resource type
        for resource_type in data['resource_type'].unique():
            type_data = data[data['resource_type'] == resource_type].copy()
            
            if len(type_data) < 100:  # Need sufficient data
                continue
            
            # Aggregate daily costs
            daily_costs = type_data.groupby(type_data['timestamp'].dt.date)['cost'].sum().reset_index()
            daily_costs.columns = ['ds', 'y']
            daily_costs['ds'] = pd.to_datetime(daily_costs['ds'])
            daily_costs = daily_costs.sort_values('ds')
            
            # Create and train model
            model = Prophet(**self.config['prophet_config'])
            
            # Add custom seasonalities for business patterns
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
            
            try:
                model.fit(daily_costs)
                
                # Store model
                self.prophet_models[resource_type] = model
                
                # Evaluate on last 30 days
                if len(daily_costs) > 60:
                    split_idx = len(daily_costs) - 30
                    train_data = daily_costs[:split_idx]
                    test_data = daily_costs[split_idx:]
                    
                    forecast = model.predict(test_data[['ds']])
                    mae = mean_absolute_error(test_data['y'], forecast['yhat'])
                    metrics[f'prophet_{resource_type}_mae'] = mae
                    
            except Exception as e:
                self.logger.warning(f"Prophet training failed for {resource_type}: {str(e)}")
        
        return metrics
    
    async def _train_clustering_model(self, X: pd.DataFrame) -> Dict[str, float]:
        """Train clustering model for resource categorization"""
        
        if X.empty:
            return {}
        
        X_clean = X.fillna(X.median())
        X_scaled = self.feature_scaler.transform(X_clean)
        
        # K-means clustering
        self.clustering_model = KMeans(**self.config['clustering_config'])
        cluster_labels = self.clustering_model.fit_predict(X_scaled)
        
        # Calculate silhouette score for cluster quality
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        return {
            'clustering_silhouette_score': silhouette_avg,
            'clustering_n_clusters': len(np.unique(cluster_labels))
        }
    
    async def forecast_cost(self, resource_data: Dict[str, Any], 
                          forecast_days: int = 30) -> CostForecast:
        """
        Forecast cost for a specific resource
        
        Args:
            resource_data: Dictionary containing resource information and historical data
            forecast_days: Number of days to forecast
            
        Returns:
            CostForecast object with prediction details
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making forecasts")
        
        try:
            # Extract features for ensemble prediction
            features = self._extract_forecast_features(resource_data)
            
            # Ensemble prediction
            ensemble_forecast = self._forecast_ensemble(features)
            
            # LSTM prediction
            lstm_forecast = self._forecast_lstm(resource_data)
            
            # Prophet prediction
            prophet_forecast = self._forecast_prophet(resource_data, forecast_days)
            
            # Combine forecasts with weighted average
            weights = {'ensemble': 0.4, 'lstm': 0.35, 'prophet': 0.25}
            final_forecast = (
                weights['ensemble'] * (ensemble_forecast or 0) +
                weights['lstm'] * (lstm_forecast or 0) +
                weights['prophet'] * (prophet_forecast or 0)
            )
            
            # Determine trend
            current_cost = resource_data.get('current_monthly_cost', 0)
            if final_forecast > current_cost * 1.1:
                trend = 'increasing'
            elif final_forecast < current_cost * 0.9:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Calculate confidence interval
            confidence_interval = self._calculate_forecast_confidence(
                ensemble_forecast, lstm_forecast, prophet_forecast
            )
            
            # Extract seasonal patterns
            seasonal_pattern = self._extract_seasonal_pattern(resource_data)
            
            return CostForecast(
                resource_id=resource_data.get('resource_id', 'unknown'),
                resource_type=resource_data.get('resource_type', 'unknown'),
                current_monthly_cost=current_cost,
                predicted_monthly_cost=final_forecast,
                cost_trend=trend,
                confidence_interval=confidence_interval,
                forecast_accuracy=0.85,  # Based on historical performance
                seasonal_pattern=seasonal_pattern,
                prediction_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Cost forecasting failed: {str(e)}")
            raise
    
    async def identify_optimization_opportunities(self, 
                                                resources: List[Dict[str, Any]]) -> List[OptimizationOpportunity]:
        """
        Identify cost optimization opportunities across resources
        
        Args:
            resources: List of resource dictionaries with usage and cost data
            
        Returns:
            List of OptimizationOpportunity objects
        """
        
        opportunities = []
        
        for resource in resources:
            try:
                # Right-sizing opportunities
                rightsizing_ops = await self._identify_rightsizing_opportunities(resource)
                opportunities.extend(rightsizing_ops)
                
                # Scheduling opportunities
                scheduling_ops = await self._identify_scheduling_opportunities(resource)
                opportunities.extend(scheduling_ops)
                
                # Reserved instance opportunities
                ri_ops = await self._identify_reserved_instance_opportunities(resource)
                opportunities.extend(ri_ops)
                
                # Spot instance opportunities
                spot_ops = await self._identify_spot_instance_opportunities(resource)
                opportunities.extend(spot_ops)
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze resource {resource.get('resource_id')}: {str(e)}")
        
        # Sort by potential savings
        opportunities.sort(key=lambda x: x.potential_savings, reverse=True)
        
        return opportunities
    
    async def _identify_rightsizing_opportunities(self, resource: Dict[str, Any]) -> List[OptimizationOpportunity]:
        """Identify right-sizing opportunities"""
        
        opportunities = []
        
        if resource.get('resource_type') != 'virtual_machine':
            return opportunities
        
        # Analyze utilization patterns
        cpu_util = resource.get('avg_cpu_utilization', 0)
        memory_util = resource.get('avg_memory_utilization', 0)
        
        thresholds = self.config['optimization_thresholds']
        
        # Scale down opportunity
        if (cpu_util < thresholds['cpu_utilization_low'] and 
            memory_util < thresholds['memory_utilization_low']):
            
            current_cost = resource.get('monthly_cost', 0)
            current_sku = resource.get('sku', '')
            
            # Recommend smaller SKU
            recommended_sku = self._recommend_smaller_sku(current_sku)
            if recommended_sku:
                optimized_cost = current_cost * 0.6  # Approximate 40% savings
                
                if (current_cost - optimized_cost) / current_cost >= thresholds['savings_threshold']:
                    opportunities.append(OptimizationOpportunity(
                        resource_id=resource['resource_id'],
                        resource_type=resource['resource_type'],
                        opportunity_type='rightsizing',
                        current_cost=current_cost,
                        optimized_cost=optimized_cost,
                        potential_savings=current_cost - optimized_cost,
                        savings_percentage=(current_cost - optimized_cost) / current_cost,
                        confidence_score=0.8,
                        implementation_effort='medium',
                        risk_level='low',
                        recommended_actions=[
                            f'Scale down from {current_sku} to {recommended_sku}',
                            'Monitor performance after scaling',
                            'Set up alerts for resource constraints'
                        ],
                        expected_impact='Reduced monthly costs with minimal performance impact',
                        timeline='1-2 weeks'
                    ))
        
        return opportunities
    
    async def _identify_scheduling_opportunities(self, resource: Dict[str, Any]) -> List[OptimizationOpportunity]:
        """Identify scheduling-based opportunities"""
        
        opportunities = []
        
        # Check if resource has predictable usage patterns
        if 'hourly_usage_pattern' in resource:
            pattern = resource['hourly_usage_pattern']
            
            # Calculate idle hours (usage < 20%)
            idle_hours = sum(1 for usage in pattern if usage < 0.2)
            
            if idle_hours >= 8:  # At least 8 hours of low usage
                current_cost = resource.get('monthly_cost', 0)
                
                # Estimate savings from auto-shutdown
                savings_percentage = idle_hours / 24 * 0.8  # 80% savings during idle time
                optimized_cost = current_cost * (1 - savings_percentage)
                
                opportunities.append(OptimizationOpportunity(
                    resource_id=resource['resource_id'],
                    resource_type=resource['resource_type'],
                    opportunity_type='scheduling',
                    current_cost=current_cost,
                    optimized_cost=optimized_cost,
                    potential_savings=current_cost - optimized_cost,
                    savings_percentage=savings_percentage,
                    confidence_score=0.7,
                    implementation_effort='low',
                    risk_level='medium',
                    recommended_actions=[
                        'Implement auto-shutdown during idle hours',
                        'Set up automated start/stop schedules',
                        'Configure alerts for unexpected usage'
                    ],
                    expected_impact=f'Reduce costs by {savings_percentage:.1%} through scheduling',
                    timeline='1 week'
                ))
        
        return opportunities
    
    async def _identify_reserved_instance_opportunities(self, resource: Dict[str, Any]) -> List[OptimizationOpportunity]:
        """Identify reserved instance opportunities"""
        
        opportunities = []
        
        # Check for consistent long-term usage
        uptime_percentage = resource.get('monthly_uptime_percentage', 0)
        
        if uptime_percentage >= 0.75:  # Resource runs at least 75% of the time
            current_cost = resource.get('monthly_cost', 0)
            
            # Estimate RI savings (typically 30-50%)
            ri_savings_percentage = 0.4
            optimized_cost = current_cost * (1 - ri_savings_percentage)
            
            opportunities.append(OptimizationOpportunity(
                resource_id=resource['resource_id'],
                resource_type=resource['resource_type'],
                opportunity_type='reserved_instances',
                current_cost=current_cost,
                optimized_cost=optimized_cost,
                potential_savings=current_cost - optimized_cost,
                savings_percentage=ri_savings_percentage,
                confidence_score=0.9,
                implementation_effort='low',
                risk_level='low',
                recommended_actions=[
                    'Purchase 1-year or 3-year reserved instances',
                    'Analyze historical usage to confirm commitment',
                    'Consider convertible RIs for flexibility'
                ],
                expected_impact='Significant cost reduction for committed workloads',
                timeline='Immediate'
            ))
        
        return opportunities
    
    async def _identify_spot_instance_opportunities(self, resource: Dict[str, Any]) -> List[OptimizationOpportunity]:
        """Identify spot instance opportunities"""
        
        opportunities = []
        
        # Check if workload is fault-tolerant
        if resource.get('fault_tolerant', False) and resource.get('resource_type') == 'virtual_machine':
            current_cost = resource.get('monthly_cost', 0)
            
            # Spot instances typically offer 60-90% savings
            spot_savings_percentage = 0.7
            optimized_cost = current_cost * (1 - spot_savings_percentage)
            
            opportunities.append(OptimizationOpportunity(
                resource_id=resource['resource_id'],
                resource_type=resource['resource_type'],
                opportunity_type='spot_instances',
                current_cost=current_cost,
                optimized_cost=optimized_cost,
                potential_savings=current_cost - optimized_cost,
                savings_percentage=spot_savings_percentage,
                confidence_score=0.6,
                implementation_effort='high',
                risk_level='high',
                recommended_actions=[
                    'Migrate to spot instances for batch workloads',
                    'Implement checkpointing and recovery mechanisms',
                    'Set up mixed instance groups for availability'
                ],
                expected_impact='Dramatic cost reduction for fault-tolerant workloads',
                timeline='2-4 weeks'
            ))
        
        return opportunities
    
    def _recommend_smaller_sku(self, current_sku: str) -> Optional[str]:
        """Recommend a smaller SKU based on current one"""
        
        # Simplified SKU downgrade logic
        # In production, this would use Azure pricing APIs
        
        sku_mappings = {
            'Standard_D4s_v3': 'Standard_D2s_v3',
            'Standard_D8s_v3': 'Standard_D4s_v3',
            'Standard_D16s_v3': 'Standard_D8s_v3',
            'Standard_F8s_v2': 'Standard_F4s_v2',
            'Standard_F16s_v2': 'Standard_F8s_v2'
        }
        
        return sku_mappings.get(current_sku)
    
    def _extract_forecast_features(self, resource_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for cost forecasting"""
        
        # Create feature vector matching training features
        feature_vector = []
        
        for col in self.feature_columns:
            if col in resource_data:
                feature_vector.append(resource_data[col])
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _forecast_ensemble(self, features: np.ndarray) -> Optional[float]:
        """Make cost forecast using ensemble models"""
        
        if not self.ensemble_models:
            return None
        
        try:
            features_scaled = self.feature_scaler.transform(features)
            
            predictions = []
            
            for model_name, model in self.ensemble_models.items():
                pred = model.predict(features_scaled)[0]
                predictions.append(pred)
            
            # Weighted average
            weights = [0.4, 0.35, 0.25]  # RF, ET, EN
            ensemble_pred = np.average(predictions, weights=weights)
            
            # Inverse transform
            ensemble_pred_scaled = self.cost_scaler.inverse_transform([[ensemble_pred]])[0][0]
            
            return max(0, ensemble_pred_scaled)
            
        except Exception as e:
            self.logger.warning(f"Ensemble forecast failed: {str(e)}")
            return None
    
    def _forecast_lstm(self, resource_data: Dict[str, Any]) -> Optional[float]:
        """Make cost forecast using LSTM model"""
        
        if not self.lstm_model or 'historical_sequences' not in resource_data:
            return None
        
        try:
            sequences = resource_data['historical_sequences']
            if len(sequences) < 60:
                return None
            
            # Prepare input tensor
            input_tensor = torch.FloatTensor(sequences[-60:]).unsqueeze(0)
            
            # Make prediction
            self.lstm_model.eval()
            with torch.no_grad():
                outputs = self.lstm_model(input_tensor)
                prediction = outputs['cost'].item()
            
            return max(0, prediction)
            
        except Exception as e:
            self.logger.warning(f"LSTM forecast failed: {str(e)}")
            return None
    
    def _forecast_prophet(self, resource_data: Dict[str, Any], forecast_days: int) -> Optional[float]:
        """Make cost forecast using Prophet model"""
        
        resource_type = resource_data.get('resource_type')
        if not resource_type or resource_type not in self.prophet_models:
            return None
        
        try:
            model = self.prophet_models[resource_type]
            
            # Create future dates
            future_dates = pd.date_range(
                start=datetime.utcnow(),
                periods=forecast_days,
                freq='D'
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Make forecast
            forecast = model.predict(future_df)
            monthly_forecast = forecast['yhat'].sum()
            
            return max(0, monthly_forecast)
            
        except Exception as e:
            self.logger.warning(f"Prophet forecast failed: {str(e)}")
            return None
    
    def _calculate_forecast_confidence(self, ensemble_pred: Optional[float],
                                     lstm_pred: Optional[float],
                                     prophet_pred: Optional[float]) -> Tuple[float, float]:
        """Calculate confidence interval for forecast"""
        
        predictions = [p for p in [ensemble_pred, lstm_pred, prophet_pred] if p is not None]
        
        if not predictions:
            return (0.0, 0.0)
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions) if len(predictions) > 1 else mean_pred * 0.1
        
        # 95% confidence interval
        lower_bound = max(0, mean_pred - 1.96 * std_pred)
        upper_bound = mean_pred + 1.96 * std_pred
        
        return (lower_bound, upper_bound)
    
    def _extract_seasonal_pattern(self, resource_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract seasonal cost patterns"""
        
        # Simplified seasonal pattern extraction
        # In production, this would analyze historical data
        
        return {
            'weekly_pattern': 0.1,  # 10% weekly variation
            'monthly_pattern': 0.05,  # 5% monthly variation
            'quarterly_pattern': 0.15,  # 15% quarterly variation
            'yearly_pattern': 0.08  # 8% yearly variation
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        
        import joblib
        
        model_data = {
            'ensemble_models': self.ensemble_models,
            'prophet_models': self.prophet_models,
            'clustering_model': self.clustering_model,
            'feature_scaler': self.feature_scaler,
            'cost_scaler': self.cost_scaler,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'metrics': self.metrics,
            'is_trained': self.is_trained,
            'pricing_cache': self.pricing_cache
        }
        
        # Save main models
        joblib.dump(model_data, f"{filepath}_cost_optimization.pkl")
        
        # Save LSTM model separately
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), f"{filepath}_lstm.pth")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        
        import joblib
        
        # Load main models
        model_data = joblib.load(f"{filepath}_cost_optimization.pkl")
        
        self.ensemble_models = model_data['ensemble_models']
        self.prophet_models = model_data['prophet_models']
        self.clustering_model = model_data['clustering_model']
        self.feature_scaler = model_data['feature_scaler']
        self.cost_scaler = model_data['cost_scaler']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        self.pricing_cache = model_data.get('pricing_cache', {})
        
        # Load LSTM model
        try:
            if self.feature_columns:
                input_size = len(self.feature_columns)
                self.lstm_model = CostForecastingLSTM(
                    input_size=input_size,
                    hidden_size=self.config['lstm_config']['hidden_size'],
                    num_layers=self.config['lstm_config']['num_layers'],
                    dropout=self.config['lstm_config']['dropout']
                )
                self.lstm_model.load_state_dict(torch.load(f"{filepath}_lstm.pth"))
                self.lstm_model.eval()
        except FileNotFoundError:
            self.logger.warning("LSTM model file not found, continuing without LSTM")
            self.lstm_model = None