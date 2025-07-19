"""
Policy Compliance Prediction Model

This module implements advanced machine learning models for predicting future policy compliance
states, identifying resources at risk of non-compliance, and recommending preventive actions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import joblib
import asyncio

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Time Series Libraries
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CompliancePrediction:
    """Data structure for compliance predictions"""
    resource_id: str
    policy_type: str
    current_compliance_score: float
    predicted_compliance_score: float
    risk_level: str
    confidence_interval: Tuple[float, float]
    recommended_actions: List[str]
    prediction_timestamp: datetime
    explanation: str

class LSTMComplianceModel(nn.Module):
    """LSTM Neural Network for time series compliance prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMComplianceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last output
        output = attn_out[:, -1, :]
        
        # Fully connected layers
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.relu(self.fc2(output))
        output = self.dropout(output)
        output = self.sigmoid(self.fc3(output))
        
        return output

class PolicyCompliancePredictor:
    """
    Advanced policy compliance prediction model using ensemble methods,
    time series analysis, and deep learning techniques.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.ensemble_model = None
        self.lstm_model = None
        self.prophet_models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Feature engineering
        self.feature_columns = []
        self.is_trained = False
        
        # Model performance metrics
        self.metrics = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the compliance predictor"""
        return {
            'ensemble_models': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 150,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'random_state': 42
                }
            },
            'lstm_config': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'prophet_config': {
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False
            },
            'prediction_horizon_days': 30,
            'confidence_threshold': 0.8,
            'risk_thresholds': {
                'low': 0.8,
                'medium': 0.6,
                'high': 0.4,
                'critical': 0.2
            }
        }
    
    async def train(self, training_data: pd.DataFrame) -> Dict[str, float]:
        """
        Train the compliance prediction models
        
        Args:
            training_data: DataFrame with compliance history and features
            
        Returns:
            Dictionary with training metrics
        """
        try:
            self.logger.info("Starting compliance prediction model training")
            
            # Prepare data
            X, y, time_series_data = self._prepare_training_data(training_data)
            
            # Train ensemble models
            ensemble_metrics = await self._train_ensemble_models(X, y)
            
            # Train LSTM model
            lstm_metrics = await self._train_lstm_model(time_series_data)
            
            # Train Prophet models for each policy type
            prophet_metrics = await self._train_prophet_models(training_data)
            
            self.is_trained = True
            self.metrics = {
                **ensemble_metrics,
                **lstm_metrics, 
                **prophet_metrics
            }
            
            self.logger.info(f"Training completed with metrics: {self.metrics}")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Prepare and engineer features for training"""
        
        # Feature engineering
        features = []
        
        # Resource characteristics
        if 'resource_type' in data.columns:
            features.extend(self._encode_categorical_features(data, ['resource_type']))
        
        # Policy features
        if 'policy_type' in data.columns:
            features.extend(self._encode_categorical_features(data, ['policy_type']))
        
        # Temporal features
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            features.extend(self._extract_temporal_features(data))
        
        # Historical compliance features
        if 'compliance_score' in data.columns:
            features.extend(self._extract_compliance_features(data))
        
        # Resource utilization features
        if 'cpu_utilization' in data.columns:
            features.extend(self._extract_utilization_features(data))
        
        # Configuration drift features
        if 'config_changes' in data.columns:
            features.extend(self._extract_config_features(data))
        
        # Prepare feature matrix
        feature_df = pd.concat(features, axis=1)
        self.feature_columns = feature_df.columns.tolist()
        
        # Target variable
        target = data['compliance_score'] if 'compliance_score' in data.columns else None
        
        # Time series data for LSTM
        time_series_data = self._prepare_time_series_data(data)
        
        return feature_df, target, time_series_data
    
    def _encode_categorical_features(self, data: pd.DataFrame, columns: List[str]) -> List[pd.DataFrame]:
        """Encode categorical features"""
        encoded_features = []
        
        for col in columns:
            if col in data.columns:
                # One-hot encoding for categorical variables
                encoded = pd.get_dummies(data[col], prefix=col)
                encoded_features.append(encoded)
        
        return encoded_features
    
    def _extract_temporal_features(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract temporal features from timestamp"""
        features = []
        
        timestamp_col = data['timestamp']
        
        # Basic temporal features
        temporal_features = pd.DataFrame({
            'hour': timestamp_col.dt.hour,
            'day_of_week': timestamp_col.dt.dayofweek,
            'day_of_month': timestamp_col.dt.day,
            'month': timestamp_col.dt.month,
            'quarter': timestamp_col.dt.quarter,
            'is_weekend': (timestamp_col.dt.dayofweek >= 5).astype(int),
            'is_business_hours': ((timestamp_col.dt.hour >= 9) & (timestamp_col.dt.hour <= 17)).astype(int)
        })
        
        features.append(temporal_features)
        
        return features
    
    def _extract_compliance_features(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract historical compliance features"""
        features = []
        
        # Rolling statistics
        compliance_features = pd.DataFrame({
            'compliance_score_lag_1': data['compliance_score'].shift(1),
            'compliance_score_lag_7': data['compliance_score'].shift(7),
            'compliance_score_rolling_mean_7': data['compliance_score'].rolling(7).mean(),
            'compliance_score_rolling_std_7': data['compliance_score'].rolling(7).std(),
            'compliance_score_rolling_mean_30': data['compliance_score'].rolling(30).mean(),
            'compliance_score_rolling_std_30': data['compliance_score'].rolling(30).std(),
            'compliance_trend_7': data['compliance_score'].rolling(7).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
            )
        })
        
        features.append(compliance_features)
        
        return features
    
    def _extract_utilization_features(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract resource utilization features"""
        features = []
        
        utilization_cols = ['cpu_utilization', 'memory_utilization', 'disk_utilization']
        utilization_features = pd.DataFrame()
        
        for col in utilization_cols:
            if col in data.columns:
                utilization_features[f'{col}_current'] = data[col]
                utilization_features[f'{col}_rolling_mean_7'] = data[col].rolling(7).mean()
                utilization_features[f'{col}_rolling_max_7'] = data[col].rolling(7).max()
                utilization_features[f'{col}_rolling_std_7'] = data[col].rolling(7).std()
        
        if not utilization_features.empty:
            features.append(utilization_features)
        
        return features
    
    def _extract_config_features(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Extract configuration change features"""
        features = []
        
        if 'config_changes' in data.columns:
            config_features = pd.DataFrame({
                'config_changes_count': data['config_changes'],
                'config_changes_rolling_sum_7': data['config_changes'].rolling(7).sum(),
                'config_changes_rolling_sum_30': data['config_changes'].rolling(30).sum(),
                'days_since_last_change': data.groupby('resource_id')['config_changes'].cumsum().diff().fillna(0)
            })
            
            features.append(config_features)
        
        return features
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for LSTM model"""
        
        if 'timestamp' not in data.columns or 'compliance_score' not in data.columns:
            return pd.DataFrame()
        
        # Sort by timestamp
        time_series = data.sort_values('timestamp')
        
        # Create sequences for LSTM
        sequence_length = 30  # 30 time steps
        sequences = []
        targets = []
        
        for resource_id in time_series['resource_id'].unique():
            resource_data = time_series[time_series['resource_id'] == resource_id].copy()
            
            if len(resource_data) < sequence_length + 1:
                continue
            
            # Prepare features for this resource
            feature_cols = ['compliance_score']
            if 'cpu_utilization' in resource_data.columns:
                feature_cols.append('cpu_utilization')
            if 'memory_utilization' in resource_data.columns:
                feature_cols.append('memory_utilization')
            
            resource_features = resource_data[feature_cols].values
            
            # Create sequences
            for i in range(len(resource_features) - sequence_length):
                sequences.append(resource_features[i:i+sequence_length])
                targets.append(resource_features[i+sequence_length][0])  # compliance_score
        
        return {
            'sequences': np.array(sequences),
            'targets': np.array(targets)
        }
    
    async def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train ensemble models (Random Forest + Gradient Boosting)"""
        
        if X.empty or y is None:
            return {}
        
        # Fill missing values
        X_clean = X.fillna(X.mean())
        y_clean = y.fillna(y.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_clean, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_config = self.config['ensemble_models']['random_forest']
        rf_model = RandomForestClassifier(**rf_config)
        rf_model.fit(X_train, (y_train > 0.5).astype(int))
        
        # Train Gradient Boosting
        gb_config = self.config['ensemble_models']['gradient_boosting']
        gb_model = GradientBoostingRegressor(**gb_config)
        gb_model.fit(X_train, y_train)
        
        # Create ensemble
        self.ensemble_model = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'scaler': self.scaler
        }
        
        # Evaluate models
        rf_score = rf_model.score(X_test, (y_test > 0.5).astype(int))
        gb_score = r2_score(y_test, gb_model.predict(X_test))
        
        return {
            'random_forest_accuracy': rf_score,
            'gradient_boosting_r2': gb_score
        }
    
    async def _train_lstm_model(self, time_series_data: Dict) -> Dict[str, float]:
        """Train LSTM model for time series prediction"""
        
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
        train_loader = DataLoader(train_dataset, batch_size=self.config['lstm_config']['batch_size'], shuffle=True)
        
        # Initialize model
        input_size = sequences.shape[2]
        self.lstm_model = LSTMComplianceModel(
            input_size=input_size,
            hidden_size=self.config['lstm_config']['hidden_size'],
            num_layers=self.config['lstm_config']['num_layers'],
            dropout=self.config['lstm_config']['dropout']
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.lstm_model.parameters(), lr=self.config['lstm_config']['learning_rate'])
        
        # Training loop
        self.lstm_model.train()
        for epoch in range(self.config['lstm_config']['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Evaluate
        self.lstm_model.eval()
        with torch.no_grad():
            test_outputs = self.lstm_model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_mae = mean_absolute_error(y_test.numpy(), test_outputs.numpy())
        
        return {
            'lstm_test_loss': test_loss,
            'lstm_test_mae': test_mae
        }
    
    async def _train_prophet_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train Prophet models for each policy type"""
        
        if 'timestamp' not in data.columns or 'compliance_score' not in data.columns:
            return {}
        
        metrics = {}
        
        # Train separate models for each policy type
        for policy_type in data['policy_type'].unique():
            policy_data = data[data['policy_type'] == policy_type].copy()
            
            if len(policy_data) < 50:  # Need enough data points
                continue
            
            # Prepare data for Prophet
            prophet_data = policy_data[['timestamp', 'compliance_score']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.sort_values('ds')
            
            # Create and train model
            model = Prophet(**self.config['prophet_config'])
            model.fit(prophet_data)
            
            # Store model
            self.prophet_models[policy_type] = model
            
            # Evaluate on last 20% of data
            split_idx = int(0.8 * len(prophet_data))
            train_data = prophet_data[:split_idx]
            test_data = prophet_data[split_idx:]
            
            if len(test_data) > 0:
                forecast = model.predict(test_data[['ds']])
                mae = mean_absolute_error(test_data['y'], forecast['yhat'])
                metrics[f'prophet_{policy_type}_mae'] = mae
        
        return metrics
    
    async def predict(self, input_data: Dict[str, Any]) -> CompliancePrediction:
        """
        Predict compliance score and risk for a resource
        
        Args:
            input_data: Dictionary containing resource information and features
            
        Returns:
            CompliancePrediction object with predictions and recommendations
        """
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Extract features
            features = self._extract_prediction_features(input_data)
            
            # Ensemble prediction
            ensemble_score = self._predict_ensemble(features)
            
            # LSTM prediction (if time series data available)
            lstm_score = self._predict_lstm(input_data)
            
            # Prophet prediction (if applicable)
            prophet_score = self._predict_prophet(input_data)
            
            # Combine predictions with weighted average
            weights = {'ensemble': 0.4, 'lstm': 0.35, 'prophet': 0.25}
            final_score = (
                weights['ensemble'] * (ensemble_score or 0) +
                weights['lstm'] * (lstm_score or 0) +
                weights['prophet'] * (prophet_score or 0)
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(final_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(input_data, final_score, risk_level)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                ensemble_score, lstm_score, prophet_score
            )
            
            # Generate explanation
            explanation = self._generate_explanation(input_data, final_score, risk_level)
            
            return CompliancePrediction(
                resource_id=input_data.get('resource_id', 'unknown'),
                policy_type=input_data.get('policy_type', 'unknown'),
                current_compliance_score=input_data.get('current_compliance_score', 0.0),
                predicted_compliance_score=final_score,
                risk_level=risk_level,
                confidence_interval=confidence_interval,
                recommended_actions=recommendations,
                prediction_timestamp=datetime.utcnow(),
                explanation=explanation
            )
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def _extract_prediction_features(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction from input data"""
        
        # Create feature vector matching training features
        feature_vector = []
        
        # Add features in the same order as training
        for col in self.feature_columns:
            if col in input_data:
                feature_vector.append(input_data[col])
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _predict_ensemble(self, features: np.ndarray) -> Optional[float]:
        """Make prediction using ensemble models"""
        
        if not self.ensemble_model:
            return None
        
        try:
            # Scale features
            features_scaled = self.ensemble_model['scaler'].transform(features)
            
            # Random Forest prediction (probability of compliance)
            rf_pred = self.ensemble_model['random_forest'].predict_proba(features_scaled)[0][1]
            
            # Gradient Boosting prediction (regression score)
            gb_pred = self.ensemble_model['gradient_boosting'].predict(features_scaled)[0]
            
            # Combine predictions
            combined_score = 0.6 * rf_pred + 0.4 * gb_pred
            
            return np.clip(combined_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Ensemble prediction failed: {str(e)}")
            return None
    
    def _predict_lstm(self, input_data: Dict[str, Any]) -> Optional[float]:
        """Make prediction using LSTM model"""
        
        if not self.lstm_model or 'historical_data' not in input_data:
            return None
        
        try:
            # Prepare sequence data
            historical_data = input_data['historical_data']
            if len(historical_data) < 30:
                return None
            
            # Convert to tensor
            sequence = torch.FloatTensor(historical_data[-30:]).unsqueeze(0)
            
            # Make prediction
            self.lstm_model.eval()
            with torch.no_grad():
                prediction = self.lstm_model(sequence)
                
            return float(prediction.item())
            
        except Exception as e:
            self.logger.warning(f"LSTM prediction failed: {str(e)}")
            return None
    
    def _predict_prophet(self, input_data: Dict[str, Any]) -> Optional[float]:
        """Make prediction using Prophet model"""
        
        policy_type = input_data.get('policy_type')
        if not policy_type or policy_type not in self.prophet_models:
            return None
        
        try:
            model = self.prophet_models[policy_type]
            
            # Create future dataframe for prediction
            future_date = datetime.utcnow() + timedelta(days=1)
            future_df = pd.DataFrame({'ds': [future_date]})
            
            # Make prediction
            forecast = model.predict(future_df)
            predicted_score = forecast['yhat'].iloc[0]
            
            return np.clip(predicted_score, 0.0, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Prophet prediction failed: {str(e)}")
            return None
    
    def _determine_risk_level(self, compliance_score: float) -> str:
        """Determine risk level based on compliance score"""
        
        thresholds = self.config['risk_thresholds']
        
        if compliance_score >= thresholds['low']:
            return 'low'
        elif compliance_score >= thresholds['medium']:
            return 'medium'
        elif compliance_score >= thresholds['high']:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendations(self, input_data: Dict[str, Any], 
                                compliance_score: float, risk_level: str) -> List[str]:
        """Generate actionable recommendations based on prediction"""
        
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.extend([
                "Immediate review of resource configuration required",
                "Schedule compliance audit within 24 hours",
                "Consider implementing automated remediation"
            ])
        
        if compliance_score < 0.6:
            recommendations.extend([
                "Update resource tags to meet policy requirements",
                "Review and update security group configurations",
                "Implement monitoring and alerting for this resource"
            ])
        
        # Resource-specific recommendations
        resource_type = input_data.get('resource_type', '')
        if 'vm' in resource_type.lower():
            recommendations.extend([
                "Ensure VM is using approved images",
                "Verify disk encryption is enabled",
                "Check for required extensions installation"
            ])
        elif 'storage' in resource_type.lower():
            recommendations.extend([
                "Enable storage account encryption",
                "Configure network access restrictions",
                "Set up blob lifecycle management"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_confidence_interval(self, ensemble_score: Optional[float],
                                     lstm_score: Optional[float],
                                     prophet_score: Optional[float]) -> Tuple[float, float]:
        """Calculate confidence interval for the prediction"""
        
        scores = [score for score in [ensemble_score, lstm_score, prophet_score] if score is not None]
        
        if not scores:
            return (0.0, 1.0)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.1
        
        # 95% confidence interval
        lower_bound = max(0.0, mean_score - 1.96 * std_score)
        upper_bound = min(1.0, mean_score + 1.96 * std_score)
        
        return (lower_bound, upper_bound)
    
    def _generate_explanation(self, input_data: Dict[str, Any], 
                            compliance_score: float, risk_level: str) -> str:
        """Generate human-readable explanation for the prediction"""
        
        explanations = []
        
        explanations.append(f"Predicted compliance score: {compliance_score:.2f}")
        explanations.append(f"Risk level: {risk_level}")
        
        if compliance_score < 0.5:
            explanations.append("Low compliance score indicates high probability of policy violations")
        elif compliance_score < 0.7:
            explanations.append("Moderate compliance score suggests some policy compliance issues")
        else:
            explanations.append("High compliance score indicates good policy adherence")
        
        # Add context about prediction confidence
        if len(self.prophet_models) > 0:
            explanations.append("Prediction includes time series trend analysis")
        
        if self.lstm_model is not None:
            explanations.append("Deep learning model used for pattern recognition")
        
        return ". ".join(explanations)
    
    async def batch_predict(self, resources: List[Dict[str, Any]]) -> List[CompliancePrediction]:
        """Predict compliance for multiple resources efficiently"""
        
        predictions = []
        
        # Process in batches for efficiency
        batch_size = 100
        for i in range(0, len(resources), batch_size):
            batch = resources[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self.predict(resource) for resource in batch]
            batch_predictions = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions and add successful predictions
            for pred in batch_predictions:
                if isinstance(pred, CompliancePrediction):
                    predictions.append(pred)
                else:
                    self.logger.warning(f"Batch prediction failed: {pred}")
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'prophet_models': self.prophet_models,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }
        
        # Save ensemble and prophet models
        joblib.dump(model_data, f"{filepath}_ensemble_prophet.pkl")
        
        # Save LSTM model separately
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), f"{filepath}_lstm.pth")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        
        # Load ensemble and prophet models
        model_data = joblib.load(f"{filepath}_ensemble_prophet.pkl")
        
        self.ensemble_model = model_data['ensemble_model']
        self.prophet_models = model_data['prophet_models']
        self.feature_columns = model_data['feature_columns']
        self.config = model_data['config']
        self.metrics = model_data['metrics']
        self.is_trained = model_data['is_trained']
        
        # Load LSTM model
        try:
            if self.feature_columns:
                input_size = len(self.feature_columns)
                self.lstm_model = LSTMComplianceModel(
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from ensemble models"""
        
        if not self.ensemble_model or not self.feature_columns:
            return {}
        
        # Get importance from Random Forest
        rf_importance = self.ensemble_model['random_forest'].feature_importances_
        
        # Get importance from Gradient Boosting
        gb_importance = self.ensemble_model['gradient_boosting'].feature_importances_
        
        # Combine importances
        combined_importance = {}
        for i, feature in enumerate(self.feature_columns):
            combined_importance[feature] = (rf_importance[i] + gb_importance[i]) / 2
        
        # Sort by importance
        return dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get comprehensive model performance metrics"""
        
        performance = {
            'is_trained': self.is_trained,
            'training_metrics': self.metrics,
            'model_components': {
                'ensemble_available': self.ensemble_model is not None,
                'lstm_available': self.lstm_model is not None,
                'prophet_models_count': len(self.prophet_models)
            }
        }
        
        if self.feature_columns:
            performance['feature_count'] = len(self.feature_columns)
            performance['top_features'] = list(self.get_feature_importance().keys())[:10]
        
        return performance