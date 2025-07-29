"""
Policy Compliance Prediction Model for PolicyCortex.

This module implements advanced ML models for predicting future policy compliance states,
    identifying resources at risk of non-compliance, and recommending preventive actions.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class PolicyComplianceAttentionModel(nn.Module):
    """
    Advanced neural network with attention mechanism for policy compliance prediction.
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.classifier = nn.Linear(hidden_size // 2, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        # Activation
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            mask: Optional attention mask

        Returns:
            Dictionary containing predictions and attention weights
        """
        batch_size, seq_len, _ = x.shape

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)

        # Apply attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=mask
        )

        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, attn_out.size(-1))
            attn_out_masked = attn_out.masked_fill(mask_expanded, 0)
            seq_lengths = (~mask).sum(dim=1, keepdim=True).float()
            pooled = attn_out_masked.sum(dim=1) / seq_lengths
        else:
            pooled = attn_out.mean(dim=1)  # (batch_size, hidden_size*2)

        # Classification layers
        x = self.dropout(pooled)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        predictions = self.classifier(x)

        return {
            'predictions': predictions,
            'attention_weights': attn_weights,
            'hidden_states': lstm_out
        }


class PolicyCompliancePredictor:
    """
    Advanced Policy Compliance Prediction system with ensemble methods,
    time series analysis, and attention-based neural networks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Policy Compliance Predictor.

        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}

        # Azure clients
        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential)

        # Initialize models
        self._initialize_models()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the model."""
        return {
            'sequence_length': 30,  # Days of history to consider
            'prediction_horizon': 7,  # Days to predict ahead
            'ensemble_weights': {
                'rf': 0.25,
                'gb': 0.25,
                'xgb': 0.25,
                'neural': 0.25
            },
            'neural_config': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'epochs': 100,
                'batch_size': 32
            },
            'feature_engineering': {
                'use_lag_features': True,
                'use_rolling_features': True,
                'use_seasonal_features': True,
                'lag_periods': [1, 3, 7, 14, 30],
                'rolling_windows': [3, 7, 14, 30]
            }
        }

    def _initialize_models(self):
        """Initialize all models in the ensemble."""
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )

        # XGBoost
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            random_state=42,
            eval_metric='mlogloss'
        )

        # Logistic Regression (for baseline)
        self.models['lr'] = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        # Scalers and encoders
        self.scalers['numerical'] = StandardScaler()
        self.scalers['imputer'] = SimpleImputer(strategy='median')
        self.encoders['label'] = LabelEncoder()
        self.encoders['categorical'] = {}

    async def prepare_training_data(self, workspace_id: str, days_back: int = 90) -> pd.DataFrame:
        """
        Prepare training data from Azure logs and policy evaluation history.

        Args:
            workspace_id: Log Analytics workspace ID
            days_back: Number of days of historical data to fetch

        Returns:
            Prepared training dataset
        """
        logger.info(f"Preparing training data for {days_back} days")

        # Query for policy evaluation history
        query = f"""
        union AppTraces, AppEvents
        | where TimeGenerated > ago({days_back}d)
        | where Message contains "PolicyEvaluation" or Name == "PolicyEvaluated"
        | extend PolicyId = tostring(Properties.PolicyId),
                 ResourceId = tostring(Properties.ResourceId),
                 ComplianceState = tostring(Properties.ComplianceState),
                 PolicyType = tostring(Properties.PolicyType),
                 Severity = tostring(Properties.Severity),
                 TenantId = tostring(Properties.TenantId)
        | where isnotempty(PolicyId) and isnotempty(ResourceId)
        | project TimeGenerated, PolicyId, ResourceId, ComplianceState, PolicyType, Severity, TenantId, Properties
        | order by TimeGenerated asc
        """

        try:
            response = await self.logs_client.query_workspace(
                workspace_id=workspace_id,
                query=query,
                timespan=timedelta(days=days_back)
            )

            # Convert to DataFrame
            df = pd.DataFrame([row for row in response.tables[0].rows])
            if not df.empty:
                df.columns = [col.name for col in response.tables[0].columns]

            logger.info(f"Retrieved {len(df)} policy evaluation records")

            # Engineer features
            df = self._engineer_features(df)

            return df

        except Exception as e:
            logger.error(f"Error fetching training data: {str(e)}")
            # Return sample data for testing
            return self._generate_sample_data(days_back)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for policy compliance prediction.

        Args:
            df: Raw policy evaluation data

        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df

        logger.info("Engineering features for policy compliance prediction")

        # Convert timestamp
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        df = df.sort_values('TimeGenerated')

        # Create compliance binary target
        df['is_compliant'] = (df['ComplianceState'].isin(['Compliant', 'compliant'])).astype(int)

        # Time-based features
        df['hour'] = df['TimeGenerated'].dt.hour
        df['day_of_week'] = df['TimeGenerated'].dt.dayofweek
        df['day_of_month'] = df['TimeGenerated'].dt.day
        df['month'] = df['TimeGenerated'].dt.month
        df['quarter'] = df['TimeGenerated'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

        # Policy-specific features
        df['policy_age_days'] = (
            (df['TimeGenerated'] - df.groupby('PolicyId')['TimeGenerated'].transform('min')).dt.days
        )
        df['resource_age_days'] = (
            (df['TimeGenerated'] - df.groupby('ResourceId')['TimeGenerated'].transform('min')).dt.days
        )

        # Compliance history features
        for window in self.config['feature_engineering']['rolling_windows']:
            df[f'compliance_rate_{window}d'] = df.groupby('ResourceId')['is_compliant'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

            df[f'policy_violations_{window}d'] = df.groupby('PolicyId')['is_compliant'].transform(
                lambda x: (1 - x).rolling(window=window, min_periods=1).sum()
            )

        # Lag features
        for lag in self.config['feature_engineering']['lag_periods']:
            df[f'compliance_lag_{lag}d'] = df.groupby('ResourceId')['is_compliant'].shift(lag)

        # Resource and policy statistics
        resource_stats = df.groupby('ResourceId').agg({
            'is_compliant': ['mean', 'std', 'count'],
            'PolicyId': 'nunique'
        }).round(4)
        resource_stats.columns = ['resource_compliance_mean', 'resource_compliance_std',
                                 'resource_eval_count', 'resource_policy_count']
        df = df.merge(resource_stats, left_on='ResourceId', right_index=True, how='left')

        policy_stats = df.groupby('PolicyId').agg({
            'is_compliant': ['mean', 'std', 'count'],
            'ResourceId': 'nunique'
        }).round(4)
        policy_stats.columns = ['policy_compliance_mean', 'policy_compliance_std',
                               'policy_eval_count', 'policy_resource_count']
        df = df.merge(policy_stats, left_on='PolicyId', right_index=True, how='left')

        # Categorical encoding
        categorical_features = ['PolicyType', 'Severity', 'TenantId']
        for feature in categorical_features:
            if feature in df.columns:
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].fillna('Unknown'))
                self.encoders['categorical'][feature] = le

        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        logger.info(f"Feature engineering completed. Dataset shape: {df.shape}")
        return df

    def _generate_sample_data(self, days_back: int) -> pd.DataFrame:
        """Generate sample data for testing when real data is not available."""
        logger.warning("Generating sample data for testing")

        np.random.seed(42)
        n_samples = days_back * 100  # Assume 100 evaluations per day

        # Generate base data
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='15min')
        policy_ids = [f'policy_{i}' for i in range(1, 21)]  # 20 policies
        resource_ids = [f'resource_{i}' for i in range(1, 101)]  # 100 resources

        data = []
        for i, date in enumerate(dates):
            policy_id = np.random.choice(policy_ids)
            resource_id = np.random.choice(resource_ids)

            # Create realistic compliance patterns
            base_compliance_rate = 0.8
            time_factor = 0.1 * np.sin(2 * np.pi * date.hour / 24)  # Time of day effect
            weekend_factor = -0.1 if date.weekday() >= 5 else 0  # Weekend effect

            compliance_prob = base_compliance_rate + time_factor + weekend_factor
            compliance_prob = max(0.1, min(0.95, compliance_prob))

            is_compliant = np.random.random() < compliance_prob

            data.append({
                'TimeGenerated': date,
                'PolicyId': policy_id,
                'ResourceId': resource_id,
                'ComplianceState': 'Compliant' if is_compliant else 'NonCompliant',
                'PolicyType': np.random.choice(['Security', 'Cost', 'Governance', 'Network']),
                'Severity': np.random.choice(['Low', 'Medium', 'High', 'Critical']),
                'TenantId': f'tenant_{np.random.randint(1, 6)}'
            })

        df = pd.DataFrame(data)
        return self._engineer_features(df)

    async def train_ensemble(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ensemble of models for policy compliance prediction.

        Args:
            training_data: Prepared training dataset

        Returns:
            Training results and model performance metrics
        """
        logger.info("Training policy compliance prediction ensemble")

        if training_data.empty:
            raise ValueError("Training data is empty")

        # Prepare features and target
        feature_columns = self._get_feature_columns(training_data)
        X = training_data[feature_columns].copy()
        y = training_data['is_compliant'].copy()

        # Handle missing values
        X = self.scalers['imputer'].fit_transform(X)
        X = pd.DataFrame(X, columns=feature_columns)

        # Scale features
        X_scaled = self.scalers['numerical'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

        # Time series split for evaluation
        tscv = TimeSeriesSplit(n_splits=5)

        training_results = {}

        # Train traditional ML models
        for model_name, model in self.models.items():
            if model_name == 'neural':
                continue

            logger.info(f"Training {model_name} model")

            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')

            # Train on full dataset
            model.fit(X_scaled, y)

            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = dict(zip(
                    feature_columns, model.feature_importances_
                ))

            # Store performance
            self.model_performance[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }

            training_results[model_name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std()
            }

            logger.info(f"{model_name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train neural network
        if 'neural' not in self.models:
            neural_results = await self._train_neural_model(X_scaled, y, feature_columns)
            training_results['neural'] = neural_results

        # Calculate ensemble performance
        ensemble_predictions = self._get_ensemble_predictions(X_scaled)
        ensemble_auc = roc_auc_score(y, ensemble_predictions)

        training_results['ensemble'] = {
            'auc': ensemble_auc,
            'feature_importance': self._calculate_ensemble_feature_importance()
        }

        logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        logger.info("Training completed successfully")

        return training_results

    async def _train_neural_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Train the neural network model with attention mechanism."""
        logger.info("Training neural network with attention mechanism")

        # Prepare sequences for neural network
        sequences, labels = self._prepare_sequences(X, y)

        if len(sequences) == 0:
            logger.warning("No sequences available for neural network training")
            return {'auc': 0.0, 'loss': float('inf')}

        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.LongTensor(labels)

        # Initialize model
        input_size = X_tensor.shape[2]
        self.models['neural'] = PolicyComplianceAttentionModel(
            input_size=input_size,
            **self.config['neural_config']
        )

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.models['neural'].parameters(),
            lr=self.config['neural_config']['learning_rate']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        batch_size = self.config['neural_config']['batch_size']
        epochs = self.config['neural_config']['epochs']

        train_losses = []
        val_aucs = []

        # Simple train/val split
        split_idx = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        self.models['neural'].train()

        for epoch in range(epochs):
            epoch_losses = []

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()

                outputs = self.models['neural'](batch_X)
                loss = criterion(outputs['predictions'], batch_y)

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)

            # Validation
            if epoch % 10 == 0:
                self.models['neural'].eval()
                with torch.no_grad():
                    val_outputs = self.models['neural'](X_val)
                    val_probs = torch.softmax(val_outputs['predictions'], dim=1)[:, 1]
                    val_auc = roc_auc_score(y_val.numpy(), val_probs.numpy())
                    val_aucs.append(val_auc)

                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")

                scheduler.step(avg_loss)
                self.models['neural'].train()

        # Final evaluation
        self.models['neural'].eval()
        with torch.no_grad():
            final_outputs = self.models['neural'](X_val)
            final_probs = torch.softmax(final_outputs['predictions'], dim=1)[:, 1]
            final_auc = roc_auc_score(y_val.numpy(), final_probs.numpy())

        self.model_performance['neural'] = {
            'final_auc': final_auc,
            'train_losses': train_losses,
            'val_aucs': val_aucs
        }

        logger.info(f"Neural network training completed. Final AUC: {final_auc:.4f}")

        return {
            'auc': final_auc,
            'final_loss': train_losses[-1] if train_losses else float('inf')
        }

    def _prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List, List]:
        """Prepare sequences for neural network training."""
        sequences = []
        labels = []

        seq_length = self.config['sequence_length']

        # Group by resource for sequence creation
        resource_groups = X.groupby(
            X.index // len(X) * len(X.index.unique()) if hasattr(X.index,
            'unique') else range(len(X))
        )

        for _, group in resource_groups:
            if len(group) < seq_length:
                continue

            for i in range(len(group) - seq_length + 1):
                seq = group.iloc[i:i+seq_length].values
                label = y.iloc[i+seq_length-1] if i+seq_length-1 < len(y) else y.iloc[-1]

                sequences.append(seq)
                labels.append(label)

        return sequences, labels

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get the list of feature columns for model training."""
        exclude_columns = {
            'TimeGenerated', 'PolicyId', 'ResourceId', 'ComplianceState',
            'is_compliant', 'PolicyType', 'Severity', 'TenantId'
        }

        feature_columns = [col for col in df.columns if col not in exclude_columns]
        return feature_columns

    def _get_ensemble_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get ensemble predictions from all trained models."""
        predictions = []
        weights = self.config['ensemble_weights']

        for model_name, model in self.models.items():
            if model_name == 'neural':
                continue

            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)

            predictions.append(weights.get(model_name, 0.25) * pred)

        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X))

    def _calculate_ensemble_feature_importance(self) -> Dict[str, float]:
        """Calculate weighted ensemble feature importance."""
        if not self.feature_importance:
            return {}

        weights = self.config['ensemble_weights']
        ensemble_importance = {}

        # Get all feature names
        all_features = set()
        for importance_dict in self.feature_importance.values():
            all_features.update(importance_dict.keys())

        # Calculate weighted importance
        for feature in all_features:
            weighted_importance = 0
            total_weight = 0

            for model_name, importance_dict in self.feature_importance.items():
                if feature in importance_dict:
                    weight = weights.get(model_name, 0.25)
                    weighted_importance += weight * importance_dict[feature]
                    total_weight += weight

            if total_weight > 0:
                ensemble_importance[feature] = weighted_importance / total_weight

        return ensemble_importance

    async def predict_compliance_risk(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict compliance risk for resources.

        Args:
            resource_data: Resource information and historical data

        Returns:
            Risk predictions and recommendations
        """
        logger.info(
            f"Predicting compliance risk for resource: {resource_data.get('resource_id', "
            f"'unknown')}"
        )

        try:
            # Prepare features
            features = self._prepare_prediction_features(resource_data)

            if features is None:
                return {
                    'risk_score': 0.5,
                    'risk_level': 'Unknown',
                    'confidence': 0.0,
                    'recommendations': ['Insufficient data for prediction']
                }

            # Get ensemble prediction
            risk_score = self._get_ensemble_predictions(features.reshape(1, -1))[0]

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = 'High'
            elif risk_score >= 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'

            # Generate recommendations
            recommendations = self._generate_recommendations(resource_data, features, risk_score)

            # Calculate confidence based on model agreement
            individual_predictions = []
            for model_name, model in self.models.items():
                if model_name == 'neural':
                    continue
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(features.reshape(1, -1))[0][1]
                    individual_predictions.append(pred)

            confidence = 1.0 - (np.std(individual_predictions) if individual_predictions else 0.5)

            return {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'recommendations': recommendations,
                'model_predictions': {
                    name: float(pred) for name, pred in zip(
                        [name for name in self.models.keys() if name != 'neural'],
                        individual_predictions
                    )
                }
            }

        except Exception as e:
            logger.error(f"Error predicting compliance risk: {str(e)}")
            return {
                'risk_score': 0.5,
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'recommendations': [f'Prediction error: {str(e)}'],
                'error': str(e)
            }

    def _prepare_prediction_features(self, resource_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare features for prediction from resource data."""
        try:
            # This would typically extract features from the resource_data
            # For now, return a mock feature vector
            n_features = 50  # Should match training data
            features = np.random.random(n_features)
            return features
        except Exception as e:
            logger.error(f"Error preparing prediction features: {str(e)}")
            return None

    def _generate_recommendations(self, resource_data: Dict[str, Any],
                                features: np.ndarray, risk_score: float) -> List[str]:
        """Generate actionable recommendations based on risk prediction."""
        recommendations = []

        if risk_score >= 0.7:
            recommendations.extend([
                "Immediate review required - high compliance risk detected",
                "Review resource configuration against policy requirements",
                "Consider implementing automated remediation",
                "Schedule urgent compliance assessment"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "Monitor resource closely for compliance drift",
                "Review policy alignment within next 7 days",
                "Consider implementing compliance automation"
            ])
        else:
            recommendations.extend([
                "Resource appears compliant",
                "Continue regular monitoring",
                "No immediate action required"
            ])

        return recommendations

    async def save_model(self, model_path: str) -> bool:
        """Save the trained model ensemble to disk."""
        try:
            model_path = Path(model_path)
            model_path.mkdir(parents=True, exist_ok=True)

            # Save traditional ML models
            for model_name, model in self.models.items():
                if model_name == 'neural':
                    # Save PyTorch model
                    torch.save(model.state_dict(), model_path / f'{model_name}_model.pth')
                else:
                    # Save sklearn models
                    with open(model_path / f'{model_name}_model.pkl', 'wb') as f:
                        pickle.dump(model, f)

            # Save scalers and encoders
            with open(model_path / 'scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)

            with open(model_path / 'encoders.pkl', 'wb') as f:
                pickle.dump(self.encoders, f)

            # Save configuration and metadata
            metadata = {
                'config': self.config,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'timestamp': datetime.now().isoformat()
            }

            with open(model_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved successfully to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    async def load_model(self, model_path: str) -> bool:
        """Load a trained model ensemble from disk."""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False

            # Load metadata
            with open(model_path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                self.config = metadata['config']
                self.feature_importance = metadata['feature_importance']
                self.model_performance = metadata['model_performance']

            # Load scalers and encoders
            with open(model_path / 'scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)

            with open(model_path / 'encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)

            # Load traditional ML models
            for model_name in ['rf', 'gb', 'xgb', 'lr']:
                model_file = model_path / f'{model_name}_model.pkl'
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

            # Load neural network model
            neural_model_file = model_path / 'neural_model.pth'
            if neural_model_file.exists():
                # Need to know the architecture to load
                # This would be stored in metadata in a real implementation
                input_size = 50  # Should come from metadata
                self.models['neural'] = PolicyComplianceAttentionModel(
                    input_size=input_size,
                    **self.config['neural_config']
                )
                self.models['neural'].load_state_dict(torch.load(neural_model_file))
                self.models['neural'].eval()

            logger.info(f"Model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize predictor
        predictor = PolicyCompliancePredictor()

        # Generate sample training data
        training_data = await predictor.prepare_training_data("sample_workspace", days_back=30)

        if not training_data.empty:
            # Train the ensemble
            results = await predictor.train_ensemble(training_data)
            print("Training Results:", results)

            # Test prediction
            sample_resource = {
                'resource_id': 'test_resource_1',
                'policy_id': 'test_policy_1',
                'resource_type': 'Microsoft.Compute/virtualMachines',
                'compliance_history': [1, 1, 0, 1, 1]
            }

            prediction = await predictor.predict_compliance_risk(sample_resource)
            print("Risk Prediction:", prediction)

            # Save model
            await predictor.save_model('./models/policy_compliance')

        else:
            print("No training data available")

    # Run the example
    asyncio.run(main())
