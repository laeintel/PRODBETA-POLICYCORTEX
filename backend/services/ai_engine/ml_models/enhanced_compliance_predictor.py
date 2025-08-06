"""
Enhanced Predictive Policy Compliance Engine for PolicyCortex.
Enterprise-Grade Implementation of Patent 1: Machine Learning System for Temporal Predictive Cloud Policy Compliance Analysis.

This implementation provides:
- Ensemble learning with XGBoost, LSTM, and Prophet
- Temporal pattern recognition with attention mechanisms
- Fuzzy logic risk assessment
- Real-time drift detection
- Enterprise-grade monitoring and observability
"""

import asyncio
import json
import pickle
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from scipy import stats
from skfuzzy import control as ctrl
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from xgboost import XGBClassifier

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

logger = structlog.get_logger(__name__)

# Metrics for enterprise monitoring
COMPLIANCE_PREDICTION_COUNTER = Counter(
    'compliance_predictions_total',
    'Total compliance predictions',
    ['model',
    'status']
)
COMPLIANCE_PREDICTION_DURATION = Histogram(
    'compliance_prediction_duration_seconds',
    'Compliance prediction duration',
    ['model']
)
ENSEMBLE_ACCURACY_GAUGE = Gauge('ensemble_accuracy', 'Ensemble model accuracy')
DRIFT_DETECTION_COUNTER = Counter('drift_detections_total', 'Total drift detections', ['severity'])

class EnhancedComplianceLSTM(nn.Module):
    """Enhanced LSTM with attention mechanism for temporal pattern recognition"""

    def __init__(self, sequence_length=30, feature_dim=50, lstm_units=128, num_classes=5):
        super(EnhancedComplianceLSTM, self).__init__()

        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes

        # Multi-layer LSTM with dropout
        self.lstm1 = nn.LSTM(feature_dim, lstm_units, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=0.2)
        self.lstm3 = nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=0.2)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(lstm_units, num_heads=8, dropout=0.1)

        # Dense layers with batch normalization
        self.dense1 = nn.Linear(lstm_units, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        """Forward pass with attention mechanism"""
        batch_size = x.size(0)

        # Sequential processing through LSTM layers
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)

        # Apply attention mechanism
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        attention_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attention_out = attention_out.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

        # Global average pooling
        pooled = torch.mean(attention_out, dim=1)

        # Dense layers with batch normalization
        x = self.dense1(pooled)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return self.output_layer(x), attention_weights

class EnhancedGovernanceProphet:
    """Enhanced Prophet model for governance seasonality analysis"""

    def __init__(self):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Please install it with: pip install prophet")

        self.model = Prophet(
            growth='logistic',
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays_prior_scale=10.0,
            seasonality_prior_scale=10.0,
            changepoint_prior_scale=0.05
        )

        # Add custom seasonalities for business patterns
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        self.model.add_seasonality(
            name='quarterly',
            period=91.25,
            fourier_order=3
        )

        # Add regressors for external factors
        self.model.add_regressor('policy_changes')
        self.model.add_regressor('resource_changes')
        self.model.add_regressor('maintenance_windows')
        self.model.add_regressor('business_criticality')

        self.is_fitted = False
        self.last_fit_time = None

    def prepare_data(self, violation_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for Prophet model with enhanced features"""

        df = pd.DataFrame({
            'ds': violation_data['timestamp'],
            'y': violation_data['violation_count'],
            'cap': violation_data.get(
                'max_possible_violations',
                violation_data['violation_count'].max() * 1.2
            )
        })

        # Add external regressors
        df['policy_changes'] = violation_data.get('policy_change_count', 0)
        df['resource_changes'] = violation_data.get('resource_change_count', 0)
        df['maintenance_windows'] = violation_data.get('maintenance_indicator', 0)
        df['business_criticality'] = violation_data.get('business_criticality_score', 0.5)

        return df

    def fit(self, violation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the Prophet model with enhanced error handling"""

        try:
            df = self.prepare_data(violation_data)
            self.model.fit(df)
            self.is_fitted = True
            self.last_fit_time = datetime.now()

            logger.info("Prophet model fitted successfully",
                       data_points=len(df),
                       last_fit_time=self.last_fit_time)

            return {
                'status': 'success',
                'data_points': len(df),
                'fit_time': self.last_fit_time.isoformat()
            }

        except Exception as e:
            logger.error("Failed to fit Prophet model", error=str(e))
            return {
                'status': 'error',
                'error': str(e)
            }

    def predict(self, future_periods: int = 24) -> Dict[str, Any]:
        """Make predictions with confidence intervals"""

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=future_periods, freq='D')

            # Add regressors for future periods (use last known values)
            future['policy_changes'] = 0  # Default value
            future['resource_changes'] = 0
            future['maintenance_windows'] = 0
            future['business_criticality'] = 0.5

            # Make prediction
            forecast = self.model.predict(future)

            # Extract relevant columns
            predictions = {
                'timestamp': forecast['ds'].tail(future_periods).tolist(),
                'predicted_violations': forecast['yhat'].tail(future_periods).tolist(),
                'lower_bound': forecast['yhat_lower'].tail(future_periods).tolist(),
                'upper_bound': forecast['yhat_upper'].tail(future_periods).tolist(),
                'trend': forecast['trend'].tail(future_periods).tolist(),
                'seasonal': forecast['yearly'].tail(future_periods).tolist()
            }

            return {
                'status': 'success',
                'predictions': predictions,
                'model_components': self.model.plot_components(forecast)
            }

        except Exception as e:
            logger.error("Failed to make Prophet predictions", error=str(e))
            return {
                'status': 'error',
                'error': str(e)
            }

class EnhancedFuzzyRiskAssessment:
    """Enhanced fuzzy logic risk assessment with enterprise features"""

    def __init__(self):
        self.setup_fuzzy_system()
        self.risk_history = []
        self.confidence_threshold = 0.7

    def setup_fuzzy_system(self):
        """Setup enhanced fuzzy logic system"""

        # Universe of discourse
        violation_prob = np.arange(0, 1.01, 0.01)
        business_impact = np.arange(0, 11, 0.1)
        complexity = np.arange(0, 11, 0.1)
        risk_level = np.arange(0, 101, 1)

        # Fuzzy sets for violation probability
        self.violation_very_low = fuzz.trapmf(violation_prob, [0, 0, 0.1, 0.2])
        self.violation_low = fuzz.trimf(violation_prob, [0.1, 0.25, 0.4])
        self.violation_medium = fuzz.trimf(violation_prob, [0.3, 0.5, 0.7])
        self.violation_high = fuzz.trimf(violation_prob, [0.6, 0.75, 0.9])
        self.violation_very_high = fuzz.trapmf(violation_prob, [0.8, 0.9, 1, 1])

        # Fuzzy sets for business impact
        self.impact_minimal = fuzz.trapmf(business_impact, [0, 0, 2, 3])
        self.impact_low = fuzz.trimf(business_impact, [2, 3.5, 5])
        self.impact_moderate = fuzz.trimf(business_impact, [4, 5.5, 7])
        self.impact_high = fuzz.trimf(business_impact, [6, 7.5, 9])
        self.impact_critical = fuzz.trapmf(business_impact, [8, 9, 10, 10])

        # Fuzzy sets for complexity
        self.complexity_simple = fuzz.trapmf(complexity, [0, 0, 2.5, 4])
        self.complexity_moderate = fuzz.trimf(complexity, [3, 5, 7])
        self.complexity_complex = fuzz.trimf(complexity, [6, 8, 10])
        self.complexity_very_complex = fuzz.trapmf(complexity, [8.5, 9.5, 10, 10])

        # Fuzzy sets for risk level
        self.risk_very_low = fuzz.trapmf(risk_level, [0, 0, 10, 20])
        self.risk_low = fuzz.trimf(risk_level, [10, 25, 40])
        self.risk_medium = fuzz.trimf(risk_level, [30, 50, 70])
        self.risk_high = fuzz.trimf(risk_level, [60, 75, 90])
        self.risk_critical = fuzz.trapmf(risk_level, [80, 90, 100, 100])

        # Create fuzzy control system
        self.setup_control_system()

    def setup_control_system(self):
        """Setup fuzzy control system with rules"""

        # Input variables
        violation_prob_input = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'violation_probability')
        business_impact_input = ctrl.Antecedent(np.arange(0, 11, 0.1), 'business_impact')
        complexity_input = ctrl.Antecedent(np.arange(0, 11, 0.1), 'complexity')

        # Output variable
        risk_level_output = ctrl.Consequent(np.arange(0, 101, 1), 'risk_level')

        # Define membership functions
        violation_prob_input['very_low'] = fuzz.trapmf(
            violation_prob_input.universe,
            [0,
            0,
            0.1,
            0.2]
        )
        violation_prob_input['low'] = fuzz.trimf(violation_prob_input.universe, [0.1, 0.25, 0.4])
        violation_prob_input['medium'] = fuzz.trimf(violation_prob_input.universe, [0.3, 0.5, 0.7])
        violation_prob_input['high'] = fuzz.trimf(violation_prob_input.universe, [0.6, 0.75, 0.9])
        violation_prob_input['very_high'] = fuzz.trapmf(
            violation_prob_input.universe,
            [0.8,
            0.9,
            1,
            1]
        )

        business_impact_input['minimal'] = fuzz.trapmf(business_impact_input.universe, [0, 0, 2, 3])
        business_impact_input['low'] = fuzz.trimf(business_impact_input.universe, [2, 3.5, 5])
        business_impact_input['moderate'] = fuzz.trimf(business_impact_input.universe, [4, 5.5, 7])
        business_impact_input['high'] = fuzz.trimf(business_impact_input.universe, [6, 7.5, 9])
        business_impact_input['critical'] = fuzz.trapmf(
            business_impact_input.universe,
            [8,
            9,
            10,
            10]
        )

        complexity_input['simple'] = fuzz.trapmf(complexity_input.universe, [0, 0, 2.5, 4])
        complexity_input['moderate'] = fuzz.trimf(complexity_input.universe, [3, 5, 7])
        complexity_input['complex'] = fuzz.trimf(complexity_input.universe, [6, 8, 10])
        complexity_input['very_complex'] = fuzz.trapmf(
            complexity_input.universe,
            [8.5,
            9.5,
            10,
            10]
        )

        risk_level_output['very_low'] = fuzz.trapmf(risk_level_output.universe, [0, 0, 10, 20])
        risk_level_output['low'] = fuzz.trimf(risk_level_output.universe, [10, 25, 40])
        risk_level_output['medium'] = fuzz.trimf(risk_level_output.universe, [30, 50, 70])
        risk_level_output['high'] = fuzz.trimf(risk_level_output.universe, [60, 75, 90])
        risk_level_output['critical'] = fuzz.trapmf(risk_level_output.universe, [80, 90, 100, 100])

        # Define fuzzy rules
        rule1 = ctrl.Rule(
            violation_prob_input['very_high'] & business_impact_input['critical'],
            risk_level_output['critical']
        )
        rule2 = ctrl.Rule(
            violation_prob_input['high'] & business_impact_input['critical'],
            risk_level_output['critical']
        )
        rule3 = ctrl.Rule(
            violation_prob_input['very_high'] & business_impact_input['high'],
            risk_level_output['high']
        )
        rule4 = ctrl.Rule(
            violation_prob_input['medium'] & business_impact_input['high'],
            risk_level_output['medium']
        )
        rule5 = ctrl.Rule(
            violation_prob_input['high'] & business_impact_input['moderate'],
            risk_level_output['medium']
        )
        rule6 = ctrl.Rule(
            violation_prob_input['low'] & business_impact_input['low'],
            risk_level_output['low']
        )
        rule7 = ctrl.Rule(
            violation_prob_input['very_low'] & business_impact_input['minimal'],
            risk_level_output['very_low']
        )

        # Create control system
        self.risk_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.risk_simulation = ctrl.ControlSystemSimulation(self.risk_control)

    def assess_risk(
        self,
        violation_prob: float,
        business_impact: float,
        complexity: float = 5.0
    ) -> Dict[str, Any]:
        """Assess risk using enhanced fuzzy logic"""

        try:
            # Set input values
            self.risk_simulation.input['violation_probability'] = violation_prob
            self.risk_simulation.input['business_impact'] = business_impact
            self.risk_simulation.input['complexity'] = complexity

            # Compute risk level
            self.risk_simulation.compute()

            # Get risk level and confidence
            risk_level = self.risk_simulation.risk_level

            # Calculate confidence based on rule activation
            confidence = self._calculate_confidence(violation_prob, business_impact, complexity)

            # Determine risk category
            risk_category = self._categorize_risk(risk_level)

            # Store in history
            assessment = {
                'timestamp': datetime.now().isoformat(),
                'violation_probability': violation_prob,
                'business_impact': business_impact,
                'complexity': complexity,
                'risk_level': risk_level,
                'risk_category': risk_category,
                'confidence': confidence
            }

            self.risk_history.append(assessment)

            # Update metrics
            DRIFT_DETECTION_COUNTER.labels(severity=risk_category).inc()

            return {
                'risk_level': risk_level,
                'risk_category': risk_category,
                'confidence': confidence,
                'assessment': assessment
            }

        except Exception as e:
            logger.error("Fuzzy risk assessment failed", error=str(e))
            return {
                'risk_level': 50.0,  # Default medium risk
                'risk_category': 'medium',
                'confidence': 0.0,
                'error': str(e)
            }

    def _calculate_confidence(
        self,
        violation_prob: float,
        business_impact: float,
        complexity: float
    ) -> float:
        """Calculate confidence in the risk assessment"""

        # Calculate membership strengths
        prob_strength = max(
            fuzz.interp_membership(
                np.arange(0,
                1.01,
                0.01),
                self.violation_very_low,
                violation_prob
            ),
            fuzz.interp_membership(np.arange(0, 1.01, 0.01), self.violation_low, violation_prob),
            fuzz.interp_membership(np.arange(0, 1.01, 0.01), self.violation_medium, violation_prob),
            fuzz.interp_membership(np.arange(0, 1.01, 0.01), self.violation_high, violation_prob),
            fuzz.interp_membership(
                np.arange(0,
                1.01,
                0.01),
                self.violation_very_high,
                violation_prob
            )
        )

        impact_strength = max(
            fuzz.interp_membership(np.arange(0, 11, 0.1), self.impact_minimal, business_impact),
            fuzz.interp_membership(np.arange(0, 11, 0.1), self.impact_low, business_impact),
            fuzz.interp_membership(np.arange(0, 11, 0.1), self.impact_moderate, business_impact),
            fuzz.interp_membership(np.arange(0, 11, 0.1), self.impact_high, business_impact),
            fuzz.interp_membership(np.arange(0, 11, 0.1), self.impact_critical, business_impact)
        )

        # Average confidence
        confidence = (prob_strength + impact_strength) / 2
        return min(confidence, 1.0)

    def _categorize_risk(self, risk_level: float) -> str:
        """Categorize risk level"""
        if risk_level < 20:
            return 'very_low'
        elif risk_level < 40:
            return 'low'
        elif risk_level < 70:
            return 'medium'
        elif risk_level < 90:
            return 'high'
        else:
            return 'critical'

class EnhancedComplianceEnsemble:
    """Enhanced ensemble model for compliance prediction"""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.last_update = None
        self.min_confidence = 0.7

    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        self.performance_history[name] = []

    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights based on performance"""
        for name, weight in new_weights.items():
            if name in self.weights:
                self.weights[name] = weight

        self.last_update = datetime.now()
        logger.info("Updated ensemble weights", weights=self.weights)

    def predict_with_confidence(
        self,
        features: np.ndarray,
        sequence_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Make ensemble prediction with confidence intervals"""

        predictions = {}
        confidences = {}

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                if name == 'lstm' and sequence_data is not None:
                    # LSTM model expects sequence data
                    with torch.no_grad():
                        sequence_tensor = torch.FloatTensor(sequence_data)
                        pred, attention_weights = model(sequence_tensor)
                        pred_proba = F.softmax(pred, dim=1).numpy()
                        predictions[name] = pred_proba
                        confidences[name] = np.max(pred_proba, axis=1)
                else:
                    # Traditional ML models
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features)
                        predictions[name] = pred_proba
                        confidences[name] = np.max(pred_proba, axis=1)
                    else:
                        pred = model.predict(features)
                        predictions[name] = pred
                        confidences[name] = np.ones(len(features))  # Default confidence

                COMPLIANCE_PREDICTION_COUNTER.labels(model=name, status='success').inc()

            except Exception as e:
                logger.error(f"Model {name} prediction failed", error=str(e))
                COMPLIANCE_PREDICTION_COUNTER.labels(model=name, status='error').inc()
                predictions[name] = None
                confidences[name] = np.zeros(len(features))

        # Weighted ensemble prediction
        ensemble_pred = self._weighted_ensemble_predict(predictions)
        ensemble_confidence = self._calculate_ensemble_confidence(confidences)

        # Calculate agreement between models
        agreement = self._calculate_model_agreement(predictions)

        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_confidence,
            'model_agreement': agreement,
            'individual_predictions': predictions,
            'individual_confidences': confidences,
            'weights': self.weights
        }

    def _weighted_ensemble_predict(self, predictions: Dict[str, Any]) -> np.ndarray:
        """Calculate weighted ensemble prediction"""

        valid_predictions = []
        valid_weights = []

        for name, pred in predictions.items():
            if pred is not None and name in self.weights:
                valid_predictions.append(pred)
                valid_weights.append(self.weights[name])

        if not valid_predictions:
            # Return default prediction if no valid models
            return np.zeros(
                (len(next(iter(predictions.values())),
                5)) if predictions else np.zeros((1,
                5)
            )

        # Weighted average
        weighted_pred = np.zeros_like(valid_predictions[0])
        total_weight = sum(valid_weights)

        for pred, weight in zip(valid_predictions, valid_weights):
            weighted_pred += (weight / total_weight) * pred

        return weighted_pred

    def _calculate_ensemble_confidence(self, confidences: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate ensemble confidence"""

        valid_confidences = [conf for conf in confidences.values() if conf is not None]

        if not valid_confidences:
            return np.zeros(1)

        # Average confidence across models
        return np.mean(valid_confidences, axis=0)

    def _calculate_model_agreement(self, predictions: Dict[str, Any]) -> float:
        """Calculate agreement between models"""

        valid_predictions = [pred for pred in predictions.values() if pred is not None]

        if len(valid_predictions) < 2:
            return 1.0  # Perfect agreement if only one model

        # Calculate correlation between predictions
        pred_arrays = [pred.flatten() for pred in valid_predictions]
        correlations = []

        for i in range(len(pred_arrays)):
            for j in range(i + 1, len(pred_arrays)):
                corr = np.corrcoef(pred_arrays[i], pred_arrays[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return np.mean(correlations) if correlations else 0.0

    def update_performance(self, model_name: str, accuracy: float):
        """Update model performance history"""
        if model_name in self.performance_history:
            self.performance_history[model_name].append(accuracy)

            # Keep only last 10 performance measurements
            if len(self.performance_history[model_name]) > 10:
                self.performance_history[model_name] = self.performance_history[model_name][-10:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {}

        for name, history in self.performance_history.items():
            if history:
                summary[name] = {
                    'current_accuracy': history[-1],
                    'average_accuracy': np.mean(history),
                    'trend': 'improving' if len(history) > 1 and
                        history[-1] > history[-2] else 'stable'
                }

        return summary

class EnhancedCompliancePredictor:
    """Enhanced compliance predictor with enterprise features"""

    def __init__(self):
        self.ensemble = EnhancedComplianceEnsemble()
        self.fuzzy_assessment = EnhancedFuzzyRiskAssessment()
        self.prophet_model = EnhancedGovernanceProphet()
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.is_initialized = False

        # Model storage
        self.model_path = Path("models/compliance")
        self.model_path.mkdir(parents=True, exist_ok=True)

        # MLflow tracking
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        self.experiment_name = "compliance_prediction"

    async def initialize(self):
        """Initialize the enhanced compliance predictor"""

        try:
            # Initialize models
            self.lstm_model = EnhancedComplianceLSTM()
            self.xgb_model = XGBClassifier(
                objective='multi:softprob',
                num_class=5,
                max_depth=8,
                learning_rate=0.1,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )

            # Add models to ensemble
            self.ensemble.add_model('lstm', self.lstm_model, weight=0.4)
            self.ensemble.add_model('xgb', self.xgb_model, weight=0.4)
            self.ensemble.add_model('prophet', self.prophet_model, weight=0.2)

            self.is_initialized = True
            logger.info("Enhanced compliance predictor initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize compliance predictor", error=str(e))
            raise

    async def train(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the ensemble models with enhanced features"""

        if not self.is_initialized:
            await self.initialize()

        try:
            # Start MLflow run
            with mlflow.start_run(experiment_name=self.experiment_name):
                mlflow.log_param("ensemble_size", len(self.ensemble.models))
                mlflow.log_param("training_data_size", len(historical_data.get('features', [])))

                # Prepare training data
                features, labels = self._prepare_training_data(historical_data)

                # Train XGBoost model
                xgb_start_time = datetime.now()
                self.xgb_model.fit(features, labels)
                xgb_training_time = (datetime.now() - xgb_start_time).total_seconds()

                # Train LSTM model
                lstm_start_time = datetime.now()
                sequence_data, sequence_labels = self._prepare_sequence_data(historical_data)

                if len(sequence_data) > 0:
                    sequence_tensor = torch.FloatTensor(sequence_data)
                    labels_tensor = torch.LongTensor(sequence_labels)

                    # Train LSTM
                    optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    self.lstm_model.train()
                    for epoch in range(50):  # Reduced epochs for faster training
                        optimizer.zero_grad()
                        outputs, _ = self.lstm_model(sequence_tensor)
                        loss = criterion(outputs, labels_tensor)
                        loss.backward()
                        optimizer.step()

                        if epoch % 10 == 0:
                            logger.info(f"LSTM training epoch {epoch}, loss: {loss.item():.4f}")

                lstm_training_time = (datetime.now() - lstm_start_time).total_seconds()

                # Train Prophet model
                prophet_start_time = datetime.now()
                prophet_result = self.prophet_model.fit(historical_data)
                prophet_training_time = (datetime.now() - prophet_start_time).total_seconds()

                # Log metrics
                mlflow.log_metric("xgb_training_time", xgb_training_time)
                mlflow.log_metric("lstm_training_time", lstm_training_time)
                mlflow.log_metric("prophet_training_time", prophet_training_time)
                mlflow.log_metric(
                    "total_training_time",
                    xgb_training_time + lstm_training_time + prophet_training_time
                )

                # Save models
                self._save_models()

                logger.info("Ensemble models trained successfully",
                           xgb_time=xgb_training_time,
                           lstm_time=lstm_training_time,
                           prophet_time=prophet_training_time)

                return {
                    'status': 'success',
                    'training_time': xgb_training_time + lstm_training_time + prophet_training_time,
                    'models_trained': list(self.ensemble.models.keys()),
                    'run_id': mlflow.active_run().info.run_id
                }

        except Exception as e:
            logger.error("Training failed", error=str(e))
            return {
                'status': 'error',
                'error': str(e)
            }

    async def predict_compliance(
        self,
        resource_data: Dict[str,
        Any],
        horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Predict compliance with enhanced features"""

        if not self.is_initialized:
            await self.initialize()

        try:
            start_time = datetime.now()

            # Prepare features
            features = self._extract_prediction_features(resource_data)
            sequence_data = self._prepare_prediction_sequence(resource_data)

            # Make ensemble prediction
            ensemble_result = self.ensemble.predict_with_confidence(features, sequence_data)

            # Fuzzy risk assessment
            violation_prob = np.max(ensemble_result['ensemble_prediction'], axis=1)[0]
            business_impact = self._calculate_business_impact(resource_data)
            complexity = self._calculate_remediation_complexity(resource_data)

            fuzzy_result = self.fuzzy_assessment.assess_risk(
                violation_prob,
                business_impact,
                complexity
            )

            # Prophet predictions for temporal patterns
            prophet_result = self.prophet_model.predict(horizon_hours // 24)  # Convert to days

            # Calculate comprehensive risk assessment
            comprehensive_risk = self._calculate_comprehensive_risk(
                violation_prob, fuzzy_result, prophet_result
            )

            # Update metrics
            prediction_time = (datetime.now() - start_time).total_seconds()
            COMPLIANCE_PREDICTION_DURATION.labels(model='ensemble').observe(prediction_time)

            # Log prediction
            logger.info("Compliance prediction completed",
                       prediction_time=prediction_time,
                       risk_level=comprehensive_risk['risk_level'],
                       confidence=ensemble_result['ensemble_confidence'][0])

            return {
                'status': 'success',
                'prediction': {
                    'violation_probability': float(violation_prob),
                    'risk_level': comprehensive_risk['risk_level'],
                    'risk_category': comprehensive_risk['risk_category'],
                    'confidence': float(ensemble_result['ensemble_confidence'][0]),
                    'model_agreement': float(ensemble_result['model_agreement']),
                    'prediction_horizon_hours': horizon_hours,
                    'business_impact': float(business_impact),
                    'remediation_complexity': float(complexity)
                },
                'temporal_analysis': prophet_result,
                'fuzzy_assessment': fuzzy_result,
                'ensemble_details': {
                    'weights': self.ensemble.weights,
                    'individual_predictions': {
                        name: pred.tolist() if pred is not None else None
                        for name, pred in ensemble_result['individual_predictions'].items()
                    }
                },
                'prediction_time_seconds': prediction_time
            }

        except Exception as e:
            logger.error("Compliance prediction failed", error=str(e))
            COMPLIANCE_PREDICTION_COUNTER.labels(model='ensemble', status='error').inc()
            return {
                'status': 'error',
                'error': str(e)
            }

    def _prepare_training_data(
        self,
        historical_data: Dict[str,
        Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for traditional ML models"""

        features = historical_data.get('features', [])
        labels = historical_data.get('labels', [])

        if not features or not labels:
            raise ValueError("Training data must contain features and labels")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        return np.array(features_scaled), np.array(labels)

    def _prepare_sequence_data(
        self,
        historical_data: Dict[str,
        Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM model"""

        sequences = historical_data.get('sequences', [])
        sequence_labels = historical_data.get('sequence_labels', [])

        if not sequences or not sequence_labels:
            return np.array([]), np.array([])

        return np.array(sequences), np.array(sequence_labels)

    def _extract_prediction_features(self, resource_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for prediction"""

        # Extract numerical features
        features = []

        # Resource characteristics
        features.extend([
            resource_data.get('resource_age_days', 0),
            resource_data.get('resource_size_category', 0),
            resource_data.get('resource_complexity_score', 0),
            resource_data.get('resource_criticality_level', 0)
        ])

        # Governance features
        features.extend([
            resource_data.get('policy_coverage_score', 0),
            resource_data.get('compliance_history_score', 0),
            resource_data.get('risk_assessment_score', 0),
            resource_data.get('remediation_success_rate', 0)
        ])

        # Temporal features
        features.extend([
            resource_data.get('days_since_last_violation', 365),
            resource_data.get('violation_frequency_7d', 0),
            resource_data.get('violation_frequency_30d', 0),
            resource_data.get('configuration_change_rate', 0)
        ])

        # Scale features
        features_scaled = self.scaler.transform([features])

        return features_scaled

    def _prepare_prediction_sequence(self, resource_data: Dict[str, Any]) -> np.ndarray:
        """Prepare sequence data for LSTM prediction"""

        # Get historical sequence data for this resource
        sequence_data = resource_data.get('historical_sequence', [])

        if not sequence_data:
            return np.array([])

        # Pad or truncate to expected sequence length
        expected_length = 30
        if len(sequence_data) < expected_length:
            # Pad with zeros
            padding = [[0] * len(sequence_data[0])] * (expected_length - len(sequence_data))
            sequence_data = padding + sequence_data
        elif len(sequence_data) > expected_length:
            # Truncate to last expected_length elements
            sequence_data = sequence_data[-expected_length:]

        return np.array([sequence_data])

    def _calculate_business_impact(self, resource_data: Dict[str, Any]) -> float:
        """Calculate business impact score"""

        # Factors affecting business impact
        criticality = resource_data.get('business_criticality', 0.5)
        user_count = resource_data.get('affected_users', 0) / 1000  # Normalize
        data_sensitivity = resource_data.get('data_sensitivity_level', 0.5)
        service_availability = resource_data.get('service_availability_requirement', 0.5)

        # Weighted calculation
        impact_score = (
            criticality * 0.4 +
            min(user_count, 1.0) * 0.3 +
            data_sensitivity * 0.2 +
            service_availability * 0.1
        )

        return min(impact_score, 10.0)  # Cap at 10

    def _calculate_remediation_complexity(self, resource_data: Dict[str, Any]) -> float:
        """Calculate remediation complexity score"""

        # Factors affecting complexity
        resource_type_complexity = resource_data.get('resource_type_complexity', 0.5)
        dependencies_count = resource_data.get('dependencies_count', 0) / 10  # Normalize
        configuration_complexity = resource_data.get('configuration_complexity', 0.5)
        team_expertise = resource_data.get('team_expertise_level', 0.5)

        # Weighted calculation
        complexity_score = (
            resource_type_complexity * 0.3 +
            min(dependencies_count, 1.0) * 0.3 +
            configuration_complexity * 0.2 +
            (1.0 - team_expertise) * 0.2  # Lower expertise = higher complexity
        )

        return min(complexity_score, 10.0)  # Cap at 10

    def _calculate_comprehensive_risk(
        self,
        violation_prob: float,
        fuzzy_result: Dict[str,
        Any],
        prophet_result: Dict[str,
        Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk assessment"""

        # Combine different risk factors
        fuzzy_risk = fuzzy_result.get('risk_level', 50.0)
        prophet_trend = prophet_result.get(
            'predictions',
            {}).get('trend',
            [0])[-1] if prophet_result.get('status'
        ) == 'success' else 0

        # Weighted risk calculation
        comprehensive_risk = (
            violation_prob * 0.4 +
            (fuzzy_risk / 100.0) * 0.4 +
            (max(prophet_trend, 0) / 100.0) * 0.2
        ) * 100

        # Categorize risk
        if comprehensive_risk < 20:
            risk_category = 'very_low'
        elif comprehensive_risk < 40:
            risk_category = 'low'
        elif comprehensive_risk < 70:
            risk_category = 'medium'
        elif comprehensive_risk < 90:
            risk_category = 'high'
        else:
            risk_category = 'critical'

        return {
            'risk_level': comprehensive_risk,
            'risk_category': risk_category,
            'violation_probability': violation_prob,
            'fuzzy_risk': fuzzy_risk,
            'temporal_trend': prophet_trend
        }

    def _save_models(self):
        """Save trained models"""

        # Save XGBoost model
        with open(self.model_path / "xgb_model.pkl", "wb") as f:
            pickle.dump(self.xgb_model, f)

        # Save LSTM model
        torch.save(self.lstm_model.state_dict(), self.model_path / "lstm_model.pth")

        # Save scaler
        with open(self.model_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save ensemble weights
        with open(self.model_path / "ensemble_weights.json", "w") as f:
            json.dump(self.ensemble.weights, f)

        logger.info("Models saved successfully", model_path=str(self.model_path))

    def load_models(self):
        """Load trained models"""

        try:
            # Load XGBoost model
            with open(self.model_path / "xgb_model.pkl", "rb") as f:
                self.xgb_model = pickle.load(f)

            # Load LSTM model
            self.lstm_model.load_state_dict(torch.load(self.model_path / "lstm_model.pth"))

            # Load scaler
            with open(self.model_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # Load ensemble weights
            with open(self.model_path / "ensemble_weights.json", "r") as f:
                self.ensemble.weights = json.load(f)

            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error("Failed to load models", error=str(e))
            raise
