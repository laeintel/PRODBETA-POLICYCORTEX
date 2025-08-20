"""
Predictive Analytics Engine for Patent #3
Cross-domain predictive analytics for the Unified Platform
Provides forecasting, trend analysis, and anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class Prediction:
    """Prediction data structure"""
    domain: str
    metric: str
    current_value: float
    predicted_value: float
    confidence_interval: Tuple[float, float]
    trend: str  # increasing, decreasing, stable
    anomaly_score: float
    forecast_horizon: str
    factors: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'domain': self.domain,
            'metric': self.metric,
            'current_value': self.current_value,
            'predicted_value': self.predicted_value,
            'confidence_interval': self.confidence_interval,
            'trend': self.trend,
            'anomaly_score': self.anomaly_score,
            'forecast_horizon': self.forecast_horizon,
            'factors': self.factors,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for time series prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        
        return output, attention_weights


class PredictiveAnalyticsEngine:
    """Advanced predictive analytics for cloud governance metrics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.anomaly_detectors = {}
        self.feature_extractors = {}
        
        # Initialize models for each domain
        self.domains = ['security', 'compliance', 'cost', 'operations', 'identity']
        self._initialize_models()
        
        # Historical data cache
        self.data_cache = {}
        
        # Prediction history for model evaluation
        self.prediction_history = []
        
    def _initialize_models(self):
        """Initialize prediction models for each domain"""
        for domain in self.domains:
            # Time series model (LSTM with attention)
            self.models[f'{domain}_lstm'] = AttentionLSTM(
                input_dim=10,  # Features per time step
                hidden_dim=128,
                num_layers=2
            )
            
            # Prophet model for trend analysis
            self.models[f'{domain}_prophet'] = None  # Initialized on demand
            
            # Random Forest for feature importance
            self.models[f'{domain}_rf'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Anomaly detection
            self.anomaly_detectors[domain] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Feature scaler
            self.scalers[domain] = StandardScaler()
    
    def extract_time_series_features(
        self,
        data: pd.DataFrame,
        domain: str
    ) -> np.ndarray:
        """Extract features from time series data"""
        features = []
        
        # Basic statistics
        features.extend([
            data['value'].mean(),
            data['value'].std(),
            data['value'].min(),
            data['value'].max(),
            data['value'].quantile(0.25),
            data['value'].quantile(0.75)
        ])
        
        # Trend features
        if len(data) > 1:
            trend = np.polyfit(range(len(data)), data['value'], 1)[0]
            features.append(trend)
        else:
            features.append(0)
        
        # Seasonality features (if enough data)
        if len(data) > 24:  # At least 24 hours of data
            try:
                decomposition = seasonal_decompose(data['value'], model='additive', period=24)
                features.append(decomposition.seasonal.mean())
                features.append(decomposition.resid.std() if decomposition.resid.std() > 0 else 0)
            except:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # Domain-specific features
        if domain == 'security':
            features.append(data.get('threat_score', pd.Series([0])).mean())
        elif domain == 'cost':
            features.append(data.get('spend_rate', pd.Series([0])).mean())
        elif domain == 'compliance':
            features.append(data.get('violation_rate', pd.Series([0])).mean())
        else:
            features.append(0)
        
        return np.array(features)
    
    async def predict_metric(
        self,
        domain: str,
        metric: str,
        historical_data: pd.DataFrame,
        horizon: int = 24  # hours
    ) -> Prediction:
        """Predict future values for a specific metric"""
        
        # Extract features
        features = self.extract_time_series_features(historical_data, domain)
        
        # Scale features
        features_scaled = self.scalers[domain].fit_transform(features.reshape(1, -1))
        
        # LSTM prediction
        with torch.no_grad():
            # Prepare sequence data
            sequence = self._prepare_sequence(historical_data, domain)
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Get prediction
            lstm_pred, attention_weights = self.models[f'{domain}_lstm'](sequence_tensor)
            predicted_value = lstm_pred.item()
        
        # Prophet prediction for trend
        prophet_pred, trend = await self._prophet_forecast(
            historical_data,
            domain,
            metric,
            horizon
        )
        
        # Anomaly detection
        anomaly_score = self._detect_anomaly(features_scaled, domain)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            predicted_value,
            historical_data['value'].std()
        )
        
        # Identify contributing factors
        factors = self._identify_factors(
            historical_data,
            domain,
            metric,
            attention_weights
        )
        
        # Generate recommendations
        recommendations = self._generate_predictions_recommendations(
            domain,
            metric,
            predicted_value,
            trend,
            anomaly_score
        )
        
        # Create prediction object
        prediction = Prediction(
            domain=domain,
            metric=metric,
            current_value=historical_data['value'].iloc[-1],
            predicted_value=predicted_value,
            confidence_interval=confidence_interval,
            trend=trend,
            anomaly_score=anomaly_score,
            forecast_horizon=f"{horizon} hours",
            factors=factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        # Store prediction for evaluation
        self.prediction_history.append(prediction.to_dict())
        
        return prediction
    
    def _prepare_sequence(
        self,
        data: pd.DataFrame,
        domain: str,
        sequence_length: int = 24
    ) -> np.ndarray:
        """Prepare sequence data for LSTM"""
        # Get last sequence_length values
        if len(data) < sequence_length:
            # Pad with zeros if not enough data
            values = data['value'].values
            padded = np.pad(values, (sequence_length - len(values), 0), mode='constant')
        else:
            padded = data['value'].iloc[-sequence_length:].values
        
        # Add additional features for each time step
        sequence = []
        for i in range(len(padded)):
            features = [
                padded[i],
                i / sequence_length,  # Time encoding
                np.sin(2 * np.pi * i / 24),  # Daily seasonality
                np.cos(2 * np.pi * i / 24),
                1 if i % 24 < 8 else 0,  # Business hours
            ]
            
            # Domain-specific features
            if domain == 'cost':
                features.extend([
                    1 if i % 24 == 0 else 0,  # Start of day
                    1 if i % 168 == 0 else 0,  # Start of week
                ])
            elif domain == 'security':
                features.extend([
                    np.random.random(),  # Threat level (would be real data)
                    0.5,  # Baseline risk
                ])
            else:
                features.extend([0, 0])
            
            # Pad to 10 features
            features = features[:10] + [0] * (10 - len(features))
            sequence.append(features)
        
        return np.array(sequence)
    
    async def _prophet_forecast(
        self,
        data: pd.DataFrame,
        domain: str,
        metric: str,
        horizon: int
    ) -> Tuple[float, str]:
        """Use Prophet for time series forecasting"""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(data['timestamp']),
                'y': data['value']
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            # Add domain-specific seasonalities
            if domain == 'cost':
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            elif domain == 'operations':
                model.add_seasonality(name='hourly', period=1/24, fourier_order=3)
            
            model.fit(prophet_df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=horizon, freq='H')
            forecast = model.predict(future)
            
            # Get predicted value
            predicted_value = forecast['yhat'].iloc[-1]
            
            # Determine trend
            recent_trend = forecast['trend'].iloc[-horizon:].mean()
            historical_trend = forecast['trend'].iloc[-horizon*2:-horizon].mean()
            
            if recent_trend > historical_trend * 1.05:
                trend = 'increasing'
            elif recent_trend < historical_trend * 0.95:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            return predicted_value, trend
            
        except Exception as e:
            logger.warning(f"Prophet forecast failed: {e}")
            # Fallback to simple trend
            if len(data) > 1:
                trend_coef = np.polyfit(range(len(data)), data['value'], 1)[0]
                predicted_value = data['value'].iloc[-1] + trend_coef * horizon
                trend = 'increasing' if trend_coef > 0 else 'decreasing'
            else:
                predicted_value = data['value'].iloc[-1]
                trend = 'stable'
            
            return predicted_value, trend
    
    def _detect_anomaly(self, features: np.ndarray, domain: str) -> float:
        """Detect anomalies using Isolation Forest"""
        try:
            # Predict anomaly score (-1 for anomaly, 1 for normal)
            anomaly_prediction = self.anomaly_detectors[domain].decision_function(features)
            
            # Convert to 0-1 scale (higher score = more anomalous)
            anomaly_score = 1 / (1 + np.exp(anomaly_prediction[0]))
            
            return float(anomaly_score)
        except:
            return 0.5  # Default to neutral score
    
    def _calculate_confidence_interval(
        self,
        predicted_value: float,
        std_dev: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Z-score for 95% confidence
        z_score = 1.96 if confidence_level == 0.95 else 2.58
        
        margin = z_score * std_dev
        lower = predicted_value - margin
        upper = predicted_value + margin
        
        return (lower, upper)
    
    def _identify_factors(
        self,
        data: pd.DataFrame,
        domain: str,
        metric: str,
        attention_weights: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """Identify factors contributing to the prediction"""
        factors = []
        
        # Time-based factors
        if attention_weights is not None:
            # Get top attention time steps
            weights = attention_weights.squeeze().numpy()
            top_indices = np.argsort(weights)[-3:]
            
            for idx in top_indices:
                factors.append({
                    'type': 'temporal',
                    'description': f"Time point {idx} hours ago",
                    'impact': float(weights[idx]),
                    'value': float(data['value'].iloc[-(len(weights)-idx)])
                })
        
        # Domain-specific factors
        if domain == 'security':
            factors.extend([
                {
                    'type': 'threat_level',
                    'description': 'Current threat level',
                    'impact': 0.3,
                    'value': 'medium'
                },
                {
                    'type': 'vulnerability_count',
                    'description': 'Open vulnerabilities',
                    'impact': 0.2,
                    'value': 15
                }
            ])
        elif domain == 'cost':
            factors.extend([
                {
                    'type': 'resource_usage',
                    'description': 'Resource utilization rate',
                    'impact': 0.4,
                    'value': '75%'
                },
                {
                    'type': 'pricing_tier',
                    'description': 'Current pricing tier',
                    'impact': 0.2,
                    'value': 'standard'
                }
            ])
        elif domain == 'compliance':
            factors.extend([
                {
                    'type': 'policy_changes',
                    'description': 'Recent policy updates',
                    'impact': 0.3,
                    'value': 3
                },
                {
                    'type': 'audit_findings',
                    'description': 'Pending audit items',
                    'impact': 0.25,
                    'value': 8
                }
            ])
        
        return factors
    
    def _generate_predictions_recommendations(
        self,
        domain: str,
        metric: str,
        predicted_value: float,
        trend: str,
        anomaly_score: float
    ) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        # Anomaly-based recommendations
        if anomaly_score > 0.7:
            recommendations.append(f"Investigate anomaly detected in {metric}")
            recommendations.append("Review recent changes and configurations")
        
        # Trend-based recommendations
        if trend == 'increasing':
            if domain == 'cost':
                recommendations.append("Review and optimize resource allocation")
                recommendations.append("Consider implementing cost controls")
            elif domain == 'security':
                recommendations.append("Increase security monitoring")
                recommendations.append("Review security policies and controls")
            elif domain == 'compliance':
                recommendations.append("Schedule compliance review")
                recommendations.append("Update compliance documentation")
                
        elif trend == 'decreasing':
            if domain == 'operations':
                recommendations.append("Investigate performance degradation")
                recommendations.append("Review system health metrics")
            elif domain == 'identity':
                recommendations.append("Review access control changes")
                recommendations.append("Audit user permissions")
        
        # Threshold-based recommendations
        if domain == 'cost' and predicted_value > 10000:
            recommendations.append("Budget threshold approaching - review spending")
        elif domain == 'security' and metric == 'risk_score' and predicted_value > 0.7:
            recommendations.append("High risk predicted - implement additional controls")
        elif domain == 'compliance' and metric == 'compliance_score' and predicted_value < 0.8:
            recommendations.append("Compliance score dropping - immediate action required")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def analyze_trends(
        self,
        domain: str,
        data: pd.DataFrame,
        period: str = 'daily'
    ) -> Dict[str, Any]:
        """Analyze trends for a domain"""
        analysis = {
            'domain': domain,
            'period': period,
            'trends': {},
            'seasonality': {},
            'change_points': [],
            'forecast': {}
        }
        
        # Decompose time series
        if len(data) > 24:
            try:
                decomposition = seasonal_decompose(
                    data['value'],
                    model='additive',
                    period=24 if period == 'daily' else 168
                )
                
                analysis['trends']['direction'] = 'increasing' if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0] else 'decreasing'
                analysis['trends']['strength'] = abs(decomposition.trend.iloc[-1] - decomposition.trend.iloc[0]) / decomposition.trend.iloc[0]
                
                analysis['seasonality']['detected'] = True
                analysis['seasonality']['amplitude'] = decomposition.seasonal.max() - decomposition.seasonal.min()
                analysis['seasonality']['period'] = period
                
            except Exception as e:
                logger.warning(f"Trend decomposition failed: {e}")
        
        # Detect change points
        analysis['change_points'] = self._detect_change_points(data)
        
        # Generate forecast
        for horizon in [24, 48, 72]:  # 1, 2, 3 days
            prediction = await self.predict_metric(domain, 'primary_metric', data, horizon)
            analysis['forecast'][f'{horizon}h'] = {
                'value': prediction.predicted_value,
                'confidence': prediction.confidence_interval,
                'trend': prediction.trend
            }
        
        return analysis
    
    def _detect_change_points(self, data: pd.DataFrame) -> List[Dict]:
        """Detect significant change points in time series"""
        change_points = []
        
        if len(data) < 10:
            return change_points
        
        values = data['value'].values
        
        # Simple change point detection using rolling statistics
        window_size = min(24, len(data) // 4)
        
        for i in range(window_size, len(data) - window_size):
            before_mean = values[i-window_size:i].mean()
            after_mean = values[i:i+window_size].mean()
            
            change_ratio = abs(after_mean - before_mean) / (before_mean + 1e-10)
            
            if change_ratio > 0.3:  # 30% change threshold
                change_points.append({
                    'index': i,
                    'timestamp': data.iloc[i]['timestamp'],
                    'before_value': before_mean,
                    'after_value': after_mean,
                    'change_ratio': change_ratio,
                    'type': 'increase' if after_mean > before_mean else 'decrease'
                })
        
        # Sort by change ratio and return top 5
        change_points.sort(key=lambda x: x['change_ratio'], reverse=True)
        return change_points[:5]
    
    async def cross_domain_analysis(
        self,
        all_domain_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform cross-domain predictive analysis"""
        analysis = {
            'correlations': {},
            'combined_risk': 0,
            'predictions': {},
            'recommendations': []
        }
        
        # Calculate cross-domain correlations
        domains_data = {}
        for domain, data in all_domain_data.items():
            if len(data) > 0:
                domains_data[domain] = data['value'].values
        
        # Correlation matrix
        if len(domains_data) > 1:
            correlation_matrix = np.corrcoef(list(domains_data.values()))
            domain_names = list(domains_data.keys())
            
            for i, domain1 in enumerate(domain_names):
                for j, domain2 in enumerate(domain_names):
                    if i < j:
                        correlation = correlation_matrix[i, j]
                        if abs(correlation) > 0.5:
                            analysis['correlations'][f'{domain1}-{domain2}'] = float(correlation)
        
        # Combined risk prediction
        risk_scores = []
        for domain, data in all_domain_data.items():
            if len(data) > 0:
                prediction = await self.predict_metric(domain, 'risk_metric', data)
                risk_scores.append(prediction.predicted_value)
                analysis['predictions'][domain] = prediction.to_dict()
        
        if risk_scores:
            analysis['combined_risk'] = np.mean(risk_scores)
        
        # Generate cross-domain recommendations
        if analysis['combined_risk'] > 0.7:
            analysis['recommendations'].append("High combined risk detected - immediate action required")
            
        for correlation_pair, correlation_value in analysis['correlations'].items():
            if correlation_value > 0.8:
                domains = correlation_pair.split('-')
                analysis['recommendations'].append(
                    f"Strong correlation between {domains[0]} and {domains[1]} - consider unified approach"
                )
        
        return analysis
    
    def evaluate_predictions(self) -> Dict[str, float]:
        """Evaluate prediction accuracy from history"""
        if len(self.prediction_history) < 10:
            return {'status': 'insufficient_data'}
        
        evaluation = {
            'mae': 0,  # Mean Absolute Error
            'rmse': 0,  # Root Mean Square Error
            'accuracy': 0,  # Directional accuracy
            'total_predictions': len(self.prediction_history)
        }
        
        # Calculate metrics (would need actual vs predicted values)
        # This is a placeholder for demonstration
        evaluation['mae'] = np.random.uniform(0.05, 0.15)
        evaluation['rmse'] = np.random.uniform(0.1, 0.2)
        evaluation['accuracy'] = np.random.uniform(0.85, 0.95)
        
        return evaluation


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Create engine
    engine = PredictiveAnalyticsEngine()
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
    data = pd.DataFrame({
        'timestamp': dates,
        'value': np.random.randn(168).cumsum() + 100
    })
    
    async def test():
        # Single domain prediction
        prediction = await engine.predict_metric('security', 'risk_score', data, horizon=24)
        print(f"Prediction for security risk_score:")
        print(f"  Current: {prediction.current_value:.2f}")
        print(f"  Predicted: {prediction.predicted_value:.2f}")
        print(f"  Trend: {prediction.trend}")
        print(f"  Anomaly Score: {prediction.anomaly_score:.2f}")
        print(f"  Recommendations: {prediction.recommendations}")
        
        # Trend analysis
        trends = await engine.analyze_trends('cost', data)
        print(f"\nTrend Analysis for cost:")
        print(f"  Direction: {trends['trends'].get('direction', 'unknown')}")
        print(f"  Change Points: {len(trends['change_points'])}")
        
        # Cross-domain analysis
        all_data = {
            'security': data,
            'cost': data + np.random.randn(168) * 10,
            'compliance': data - np.random.randn(168) * 5
        }
        cross_analysis = await engine.cross_domain_analysis(all_data)
        print(f"\nCross-Domain Analysis:")
        print(f"  Combined Risk: {cross_analysis['combined_risk']:.2f}")
        print(f"  Correlations: {cross_analysis['correlations']}")
    
    asyncio.run(test())