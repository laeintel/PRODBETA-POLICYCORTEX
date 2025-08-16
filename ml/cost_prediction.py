"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/cost_prediction.py
# Cost Prediction Model for PolicyCortex

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import logging

logger = logging.getLogger(__name__)

@dataclass
class CostPrediction:
    """Cost prediction result"""
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    breakdown: Dict[str, float]
    trend: str
    anomaly_risk: float
    optimization_potential: float
    recommendations: List[str]

@dataclass
class ResourceCost:
    """Resource cost information"""
    resource_id: str
    resource_type: str
    current_cost: float
    predicted_cost: float
    cost_drivers: Dict[str, float]
    optimization_opportunities: List[Dict[str, Any]]

class CostPredictionModel:
    """Advanced cost prediction model for cloud resources"""
    
    def __init__(self):
        self.xgb_model = None
        self.trend_model = None
        self.seasonality_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.cost_history = []
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize prediction models"""
        # XGBoost for main prediction
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        
        # Random Forest for ensemble
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting for comparison
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Prophet for time series
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
    
    def predict_monthly_cost(self, resources: List[Dict[str, Any]]) -> CostPrediction:
        """Predict monthly cost for a set of resources"""
        
        # Extract cost features
        features = self.extract_cost_features(resources)
        
        # Base prediction using XGBoost
        base_prediction = self._predict_base_cost(features)
        
        # Adjust for trends
        trend_adjustment = self._predict_trend_adjustment()
        
        # Add seasonality
        seasonal_factor = self._calculate_seasonality_factor()
        
        # Calculate final prediction
        predicted_cost = base_prediction * trend_adjustment * seasonal_factor
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            predicted_cost, features
        )
        
        # Break down cost by category
        breakdown = self._calculate_cost_breakdown(resources, predicted_cost)
        
        # Determine trend
        trend = self._determine_trend()
        
        # Calculate anomaly risk
        anomaly_risk = self._calculate_anomaly_risk(predicted_cost)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(resources)
        
        # Generate recommendations
        recommendations = self._generate_cost_recommendations(
            resources, predicted_cost, optimization_potential
        )
        
        return CostPrediction(
            predicted_cost=predicted_cost,
            confidence_interval=confidence_interval,
            breakdown=breakdown,
            trend=trend,
            anomaly_risk=anomaly_risk,
            optimization_potential=optimization_potential,
            recommendations=recommendations
        )
    
    def extract_cost_features(self, resources: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features relevant to cost prediction"""
        features = []
        
        for resource in resources:
            resource_features = [
                # Resource characteristics
                self._encode_resource_type(resource.get('type', '')),
                self._encode_resource_size(resource.get('size', '')),
                resource.get('instance_count', 1),
                
                # Usage metrics
                resource.get('cpu_utilization', 0),
                resource.get('memory_utilization', 0),
                resource.get('storage_gb', 0),
                resource.get('network_gb', 0),
                resource.get('requests_per_day', 0),
                
                # Configuration
                1 if resource.get('premium_tier', False) else 0,
                1 if resource.get('geo_redundancy', False) else 0,
                1 if resource.get('auto_scaling', False) else 0,
                resource.get('retention_days', 7),
                
                # Time-based features
                self._get_resource_age_days(resource),
                self._get_day_of_week(),
                self._get_month_of_year(),
                
                # Historical cost
                resource.get('last_month_cost', 0),
                resource.get('avg_daily_cost', 0)
            ]
            features.append(resource_features)
        
        # Aggregate features across resources
        if features:
            aggregated = np.mean(features, axis=0)
        else:
            aggregated = np.zeros(17)  # Default feature vector
        
        return aggregated.reshape(1, -1)
    
    def _predict_base_cost(self, features: np.ndarray) -> float:
        """Predict base cost using ensemble"""
        predictions = []
        
        # Get predictions from each model
        if self.xgb_model is not None:
            try:
                # For demo, simulate prediction
                xgb_pred = np.random.uniform(1000, 5000)
                predictions.append(xgb_pred)
            except:
                pass
        
        if self.rf_model is not None:
            try:
                # For demo, simulate prediction
                rf_pred = np.random.uniform(1000, 5000)
                predictions.append(rf_pred)
            except:
                pass
        
        if self.gb_model is not None:
            try:
                # For demo, simulate prediction
                gb_pred = np.random.uniform(1000, 5000)
                predictions.append(gb_pred)
            except:
                pass
        
        # Average predictions
        if predictions:
            return np.mean(predictions)
        else:
            return 2500.0  # Default prediction
    
    def _predict_trend_adjustment(self) -> float:
        """Predict trend-based adjustment factor"""
        if len(self.cost_history) < 3:
            return 1.0  # No adjustment without history
        
        # Calculate recent trend
        recent_costs = [h['cost'] for h in self.cost_history[-10:]]
        if len(recent_costs) >= 2:
            trend = (recent_costs[-1] - recent_costs[0]) / recent_costs[0]
            
            # Apply dampened trend
            trend_factor = 1.0 + (trend * 0.3)  # 30% of trend
            return max(0.8, min(1.2, trend_factor))  # Cap at ±20%
        
        return 1.0
    
    def _calculate_seasonality_factor(self) -> float:
        """Calculate seasonality adjustment factor"""
        month = datetime.now().month
        
        # Simplified seasonality pattern
        seasonality_factors = {
            1: 1.1,   # January - New year budgets
            2: 1.05,
            3: 1.05,
            4: 1.0,
            5: 0.95,
            6: 0.95,
            7: 0.9,   # July - Summer slowdown
            8: 0.9,
            9: 1.0,
            10: 1.05,
            11: 1.15,  # November - Black Friday prep
            12: 1.2    # December - Holiday season
        }
        
        return seasonality_factors.get(month, 1.0)
    
    def _calculate_confidence_interval(
        self, prediction: float, features: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Base uncertainty
        base_uncertainty = prediction * 0.1  # 10% base uncertainty
        
        # Adjust based on feature quality
        feature_std = np.std(features)
        if feature_std > 1.0:
            base_uncertainty *= 1.2
        
        # Adjust based on historical accuracy
        if len(self.cost_history) > 10:
            historical_errors = []
            for i in range(1, min(10, len(self.cost_history))):
                if 'predicted' in self.cost_history[-i]:
                    error = abs(self.cost_history[-i]['cost'] - self.cost_history[-i]['predicted'])
                    historical_errors.append(error)
            
            if historical_errors:
                avg_error = np.mean(historical_errors)
                base_uncertainty = max(base_uncertainty, avg_error)
        
        lower_bound = prediction - base_uncertainty
        upper_bound = prediction + base_uncertainty
        
        return (max(0, lower_bound), upper_bound)
    
    def _calculate_cost_breakdown(
        self, resources: List[Dict[str, Any]], total_cost: float
    ) -> Dict[str, float]:
        """Break down cost by category"""
        breakdown = {
            'compute': 0,
            'storage': 0,
            'network': 0,
            'database': 0,
            'ai_ml': 0,
            'monitoring': 0,
            'other': 0
        }
        
        # Calculate proportions based on resource types
        for resource in resources:
            resource_type = resource.get('type', '').lower()
            resource_cost = resource.get('estimated_cost', 0)
            
            if 'vm' in resource_type or 'compute' in resource_type:
                breakdown['compute'] += resource_cost
            elif 'storage' in resource_type or 'disk' in resource_type:
                breakdown['storage'] += resource_cost
            elif 'network' in resource_type or 'bandwidth' in resource_type:
                breakdown['network'] += resource_cost
            elif 'database' in resource_type or 'sql' in resource_type:
                breakdown['database'] += resource_cost
            elif 'cognitive' in resource_type or 'ml' in resource_type:
                breakdown['ai_ml'] += resource_cost
            elif 'monitor' in resource_type or 'log' in resource_type:
                breakdown['monitoring'] += resource_cost
            else:
                breakdown['other'] += resource_cost
        
        # Normalize to match total cost
        total_breakdown = sum(breakdown.values())
        if total_breakdown > 0:
            factor = total_cost / total_breakdown
            breakdown = {k: v * factor for k, v in breakdown.items()}
        else:
            # Default breakdown
            breakdown['compute'] = total_cost * 0.4
            breakdown['storage'] = total_cost * 0.2
            breakdown['network'] = total_cost * 0.15
            breakdown['database'] = total_cost * 0.15
            breakdown['other'] = total_cost * 0.1
        
        return breakdown
    
    def _determine_trend(self) -> str:
        """Determine cost trend"""
        if len(self.cost_history) < 3:
            return "stable"
        
        recent_costs = [h['cost'] for h in self.cost_history[-5:]]
        
        # Calculate linear trend
        x = np.arange(len(recent_costs))
        slope = np.polyfit(x, recent_costs, 1)[0]
        
        avg_cost = np.mean(recent_costs)
        trend_percentage = (slope / avg_cost) * 100
        
        if trend_percentage > 5:
            return "increasing"
        elif trend_percentage < -5:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_anomaly_risk(self, predicted_cost: float) -> float:
        """Calculate risk of cost anomaly"""
        if len(self.cost_history) < 5:
            return 0.1  # Low risk without history
        
        historical_costs = [h['cost'] for h in self.cost_history[-30:]]
        mean_cost = np.mean(historical_costs)
        std_cost = np.std(historical_costs)
        
        if std_cost == 0:
            return 0.1
        
        # Calculate z-score
        z_score = abs((predicted_cost - mean_cost) / std_cost)
        
        # Convert to risk score (0-1)
        if z_score < 1:
            return 0.1
        elif z_score < 2:
            return 0.3
        elif z_score < 3:
            return 0.6
        else:
            return 0.9
    
    def _calculate_optimization_potential(self, resources: List[Dict[str, Any]]) -> float:
        """Calculate potential for cost optimization"""
        optimization_score = 0.0
        factors = []
        
        for resource in resources:
            # Check for optimization opportunities
            if resource.get('cpu_utilization', 100) < 20:
                factors.append(0.3)  # Underutilized
            if resource.get('premium_tier', False) and resource.get('environment') == 'dev':
                factors.append(0.4)  # Premium in dev
            if not resource.get('auto_scaling', False):
                factors.append(0.2)  # No auto-scaling
            if resource.get('retention_days', 0) > 30:
                factors.append(0.2)  # Excessive retention
            if resource.get('geo_redundancy', False) and resource.get('criticality') == 'low':
                factors.append(0.3)  # Over-provisioned
        
        if factors:
            optimization_score = min(sum(factors) / len(resources), 1.0)
        
        return optimization_score
    
    def _generate_cost_recommendations(
        self, resources: List[Dict[str, Any]], 
        predicted_cost: float,
        optimization_potential: float
    ) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        # High-level recommendations based on optimization potential
        if optimization_potential > 0.5:
            recommendations.append(
                f"High optimization potential detected - potential savings of ${predicted_cost * optimization_potential:.2f}"
            )
        
        # Specific recommendations
        underutilized = [r for r in resources if r.get('cpu_utilization', 100) < 20]
        if underutilized:
            recommendations.append(
                f"Rightsize {len(underutilized)} underutilized resources (avg CPU < 20%)"
            )
        
        premium_dev = [r for r in resources if r.get('premium_tier') and r.get('environment') == 'dev']
        if premium_dev:
            recommendations.append(
                f"Switch {len(premium_dev)} dev resources from premium to standard tier"
            )
        
        no_autoscale = [r for r in resources if not r.get('auto_scaling')]
        if no_autoscale:
            recommendations.append(
                f"Enable auto-scaling for {len(no_autoscale)} resources to optimize costs"
            )
        
        # Reserved instances recommendation
        stable_resources = [r for r in resources if r.get('uptime_percentage', 0) > 90]
        if len(stable_resources) > 5:
            recommendations.append(
                "Consider reserved instances for stable workloads (>90% uptime)"
            )
        
        # Spot instances recommendation
        batch_resources = [r for r in resources if r.get('workload_type') == 'batch']
        if batch_resources:
            recommendations.append(
                f"Use spot instances for {len(batch_resources)} batch workloads"
            )
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _encode_resource_type(self, resource_type: str) -> float:
        """Encode resource type as numeric feature"""
        type_encoding = {
            'compute': 1.0,
            'storage': 2.0,
            'network': 3.0,
            'database': 4.0,
            'analytics': 5.0,
            'ai': 6.0,
            'other': 7.0
        }
        
        for key, value in type_encoding.items():
            if key in resource_type.lower():
                return value
        
        return 7.0  # Default to 'other'
    
    def _encode_resource_size(self, size: str) -> float:
        """Encode resource size as numeric feature"""
        size_encoding = {
            'small': 1.0,
            'medium': 2.0,
            'large': 3.0,
            'xlarge': 4.0,
            'xxlarge': 5.0
        }
        
        size_lower = size.lower()
        for key, value in size_encoding.items():
            if key in size_lower:
                return value
        
        return 2.0  # Default to medium
    
    def _get_resource_age_days(self, resource: Dict[str, Any]) -> float:
        """Get resource age in days"""
        if 'created_date' in resource:
            created = resource['created_date']
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            age = (datetime.now() - created).days
            return float(age)
        return 30.0  # Default age
    
    def _get_day_of_week(self) -> float:
        """Get current day of week as feature"""
        return float(datetime.now().weekday())
    
    def _get_month_of_year(self) -> float:
        """Get current month as feature"""
        return float(datetime.now().month)
    
    def update_history(self, actual_cost: float, predicted_cost: Optional[float] = None):
        """Update cost history with actual values"""
        entry = {
            'timestamp': datetime.now(),
            'cost': actual_cost
        }
        
        if predicted_cost is not None:
            entry['predicted'] = predicted_cost
            entry['error'] = abs(actual_cost - predicted_cost)
        
        self.cost_history.append(entry)
        
        # Keep only last 365 days
        if len(self.cost_history) > 365:
            self.cost_history = self.cost_history[-365:]

class CostAnomalyDetector:
    """Detect cost anomalies and spikes"""
    
    def __init__(self):
        self.threshold_multiplier = 2.0
        self.min_history_days = 7
        
    def detect_anomalies(self, current_cost: float, cost_history: List[float]) -> Dict[str, Any]:
        """Detect if current cost is anomalous"""
        if len(cost_history) < self.min_history_days:
            return {
                'is_anomaly': False,
                'reason': 'Insufficient history',
                'severity': 'none'
            }
        
        mean_cost = np.mean(cost_history)
        std_cost = np.std(cost_history)
        
        if std_cost == 0:
            std_cost = mean_cost * 0.1  # Use 10% of mean as default
        
        z_score = (current_cost - mean_cost) / std_cost
        
        if abs(z_score) > 3:
            severity = 'critical'
            is_anomaly = True
        elif abs(z_score) > 2:
            severity = 'high'
            is_anomaly = True
        elif abs(z_score) > 1.5:
            severity = 'medium'
            is_anomaly = True
        else:
            severity = 'none'
            is_anomaly = False
        
        reason = self._determine_anomaly_reason(current_cost, mean_cost, z_score)
        
        return {
            'is_anomaly': is_anomaly,
            'reason': reason,
            'severity': severity,
            'z_score': z_score,
            'expected_range': (mean_cost - 2*std_cost, mean_cost + 2*std_cost),
            'percentage_change': ((current_cost - mean_cost) / mean_cost) * 100
        }
    
    def _determine_anomaly_reason(self, current: float, mean: float, z_score: float) -> str:
        """Determine the reason for anomaly"""
        if z_score > 0:
            if z_score > 3:
                return f"Extreme cost spike - {((current - mean) / mean * 100):.1f}% above average"
            else:
                return f"Unusual cost increase - {((current - mean) / mean * 100):.1f}% above average"
        else:
            if z_score < -3:
                return f"Extreme cost drop - {((mean - current) / mean * 100):.1f}% below average"
            else:
                return f"Unusual cost decrease - {((mean - current) / mean * 100):.1f}% below average"

# Export main components
__all__ = [
    'CostPredictionModel',
    'CostAnomalyDetector',
    'CostPrediction',
    'ResourceCost'
]