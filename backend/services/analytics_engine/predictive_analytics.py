"""
Predictive Analytics Engine
Provides ML-based predictions for various metrics
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class PredictiveAnalytics:
    """
    Advanced predictive analytics using ensemble ML models
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for different prediction tasks"""

        # Cost prediction model
        self.models["cost"] = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "gradient_boost": GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42
            ),
        }

        # Performance prediction model
        self.models["performance"] = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42
            )
        }

        # Compliance prediction model
        self.models["compliance"] = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
        }

        # Initialize scalers
        self.scalers["cost"] = StandardScaler()
        self.scalers["performance"] = StandardScaler()
        self.scalers["compliance"] = StandardScaler()

    async def predict_metric(
        self, metric_type: str, historical_data: pd.DataFrame, forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Predict future values for a specific metric

        Args:
            metric_type: Type of metric (cost, performance, compliance, etc.)
            historical_data: Historical data for training
            forecast_horizon: Number of days to forecast

        Returns:
            Predictions with confidence intervals
        """

        if metric_type not in self.models:
            raise ValueError(f"Unsupported metric type: {metric_type}")

        # Prepare features
        X, y = self._prepare_features(historical_data, metric_type)

        # Train models
        predictions = {}
        confidence_scores = {}

        for model_name, model in self.models[metric_type].items():
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scalers[metric_type].fit_transform(X_train)
            X_test_scaled = self.scalers[metric_type].transform(X_test)

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            score = model.score(X_test_scaled, y_test)
            confidence_scores[model_name] = score

            # Generate predictions
            future_X = self._generate_future_features(
                historical_data, forecast_horizon, metric_type
            )
            future_X_scaled = self.scalers[metric_type].transform(future_X)

            predictions[model_name] = model.predict(future_X_scaled)

        # Ensemble predictions
        ensemble_prediction = np.mean(list(predictions.values()), axis=0)

        # Calculate confidence intervals
        prediction_std = np.std(list(predictions.values()), axis=0)
        lower_bound = ensemble_prediction - 1.96 * prediction_std
        upper_bound = ensemble_prediction + 1.96 * prediction_std

        # Generate forecast dates
        last_date = pd.to_datetime(historical_data["date"].max())
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1), periods=forecast_horizon, freq="D"
        )

        return {
            "metric_type": metric_type,
            "predictions": ensemble_prediction.tolist(),
            "confidence_interval": {"lower": lower_bound.tolist(), "upper": upper_bound.tolist()},
            "dates": forecast_dates.strftime("%Y-%m-%d").tolist(),
            "model_confidence": np.mean(list(confidence_scores.values())),
            "individual_predictions": {name: pred.tolist() for name, pred in predictions.items()},
        }

    async def predict_anomalies(
        self, data: pd.DataFrame, sensitivity: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in time series data

        Args:
            data: Time series data
            sensitivity: Sensitivity threshold (0-1)

        Returns:
            List of detected anomalies
        """
        from sklearn.ensemble import IsolationForest

        # Prepare features
        features = self._extract_time_series_features(data)

        # Train isolation forest
        clf = IsolationForest(contamination=1 - sensitivity, random_state=42)

        predictions = clf.fit_predict(features)
        anomaly_scores = clf.score_samples(features)

        # Identify anomalies
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:  # Anomaly
                anomalies.append(
                    {
                        "index": i,
                        "date": data.iloc[i]["date"],
                        "value": data.iloc[i]["value"],
                        "anomaly_score": float(score),
                        "severity": self._calculate_anomaly_severity(score),
                    }
                )

        return anomalies

    async def forecast_with_prophet(
        self, data: pd.DataFrame, periods: int = 30, include_seasonality: bool = True
    ) -> Dict[str, Any]:
        """
        Use Prophet for time series forecasting with seasonality

        Args:
            data: Historical time series data
            periods: Number of periods to forecast
            include_seasonality: Whether to include seasonal components

        Returns:
            Forecast with components
        """

        # Prepare data for Prophet
        df = pd.DataFrame({"ds": pd.to_datetime(data["date"]), "y": data["value"]})

        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=include_seasonality,
            weekly_seasonality=include_seasonality,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.95,
        )

        # Add custom seasonalities if needed
        if include_seasonality:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        # Fit model
        model.fit(df)

        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Generate forecast
        forecast = model.predict(future)

        # Extract components
        components = {
            "trend": forecast["trend"].tolist(),
            "yearly": forecast.get("yearly", []).tolist() if include_seasonality else [],
            "weekly": forecast.get("weekly", []).tolist() if include_seasonality else [],
            "monthly": forecast.get("monthly", []).tolist() if include_seasonality else [],
        }

        return {
            "forecast": forecast["yhat"].tolist(),
            "lower_bound": forecast["yhat_lower"].tolist(),
            "upper_bound": forecast["yhat_upper"].tolist(),
            "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "components": components,
            "changepoints": (
                model.changepoints.dt.strftime("%Y-%m-%d").tolist()
                if len(model.changepoints) > 0
                else []
            ),
        }

    async def predict_capacity_needs(
        self, resource_usage: pd.DataFrame, growth_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Predict future capacity requirements

        Args:
            resource_usage: Historical resource usage data
            growth_rate: Expected growth rate (if known)

        Returns:
            Capacity predictions and recommendations
        """

        # Calculate current utilization trends
        current_utilization = resource_usage["utilization"].mean()
        utilization_trend = resource_usage["utilization"].pct_change().mean()

        # If growth rate not provided, estimate from data
        if growth_rate is None:
            growth_rate = self._estimate_growth_rate(resource_usage)

        # Project future capacity needs
        forecast_periods = 90  # 3 months
        future_dates = pd.date_range(
            start=resource_usage["date"].max() + timedelta(days=1),
            periods=forecast_periods,
            freq="D",
        )

        # Simple projection model
        projected_usage = []
        current = resource_usage["utilization"].iloc[-1]

        for i in range(forecast_periods):
            daily_growth = (1 + growth_rate) ** (1 / 365)
            current = current * daily_growth
            projected_usage.append(current)

        # Identify capacity breach points
        capacity_threshold = 80  # 80% utilization threshold
        breach_date = None

        for i, usage in enumerate(projected_usage):
            if usage > capacity_threshold:
                breach_date = future_dates[i]
                break

        # Generate recommendations
        recommendations = []

        if breach_date:
            days_until_breach = (breach_date - datetime.now()).days

            if days_until_breach < 30:
                recommendations.append(
                    {
                        "priority": "critical",
                        "action": "Immediate capacity expansion required",
                        "details": f"Capacity threshold will be breached in {days_until_breach} days",
                    }
                )
            elif days_until_breach < 60:
                recommendations.append(
                    {
                        "priority": "high",
                        "action": "Plan capacity expansion",
                        "details": f"Capacity threshold will be breached in {days_until_breach} days",
                    }
                )
            else:
                recommendations.append(
                    {
                        "priority": "medium",
                        "action": "Monitor capacity trends",
                        "details": f"Capacity threshold projected to be breached in {days_until_breach} days",
                    }
                )

        return {
            "current_utilization": float(current_utilization),
            "growth_rate": float(growth_rate),
            "projected_usage": projected_usage,
            "dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "breach_date": breach_date.strftime("%Y-%m-%d") if breach_date else None,
            "recommendations": recommendations,
        }

    def _prepare_features(
        self, data: pd.DataFrame, metric_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML models"""

        # Extract time-based features
        data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek
        data["day_of_month"] = pd.to_datetime(data["date"]).dt.day
        data["month"] = pd.to_datetime(data["date"]).dt.month
        data["quarter"] = pd.to_datetime(data["date"]).dt.quarter

        # Add lag features
        for lag in [1, 7, 30]:
            data[f"lag_{lag}"] = data["value"].shift(lag)

        # Add rolling statistics
        for window in [7, 30]:
            data[f"rolling_mean_{window}"] = data["value"].rolling(window).mean()
            data[f"rolling_std_{window}"] = data["value"].rolling(window).std()

        # Remove NaN values
        data = data.dropna()

        # Select features based on metric type
        if metric_type == "cost":
            feature_cols = [
                "day_of_week",
                "day_of_month",
                "month",
                "quarter",
                "lag_1",
                "lag_7",
                "lag_30",
                "rolling_mean_7",
                "rolling_std_7",
                "rolling_mean_30",
                "rolling_std_30",
            ]
        else:
            feature_cols = [
                "day_of_week",
                "month",
                "lag_1",
                "lag_7",
                "rolling_mean_7",
                "rolling_mean_30",
            ]

        X = data[feature_cols].values
        y = data["value"].values

        return X, y

    def _generate_future_features(
        self, historical_data: pd.DataFrame, periods: int, metric_type: str
    ) -> np.ndarray:
        """Generate features for future predictions"""

        last_date = pd.to_datetime(historical_data["date"].max())
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq="D")

        # Create future dataframe
        future_df = pd.DataFrame({"date": future_dates})

        # Add time-based features
        future_df["day_of_week"] = future_df["date"].dt.dayofweek
        future_df["day_of_month"] = future_df["date"].dt.day
        future_df["month"] = future_df["date"].dt.month
        future_df["quarter"] = future_df["date"].dt.quarter

        # Use last known values for lag features
        last_values = historical_data["value"].tail(30).values

        # Simplified feature generation for future data
        features = []
        for i in range(periods):
            if metric_type == "cost":
                feature = [
                    future_df.iloc[i]["day_of_week"],
                    future_df.iloc[i]["day_of_month"],
                    future_df.iloc[i]["month"],
                    future_df.iloc[i]["quarter"],
                    last_values[-1] if len(last_values) >= 1 else 0,  # lag_1
                    last_values[-7] if len(last_values) >= 7 else 0,  # lag_7
                    last_values[-30] if len(last_values) >= 30 else 0,  # lag_30
                    np.mean(last_values[-7:]) if len(last_values) >= 7 else 0,  # rolling_mean_7
                    np.std(last_values[-7:]) if len(last_values) >= 7 else 0,  # rolling_std_7
                    np.mean(last_values) if len(last_values) >= 30 else 0,  # rolling_mean_30
                    np.std(last_values) if len(last_values) >= 30 else 0,  # rolling_std_30
                ]
            else:
                feature = [
                    future_df.iloc[i]["day_of_week"],
                    future_df.iloc[i]["month"],
                    last_values[-1] if len(last_values) >= 1 else 0,  # lag_1
                    last_values[-7] if len(last_values) >= 7 else 0,  # lag_7
                    np.mean(last_values[-7:]) if len(last_values) >= 7 else 0,  # rolling_mean_7
                    np.mean(last_values) if len(last_values) >= 30 else 0,  # rolling_mean_30
                ]

            features.append(feature)

        return np.array(features)

    def _extract_time_series_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for anomaly detection"""

        features = []

        for i in range(len(data)):
            # Get surrounding context
            window_size = 5
            start_idx = max(0, i - window_size)
            end_idx = min(len(data), i + window_size + 1)

            window_data = data.iloc[start_idx:end_idx]["value"]

            feature = [
                data.iloc[i]["value"],
                window_data.mean(),
                window_data.std() if len(window_data) > 1 else 0,
                window_data.min(),
                window_data.max(),
                data.iloc[i]["value"] - window_data.mean(),  # Deviation from local mean
            ]

            features.append(feature)

        return np.array(features)

    def _calculate_anomaly_severity(self, score: float) -> str:
        """Calculate anomaly severity based on score"""

        # Score is typically negative for anomalies
        abs_score = abs(score)

        if abs_score > 0.5:
            return "critical"
        elif abs_score > 0.3:
            return "high"
        elif abs_score > 0.15:
            return "medium"
        else:
            return "low"

    def _estimate_growth_rate(self, data: pd.DataFrame) -> float:
        """Estimate growth rate from historical data"""

        # Simple linear regression on log scale
        from scipy import stats

        # Convert to numeric index
        x = np.arange(len(data))
        y = np.log(data["utilization"].values + 1)  # Add 1 to avoid log(0)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Convert slope to annual growth rate
        daily_growth = np.exp(slope) - 1
        annual_growth = (1 + daily_growth) ** 365 - 1

        return annual_growth
