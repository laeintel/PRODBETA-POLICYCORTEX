"""
Production ML Monitoring with Evidently AI
Implements drift detection, performance monitoring, and automated retraining triggers
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *
import mlflow
from prometheus_client import Counter, Histogram, Gauge
import asyncio
import aiohttp
from sqlalchemy import create_engine, Column, Float, String, DateTime, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mlops")

# Metrics
DRIFT_DETECTED = Counter('drift_detected_total', 'Total drift detections', ['model', 'drift_type'])
PERFORMANCE_DEGRADATION = Counter('performance_degradation_total', 'Performance degradations', ['model', 'metric'])
RETRAINING_TRIGGERED = Counter('retraining_triggered_total', 'Retraining triggers', ['model', 'reason'])
MONITORING_ERRORS = Counter('monitoring_errors_total', 'Monitoring errors', ['model', 'error_type'])

@dataclass
class DriftMetrics:
    """Drift detection metrics"""
    dataset_drift: bool
    dataset_drift_score: float
    feature_drift: Dict[str, float]
    target_drift: Optional[float]
    prediction_drift: Optional[float]
    data_quality_issues: List[str]
    timestamp: datetime
    
    def requires_retraining(self) -> bool:
        """Check if drift requires retraining"""
        # PSI > 0.2 triggers alert (Patent requirement)
        high_psi_features = [
            feature for feature, psi in self.feature_drift.items()
            if psi > 0.2
        ]
        
        # KS > 0.1 triggers investigation (Patent requirement)
        return (
            self.dataset_drift or
            len(high_psi_features) > 0 or
            (self.target_drift and self.target_drift > 0.15) or
            (self.prediction_drift and self.prediction_drift > 0.1)
        )

@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    mae: Optional[float]
    rmse: Optional[float]
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    timestamp: datetime
    
    def has_degraded(self, baseline: 'PerformanceMetrics', threshold: float = 0.05) -> bool:
        """Check if performance has degraded from baseline"""
        # 5% degradation triggers retraining
        degradations = []
        
        if self.accuracy < baseline.accuracy * (1 - threshold):
            degradations.append("accuracy")
        if self.f1_score < baseline.f1_score * (1 - threshold):
            degradations.append("f1_score")
        if self.latency_p95 > baseline.latency_p95 * (1 + threshold * 2):
            degradations.append("latency")
        if self.error_rate > baseline.error_rate * (1 + threshold):
            degradations.append("error_rate")
        
        return len(degradations) > 0

@dataclass
class BiasMetrics:
    """Fairness and bias metrics"""
    protected_groups: List[str]
    demographic_parity_difference: Dict[str, float]
    equal_opportunity_difference: Dict[str, float]
    disparate_impact: Dict[str, float]
    fairness_violations: List[str]
    timestamp: datetime

class MonitoringRecord(Base):
    """Database model for monitoring records"""
    __tablename__ = 'monitoring_records'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    drift_metrics = Column(JSON)
    performance_metrics = Column(JSON)
    bias_metrics = Column(JSON)
    retraining_triggered = Column(String)
    alert_sent = Column(String)

class MLMonitor:
    """
    Comprehensive ML monitoring system with drift detection and automated retraining
    Implements NIST AI RMF Measure function
    """
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.reference_data = None
        self.current_data = None
        self.column_mapping = None
        self.performance_baseline = None
        self.drift_thresholds = {
            'psi': 0.2,  # Patent requirement
            'ks': 0.1,   # Patent requirement
            'wasserstein': 0.1,
            'jensen_shannon': 0.15
        }
        self.retraining_rules = []
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database connection"""
        try:
            self.engine = create_engine(DATABASE_URL)
            Base.metadata.create_all(self.engine)
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.session = None
    
    def set_reference_data(
        self,
        reference_df: pd.DataFrame,
        target_col: Optional[str] = None,
        prediction_col: Optional[str] = None,
        protected_attributes: Optional[List[str]] = None
    ):
        """Set reference data for drift detection"""
        self.reference_data = reference_df
        self.column_mapping = ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            numerical_features=list(reference_df.select_dtypes(include=[np.number]).columns),
            categorical_features=list(reference_df.select_dtypes(include=['object']).columns)
        )
        
        if protected_attributes:
            self.protected_attributes = protected_attributes
        
        logger.info(f"Reference data set with {len(reference_df)} samples")
    
    def set_performance_baseline(self, metrics: PerformanceMetrics):
        """Set baseline performance metrics"""
        self.performance_baseline = metrics
        logger.info(f"Performance baseline set: Accuracy={metrics.accuracy:.4f}")
    
    def calculate_drift(self, current_df: pd.DataFrame) -> DriftMetrics:
        """
        Calculate drift metrics using Evidently
        
        Args:
            current_df: Current production data
            
        Returns:
            DriftMetrics object with all drift scores
        """
        self.current_data = current_df
        
        # Create drift report
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DatasetCorrelationsMetric(),
            ColumnDriftMetric(column_name=col) 
            for col in self.reference_data.columns[:10]  # Top 10 features
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping
        )
        
        # Extract results
        report_dict = drift_report.as_dict()
        
        # Parse drift metrics
        dataset_drift = False
        dataset_drift_score = 0.0
        feature_drift = {}
        
        if 'metrics' in report_dict:
            for metric in report_dict['metrics']:
                if metric.get('metric') == 'DatasetDriftMetric':
                    result = metric.get('result', {})
                    dataset_drift = result.get('dataset_drift', False)
                    dataset_drift_score = result.get('drift_share', 0.0)
                elif metric.get('metric') == 'ColumnDriftMetric':
                    result = metric.get('result', {})
                    column_name = result.get('column_name')
                    if column_name:
                        feature_drift[column_name] = result.get('drift_score', 0.0)
        
        # Calculate PSI for features (Patent requirement)
        psi_scores = self._calculate_psi(self.reference_data, current_df)
        feature_drift.update(psi_scores)
        
        # Calculate target drift if available
        target_drift = None
        if self.column_mapping.target:
            target_drift = self._calculate_ks_statistic(
                self.reference_data[self.column_mapping.target],
                current_df[self.column_mapping.target]
            )
        
        # Calculate prediction drift if available
        prediction_drift = None
        if self.column_mapping.prediction:
            prediction_drift = self._calculate_ks_statistic(
                self.reference_data[self.column_mapping.prediction],
                current_df[self.column_mapping.prediction]
            )
        
        # Identify data quality issues
        quality_issues = self._check_data_quality(current_df)
        
        drift_metrics = DriftMetrics(
            dataset_drift=dataset_drift,
            dataset_drift_score=dataset_drift_score,
            feature_drift=feature_drift,
            target_drift=target_drift,
            prediction_drift=prediction_drift,
            data_quality_issues=quality_issues,
            timestamp=datetime.utcnow()
        )
        
        # Log drift detection
        if drift_metrics.requires_retraining():
            DRIFT_DETECTED.labels(model=self.model_name, drift_type="feature").inc()
            logger.warning(f"Drift detected for {self.model_name}: {drift_metrics}")
        
        return drift_metrics
    
    def calculate_performance(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        latencies: List[float]
    ) -> PerformanceMetrics:
        """
        Calculate current performance metrics
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            latencies: Inference latencies in ms
            
        Returns:
            PerformanceMetrics object
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, mean_absolute_error,
            mean_squared_error
        )
        
        # Classification metrics
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average='weighted', zero_division=0)
        recall = recall_score(actuals, predictions, average='weighted', zero_division=0)
        f1 = f1_score(actuals, predictions, average='weighted', zero_division=0)
        
        # Try AUC if binary classification
        try:
            auc = roc_auc_score(actuals, predictions)
        except:
            auc = None
        
        # Regression metrics (if applicable)
        try:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
        except:
            mae = None
            rmse = None
        
        # Latency metrics
        latencies_array = np.array(latencies)
        latency_p50 = np.percentile(latencies_array, 50)
        latency_p95 = np.percentile(latencies_array, 95)
        latency_p99 = np.percentile(latencies_array, 99)
        
        # Throughput (requests per second)
        throughput = 1000 / np.mean(latencies_array) if len(latencies_array) > 0 else 0
        
        # Error rate
        error_rate = 1 - accuracy
        
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc,
            mae=mae,
            rmse=rmse,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=datetime.utcnow()
        )
        
        # Check for degradation
        if self.performance_baseline and metrics.has_degraded(self.performance_baseline):
            PERFORMANCE_DEGRADATION.labels(model=self.model_name, metric="accuracy").inc()
            logger.warning(f"Performance degradation detected for {self.model_name}")
        
        return metrics
    
    def calculate_bias(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        protected_attributes: List[str]
    ) -> BiasMetrics:
        """
        Calculate fairness and bias metrics
        
        Args:
            predictions: Predictions with protected attributes
            actuals: Actual values
            protected_attributes: List of protected attribute columns
            
        Returns:
            BiasMetrics object
        """
        fairness_violations = []
        demographic_parity = {}
        equal_opportunity = {}
        disparate_impact = {}
        
        for attr in protected_attributes:
            if attr not in predictions.columns:
                continue
            
            # Group by protected attribute
            groups = predictions.groupby(attr)
            
            # Demographic parity difference
            positive_rates = groups.apply(
                lambda x: (x['prediction'] == 1).mean() if 'prediction' in x else 0
            )
            if len(positive_rates) > 1:
                dp_diff = positive_rates.max() - positive_rates.min()
                demographic_parity[attr] = dp_diff
                
                if dp_diff > 0.1:  # 10% threshold
                    fairness_violations.append(f"Demographic parity violation for {attr}")
            
            # Equal opportunity difference (true positive rate difference)
            tpr_by_group = {}
            for group_name, group_data in groups:
                if 'actual' in group_data:
                    tp = ((group_data['prediction'] == 1) & (group_data['actual'] == 1)).sum()
                    fn = ((group_data['prediction'] == 0) & (group_data['actual'] == 1)).sum()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    tpr_by_group[group_name] = tpr
            
            if len(tpr_by_group) > 1:
                eo_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
                equal_opportunity[attr] = eo_diff
                
                if eo_diff > 0.1:
                    fairness_violations.append(f"Equal opportunity violation for {attr}")
            
            # Disparate impact (ratio of positive rates)
            if len(positive_rates) > 1:
                min_rate = positive_rates.min()
                max_rate = positive_rates.max()
                di_ratio = min_rate / max_rate if max_rate > 0 else 0
                disparate_impact[attr] = di_ratio
                
                if di_ratio < 0.8:  # 80% rule
                    fairness_violations.append(f"Disparate impact violation for {attr}")
        
        return BiasMetrics(
            protected_groups=protected_attributes,
            demographic_parity_difference=demographic_parity,
            equal_opportunity_difference=equal_opportunity,
            disparate_impact=disparate_impact,
            fairness_violations=fairness_violations,
            timestamp=datetime.utcnow()
        )
    
    def check_retraining_triggers(
        self,
        drift_metrics: DriftMetrics,
        performance_metrics: PerformanceMetrics,
        bias_metrics: Optional[BiasMetrics] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if retraining should be triggered
        
        Args:
            drift_metrics: Current drift metrics
            performance_metrics: Current performance metrics
            bias_metrics: Optional bias metrics
            
        Returns:
            Tuple of (should_retrain, reasons)
        """
        reasons = []
        
        # Check drift triggers (Patent requirements)
        if drift_metrics.requires_retraining():
            # PSI > 0.2 check
            high_psi_features = [
                f for f, psi in drift_metrics.feature_drift.items()
                if psi > self.drift_thresholds['psi']
            ]
            if high_psi_features:
                reasons.append(f"PSI exceeded for features: {high_psi_features[:3]}")
            
            # KS > 0.1 check
            if drift_metrics.prediction_drift and drift_metrics.prediction_drift > self.drift_thresholds['ks']:
                reasons.append(f"KS statistic {drift_metrics.prediction_drift:.3f} exceeds threshold")
            
            # Dataset drift
            if drift_metrics.dataset_drift:
                reasons.append(f"Dataset drift detected (score: {drift_metrics.dataset_drift_score:.3f})")
        
        # Check performance triggers
        if self.performance_baseline:
            if performance_metrics.has_degraded(self.performance_baseline, threshold=0.05):
                reasons.append(f"Performance degradation: accuracy dropped to {performance_metrics.accuracy:.3f}")
            
            # AUC drop check (Patent requirement)
            if self.performance_baseline.auc_roc and performance_metrics.auc_roc:
                auc_drop = self.performance_baseline.auc_roc - performance_metrics.auc_roc
                if auc_drop > 0.05:
                    reasons.append(f"AUC dropped by {auc_drop:.3f}")
        
        # Check bias triggers
        if bias_metrics and bias_metrics.fairness_violations:
            reasons.append(f"Fairness violations: {', '.join(bias_metrics.fairness_violations[:2])}")
        
        # Check consecutive drift (3 days)
        if self._check_consecutive_drift(3):
            reasons.append("Drift detected for 3 consecutive days")
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            RETRAINING_TRIGGERED.labels(model=self.model_name, reason=reasons[0]).inc()
            logger.info(f"Retraining triggered for {self.model_name}: {reasons}")
        
        return should_retrain, reasons
    
    def save_monitoring_record(
        self,
        drift_metrics: DriftMetrics,
        performance_metrics: PerformanceMetrics,
        bias_metrics: Optional[BiasMetrics] = None,
        retraining_triggered: bool = False
    ):
        """Save monitoring record to database"""
        if not self.session:
            return
        
        try:
            record = MonitoringRecord(
                model_name=self.model_name,
                model_version=self.model_version,
                timestamp=datetime.utcnow(),
                drift_metrics=asdict(drift_metrics),
                performance_metrics=asdict(performance_metrics),
                bias_metrics=asdict(bias_metrics) if bias_metrics else None,
                retraining_triggered="Yes" if retraining_triggered else "No",
                alert_sent="No"
            )
            
            self.session.add(record)
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save monitoring record: {e}")
            self.session.rollback()
    
    def _calculate_psi(self, reference: pd.DataFrame, current: pd.DataFrame) -> Dict[str, float]:
        """Calculate Population Stability Index (PSI) for features"""
        psi_scores = {}
        
        for column in reference.select_dtypes(include=[np.number]).columns:
            if column not in current.columns:
                continue
            
            # Create bins from reference data
            _, bin_edges = np.histogram(reference[column].dropna(), bins=10)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference[column].dropna(), bins=bin_edges)
            curr_counts, _ = np.histogram(current[column].dropna(), bins=bin_edges)
            
            # Normalize to proportions
            ref_prop = (ref_counts + 1) / (ref_counts.sum() + 10)
            curr_prop = (curr_counts + 1) / (curr_counts.sum() + 10)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            psi_scores[f"{column}_psi"] = psi
        
        return psi_scores
    
    def _calculate_ks_statistic(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        from scipy import stats
        
        try:
            ks_stat, _ = stats.ks_2samp(reference.dropna(), current.dropna())
            return ks_stat
        except:
            return 0.0
    
    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check for data quality issues"""
        issues = []
        
        # Check for high missing values
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.1].index.tolist()
        if high_missing:
            issues.append(f"High missing values in: {high_missing[:3]}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols[:3]}")
        
        # Check for duplicates
        if df.duplicated().sum() > len(df) * 0.05:
            issues.append("High duplicate rate")
        
        return issues
    
    def _check_consecutive_drift(self, days: int) -> bool:
        """Check if drift has been detected for consecutive days"""
        if not self.session:
            return False
        
        try:
            # Query recent records
            cutoff = datetime.utcnow() - timedelta(days=days)
            records = self.session.query(MonitoringRecord).filter(
                MonitoringRecord.model_name == self.model_name,
                MonitoringRecord.timestamp >= cutoff
            ).order_by(MonitoringRecord.timestamp.desc()).limit(days).all()
            
            if len(records) < days:
                return False
            
            # Check if all have drift
            for record in records:
                drift_metrics = record.drift_metrics
                if not drift_metrics or not drift_metrics.get('dataset_drift'):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check consecutive drift: {e}")
            return False

class MonitoringOrchestrator:
    """
    Orchestrates monitoring across multiple models and triggers actions
    """
    
    def __init__(self):
        self.monitors = {}
        self.alert_manager = AlertManager()
        self.retraining_queue = []
        
    def register_model(
        self,
        model_name: str,
        model_version: str,
        reference_data: pd.DataFrame,
        baseline_metrics: PerformanceMetrics
    ):
        """Register a model for monitoring"""
        monitor = MLMonitor(model_name, model_version)
        monitor.set_reference_data(reference_data)
        monitor.set_performance_baseline(baseline_metrics)
        
        self.monitors[model_name] = monitor
        logger.info(f"Registered {model_name} for monitoring")
    
    async def monitor_production(
        self,
        model_name: str,
        current_data: pd.DataFrame,
        predictions: np.ndarray,
        actuals: np.ndarray,
        latencies: List[float]
    ):
        """Monitor production model performance"""
        if model_name not in self.monitors:
            logger.error(f"Model {model_name} not registered")
            return
        
        monitor = self.monitors[model_name]
        
        # Calculate metrics
        drift_metrics = monitor.calculate_drift(current_data)
        performance_metrics = monitor.calculate_performance(predictions, actuals, latencies)
        
        # Check for bias if protected attributes available
        bias_metrics = None
        if hasattr(monitor, 'protected_attributes'):
            predictions_df = pd.DataFrame({'prediction': predictions})
            actuals_df = pd.DataFrame({'actual': actuals})
            combined_df = pd.concat([predictions_df, actuals_df, current_data], axis=1)
            bias_metrics = monitor.calculate_bias(
                combined_df, combined_df, monitor.protected_attributes
            )
        
        # Check retraining triggers
        should_retrain, reasons = monitor.check_retraining_triggers(
            drift_metrics, performance_metrics, bias_metrics
        )
        
        # Save monitoring record
        monitor.save_monitoring_record(
            drift_metrics, performance_metrics, bias_metrics, should_retrain
        )
        
        # Send alerts if needed
        if should_retrain:
            await self.alert_manager.send_alert(
                model_name,
                "Retraining Required",
                f"Reasons: {', '.join(reasons)}"
            )
            self.retraining_queue.append({
                'model_name': model_name,
                'timestamp': datetime.utcnow(),
                'reasons': reasons
            })
        
        # Log to MLflow
        self._log_to_mlflow(model_name, drift_metrics, performance_metrics)
    
    def _log_to_mlflow(
        self,
        model_name: str,
        drift_metrics: DriftMetrics,
        performance_metrics: PerformanceMetrics
    ):
        """Log monitoring metrics to MLflow"""
        try:
            mlflow.set_experiment(f"{model_name}_monitoring")
            
            with mlflow.start_run():
                # Log drift metrics
                mlflow.log_metric("dataset_drift", float(drift_metrics.dataset_drift))
                mlflow.log_metric("drift_score", drift_metrics.dataset_drift_score)
                
                # Log performance metrics
                mlflow.log_metric("accuracy", performance_metrics.accuracy)
                mlflow.log_metric("f1_score", performance_metrics.f1_score)
                mlflow.log_metric("latency_p95", performance_metrics.latency_p95)
                mlflow.log_metric("throughput", performance_metrics.throughput)
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")

class AlertManager:
    """Manages monitoring alerts"""
    
    async def send_alert(self, model_name: str, alert_type: str, message: str):
        """Send monitoring alert"""
        alert = {
            'model_name': model_name,
            'alert_type': alert_type,
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Log alert
        logger.warning(f"ALERT [{alert_type}] for {model_name}: {message}")
        
        # In production, send to alerting service (PagerDuty, Slack, etc.)
        # For now, just log
        return alert

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000),
        'feature_3': np.random.randn(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # Simulate drift in current data
    current_data = pd.DataFrame({
        'feature_1': np.random.randn(100) + 0.5,  # Drift
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Initialize monitor
    monitor = MLMonitor("test_model", "1.0")
    monitor.set_reference_data(reference_data, target_col='target')
    
    # Calculate drift
    drift_metrics = monitor.calculate_drift(current_data)
    print(f"Drift detected: {drift_metrics.dataset_drift}")
    print(f"Drift score: {drift_metrics.dataset_drift_score}")
    print(f"Requires retraining: {drift_metrics.requires_retraining()}")