"""
Simple ML Service for PolicyCortex
Lightweight implementation using scikit-learn for quick deployment
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_dir: str = os.path.join(os.path.dirname(__file__), "models_cache")
    auto_train: bool = True
    min_samples_for_training: int = 100
    retrain_interval_days: int = 7

class SimpleMlService:
    """
    Lightweight ML service for quick deployment
    Focuses on core predictions: compliance, anomaly detection, cost optimization
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        
        # Create model directory if it doesn't exist
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load existing models"""
        logger.info("Initializing ML models...")
        
        # Compliance Prediction Model
        compliance_model_path = os.path.join(self.config.model_dir, "compliance_model.pkl")
        if os.path.exists(compliance_model_path):
            self._load_compliance_model()
        elif self.config.auto_train:
            self._train_compliance_model()
        
        # Anomaly Detection Model
        anomaly_model_path = os.path.join(self.config.model_dir, "anomaly_model.pkl")
        if os.path.exists(anomaly_model_path):
            self._load_anomaly_model()
        elif self.config.auto_train:
            self._train_anomaly_model()
        
        # Cost Optimization Model
        cost_model_path = os.path.join(self.config.model_dir, "cost_model.pkl")
        if os.path.exists(cost_model_path):
            self._load_cost_model()
        elif self.config.auto_train:
            self._train_cost_model()
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def _load_compliance_model(self):
        """Load existing compliance model"""
        try:
            model_path = os.path.join(self.config.model_dir, "compliance_model.pkl")
            scaler_path = os.path.join(self.config.model_dir, "compliance_scaler.pkl")
            metadata_path = os.path.join(self.config.model_dir, "compliance_metadata.json")
            
            self.models['compliance'] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers['compliance'] = joblib.load(scaler_path)
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata['compliance'] = json.load(f)
            
            logger.info("Loaded compliance model")
        except Exception as e:
            logger.error(f"Failed to load compliance model: {e}")
    
    def _load_anomaly_model(self):
        """Load existing anomaly model"""
        try:
            model_path = os.path.join(self.config.model_dir, "anomaly_model.pkl")
            scaler_path = os.path.join(self.config.model_dir, "anomaly_scaler.pkl")
            
            self.models['anomaly'] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers['anomaly'] = joblib.load(scaler_path)
            
            logger.info("Loaded anomaly model")
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
    
    def _load_cost_model(self):
        """Load existing cost model"""
        try:
            model_path = os.path.join(self.config.model_dir, "cost_model.pkl")
            scaler_path = os.path.join(self.config.model_dir, "cost_scaler.pkl")
            
            self.models['cost'] = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scalers['cost'] = joblib.load(scaler_path)
            
            logger.info("Loaded cost model")
        except Exception as e:
            logger.error(f"Failed to load cost model: {e}")
    
    def _train_compliance_model(self):
        """Train a new compliance prediction model"""
        logger.info("Training compliance prediction model...")
        
        # Generate synthetic training data
        X, y = self._generate_compliance_training_data(n_samples=1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Save model and metadata
        self.models['compliance'] = model
        self.scalers['compliance'] = scaler
        self.model_metadata['compliance'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'trained_at': datetime.utcnow().isoformat(),
            'n_samples': len(X),
            'feature_importance': dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                model.feature_importances_.tolist()
            ))
        }
        
        # Persist to disk
        joblib.dump(model, os.path.join(self.config.model_dir, "compliance_model.pkl"))
        joblib.dump(scaler, os.path.join(self.config.model_dir, "compliance_scaler.pkl"))
        with open(os.path.join(self.config.model_dir, "compliance_metadata.json"), 'w') as f:
            json.dump(self.model_metadata['compliance'], f, indent=2)
        
        logger.info(f"Compliance model trained - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    def _train_anomaly_model(self):
        """Train anomaly detection model"""
        logger.info("Training anomaly detection model...")
        
        # Generate training data
        X = self._generate_anomaly_training_data(n_samples=1000)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        # Save model
        self.models['anomaly'] = model
        self.scalers['anomaly'] = scaler
        
        joblib.dump(model, os.path.join(self.config.model_dir, "anomaly_model.pkl"))
        joblib.dump(scaler, os.path.join(self.config.model_dir, "anomaly_scaler.pkl"))
        
        logger.info("Anomaly detection model trained")
    
    def _train_cost_model(self):
        """Train cost optimization model"""
        logger.info("Training cost optimization model...")
        
        # Generate training data
        X, y = self._generate_cost_training_data(n_samples=1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Save model
        self.models['cost'] = model
        self.scalers['cost'] = scaler
        
        joblib.dump(model, os.path.join(self.config.model_dir, "cost_model.pkl"))
        joblib.dump(scaler, os.path.join(self.config.model_dir, "cost_scaler.pkl"))
        
        logger.info(f"Cost optimization model trained - RMSE: ${rmse:.2f}")
    
    def predict_compliance(self, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict compliance status for a resource
        
        Args:
            resource_data: Dictionary containing resource information
            
        Returns:
            Prediction result with confidence score
        """
        try:
            if 'compliance' not in self.models:
                return self._default_compliance_prediction()
            
            # Extract features
            features = self._extract_compliance_features(resource_data)
            
            # Scale features
            if 'compliance' in self.scalers:
                features_scaled = self.scalers['compliance'].transform([features])
            else:
                features_scaled = [features]
            
            # Predict
            model = self.models['compliance']
            prediction = model.predict(features_scaled)[0]
            
            # Get probability scores if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = float(max(probabilities))
            else:
                confidence = 0.85  # Default confidence
            
            # Map prediction to status
            status_map = {0: "Non-Compliant", 1: "Compliant", 2: "Needs Review"}
            status = status_map.get(int(prediction), "Unknown")
            
            # Generate recommendations based on prediction
            recommendations = self._generate_compliance_recommendations(status, resource_data)
            
            return {
                "resource_id": resource_data.get("id", "unknown"),
                "status": status,
                "confidence": confidence,
                "risk_level": self._calculate_risk_level(confidence, status),
                "recommendations": recommendations,
                "predicted_at": datetime.utcnow().isoformat(),
                "model_version": self.model_metadata.get('compliance', {}).get('trained_at', 'v1.0')
            }
            
        except Exception as e:
            logger.error(f"Compliance prediction failed: {e}")
            return self._default_compliance_prediction()
    
    def detect_anomalies(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect anomalies in metrics data
        
        Args:
            metrics_data: List of metric dictionaries with timestamps and values
            
        Returns:
            Anomaly detection results
        """
        try:
            if 'anomaly' not in self.models:
                return self._default_anomaly_detection()
            
            # Extract features from metrics
            features = self._extract_anomaly_features(metrics_data)
            
            # Scale features
            if 'anomaly' in self.scalers:
                features_scaled = self.scalers['anomaly'].transform(features)
            else:
                features_scaled = features
            
            # Detect anomalies
            model = self.models['anomaly']
            predictions = model.predict(features_scaled)
            
            # Get anomaly scores if available
            if hasattr(model, 'score_samples'):
                scores = model.score_samples(features_scaled)
            else:
                scores = predictions
            
            # Process results
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly detected
                    anomalies.append({
                        "timestamp": metrics_data[i].get("timestamp", datetime.utcnow().isoformat()),
                        "value": metrics_data[i].get("value", 0),
                        "anomaly_score": float(abs(score)),
                        "severity": self._calculate_anomaly_severity(score)
                    })
            
            return {
                "anomalies_detected": len(anomalies),
                "total_points": len(metrics_data),
                "anomaly_rate": len(anomalies) / len(metrics_data) if metrics_data else 0,
                "anomalies": anomalies[:10],  # Limit to top 10
                "summary": self._generate_anomaly_summary(anomalies),
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return self._default_anomaly_detection()
    
    def optimize_costs(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate cost optimization recommendations
        
        Args:
            usage_data: Dictionary containing resource usage information
            
        Returns:
            Cost optimization recommendations
        """
        try:
            if 'cost' not in self.models:
                return self._default_cost_optimization()
            
            # Extract features
            features = self._extract_cost_features(usage_data)
            
            # Scale features
            if 'cost' in self.scalers:
                features_scaled = self.scalers['cost'].transform([features])
            else:
                features_scaled = [features]
            
            # Predict potential savings
            model = self.models['cost']
            predicted_savings = model.predict(features_scaled)[0]
            
            # Generate recommendations
            current_cost = usage_data.get("monthly_cost", 0)
            recommendations = self._generate_cost_recommendations(
                current_cost, 
                predicted_savings,
                usage_data
            )
            
            return {
                "current_monthly_cost": current_cost,
                "predicted_monthly_cost": max(0, current_cost - predicted_savings),
                "estimated_savings": max(0, predicted_savings),
                "savings_percentage": (predicted_savings / current_cost * 100) if current_cost > 0 else 0,
                "recommendations": recommendations,
                "confidence": 0.85,
                "analyzed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cost optimization failed: {e}")
            return self._default_cost_optimization()
    
    # Feature extraction methods
    
    def _extract_compliance_features(self, resource_data: Dict) -> np.ndarray:
        """Extract features for compliance prediction"""
        features = []
        
        # Basic resource attributes
        features.append(1 if resource_data.get("encryption_enabled", False) else 0)
        features.append(1 if resource_data.get("backup_enabled", False) else 0)
        features.append(1 if resource_data.get("monitoring_enabled", False) else 0)
        features.append(1 if resource_data.get("public_access", False) else 0)
        
        # Tags and metadata
        tags = resource_data.get("tags", {})
        features.append(len(tags))
        features.append(1 if "Environment" in tags else 0)
        features.append(1 if "Owner" in tags else 0)
        features.append(1 if "CostCenter" in tags else 0)
        
        # Configuration scores
        config = resource_data.get("configuration", {})
        features.append(len(config))
        features.append(1 if config.get("tls_version", "") >= "1.2" else 0)
        
        # Age and modification
        created_days_ago = resource_data.get("age_days", 0)
        features.append(min(created_days_ago, 365))  # Cap at 1 year
        features.append(resource_data.get("modifications_last_30_days", 0))
        
        return np.array(features)
    
    def _extract_anomaly_features(self, metrics_data: List[Dict]) -> np.ndarray:
        """Extract features for anomaly detection"""
        if not metrics_data:
            return np.array([[0] * 10])
        
        features = []
        for metric in metrics_data:
            value = metric.get("value", 0)
            
            # Basic statistical features
            feature_vector = [
                value,
                abs(value),
                value ** 2 if abs(value) < 100 else 10000,  # Squared value (capped)
                np.log1p(abs(value)),  # Log transform
                1 if value > 0 else 0,  # Sign
            ]
            
            # Add time-based features if available
            if "timestamp" in metric:
                try:
                    dt = datetime.fromisoformat(metric["timestamp"].replace("Z", "+00:00"))
                    feature_vector.extend([
                        dt.hour,
                        dt.weekday(),
                        dt.day,
                    ])
                except:
                    feature_vector.extend([0, 0, 0])
            else:
                feature_vector.extend([0, 0, 0])
            
            # Add context features
            feature_vector.append(metric.get("resource_count", 1))
            feature_vector.append(metric.get("alert_count", 0))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_cost_features(self, usage_data: Dict) -> np.ndarray:
        """Extract features for cost optimization"""
        features = []
        
        # Utilization metrics
        features.append(usage_data.get("cpu_utilization", 50))
        features.append(usage_data.get("memory_utilization", 50))
        features.append(usage_data.get("storage_utilization", 50))
        features.append(usage_data.get("network_utilization", 50))
        
        # Cost metrics
        features.append(usage_data.get("monthly_cost", 0))
        features.append(usage_data.get("compute_cost", 0))
        features.append(usage_data.get("storage_cost", 0))
        features.append(usage_data.get("network_cost", 0))
        
        # Resource characteristics
        features.append(usage_data.get("instance_count", 1))
        features.append(usage_data.get("average_instance_age_days", 30))
        features.append(1 if usage_data.get("reserved_instances", False) else 0)
        features.append(1 if usage_data.get("spot_instances", False) else 0)
        
        return np.array(features)
    
    # Training data generation methods
    
    def _generate_compliance_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic compliance training data"""
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate features
            encryption = np.random.choice([0, 1], p=[0.3, 0.7])
            backup = np.random.choice([0, 1], p=[0.4, 0.6])
            monitoring = np.random.choice([0, 1], p=[0.35, 0.65])
            public_access = np.random.choice([0, 1], p=[0.8, 0.2])
            
            num_tags = np.random.poisson(3)
            has_env_tag = np.random.choice([0, 1], p=[0.3, 0.7])
            has_owner_tag = np.random.choice([0, 1], p=[0.4, 0.6])
            has_cost_tag = np.random.choice([0, 1], p=[0.5, 0.5])
            
            config_items = np.random.poisson(5)
            tls_compliant = np.random.choice([0, 1], p=[0.2, 0.8])
            
            age_days = np.random.exponential(100)
            modifications = np.random.poisson(2)
            
            features = [
                encryption, backup, monitoring, public_access,
                num_tags, has_env_tag, has_owner_tag, has_cost_tag,
                config_items, tls_compliant, age_days, modifications
            ]
            
            # Determine compliance based on features
            compliance_score = (
                encryption * 0.2 +
                backup * 0.15 +
                monitoring * 0.15 +
                (1 - public_access) * 0.2 +
                has_env_tag * 0.1 +
                has_owner_tag * 0.1 +
                tls_compliant * 0.1
            )
            
            if compliance_score > 0.7:
                label = 1  # Compliant
            elif compliance_score < 0.4:
                label = 0  # Non-compliant
            else:
                label = 2  # Needs review
            
            X.append(features)
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def _generate_anomaly_training_data(self, n_samples: int = 1000) -> np.ndarray:
        """Generate synthetic anomaly training data"""
        np.random.seed(42)
        
        # Generate normal data (90%)
        normal_samples = int(n_samples * 0.9)
        normal_data = np.random.randn(normal_samples, 10) * 10 + 50
        
        # Generate anomalous data (10%)
        anomaly_samples = n_samples - normal_samples
        anomaly_data = np.concatenate([
            np.random.randn(anomaly_samples // 2, 10) * 50 + 100,  # High outliers
            np.random.randn(anomaly_samples // 2, 10) * 5 - 20     # Low outliers
        ])
        
        # Combine and shuffle
        data = np.vstack([normal_data, anomaly_data])
        np.random.shuffle(data)
        
        return data
    
    def _generate_cost_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic cost training data"""
        np.random.seed(42)
        
        X = []
        y = []
        
        for _ in range(n_samples):
            # Generate utilization features
            cpu_util = np.random.beta(2, 5) * 100  # Typically low
            memory_util = np.random.beta(2, 5) * 100
            storage_util = np.random.beta(3, 3) * 100  # More balanced
            network_util = np.random.beta(2, 6) * 100
            
            # Generate cost features
            monthly_cost = np.random.exponential(500) + 100
            compute_cost = monthly_cost * np.random.beta(5, 3)
            storage_cost = monthly_cost * np.random.beta(3, 7)
            network_cost = monthly_cost - compute_cost - storage_cost
            
            # Generate resource features
            instance_count = np.random.poisson(5) + 1
            age_days = np.random.exponential(60)
            has_reserved = np.random.choice([0, 1], p=[0.7, 0.3])
            has_spot = np.random.choice([0, 1], p=[0.8, 0.2])
            
            features = [
                cpu_util, memory_util, storage_util, network_util,
                monthly_cost, compute_cost, storage_cost, network_cost,
                instance_count, age_days, has_reserved, has_spot
            ]
            
            # Calculate potential savings based on utilization
            savings = 0
            if cpu_util < 20 and memory_util < 30:
                savings += monthly_cost * 0.3  # Rightsizing opportunity
            if not has_reserved and age_days > 90:
                savings += monthly_cost * 0.25  # Reserved instance opportunity
            if cpu_util < 10:
                savings += monthly_cost * 0.15  # Shutdown opportunity
            
            X.append(features)
            y.append(savings)
        
        return np.array(X), np.array(y)
    
    # Helper methods
    
    def _calculate_risk_level(self, confidence: float, status: str) -> str:
        """Calculate risk level based on confidence and status"""
        if status == "Non-Compliant":
            if confidence > 0.8:
                return "Critical"
            elif confidence > 0.6:
                return "High"
            else:
                return "Medium"
        elif status == "Needs Review":
            return "Medium"
        else:
            return "Low"
    
    def _calculate_anomaly_severity(self, score: float) -> str:
        """Calculate anomaly severity based on score"""
        abs_score = abs(score)
        if abs_score > 3:
            return "Critical"
        elif abs_score > 2:
            return "High"
        elif abs_score > 1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_compliance_recommendations(self, status: str, resource_data: Dict) -> List[str]:
        """Generate compliance recommendations based on status"""
        recommendations = []
        
        if status == "Non-Compliant":
            if not resource_data.get("encryption_enabled"):
                recommendations.append("Enable encryption for data at rest and in transit")
            if not resource_data.get("backup_enabled"):
                recommendations.append("Configure automated backups with appropriate retention")
            if not resource_data.get("monitoring_enabled"):
                recommendations.append("Enable comprehensive monitoring and alerting")
            if resource_data.get("public_access"):
                recommendations.append("Review and restrict public access settings")
            
            tags = resource_data.get("tags", {})
            if "Owner" not in tags:
                recommendations.append("Add Owner tag for accountability")
            if "Environment" not in tags:
                recommendations.append("Add Environment tag for proper classification")
        
        elif status == "Needs Review":
            recommendations.append("Schedule compliance review with security team")
            recommendations.append("Verify configuration against compliance framework")
            recommendations.append("Update resource documentation")
        
        else:  # Compliant
            recommendations.append("Continue regular compliance monitoring")
            recommendations.append("Document current configuration as baseline")
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_anomaly_summary(self, anomalies: List[Dict]) -> str:
        """Generate summary of detected anomalies"""
        if not anomalies:
            return "No anomalies detected in the analyzed period"
        
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "Unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary_parts = []
        if severity_counts.get("Critical", 0) > 0:
            summary_parts.append(f"{severity_counts['Critical']} critical")
        if severity_counts.get("High", 0) > 0:
            summary_parts.append(f"{severity_counts['High']} high")
        if severity_counts.get("Medium", 0) > 0:
            summary_parts.append(f"{severity_counts['Medium']} medium")
        
        if summary_parts:
            return f"Detected {', '.join(summary_parts)} severity anomalies"
        else:
            return f"Detected {len(anomalies)} low severity anomalies"
    
    def _generate_cost_recommendations(self, current_cost: float, predicted_savings: float, usage_data: Dict) -> List[Dict]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if usage_data.get("cpu_utilization", 100) < 20:
            recommendations.append({
                "action": "Rightsize compute resources",
                "description": "CPU utilization is below 20%. Consider using smaller instance types.",
                "estimated_savings": current_cost * 0.15,
                "priority": "High"
            })
        
        if usage_data.get("memory_utilization", 100) < 30:
            recommendations.append({
                "action": "Optimize memory allocation",
                "description": "Memory utilization is below 30%. Review memory requirements.",
                "estimated_savings": current_cost * 0.10,
                "priority": "Medium"
            })
        
        if not usage_data.get("reserved_instances") and usage_data.get("average_instance_age_days", 0) > 90:
            recommendations.append({
                "action": "Purchase Reserved Instances",
                "description": "Long-running instances detected. Reserved Instances can save up to 72%.",
                "estimated_savings": current_cost * 0.25,
                "priority": "High"
            })
        
        if usage_data.get("storage_utilization", 100) < 40:
            recommendations.append({
                "action": "Optimize storage tiers",
                "description": "Low storage utilization. Consider archival or cold storage tiers.",
                "estimated_savings": current_cost * 0.05,
                "priority": "Low"
            })
        
        if usage_data.get("instance_count", 0) > 5:
            recommendations.append({
                "action": "Implement auto-scaling",
                "description": "Multiple instances detected. Auto-scaling can optimize resource usage.",
                "estimated_savings": current_cost * 0.20,
                "priority": "Medium"
            })
        
        # Sort by estimated savings
        recommendations.sort(key=lambda x: x["estimated_savings"], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    # Default/fallback methods
    
    def _default_compliance_prediction(self) -> Dict[str, Any]:
        """Default compliance prediction when model is not available"""
        return {
            "resource_id": "unknown",
            "status": "Needs Review",
            "confidence": 0.5,
            "risk_level": "Medium",
            "recommendations": [
                "Unable to perform ML-based prediction",
                "Review resource configuration manually",
                "Ensure basic security controls are in place"
            ],
            "predicted_at": datetime.utcnow().isoformat(),
            "model_version": "fallback"
        }
    
    def _default_anomaly_detection(self) -> Dict[str, Any]:
        """Default anomaly detection when model is not available"""
        return {
            "anomalies_detected": 0,
            "total_points": 0,
            "anomaly_rate": 0,
            "anomalies": [],
            "summary": "Anomaly detection model not available",
            "analyzed_at": datetime.utcnow().isoformat()
        }
    
    def _default_cost_optimization(self) -> Dict[str, Any]:
        """Default cost optimization when model is not available"""
        return {
            "current_monthly_cost": 0,
            "predicted_monthly_cost": 0,
            "estimated_savings": 0,
            "savings_percentage": 0,
            "recommendations": [
                {
                    "action": "Review resource usage",
                    "description": "Cost optimization model not available. Manual review recommended.",
                    "estimated_savings": 0,
                    "priority": "Medium"
                }
            ],
            "confidence": 0.5,
            "analyzed_at": datetime.utcnow().isoformat()
        }

# Create singleton instance
simple_ml_service = SimpleMlService()

# Export main prediction functions
def predict_compliance(resource_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict compliance for a resource"""
    return simple_ml_service.predict_compliance(resource_data)

def detect_anomalies(metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect anomalies in metrics"""
    return simple_ml_service.detect_anomalies(metrics_data)

def optimize_costs(usage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cost optimization recommendations"""
    return simple_ml_service.optimize_costs(usage_data)