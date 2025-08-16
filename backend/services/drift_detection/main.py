#!/usr/bin/env python3
"""
PolicyCortex Model Drift Detection Service
Real-time drift detection and auto-retrain pipeline using River ML
"""

import asyncio
import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from river import drift, stats, metrics
from river.drift import ADWIN, PageHinkley, KSWIN
import asyncpg
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyCortex Drift Detection Service", version="1.0.0")

class DriftAlert(BaseModel):
    model_id: str
    drift_type: str  # data, concept, performance
    severity: str    # low, medium, high, critical
    detection_time: datetime
    drift_score: float
    affected_features: List[str]
    recommended_actions: List[str]
    metadata: Dict[str, Any]

class DriftConfig(BaseModel):
    model_id: str
    drift_detectors: Dict[str, Any]
    thresholds: Dict[str, float]
    monitoring_window: int = 1000
    alert_settings: Dict[str, Any]

class RetrainRequest(BaseModel):
    model_id: str
    trigger_reason: str
    priority: str = "normal"  # low, normal, high, urgent
    config_overrides: Optional[Dict[str, Any]] = None

class DriftDetectionService:
    """Main drift detection service"""
    
    def __init__(self):
        self.config = self._load_config()
        self.db_pool = None
        self.drift_detectors = {}
        self.performance_trackers = {}
        self.retrain_queue = asyncio.Queue()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        return {
            'database': {
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'policycortex'),
                'user': os.getenv('DATABASE_USER', 'postgres'),
                'password': os.getenv('DATABASE_PASSWORD', 'postgres'),
            },
            'drift': {
                'check_interval': int(os.getenv('DRIFT_CHECK_INTERVAL', 300)),  # 5 minutes
                'data_drift_threshold': float(os.getenv('DATA_DRIFT_THRESHOLD', 0.05)),
                'concept_drift_threshold': float(os.getenv('CONCEPT_DRIFT_THRESHOLD', 0.1)),
                'performance_drift_threshold': float(os.getenv('PERFORMANCE_DRIFT_THRESHOLD', 0.05)),
            },
            'retraining': {
                'auto_retrain_enabled': os.getenv('AUTO_RETRAIN_ENABLED', 'true').lower() == 'true',
                'retrain_service_url': os.getenv('RETRAIN_SERVICE_URL', 'http://localhost:8082'),
                'max_concurrent_retrains': int(os.getenv('MAX_CONCURRENT_RETRAINS', 2)),
            },
            'alerting': {
                'webhook_url': os.getenv('ALERT_WEBHOOK_URL'),
                'email_enabled': os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true',
            }
        }
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing drift detection service...")
        
        # Initialize database pool
        self.db_pool = await asyncpg.create_pool(
            host=self.config['database']['host'],
            port=self.config['database']['port'],
            database=self.config['database']['name'],
            user=self.config['database']['user'],
            password=self.config['database']['password'],
            min_size=2,
            max_size=10
        )
        
        # Load existing drift detectors
        await self._load_drift_detectors()
        
        # Start drift monitoring
        asyncio.create_task(self._drift_monitoring_loop())
        
        # Start retrain processing
        asyncio.create_task(self._retrain_processor())
        
        logger.info("Drift detection service initialized")
    
    async def _load_drift_detectors(self):
        """Load drift detectors for active models"""
        try:
            async with self.db_pool.acquire() as conn:
                models = await conn.fetch(
                    "SELECT model_id, drift_config FROM ml_models WHERE status = 'active'"
                )
                
                for model in models:
                    model_id = model['model_id']
                    drift_config = json.loads(model['drift_config'] or '{}')
                    
                    await self._initialize_model_detectors(model_id, drift_config)
                    
        except Exception as e:
            logger.error(f"Failed to load drift detectors: {e}")
    
    async def _initialize_model_detectors(self, model_id: str, config: Dict[str, Any]):
        """Initialize drift detectors for a model"""
        detectors = {}
        
        # Data drift detectors (for each feature)
        data_detectors = {}
        for feature in config.get('features', []):
            # ADWIN for detecting changes in data streams
            data_detectors[feature] = {
                'adwin': ADWIN(delta=0.002),
                'kswin': KSWIN(alpha=0.005, window_size=100),
                'stats': stats.Mean()
            }
        
        detectors['data_drift'] = data_detectors
        
        # Concept drift detector
        detectors['concept_drift'] = {
            'page_hinkley': PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=1-0.0001),
            'adwin': ADWIN(delta=0.002)
        }
        
        # Performance drift tracker
        performance_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        detectors['performance_drift'] = {}
        for metric in performance_metrics:
            detectors['performance_drift'][metric] = {
                'adwin': ADWIN(delta=0.002),
                'stats': stats.Mean(),
                'threshold': config.get('thresholds', {}).get(metric, 0.05)
            }
        
        self.drift_detectors[model_id] = detectors
        self.performance_trackers[model_id] = {
            'baseline_performance': config.get('baseline_performance', {}),
            'current_performance': {},
            'performance_history': [],
            'last_evaluation': None
        }
        
        logger.info(f"Initialized drift detectors for model: {model_id}")
    
    async def process_prediction(self, model_id: str, prediction_data: Dict[str, Any]):
        """Process a new prediction for drift detection"""
        if model_id not in self.drift_detectors:
            logger.warning(f"No drift detectors for model: {model_id}")
            return
        
        detectors = self.drift_detectors[model_id]
        
        try:
            # Extract features and prediction
            features = prediction_data.get('features', {})
            prediction = prediction_data.get('prediction')
            confidence = prediction_data.get('confidence', 0.0)
            
            # Update data drift detectors
            data_drift_detected = await self._check_data_drift(
                model_id, features, detectors['data_drift']
            )
            
            # Update concept drift detectors
            concept_drift_detected = await self._check_concept_drift(
                model_id, prediction, confidence, detectors['concept_drift']
            )
            
            # Check performance drift (if ground truth available)
            actual_value = prediction_data.get('actual_value')
            if actual_value is not None:
                performance_drift_detected = await self._check_performance_drift(
                    model_id, prediction, actual_value, detectors['performance_drift']
                )
            else:
                performance_drift_detected = False
            
            # Handle drift detection
            if data_drift_detected or concept_drift_detected or performance_drift_detected:
                await self._handle_drift_detection(
                    model_id, data_drift_detected, concept_drift_detected, performance_drift_detected
                )
            
        except Exception as e:
            logger.error(f"Error processing prediction for drift detection: {e}")
    
    async def _check_data_drift(self, model_id: str, features: Dict[str, Any], 
                              detectors: Dict[str, Any]) -> bool:
        """Check for data drift in features"""
        drift_detected = False
        
        for feature_name, value in features.items():
            if feature_name in detectors:
                feature_detectors = detectors[feature_name]
                
                try:
                    # Convert value to float
                    numeric_value = float(value) if value is not None else 0.0
                    
                    # Update ADWIN detector
                    feature_detectors['adwin'].update(numeric_value)
                    if feature_detectors['adwin'].drift_detected:
                        logger.warning(f"ADWIN data drift detected in {feature_name} for model {model_id}")
                        drift_detected = True
                    
                    # Update KSWIN detector
                    feature_detectors['kswin'].update(numeric_value)
                    if feature_detectors['kswin'].drift_detected:
                        logger.warning(f"KSWIN data drift detected in {feature_name} for model {model_id}")
                        drift_detected = True
                    
                    # Update statistics
                    feature_detectors['stats'].update(numeric_value)
                    
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert feature {feature_name} to numeric: {value}")
        
        return drift_detected
    
    async def _check_concept_drift(self, model_id: str, prediction: Any, confidence: float,
                                 detectors: Dict[str, Any]) -> bool:
        """Check for concept drift"""
        drift_detected = False
        
        try:
            # Use prediction confidence as a proxy for concept stability
            # Lower confidence might indicate concept drift
            
            # Update Page-Hinkley detector
            ph_detector = detectors['page_hinkley']
            ph_detector.update(confidence)
            if ph_detector.drift_detected:
                logger.warning(f"Page-Hinkley concept drift detected for model {model_id}")
                drift_detected = True
            
            # Update ADWIN detector with confidence
            adwin_detector = detectors['adwin']
            adwin_detector.update(confidence)
            if adwin_detector.drift_detected:
                logger.warning(f"ADWIN concept drift detected for model {model_id}")
                drift_detected = True
            
        except Exception as e:
            logger.error(f"Error checking concept drift: {e}")
        
        return drift_detected
    
    async def _check_performance_drift(self, model_id: str, prediction: Any, actual: Any,
                                     detectors: Dict[str, Any]) -> bool:
        """Check for performance drift"""
        drift_detected = False
        
        try:
            # Calculate prediction accuracy (simplified binary classification)
            is_correct = 1.0 if prediction == actual else 0.0
            
            # Update performance detectors
            for metric_name, metric_detector in detectors.items():
                if metric_name == 'accuracy':
                    metric_detector['adwin'].update(is_correct)
                    metric_detector['stats'].update(is_correct)
                    
                    if metric_detector['adwin'].drift_detected:
                        logger.warning(f"Performance drift detected in {metric_name} for model {model_id}")
                        drift_detected = True
            
            # Update performance tracker
            tracker = self.performance_trackers[model_id]
            tracker['performance_history'].append({
                'timestamp': datetime.utcnow(),
                'accuracy': is_correct,
                'prediction': prediction,
                'actual': actual
            })
            
            # Limit history size
            if len(tracker['performance_history']) > 10000:
                tracker['performance_history'] = tracker['performance_history'][-5000:]
            
        except Exception as e:
            logger.error(f"Error checking performance drift: {e}")
        
        return drift_detected
    
    async def _handle_drift_detection(self, model_id: str, data_drift: bool, 
                                    concept_drift: bool, performance_drift: bool):
        """Handle detected drift"""
        drift_types = []
        severity = "low"
        
        if data_drift:
            drift_types.append("data")
        if concept_drift:
            drift_types.append("concept")
        if performance_drift:
            drift_types.append("performance")
            severity = "high"  # Performance drift is more critical
        
        # Determine severity
        if len(drift_types) > 1:
            severity = "critical"
        elif concept_drift or performance_drift:
            severity = "high"
        elif data_drift:
            severity = "medium"
        
        # Create drift alert
        alert = DriftAlert(
            model_id=model_id,
            drift_type=", ".join(drift_types),
            severity=severity,
            detection_time=datetime.utcnow(),
            drift_score=0.8,  # Simplified score
            affected_features=["feature1", "feature2"],  # Would be determined by analysis
            recommended_actions=self._get_recommended_actions(drift_types, severity),
            metadata={
                'detection_method': 'river_ml',
                'data_drift': data_drift,
                'concept_drift': concept_drift,
                'performance_drift': performance_drift
            }
        )
        
        # Store alert
        await self._store_drift_alert(alert)
        
        # Send notifications
        await self._send_drift_alert(alert)
        
        # Trigger auto-retrain if enabled and severity is high enough
        if (self.config['retraining']['auto_retrain_enabled'] and 
            severity in ['high', 'critical']):
            await self._trigger_auto_retrain(model_id, alert)
    
    def _get_recommended_actions(self, drift_types: List[str], severity: str) -> List[str]:
        """Get recommended actions for drift"""
        actions = []
        
        if "data" in drift_types:
            actions.append("Review input data quality and preprocessing")
            actions.append("Check for changes in data sources")
        
        if "concept" in drift_types:
            actions.append("Retrain model with recent data")
            actions.append("Review model assumptions and target definition")
        
        if "performance" in drift_types:
            actions.append("Immediate model retraining recommended")
            actions.append("Consider ensemble or online learning approaches")
        
        if severity in ['high', 'critical']:
            actions.append("Consider temporarily disabling automated decisions")
            actions.append("Increase monitoring frequency")
        
        return actions
    
    async def _store_drift_alert(self, alert: DriftAlert):
        """Store drift alert in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO drift_alerts 
                    (model_id, drift_type, severity, detection_time, drift_score,
                     affected_features, recommended_actions, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    alert.model_id,
                    alert.drift_type,
                    alert.severity,
                    alert.detection_time,
                    alert.drift_score,
                    json.dumps(alert.affected_features),
                    json.dumps(alert.recommended_actions),
                    json.dumps(alert.metadata)
                )
        except Exception as e:
            logger.error(f"Failed to store drift alert: {e}")
    
    async def _send_drift_alert(self, alert: DriftAlert):
        """Send drift alert notifications"""
        try:
            # Send webhook notification
            if self.config['alerting']['webhook_url']:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        self.config['alerting']['webhook_url'],
                        json=alert.dict(),
                        timeout=10.0
                    )
            
            # Email notification would be implemented here
            if self.config['alerting']['email_enabled']:
                # Send email logic
                pass
            
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    
    async def _trigger_auto_retrain(self, model_id: str, alert: DriftAlert):
        """Trigger automatic model retraining"""
        retrain_request = RetrainRequest(
            model_id=model_id,
            trigger_reason=f"Drift detected: {alert.drift_type}",
            priority="high" if alert.severity == "critical" else "normal"
        )
        
        await self.retrain_queue.put(retrain_request)
        logger.info(f"Queued auto-retrain for model {model_id}")
    
    async def _retrain_processor(self):
        """Process retrain queue"""
        while True:
            try:
                # Get retrain request from queue
                retrain_request = await asyncio.wait_for(
                    self.retrain_queue.get(), timeout=60.0
                )
                
                # Send to retraining service
                await self._send_retrain_request(retrain_request)
                
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing retrain queue: {e}")
                await asyncio.sleep(10)
    
    async def _send_retrain_request(self, request: RetrainRequest):
        """Send retrain request to retraining service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config['retraining']['retrain_service_url']}/retrain",
                    json=request.dict(),
                    timeout=30.0
                )
                response.raise_for_status()
                
                logger.info(f"Retrain request sent for model {request.model_id}")
                
        except Exception as e:
            logger.error(f"Failed to send retrain request: {e}")
    
    async def _drift_monitoring_loop(self):
        """Main drift monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.config['drift']['check_interval'])
                
                # Periodic drift analysis
                await self._periodic_drift_analysis()
                
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_drift_analysis(self):
        """Perform periodic drift analysis"""
        logger.info("Running periodic drift analysis...")
        
        try:
            # Analyze historical performance trends
            for model_id in self.performance_trackers:
                await self._analyze_performance_trends(model_id)
            
            # Clean up old data
            await self._cleanup_old_data()
            
        except Exception as e:
            logger.error(f"Error in periodic drift analysis: {e}")
    
    async def _analyze_performance_trends(self, model_id: str):
        """Analyze performance trends for a model"""
        tracker = self.performance_trackers[model_id]
        history = tracker['performance_history']
        
        if len(history) < 100:  # Need sufficient data
            return
        
        # Calculate recent performance
        recent_data = history[-100:]  # Last 100 predictions
        recent_accuracy = sum(item['accuracy'] for item in recent_data) / len(recent_data)
        
        # Compare with baseline
        baseline_accuracy = tracker['baseline_performance'].get('accuracy', 0.8)
        
        performance_drop = baseline_accuracy - recent_accuracy
        if performance_drop > self.config['drift']['performance_drift_threshold']:
            logger.warning(f"Performance trend drift detected for model {model_id}")
            
            # Trigger drift handling
            await self._handle_drift_detection(model_id, False, False, True)
    
    async def _cleanup_old_data(self):
        """Clean up old drift data"""
        try:
            async with self.db_pool.acquire() as conn:
                # Delete old drift alerts (keep 90 days)
                await conn.execute(
                    "DELETE FROM drift_alerts WHERE detection_time < NOW() - INTERVAL '90 days'"
                )
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# Global service instance
drift_service = DriftDetectionService()

@app.on_event("startup")
async def startup_event():
    await drift_service.initialize()

@app.post("/models/{model_id}/prediction")
async def process_prediction(model_id: str, prediction_data: Dict[str, Any]):
    """Process a prediction for drift detection"""
    await drift_service.process_prediction(model_id, prediction_data)
    return {"status": "processed"}

@app.post("/models/{model_id}/configure")
async def configure_drift_detection(model_id: str, config: DriftConfig):
    """Configure drift detection for a model"""
    await drift_service._initialize_model_detectors(model_id, config.dict())
    return {"status": "configured"}

@app.get("/models/{model_id}/alerts")
async def get_drift_alerts(model_id: str, limit: int = 100):
    """Get drift alerts for a model"""
    async with drift_service.db_pool.acquire() as conn:
        alerts = await conn.fetch(
            """
            SELECT * FROM drift_alerts 
            WHERE model_id = $1 
            ORDER BY detection_time DESC 
            LIMIT $2
            """,
            model_id, limit
        )
        return [dict(alert) for alert in alerts]

@app.post("/retrain/trigger")
async def trigger_retrain(request: RetrainRequest):
    """Manually trigger model retraining"""
    await drift_service.retrain_queue.put(request)
    return {"status": "queued"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "drift-detection"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082)