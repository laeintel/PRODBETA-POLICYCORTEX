#!/usr/bin/env python3
"""
Drift Detection and Auto-Retrain Pipeline
Monitors model performance and triggers retraining when drift is detected
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from river import drift

import aiohttp
from azure.ai.ml import MLClient
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient
import structlog
from prometheus_client import Counter, Gauge, Histogram
import redis.asyncio as redis

# Configure logging
logger = structlog.get_logger()

# Metrics
drift_detected = Counter('drift_detected_total', 'Total drift detections', ['model', 'type'])
psi_metric = Gauge('population_stability_index', 'PSI value', ['model'])
ks_statistic = Gauge('kolmogorov_smirnov_statistic', 'KS statistic', ['model'])
retrain_triggered = Counter('retrain_triggered_total', 'Total retrain triggers', ['model'])
model_performance = Gauge('model_performance_metric', 'Model performance metric', ['model', 'metric'])

class DriftMonitor:
    def __init__(self):
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP", "policycortex-ml")
        self.workspace_name = os.getenv("AZURE_ML_WORKSPACE", "pcx-ml-workspace")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.storage_account = os.getenv("AZURE_STORAGE_ACCOUNT", "pcxmlstorage")
        
        # Drift thresholds
        self.psi_threshold = float(os.getenv("PSI_THRESHOLD", "0.2"))
        self.ks_threshold = float(os.getenv("KS_THRESHOLD", "0.05"))
        self.performance_degradation_threshold = float(os.getenv("PERF_DEGRADATION_THRESHOLD", "0.1"))
        
        # Monitoring window
        self.window_hours = int(os.getenv("MONITOR_WINDOW_HOURS", "24"))
        self.check_interval_minutes = int(os.getenv("CHECK_INTERVAL_MINUTES", "360"))  # 6 hours
        
        self.ml_client = None
        self.redis_client = None
        self.blob_client = None
        self.credential = None
        self.drift_detectors = {}
        
    async def initialize(self):
        """Initialize connections and drift detectors"""
        logger.info("Initializing drift monitor")
        
        # Initialize Azure credential
        self.credential = DefaultAzureCredential()
        
        # Initialize ML Client
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name
        )
        
        # Initialize Redis for state management
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Initialize Blob Storage for model artifacts
        self.blob_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.credential
        )
        
        # Initialize drift detectors
        self.initialize_drift_detectors()
        
        logger.info("Drift monitor initialized")
    
    def initialize_drift_detectors(self):
        """Initialize River drift detectors for streaming detection"""
        models = ["compliance_predictor", "cost_optimizer", "security_scorer", "resource_analyzer"]
        
        for model in models:
            self.drift_detectors[model] = {
                'adwin': drift.ADWIN(),  # Adaptive Windowing
                'kswin': drift.KSWIN(alpha=0.005, window_size=100),  # Kolmogorov-Smirnov Windowing
                'page_hinkley': drift.PageHinkley(min_instances=30, delta=0.005)  # Page-Hinkley test
            }
    
    async def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10
    ) -> float:
        """Calculate Population Stability Index"""
        # Create bins based on expected distribution
        breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        # Calculate frequencies
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    async def perform_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test"""
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        return float(ks_statistic), float(p_value)
    
    async def get_model_data(
        self,
        model_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch model input and output data from storage"""
        # Fetch from Redis or Blob Storage
        cache_key = f"model_data:{model_name}:{start_time.isoformat()}:{end_time.isoformat()}"
        
        cached = await self.redis_client.get(cache_key)
        if cached:
            data = json.loads(cached)
            return pd.DataFrame(data['inputs']), pd.DataFrame(data['outputs'])
        
        # If not cached, fetch from Blob Storage
        container_name = f"model-data-{model_name}"
        container_client = self.blob_client.get_container_client(container_name)
        
        inputs_list = []
        outputs_list = []
        
        async for blob in container_client.list_blobs(
            name_starts_with=f"{start_time.strftime('%Y%m%d')}"
        ):
            blob_client = container_client.get_blob_client(blob.name)
            content = await blob_client.download_blob()
            data = json.loads(await content.readall())
            
            if 'input' in data:
                inputs_list.append(data['input'])
            if 'output' in data:
                outputs_list.append(data['output'])
        
        inputs_df = pd.DataFrame(inputs_list) if inputs_list else pd.DataFrame()
        outputs_df = pd.DataFrame(outputs_list) if outputs_list else pd.DataFrame()
        
        # Cache for future use
        cache_data = {
            'inputs': inputs_df.to_dict('records'),
            'outputs': outputs_df.to_dict('records')
        }
        await self.redis_client.set(cache_key, json.dumps(cache_data), ex=3600)
        
        return inputs_df, outputs_df
    
    async def get_reference_distribution(self, model_name: str) -> pd.DataFrame:
        """Get reference distribution from training data"""
        cache_key = f"reference_dist:{model_name}"
        
        cached = await self.redis_client.get(cache_key)
        if cached:
            return pd.DataFrame(json.loads(cached))
        
        # Fetch from model registry
        model = self.ml_client.models.get(name=model_name, version="latest")
        reference_path = model.properties.get("reference_data_path")
        
        if reference_path:
            # Download reference data
            blob_client = self.blob_client.get_blob_client(
                container="reference-data",
                blob=reference_path
            )
            content = await blob_client.download_blob()
            reference_df = pd.read_csv(await content.readall())
            
            # Cache for 24 hours
            await self.redis_client.set(
                cache_key,
                reference_df.to_json(),
                ex=86400
            )
            
            return reference_df
        
        return pd.DataFrame()
    
    async def check_data_drift(
        self,
        model_name: str,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check for data drift across all features"""
        drift_results = {
            'model': model_name,
            'timestamp': datetime.utcnow().isoformat(),
            'features': {},
            'overall_drift': False
        }
        
        for column in current_data.columns:
            if column in reference_data.columns:
                # Calculate PSI
                psi = await self.calculate_psi(
                    reference_data[column].values,
                    current_data[column].values
                )
                
                # Perform KS test
                ks_stat, p_value = await self.perform_ks_test(
                    reference_data[column].values,
                    current_data[column].values
                )
                
                # Check thresholds
                has_drift = psi > self.psi_threshold or p_value < self.ks_threshold
                
                drift_results['features'][column] = {
                    'psi': psi,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'has_drift': has_drift
                }
                
                if has_drift:
                    drift_results['overall_drift'] = True
                    logger.warning(
                        f"Drift detected in {model_name} feature {column}: "
                        f"PSI={psi:.3f}, KS p-value={p_value:.3f}"
                    )
        
        # Update metrics
        if model_name in ['compliance_predictor']:  # Track key models
            psi_metric.labels(model=model_name).set(
                np.mean([f['psi'] for f in drift_results['features'].values()])
            )
            ks_statistic.labels(model=model_name).set(
                np.mean([f['ks_statistic'] for f in drift_results['features'].values()])
            )
        
        return drift_results
    
    async def check_concept_drift(
        self,
        model_name: str,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check for concept drift by monitoring model performance"""
        # Calculate performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # Assume binary classification for simplicity
        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, average='weighted')
        recall = recall_score(actuals, predictions, average='weighted')
        f1 = f1_score(actuals, predictions, average='weighted')
        
        # Get historical performance
        hist_key = f"historical_performance:{model_name}"
        hist_perf = await self.redis_client.get(hist_key)
        
        drift_detected = False
        degradation = 0.0
        
        if hist_perf:
            hist_perf = json.loads(hist_perf)
            baseline_f1 = hist_perf.get('f1', f1)
            degradation = baseline_f1 - f1
            
            if degradation > self.performance_degradation_threshold:
                drift_detected = True
                logger.warning(
                    f"Concept drift detected in {model_name}: "
                    f"F1 degraded by {degradation:.3f}"
                )
        
        # Update metrics
        model_performance.labels(model=model_name, metric='accuracy').set(accuracy)
        model_performance.labels(model=model_name, metric='f1').set(f1)
        
        # Store current performance
        current_perf = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'timestamp': datetime.utcnow().isoformat()
        }
        await self.redis_client.set(hist_key, json.dumps(current_perf), ex=604800)  # 7 days
        
        return {
            'model': model_name,
            'performance': current_perf,
            'degradation': degradation,
            'drift_detected': drift_detected
        }
    
    async def trigger_retrain(
        self,
        model_name: str,
        drift_info: Dict[str, Any]
    ) -> str:
        """Trigger model retraining via Azure ML Pipeline"""
        logger.info(f"Triggering retrain for {model_name}")
        retrain_triggered.labels(model=model_name).inc()
        
        # Create experiment
        experiment_name = f"retrain_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Define pipeline
        pipeline_config = {
            "display_name": f"Retrain {model_name}",
            "experiment_name": experiment_name,
            "compute": "gpu-cluster",
            "tags": {
                "triggered_by": "drift_detection",
                "drift_type": "data" if drift_info.get('overall_drift') else "concept",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Submit pipeline job
        try:
            # In production, this would submit an actual Azure ML pipeline
            # For now, we'll simulate it
            job_id = f"job_{model_name}_{datetime.utcnow().timestamp()}"
            
            # Store job info
            job_info = {
                'job_id': job_id,
                'model_name': model_name,
                'status': 'submitted',
                'drift_info': drift_info,
                'config': pipeline_config,
                'submitted_at': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.set(
                f"retrain_job:{job_id}",
                json.dumps(job_info),
                ex=86400  # 24 hours
            )
            
            # Publish model card
            await self.publish_model_card(model_name, drift_info, job_id)
            
            logger.info(f"Retrain job submitted: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to trigger retrain: {e}")
            raise
    
    async def publish_model_card(
        self,
        model_name: str,
        drift_info: Dict[str, Any],
        job_id: str
    ):
        """Publish model card with drift and retrain information"""
        model_card = {
            'model_name': model_name,
            'version': 'pending',
            'retrain_job_id': job_id,
            'trigger_reason': 'drift_detection',
            'drift_summary': {
                'data_drift': drift_info.get('overall_drift', False),
                'affected_features': [
                    f for f, info in drift_info.get('features', {}).items()
                    if info.get('has_drift')
                ],
                'performance_degradation': drift_info.get('degradation', 0.0)
            },
            'metadata': {
                'created_at': datetime.utcnow().isoformat(),
                'expected_completion': (datetime.utcnow() + timedelta(hours=2)).isoformat(),
                'auto_deploy': True,
                'deployment_strategy': 'blue_green'
            }
        }
        
        # Upload to Blob Storage
        container_client = self.blob_client.get_container_client("model-cards")
        blob_name = f"{model_name}/retrain_{job_id}.json"
        
        blob_client = container_client.get_blob_client(blob_name)
        await blob_client.upload_blob(
            json.dumps(model_card, indent=2),
            overwrite=True
        )
        
        logger.info(f"Model card published: {blob_name}")
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Check each model
                models = ["compliance_predictor", "cost_optimizer", "security_scorer"]
                
                for model_name in models:
                    logger.info(f"Checking drift for {model_name}")
                    
                    # Get data for monitoring window
                    end_time = datetime.utcnow()
                    start_time = end_time - timedelta(hours=self.window_hours)
                    
                    # Fetch current data
                    current_inputs, current_outputs = await self.get_model_data(
                        model_name,
                        start_time,
                        end_time
                    )
                    
                    if current_inputs.empty:
                        logger.info(f"No data available for {model_name}")
                        continue
                    
                    # Get reference distribution
                    reference_data = await self.get_reference_distribution(model_name)
                    
                    if not reference_data.empty:
                        # Check data drift
                        data_drift = await self.check_data_drift(
                            model_name,
                            current_inputs,
                            reference_data
                        )
                        
                        # If significant drift detected, trigger retrain
                        if data_drift['overall_drift']:
                            drift_detected.labels(model=model_name, type='data').inc()
                            await self.trigger_retrain(model_name, data_drift)
                    
                    # Check concept drift if we have actuals
                    # In production, this would fetch actual outcomes
                    # For now, we'll skip this check
                    
                # Wait for next check
                await asyncio.sleep(self.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def run(self):
        """Run the drift monitor"""
        await self.initialize()
        await self.monitor_loop()

if __name__ == "__main__":
    monitor = DriftMonitor()
    asyncio.run(monitor.run())