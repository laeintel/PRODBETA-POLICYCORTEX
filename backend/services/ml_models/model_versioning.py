"""
Model Versioning and Rollback System
Manages model versions, deployment, and rollback capabilities
Author: PolicyCortex ML Team
Date: January 2025
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import mlflow
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    tenant_id: str
    model_type: str
    status: ModelStatus
    metrics: Dict[str, float]
    created_at: datetime
    deployed_at: Optional[datetime]
    retired_at: Optional[datetime]
    parent_version: Optional[str]
    changelog: str
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict:
        return {
            'model_id': self.model_id,
            'version': self.version,
            'tenant_id': self.tenant_id,
            'model_type': self.model_type,
            'status': self.status.value,
            'metrics': self.metrics,
            'created_at': self.created_at.isoformat(),
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'retired_at': self.retired_at.isoformat() if self.retired_at else None,
            'parent_version': self.parent_version,
            'changelog': self.changelog,
            'tags': self.tags
        }


class ModelVersionManager:
    """Manages model versions and deployments"""
    
    def __init__(self, database_url: str, model_storage_path: str = "/models"):
        self.database_url = database_url
        self.model_storage_path = model_storage_path
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # MLflow setup
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Version comparison thresholds
        self.promotion_thresholds = {
            'accuracy': 0.99,  # Minimum accuracy for production
            'false_positive_rate': 0.02,  # Maximum FPR
            'inference_time_ms': 100  # Maximum latency
        }
        
        # Rollback conditions
        self.rollback_thresholds = {
            'error_rate': 0.05,  # 5% error rate triggers rollback
            'latency_spike': 2.0,  # 2x latency spike
            'accuracy_drop': 0.05  # 5% accuracy drop
        }
    
    def create_version(self, model: Any, tenant_id: str, model_type: str,
                      metrics: Dict[str, float], parent_version: Optional[str] = None,
                      changelog: str = "", tags: Optional[Dict[str, str]] = None) -> ModelVersion:
        """Create a new model version"""
        
        # Generate version number
        version = self._generate_version_number(tenant_id, model_type, parent_version)
        
        # Generate model ID
        model_id = f"{model_type}_{tenant_id}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create version object
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            tenant_id=tenant_id,
            model_type=model_type,
            status=ModelStatus.TRAINING,
            metrics=metrics,
            created_at=datetime.now(),
            deployed_at=None,
            retired_at=None,
            parent_version=parent_version,
            changelog=changelog,
            tags=tags or {}
        )
        
        # Save model to storage
        self._save_model_to_storage(model, model_version)
        
        # Save metadata to database
        self._save_version_metadata(model_version)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params({
                'model_id': model_id,
                'version': version,
                'tenant_id': tenant_id,
                'model_type': model_type
            })
            mlflow.log_metrics(metrics)
            
            # Log model
            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, f"models/{model_id}")
        
        logger.info(f"Created model version {version} for tenant {tenant_id}")
        
        return model_version
    
    def _generate_version_number(self, tenant_id: str, model_type: str, 
                                parent_version: Optional[str]) -> str:
        """Generate semantic version number"""
        session = self.Session()
        
        try:
            # Get latest version for this tenant and model type
            result = session.execute(
                """
                SELECT version FROM ml_models
                WHERE tenant_id = :tenant_id AND model_type = :model_type
                ORDER BY created_at DESC
                LIMIT 1
                """,
                {'tenant_id': tenant_id, 'model_type': model_type}
            ).fetchone()
            
            if result:
                latest_version = result[0]
                # Parse semantic version (e.g., "1.2.3")
                major, minor, patch = map(int, latest_version.split('.'))
                
                if parent_version:
                    # Increment patch for updates
                    return f"{major}.{minor}.{patch + 1}"
                else:
                    # Increment minor for new features
                    return f"{major}.{minor + 1}.0"
            else:
                # First version
                return "1.0.0"
                
        finally:
            session.close()
    
    def promote_to_staging(self, model_id: str) -> bool:
        """Promote model to staging environment"""
        session = self.Session()
        
        try:
            # Get model version
            model_version = self._get_model_version(model_id)
            
            if not model_version:
                logger.error(f"Model {model_id} not found")
                return False
            
            # Validate metrics
            if not self._validate_promotion_criteria(model_version.metrics):
                logger.warning(f"Model {model_id} does not meet promotion criteria")
                return False
            
            # Update status
            session.execute(
                """
                UPDATE ml_models
                SET status = :status, updated_at = :updated_at
                WHERE model_id = :model_id
                """,
                {
                    'status': ModelStatus.STAGING.value,
                    'updated_at': datetime.now(),
                    'model_id': model_id
                }
            )
            
            session.commit()
            logger.info(f"Promoted model {model_id} to staging")
            
            # Run A/B test in staging
            asyncio.create_task(self._run_staging_tests(model_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def deploy_to_production(self, model_id: str, 
                            canary_percentage: float = 0.1) -> bool:
        """Deploy model to production with canary deployment"""
        session = self.Session()
        
        try:
            # Get model version
            model_version = self._get_model_version(model_id)
            
            if not model_version:
                logger.error(f"Model {model_id} not found")
                return False
            
            if model_version.status != ModelStatus.STAGING:
                logger.error(f"Model {model_id} must be in staging before production deployment")
                return False
            
            # Get current production model
            current_prod = session.execute(
                """
                SELECT model_id FROM ml_models
                WHERE tenant_id = :tenant_id 
                AND model_type = :model_type
                AND status = :status
                """,
                {
                    'tenant_id': model_version.tenant_id,
                    'model_type': model_version.model_type,
                    'status': ModelStatus.PRODUCTION.value
                }
            ).fetchone()
            
            # Archive current production model
            if current_prod:
                self._archive_model(current_prod[0])
            
            # Deploy new model with canary
            session.execute(
                """
                UPDATE ml_models
                SET status = :status, 
                    deployed_at = :deployed_at,
                    updated_at = :updated_at,
                    parameters = jsonb_set(parameters, '{canary_percentage}', :canary)
                WHERE model_id = :model_id
                """,
                {
                    'status': ModelStatus.PRODUCTION.value,
                    'deployed_at': datetime.now(),
                    'updated_at': datetime.now(),
                    'canary': json.dumps(canary_percentage),
                    'model_id': model_id
                }
            )
            
            session.commit()
            logger.info(f"Deployed model {model_id} to production with {canary_percentage*100}% canary")
            
            # Start monitoring for rollback conditions
            asyncio.create_task(self._monitor_production_model(model_id))
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def rollback(self, current_model_id: str, target_version: Optional[str] = None) -> bool:
        """Rollback to previous model version"""
        session = self.Session()
        
        try:
            # Get current model
            current_model = self._get_model_version(current_model_id)
            
            if not current_model:
                logger.error(f"Model {current_model_id} not found")
                return False
            
            # Find target version to rollback to
            if target_version:
                # Rollback to specific version
                target_model_id = session.execute(
                    """
                    SELECT model_id FROM ml_models
                    WHERE tenant_id = :tenant_id
                    AND model_type = :model_type
                    AND version = :version
                    """,
                    {
                        'tenant_id': current_model.tenant_id,
                        'model_type': current_model.model_type,
                        'version': target_version
                    }
                ).fetchone()
                
                if not target_model_id:
                    logger.error(f"Target version {target_version} not found")
                    return False
                    
                target_model_id = target_model_id[0]
            else:
                # Rollback to previous production version
                target_model_id = current_model.parent_version
                
                if not target_model_id:
                    # Find last archived production model
                    result = session.execute(
                        """
                        SELECT model_id FROM ml_models
                        WHERE tenant_id = :tenant_id
                        AND model_type = :model_type
                        AND status = :status
                        ORDER BY retired_at DESC
                        LIMIT 1
                        """,
                        {
                            'tenant_id': current_model.tenant_id,
                            'model_type': current_model.model_type,
                            'status': ModelStatus.ARCHIVED.value
                        }
                    ).fetchone()
                    
                    if result:
                        target_model_id = result[0]
                    else:
                        logger.error("No previous version found for rollback")
                        return False
            
            # Mark current model as rolled back
            session.execute(
                """
                UPDATE ml_models
                SET status = :status, retired_at = :retired_at
                WHERE model_id = :model_id
                """,
                {
                    'status': ModelStatus.ROLLED_BACK.value,
                    'retired_at': datetime.now(),
                    'model_id': current_model_id
                }
            )
            
            # Restore target model to production
            session.execute(
                """
                UPDATE ml_models
                SET status = :status, 
                    deployed_at = :deployed_at,
                    retired_at = NULL
                WHERE model_id = :model_id
                """,
                {
                    'status': ModelStatus.PRODUCTION.value,
                    'deployed_at': datetime.now(),
                    'model_id': target_model_id
                }
            )
            
            session.commit()
            
            # Log rollback event
            self._log_rollback_event(current_model_id, target_model_id)
            
            logger.info(f"Rolled back from {current_model_id} to {target_model_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def _validate_promotion_criteria(self, metrics: Dict[str, float]) -> bool:
        """Validate if model meets promotion criteria"""
        
        # Check accuracy
        if metrics.get('accuracy', 0) < self.promotion_thresholds['accuracy']:
            logger.warning(f"Accuracy {metrics.get('accuracy')} below threshold")
            return False
        
        # Check false positive rate
        if metrics.get('false_positive_rate', 1) > self.promotion_thresholds['false_positive_rate']:
            logger.warning(f"FPR {metrics.get('false_positive_rate')} above threshold")
            return False
        
        # Check inference latency
        if metrics.get('inference_time_ms', 1000) > self.promotion_thresholds['inference_time_ms']:
            logger.warning(f"Latency {metrics.get('inference_time_ms')}ms above threshold")
            return False
        
        return True
    
    async def _monitor_production_model(self, model_id: str):
        """Monitor production model for rollback conditions"""
        monitoring_duration = 3600  # Monitor for 1 hour
        check_interval = 60  # Check every minute
        
        start_time = datetime.now()
        baseline_metrics = self._get_model_metrics(model_id)
        
        while (datetime.now() - start_time).total_seconds() < monitoring_duration:
            await asyncio.sleep(check_interval)
            
            # Get current metrics
            current_metrics = self._get_real_time_metrics(model_id)
            
            # Check for rollback conditions
            if self._should_rollback(baseline_metrics, current_metrics):
                logger.warning(f"Rollback conditions met for model {model_id}")
                self.rollback(model_id)
                break
            
            # Gradually increase canary percentage if performing well
            if self._is_performing_well(current_metrics):
                self._increase_canary_percentage(model_id)
    
    def _should_rollback(self, baseline: Dict, current: Dict) -> bool:
        """Check if rollback conditions are met"""
        
        # Check error rate
        if current.get('error_rate', 0) > self.rollback_thresholds['error_rate']:
            return True
        
        # Check latency spike
        baseline_latency = baseline.get('inference_time_ms', 100)
        current_latency = current.get('inference_time_ms', 100)
        if current_latency > baseline_latency * self.rollback_thresholds['latency_spike']:
            return True
        
        # Check accuracy drop
        baseline_accuracy = baseline.get('accuracy', 0.99)
        current_accuracy = current.get('accuracy', 0.99)
        if baseline_accuracy - current_accuracy > self.rollback_thresholds['accuracy_drop']:
            return True
        
        return False
    
    def _get_model_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get model version from database"""
        session = self.Session()
        
        try:
            result = session.execute(
                """
                SELECT model_id, version, tenant_id, model_type, status,
                       metrics, created_at, deployed_at, retired_at,
                       parameters
                FROM ml_models
                WHERE model_id = :model_id
                """,
                {'model_id': model_id}
            ).fetchone()
            
            if result:
                return ModelVersion(
                    model_id=result[0],
                    version=result[1],
                    tenant_id=result[2],
                    model_type=result[3],
                    status=ModelStatus(result[4]),
                    metrics=result[5] or {},
                    created_at=result[6],
                    deployed_at=result[7],
                    retired_at=result[8],
                    parent_version=None,
                    changelog="",
                    tags={}
                )
            return None
            
        finally:
            session.close()
    
    def _save_model_to_storage(self, model: Any, version: ModelVersion):
        """Save model to file storage"""
        model_path = os.path.join(
            self.model_storage_path,
            version.tenant_id,
            version.model_type,
            version.version
        )
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save model file
        model_file = os.path.join(model_path, "model.pkl")
        
        if isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), model_file)
        else:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata_file = os.path.join(model_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        logger.info(f"Saved model to {model_path}")
    
    def _save_version_metadata(self, version: ModelVersion):
        """Save version metadata to database"""
        session = self.Session()
        
        try:
            session.execute(
                """
                INSERT INTO ml_models
                (model_id, tenant_id, model_name, model_type, version,
                 parameters, metrics, status, created_at)
                VALUES
                (:model_id, :tenant_id, :model_name, :model_type, :version,
                 :parameters, :metrics, :status, :created_at)
                """,
                {
                    'model_id': version.model_id,
                    'tenant_id': version.tenant_id,
                    'model_name': f"{version.model_type}_v{version.version}",
                    'model_type': version.model_type,
                    'version': version.version,
                    'parameters': json.dumps({'tags': version.tags}),
                    'metrics': json.dumps(version.metrics),
                    'status': version.status.value,
                    'created_at': version.created_at
                }
            )
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error saving version metadata: {e}")
            session.rollback()
            raise
        finally:
            session.close()
    
    def list_versions(self, tenant_id: str, model_type: Optional[str] = None,
                     status: Optional[ModelStatus] = None) -> List[ModelVersion]:
        """List model versions with optional filters"""
        session = self.Session()
        
        try:
            query = """
                SELECT model_id, version, tenant_id, model_type, status,
                       metrics, created_at, deployed_at, retired_at
                FROM ml_models
                WHERE tenant_id = :tenant_id
            """
            
            params = {'tenant_id': tenant_id}
            
            if model_type:
                query += " AND model_type = :model_type"
                params['model_type'] = model_type
            
            if status:
                query += " AND status = :status"
                params['status'] = status.value
            
            query += " ORDER BY created_at DESC"
            
            results = session.execute(query, params).fetchall()
            
            versions = []
            for result in results:
                versions.append(ModelVersion(
                    model_id=result[0],
                    version=result[1],
                    tenant_id=result[2],
                    model_type=result[3],
                    status=ModelStatus(result[4]),
                    metrics=result[5] or {},
                    created_at=result[6],
                    deployed_at=result[7],
                    retired_at=result[8],
                    parent_version=None,
                    changelog="",
                    tags={}
                ))
            
            return versions
            
        finally:
            session.close()
    
    def compare_versions(self, version1_id: str, version2_id: str) -> Dict[str, Any]:
        """Compare two model versions"""
        v1 = self._get_model_version(version1_id)
        v2 = self._get_model_version(version2_id)
        
        if not v1 or not v2:
            return {"error": "One or both versions not found"}
        
        comparison = {
            'version1': v1.to_dict(),
            'version2': v2.to_dict(),
            'metrics_diff': {},
            'performance_improvement': {}
        }
        
        # Compare metrics
        for metric in set(v1.metrics.keys()) | set(v2.metrics.keys()):
            v1_val = v1.metrics.get(metric, 0)
            v2_val = v2.metrics.get(metric, 0)
            
            comparison['metrics_diff'][metric] = {
                'v1': v1_val,
                'v2': v2_val,
                'diff': v2_val - v1_val,
                'improvement_pct': ((v2_val - v1_val) / v1_val * 100) if v1_val != 0 else 0
            }
        
        # Overall performance assessment
        comparison['performance_improvement']['accuracy'] = (
            comparison['metrics_diff'].get('accuracy', {}).get('improvement_pct', 0) > 0
        )
        comparison['performance_improvement']['latency'] = (
            comparison['metrics_diff'].get('inference_time_ms', {}).get('improvement_pct', 0) < 0
        )
        
        return comparison