"""
MLflow Model Registry and Versioning System
Implements model lifecycle management with staging progression
"""

import os
import json
import time
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    latency_p95_ms: float
    throughput_rps: float
    drift_score: float = 0.0
    
    def meets_production_criteria(self) -> bool:
        """Check if metrics meet production requirements"""
        return (
            self.accuracy >= 0.92 and  # Patent #4 requirement: 99.2% target, 92% minimum
            self.precision >= 0.90 and
            self.recall >= 0.88 and
            self.latency_p95_ms < 100 and  # <100ms latency requirement
            self.drift_score < 0.15
        )

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    version: str
    model_type: str
    framework: str
    requirements: List[str]
    hyperparameters: Dict[str, Any]
    training_data_version: str
    feature_columns: List[str]
    target_column: str
    
class ModelRegistry:
    """
    MLflow-based model registry with automated lifecycle management
    Implements staging progression: Dev → Staging → Production
    """
    
    def __init__(self):
        self.client = MlflowClient()
        self.experiment_name = "policycortex_governance"
        self._ensure_experiment()
        
    def _ensure_experiment(self):
        """Ensure MLflow experiment exists"""
        try:
            self.experiment = self.client.get_experiment_by_name(self.experiment_name)
            if not self.experiment:
                self.experiment_id = self.client.create_experiment(
                    self.experiment_name,
                    tags={"project": "PolicyCortex", "team": "ML Platform"}
                )
            else:
                self.experiment_id = self.experiment.experiment_id
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            self.experiment_id = "0"  # Default experiment
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: ModelMetrics,
        config: ModelConfig,
        artifacts: Optional[Dict[str, Any]] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT
    ) -> str:
        """
        Register a new model version with MLflow
        
        Args:
            model: Trained model object
            model_name: Name for the model in registry
            metrics: Model performance metrics
            config: Model configuration
            artifacts: Additional artifacts to log
            stage: Initial stage for the model
            
        Returns:
            Model version string
        """
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run() as run:
            # Log metrics
            for key, value in asdict(metrics).items():
                mlflow.log_metric(key, value)
            
            # Log parameters
            for key, value in config.hyperparameters.items():
                mlflow.log_param(key, value)
            
            # Log model configuration
            mlflow.log_dict(asdict(config), "model_config.json")
            
            # Create model signature
            signature = self._create_signature(config)
            
            # Log model based on framework
            if config.framework == "pytorch":
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=model_name,
                    code_paths=["backend/services/model_server/"],
                    pip_requirements=config.requirements
                )
            elif config.framework == "sklearn":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    signature=signature,
                    registered_model_name=model_name,
                    pip_requirements=config.requirements
                )
            else:
                # Generic Python function model
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    signature=signature,
                    registered_model_name=model_name,
                    pip_requirements=config.requirements
                )
            
            # Log additional artifacts
            if artifacts:
                for name, artifact in artifacts.items():
                    if isinstance(artifact, dict):
                        mlflow.log_dict(artifact, f"{name}.json")
                    elif isinstance(artifact, pd.DataFrame):
                        mlflow.log_table(artifact, f"{name}.csv")
                    else:
                        mlflow.log_text(str(artifact), f"{name}.txt")
            
            # Log model cards (documentation)
            model_card = self._generate_model_card(model_name, metrics, config)
            mlflow.log_text(model_card, "MODEL_CARD.md")
            
            # Get the latest model version
            model_version = self._get_latest_version(model_name)
            
            # Transition to specified stage
            if stage != ModelStage.DEVELOPMENT:
                self.transition_model_stage(model_name, model_version, stage)
            
            # Add tags
            self.client.set_model_version_tag(
                model_name, model_version,
                "deployment_ready", str(metrics.meets_production_criteria())
            )
            self.client.set_model_version_tag(
                model_name, model_version,
                "registered_at", datetime.utcnow().isoformat()
            )
            
            logger.info(f"Registered model {model_name} version {model_version}")
            return model_version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True
    ) -> bool:
        """
        Transition model to a new stage with validation
        
        Args:
            model_name: Name of the model
            version: Version to transition
            stage: Target stage
            archive_existing: Whether to archive existing production models
            
        Returns:
            Success status
        """
        try:
            # Validate transition
            if not self._validate_transition(model_name, version, stage):
                logger.warning(f"Transition validation failed for {model_name}:{version}")
                return False
            
            # Archive existing production models if transitioning to production
            if stage == ModelStage.PRODUCTION and archive_existing:
                existing_prod = self.client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                for model in existing_prod:
                    self.client.transition_model_version_stage(
                        model_name,
                        model.version,
                        ModelStage.ARCHIVED.value,
                        archive_existing_versions=False
                    )
                    logger.info(f"Archived {model_name}:{model.version}")
            
            # Transition to new stage
            self.client.transition_model_version_stage(
                model_name,
                version,
                stage.value,
                archive_existing_versions=False
            )
            
            # Log transition
            self.client.set_model_version_tag(
                model_name, version,
                f"transitioned_to_{stage.value}", datetime.utcnow().isoformat()
            )
            
            logger.info(f"Transitioned {model_name}:{version} to {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Get the current production model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model or None
        """
        try:
            model_version = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            if not model_version:
                logger.warning(f"No production model found for {model_name}")
                return None
            
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded production model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load production model: {e}")
            return None
    
    def compare_models(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        try:
            # Get model versions
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)
            
            # Get runs
            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)
            
            # Compare metrics
            metrics1 = run1.data.metrics
            metrics2 = run2.data.metrics
            
            comparison = {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "metrics_comparison": {},
                "improvements": [],
                "regressions": []
            }
            
            for metric in metrics1:
                if metric in metrics2:
                    diff = metrics2[metric] - metrics1[metric]
                    pct_change = (diff / metrics1[metric]) * 100 if metrics1[metric] != 0 else 0
                    
                    comparison["metrics_comparison"][metric] = {
                        "v1": metrics1[metric],
                        "v2": metrics2[metric],
                        "difference": diff,
                        "pct_change": pct_change
                    }
                    
                    # Classify as improvement or regression
                    if metric in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "throughput_rps"]:
                        if diff > 0:
                            comparison["improvements"].append(metric)
                        elif diff < 0:
                            comparison["regressions"].append(metric)
                    elif metric in ["latency_p95_ms", "drift_score"]:
                        if diff < 0:
                            comparison["improvements"].append(metric)
                        elif diff > 0:
                            comparison["regressions"].append(metric)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return {"error": str(e)}
    
    def get_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get model lineage and history
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model versions with metadata
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            lineage = []
            for version in versions:
                run = self.client.get_run(version.run_id)
                
                lineage.append({
                    "version": version.version,
                    "stage": version.current_stage,
                    "created_at": version.creation_timestamp,
                    "updated_at": version.last_updated_timestamp,
                    "metrics": run.data.metrics,
                    "parameters": run.data.params,
                    "tags": version.tags,
                    "run_id": version.run_id
                })
            
            # Sort by version
            lineage.sort(key=lambda x: int(x["version"]))
            
            return lineage
            
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return []
    
    def _validate_transition(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage
    ) -> bool:
        """
        Validate stage transition based on business rules
        
        Args:
            model_name: Name of the model
            version: Version to transition
            target_stage: Target stage
            
        Returns:
            Validation result
        """
        try:
            model_version = self.client.get_model_version(model_name, version)
            current_stage = model_version.current_stage
            
            # Development -> Staging: Always allowed
            if current_stage == "None" and target_stage == ModelStage.STAGING:
                return True
            
            # Staging -> Production: Requires metric validation
            if current_stage == "Staging" and target_stage == ModelStage.PRODUCTION:
                run = self.client.get_run(model_version.run_id)
                metrics = run.data.metrics
                
                # Check production criteria
                required_metrics = ["accuracy", "latency_p95_ms"]
                for metric in required_metrics:
                    if metric not in metrics:
                        logger.warning(f"Missing required metric: {metric}")
                        return False
                
                if metrics.get("accuracy", 0) < 0.90:
                    logger.warning(f"Accuracy {metrics.get('accuracy')} below threshold")
                    return False
                
                if metrics.get("latency_p95_ms", 1000) > 100:
                    logger.warning(f"Latency {metrics.get('latency_p95_ms')}ms exceeds limit")
                    return False
                
                return True
            
            # Production -> Archived: Always allowed
            if current_stage == "Production" and target_stage == ModelStage.ARCHIVED:
                return True
            
            # All other transitions require manual approval
            logger.info(f"Transition from {current_stage} to {target_stage} requires approval")
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def _create_signature(self, config: ModelConfig) -> ModelSignature:
        """Create model signature for MLflow"""
        # Define input schema based on feature columns
        input_schema = Schema([
            ColSpec("double", name) for name in config.feature_columns
        ])
        
        # Define output schema
        if config.model_type == "classification":
            output_schema = Schema([
                ColSpec("long", "prediction"),
                ColSpec("double", "probability")
            ])
        else:
            output_schema = Schema([
                ColSpec("double", "prediction")
            ])
        
        return ModelSignature(inputs=input_schema, outputs=output_schema)
    
    def _generate_model_card(
        self,
        model_name: str,
        metrics: ModelMetrics,
        config: ModelConfig
    ) -> str:
        """Generate model card documentation"""
        return f"""# Model Card: {model_name}

## Model Details
- **Name**: {model_name}
- **Version**: {config.version}
- **Type**: {config.model_type}
- **Framework**: {config.framework}
- **Training Data Version**: {config.training_data_version}

## Intended Use
This model is designed for cloud governance compliance prediction and policy enforcement
within the PolicyCortex platform.

## Performance Metrics
- **Accuracy**: {metrics.accuracy:.4f}
- **Precision**: {metrics.precision:.4f}
- **Recall**: {metrics.recall:.4f}
- **F1 Score**: {metrics.f1_score:.4f}
- **AUC-ROC**: {metrics.auc_roc:.4f}
- **P95 Latency**: {metrics.latency_p95_ms:.2f}ms
- **Throughput**: {metrics.throughput_rps:.0f} RPS
- **Drift Score**: {metrics.drift_score:.4f}

## Limitations
- Model performance may degrade with significant data drift
- Requires retraining when drift score exceeds 0.15
- Latency requirements may not be met under extreme load

## Ethical Considerations
- Model decisions should be reviewed for bias across different resource types
- Compliance predictions should not replace human judgment for critical resources
- Regular audits should be conducted to ensure fairness

## Hyperparameters
{json.dumps(config.hyperparameters, indent=2)}

## Feature Columns
{', '.join(config.feature_columns)}

## Target Column
{config.target_column}

## Production Criteria Met
{metrics.meets_production_criteria()}

## Registered At
{datetime.utcnow().isoformat()}
"""
    
    def _get_latest_version(self, model_name: str) -> str:
        """Get latest version number for a model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                return str(max([int(v.version) for v in versions]))
            return "1"
        except:
            return "1"

class ModelDeploymentManager:
    """
    Manages model deployment with A/B testing and canary deployments
    """
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.deployments = {}
        
    def create_canary_deployment(
        self,
        model_name: str,
        production_version: str,
        canary_version: str,
        canary_weight: float = 0.1
    ) -> Dict[str, Any]:
        """
        Create canary deployment configuration
        
        Args:
            model_name: Name of the model
            production_version: Current production version
            canary_version: New version to test
            canary_weight: Traffic percentage for canary (0.0-1.0)
            
        Returns:
            Deployment configuration
        """
        deployment = {
            "model_name": model_name,
            "type": "canary",
            "production": {
                "version": production_version,
                "weight": 1.0 - canary_weight
            },
            "canary": {
                "version": canary_version,
                "weight": canary_weight
            },
            "created_at": datetime.utcnow().isoformat(),
            "metrics": {
                "production": {"requests": 0, "errors": 0, "latency_sum": 0},
                "canary": {"requests": 0, "errors": 0, "latency_sum": 0}
            }
        }
        
        deployment_id = hashlib.md5(
            f"{model_name}_{canary_version}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        self.deployments[deployment_id] = deployment
        
        logger.info(f"Created canary deployment {deployment_id} for {model_name}")
        return {"deployment_id": deployment_id, "config": deployment}
    
    def update_deployment_metrics(
        self,
        deployment_id: str,
        variant: str,
        latency_ms: float,
        success: bool
    ):
        """Update deployment metrics for monitoring"""
        if deployment_id not in self.deployments:
            return
        
        metrics = self.deployments[deployment_id]["metrics"][variant]
        metrics["requests"] += 1
        if not success:
            metrics["errors"] += 1
        metrics["latency_sum"] += latency_ms
    
    def evaluate_canary(
        self,
        deployment_id: str,
        min_requests: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate canary deployment performance
        
        Args:
            deployment_id: Deployment identifier
            min_requests: Minimum requests before evaluation
            
        Returns:
            Evaluation results and recommendation
        """
        if deployment_id not in self.deployments:
            return {"error": "Deployment not found"}
        
        deployment = self.deployments[deployment_id]
        prod_metrics = deployment["metrics"]["production"]
        canary_metrics = deployment["metrics"]["canary"]
        
        # Check minimum requests
        if canary_metrics["requests"] < min_requests:
            return {
                "status": "insufficient_data",
                "canary_requests": canary_metrics["requests"],
                "required_requests": min_requests
            }
        
        # Calculate metrics
        prod_error_rate = prod_metrics["errors"] / max(prod_metrics["requests"], 1)
        canary_error_rate = canary_metrics["errors"] / max(canary_metrics["requests"], 1)
        
        prod_avg_latency = prod_metrics["latency_sum"] / max(prod_metrics["requests"], 1)
        canary_avg_latency = canary_metrics["latency_sum"] / max(canary_metrics["requests"], 1)
        
        # Evaluation criteria
        promote = (
            canary_error_rate <= prod_error_rate * 1.1 and  # Error rate not 10% worse
            canary_avg_latency <= prod_avg_latency * 1.2     # Latency not 20% worse
        )
        
        return {
            "status": "ready",
            "recommendation": "promote" if promote else "rollback",
            "production": {
                "error_rate": prod_error_rate,
                "avg_latency_ms": prod_avg_latency
            },
            "canary": {
                "error_rate": canary_error_rate,
                "avg_latency_ms": canary_avg_latency
            },
            "improvement": {
                "error_rate": (prod_error_rate - canary_error_rate) / max(prod_error_rate, 0.001),
                "latency": (prod_avg_latency - canary_avg_latency) / max(prod_avg_latency, 1)
            }
        }
    
    def promote_canary(self, deployment_id: str) -> bool:
        """Promote canary to production"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        model_name = deployment["model_name"]
        canary_version = deployment["canary"]["version"]
        
        # Transition canary to production
        success = self.registry.transition_model_stage(
            model_name,
            canary_version,
            ModelStage.PRODUCTION,
            archive_existing=True
        )
        
        if success:
            deployment["status"] = "promoted"
            deployment["promoted_at"] = datetime.utcnow().isoformat()
            logger.info(f"Promoted canary {canary_version} to production")
        
        return success
    
    def rollback_canary(self, deployment_id: str) -> bool:
        """Rollback canary deployment"""
        if deployment_id not in self.deployments:
            return False
        
        deployment = self.deployments[deployment_id]
        deployment["status"] = "rolled_back"
        deployment["rolled_back_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Rolled back canary deployment {deployment_id}")
        return True

# Example usage
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # Example model registration
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    
    metrics = ModelMetrics(
        accuracy=0.95,
        precision=0.93,
        recall=0.91,
        f1_score=0.92,
        auc_roc=0.96,
        latency_p95_ms=45,
        throughput_rps=1500,
        drift_score=0.05
    )
    
    config = ModelConfig(
        name="compliance_predictor",
        version="1.0.0",
        model_type="classification",
        framework="pytorch",
        requirements=["torch>=1.9.0", "numpy>=1.19.0"],
        hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        training_data_version="2024.1",
        feature_columns=["feature_" + str(i) for i in range(10)],
        target_column="compliant"
    )
    
    # Register model
    version = registry.register_model(
        model, "compliance_predictor", metrics, config,
        stage=ModelStage.STAGING
    )
    
    print(f"Registered model version: {version}")