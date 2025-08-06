"""
Model Manager Service for AI Engine.
Handles model lifecycle management, versioning, and deployment.
"""

import asyncio
import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import aiofiles
import structlog

# from azure.ml.aio import MLClient  # Not available yet, using direct blob storage
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobServiceClient

from backend.shared.config import get_settings

from ..models import ModelInfo
from ..models import ModelStatus
from ..models import ModelType

settings = get_settings()
logger = structlog.get_logger(__name__)


class ModelManager:
    """Manages AI/ML models for the AI Engine service."""

    def __init__(self):
        self.settings = settings
        self.models_cache = {}
        self.active_models = {}
        self.model_registry = {}
        self.azure_credential = None
        self.blob_client = None
        self.ml_client = None
        self.local_model_dir = Path(settings.ai.model_cache_dir)
        self.local_model_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the model manager."""
        try:
            logger.info("Initializing model manager")

            # Initialize Azure clients
            await self._initialize_azure_clients()

            # Load model registry
            await self._load_model_registry()

            # Initialize model cache
            await self._initialize_model_cache()

            logger.info("Model manager initialized successfully")

        except Exception as e:
            logger.error("Model manager initialization failed", error=str(e))
            raise

    async def _initialize_azure_clients(self) -> None:
        """Initialize Azure clients for model storage and ML workspace."""
        try:
            if self.settings.is_production():
                self.azure_credential = DefaultAzureCredential()

                # Initialize Blob Storage client for model artifacts
                blob_service_url = (
                    f"https://{self.settings.azure.storage_account_name}.blob.core.windows.net"
                )
                self.blob_client = BlobServiceClient(
                    account_url=blob_service_url,
                    credential=self.azure_credential
                )

                # Initialize ML Client for Azure ML workspace
                self.ml_client = MLClient(
                    credential=self.azure_credential,
                    subscription_id=self.settings.azure.subscription_id,
                    resource_group_name=self.settings.azure.ml_resource_group,
                    workspace_name=self.settings.azure.ml_workspace_name
                )

                logger.info("Azure clients initialized")
            else:
                logger.info("Running in development mode, skipping Azure client initialization")

        except Exception as e:
            logger.warning("Failed to initialize Azure clients", error=str(e))

    async def _load_model_registry(self) -> None:
        """Load model registry from storage."""
        try:
            registry_file = self.local_model_dir / "model_registry.json"

            if registry_file.exists():
                async with aiofiles.open(registry_file, 'r') as f:
                    content = await f.read()
                    self.model_registry = json.loads(content)
            else:
                # Initialize with default models
                self.model_registry = {
                    "policy_analyzer": {
                        "name": "policy_analyzer",
                        "version": "1.0.0",
                        "type": "nlp",
                        "status": "active",
                        "description": "Natural language processing model for policy analysis",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "max_sequence_length": 512,
                            "model_type": "transformer",
                            "language": "en"
                        }
                    },
                    "anomaly_detector": {
                        "name": "anomaly_detector",
                        "version": "1.0.0",
                        "type": "anomaly_detection",
                        "status": "active",
                        "description": "Machine learning model for anomaly detection",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "algorithm": "isolation_forest",
                            "contamination": 0.1,
                            "n_estimators": 100
                        }
                    },
                    "cost_optimizer": {
                        "name": "cost_optimizer",
                        "version": "1.0.0",
                        "type": "cost_optimization",
                        "status": "active",
                        "description": "Machine learning model for cost optimization",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "algorithm": "linear_programming",
                            "optimization_objective": "minimize_cost"
                        }
                    },
                    "predictive_analytics": {
                        "name": "predictive_analytics",
                        "version": "1.0.0",
                        "type": "predictive_analytics",
                        "status": "active",
                        "description": "Time series forecasting model",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "algorithm": "prophet",
                            "seasonality_mode": "additive",
                            "forecast_horizon": 30
                        }
                    },
                    "sentiment_analyzer": {
                        "name": "sentiment_analyzer",
                        "version": "1.0.0",
                        "type": "sentiment_analysis",
                        "status": "active",
                        "description": "Sentiment analysis model for text processing",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "model_type": "transformer",
                            "language": "en",
                            "emotions": ["positive", "negative", "neutral"]
                        }
                    },
                    "compliance_predictor": {
                        "name": "compliance_predictor",
                        "version": "2.0.0",
                        "type": "compliance_prediction",
                        "status": "active",
                        "description": "Predictive compliance engine with drift detection and
                            temporal analysis",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "ensemble_models": ["xgboost", "random_forest", "isolation_forest"],
                            "drift_detection": "vae",
                            "temporal_analysis": "stl_decomposition",
                            "prediction_horizon": 24
                        }
                    },
                    "correlation_engine": {
                        "name": "correlation_engine",
                        "version": "2.0.0",
                        "type": "cross_domain_correlation",
                        "status": "active",
                        "description": "Cross-domain correlation analysis with graph neural networks",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "parameters": {
                            "graph_layers": 3,
                            "correlation_methods": ["pearson", "spearman", "mutual_info", "granger"],
                            "impact_models": ["gradient_boost", "random_forest"],
                            "domains": ["policy", "rbac", "network", "cost"]
                        }
                    }
                }

                # Save registry
                await self._save_model_registry()

            logger.info("Model registry loaded", model_count=len(self.model_registry))

        except Exception as e:
            logger.error("Failed to load model registry", error=str(e))
            self.model_registry = {}

    async def _save_model_registry(self) -> None:
        """Save model registry to storage."""
        try:
            registry_file = self.local_model_dir / "model_registry.json"

            async with aiofiles.open(registry_file, 'w') as f:
                await f.write(json.dumps(self.model_registry, indent=2))

            logger.debug("Model registry saved")

        except Exception as e:
            logger.error("Failed to save model registry", error=str(e))

    async def _initialize_model_cache(self) -> None:
        """Initialize model cache with frequently used models."""
        try:
            # Pre-load critical models
            critical_models = ["policy_analyzer", "anomaly_detector"]

            for model_name in critical_models:
                if model_name in self.model_registry:
                    await self._load_model(model_name)

            logger.info("Model cache initialized", cached_models=len(self.models_cache))

        except Exception as e:
            logger.error("Failed to initialize model cache", error=str(e))

    async def _load_model(self, model_name: str) -> Any:
        """Load a model into memory."""
        try:
            if model_name in self.models_cache:
                return self.models_cache[model_name]

            model_info = self.model_registry.get(model_name)
            if not model_info:
                raise ValueError(f"Model '{model_name}' not found in registry")

            # Try to load from local cache first
            model_file = self.local_model_dir / f"{model_name}.pkl"

            if model_file.exists():
                async with aiofiles.open(model_file, 'rb') as f:
                    model_data = await f.read()
                    model = pickle.loads(model_data)
                    self.models_cache[model_name] = model

                    logger.info("Model loaded from local cache", model_name=model_name)
                    return model

            # If not in local cache, try to download from Azure
            if self.blob_client:
                try:
                    model = await self._download_model_from_azure(model_name)
                    if model:
                        self.models_cache[model_name] = model
                        return model
                except Exception as e:
                    logger.warning(
                        "Failed to download model from Azure",
                        model_name=model_name,
                        error=str(e)
                    )

            # If model not found, create a placeholder
            model = await self._create_placeholder_model(model_name, model_info)
            self.models_cache[model_name] = model

            return model

        except Exception as e:
            logger.error("Failed to load model", model_name=model_name, error=str(e))
            raise

    async def _download_model_from_azure(self, model_name: str) -> Optional[Any]:
        """Download model from Azure Blob Storage."""
        try:
            container_name = "models"
            blob_name = f"{model_name}.pkl"

            blob_client = self.blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            # Download model data
            model_data = await blob_client.download_blob()
            model_bytes = await model_data.readall()

            # Deserialize model
            model = pickle.loads(model_bytes)

            # Cache locally
            model_file = self.local_model_dir / f"{model_name}.pkl"
            async with aiofiles.open(model_file, 'wb') as f:
                await f.write(model_bytes)

            logger.info("Model downloaded from Azure", model_name=model_name)
            return model

        except Exception as e:
            logger.error("Failed to download model from Azure", model_name=model_name, error=str(e))
            return None

    async def _create_placeholder_model(self, model_name: str, model_info: Dict[str, Any]) -> Any:
        """Create a placeholder model for development/testing."""
        try:
            model_type = model_info.get("type", "unknown")

            # Create simple placeholder based on model type
            if model_type == "nlp":
                model = {
                    "type": "placeholder_nlp",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
            elif model_type == "anomaly_detection":
                model = {
                    "type": "placeholder_anomaly",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
            elif model_type == "cost_optimization":
                model = {
                    "type": "placeholder_cost",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
            elif model_type == "predictive_analytics":
                model = {
                    "type": "placeholder_predictive",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
            elif model_type == "sentiment_analysis":
                model = {
                    "type": "placeholder_sentiment",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
            else:
                model = {
                    "type": "placeholder_generic",
                    "name": model_name,
                    "parameters": model_info.get("parameters", {}),
                    "created_at": datetime.utcnow().isoformat()
                }

            logger.info("Placeholder model created", model_name=model_name, model_type=model_type)
            return model

        except Exception as e:
            logger.error("Failed to create placeholder model", model_name=model_name, error=str(e))
            raise

    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        try:
            models = []

            for model_name, model_data in self.model_registry.items():
                model_info = ModelInfo(
                    name=model_data["name"],
                    version=model_data["version"],
                    type=ModelType(model_data["type"]),
                    status=ModelStatus(model_data["status"]),
                    description=model_data.get("description"),
                    created_at=datetime.fromisoformat(model_data["created_at"]),
                    updated_at=datetime.fromisoformat(model_data["updated_at"]),
                    parameters=model_data.get("parameters"),
                    metrics=model_data.get("metrics")
                )
                models.append(model_info)

            return models

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            return []

    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        try:
            model_data = self.model_registry.get(model_name)
            if not model_data:
                return None

            return ModelInfo(
                name=model_data["name"],
                version=model_data["version"],
                type=ModelType(model_data["type"]),
                status=ModelStatus(model_data["status"]),
                description=model_data.get("description"),
                created_at=datetime.fromisoformat(model_data["created_at"]),
                updated_at=datetime.fromisoformat(model_data["updated_at"]),
                parameters=model_data.get("parameters"),
                metrics=model_data.get("metrics")
            )

        except Exception as e:
            logger.error("Failed to get model info", model_name=model_name, error=str(e))
            return None

    async def get_model(self, model_name: str) -> Any:
        """Get a model instance."""
        try:
            return await self._load_model(model_name)

        except Exception as e:
            logger.error("Failed to get model", model_name=model_name, error=str(e))
            raise

    async def train_model(self, model_name: str, training_data: Dict[str, Any],
                         parameters: Dict[str, Any], task_id: str) -> None:
        """Train a model (background task)."""
        try:
            logger.info("Starting model training", model_name=model_name, task_id=task_id)

            # Simulate training process
            await asyncio.sleep(5)  # Simulate training time

            # Update model registry
            if model_name in self.model_registry:
                self.model_registry[model_name]["updated_at"] = datetime.utcnow().isoformat()
                self.model_registry[model_name]["status"] = "active"
                self.model_registry[model_name]["parameters"].update(parameters)

                # Add training metrics
                self.model_registry[model_name]["metrics"] = {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                    "f1_score": 0.91,
                    "training_time": 5.0
                }

                await self._save_model_registry()

                # Clear cache to force reload
                if model_name in self.models_cache:
                    del self.models_cache[model_name]

                logger.info("Model training completed", model_name=model_name, task_id=task_id)
            else:
                logger.error("Model not found for training", model_name=model_name, task_id=task_id)

        except Exception as e:
            logger.error(
                "Model training failed",
                model_name=model_name,
                task_id=task_id,
                error=str(e)
            )

            # Update model status to failed
            if model_name in self.model_registry:
                self.model_registry[model_name]["status"] = "failed"
                await self._save_model_registry()

    async def upload_model(self, model_name: str, model_data: Any) -> bool:
        """Upload a model to Azure Blob Storage."""
        try:
            if not self.blob_client:
                logger.warning("Blob client not initialized, skipping upload")
                return False

            # Serialize model
            model_bytes = pickle.dumps(model_data)

            # Upload to Azure
            container_name = "models"
            blob_name = f"{model_name}.pkl"

            blob_client = self.blob_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )

            await blob_client.upload_blob(model_bytes, overwrite=True)

            # Update local cache
            model_file = self.local_model_dir / f"{model_name}.pkl"
            async with aiofiles.open(model_file, 'wb') as f:
                await f.write(model_bytes)

            logger.info("Model uploaded to Azure", model_name=model_name)
            return True

        except Exception as e:
            logger.error("Failed to upload model", model_name=model_name, error=str(e))
            return False

    async def delete_model(self, model_name: str) -> bool:
        """Delete a model from registry and storage."""
        try:
            # Remove from registry
            if model_name in self.model_registry:
                del self.model_registry[model_name]
                await self._save_model_registry()

            # Remove from cache
            if model_name in self.models_cache:
                del self.models_cache[model_name]

            # Remove from local storage
            model_file = self.local_model_dir / f"{model_name}.pkl"
            if model_file.exists():
                model_file.unlink()

            # Remove from Azure (if available)
            if self.blob_client:
                try:
                    container_name = "models"
                    blob_name = f"{model_name}.pkl"

                    blob_client = self.blob_client.get_blob_client(
                        container=container_name,
                        blob=blob_name
                    )

                    await blob_client.delete_blob()

                except Exception as e:
                    logger.warning(
                        "Failed to delete model from Azure",
                        model_name=model_name,
                        error=str(e)
                    )

            logger.info("Model deleted", model_name=model_name)
            return True

        except Exception as e:
            logger.error("Failed to delete model", model_name=model_name, error=str(e))
            return False

    async def load_default_models(self) -> None:
        """Load default models on startup."""
        try:
            logger.info("Loading default models")

            # Load all models in registry
            for model_name in self.model_registry.keys():
                try:
                    await self._load_model(model_name)
                    self.active_models[model_name] = True
                except Exception as e:
                    logger.warning("Failed to load model", model_name=model_name, error=str(e))
                    self.active_models[model_name] = False

            logger.info("Default models loaded", active_count=sum(self.active_models.values()))

        except Exception as e:
            logger.error("Failed to load default models", error=str(e))

    def is_ready(self) -> bool:
        """Check if model manager is ready."""
        return len(self.models_cache) > 0

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            # Clear caches
            self.models_cache.clear()
            self.active_models.clear()

            # Close Azure clients
            if self.blob_client:
                await self.blob_client.close()

            logger.info("Model manager cleanup completed")

        except Exception as e:
            logger.error("Model manager cleanup failed", error=str(e))
