"""
PATENT NOTICE: ML Explainability Sidecar Service
Implements explainability for Patents #3 & #4
© 2024 PolicyCortex. All rights reserved.
"""

import asyncio
import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import shap
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    NoiseTunnel,
    FeatureAblation,
    Saliency
)
from captum.attr import visualization as viz
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog
import uvicorn
from sklearn.preprocessing import StandardScaler
import joblib

# Configure structured logging
logger = structlog.get_logger()

# Metrics
explanation_requests = Counter('explainability_requests_total', 'Total explanation requests', ['model', 'method'])
explanation_duration = Histogram('explainability_duration_seconds', 'Explanation computation time', ['method'])
cache_hits = Counter('explainability_cache_hits_total', 'Cache hits', ['model'])
cache_misses = Counter('explainability_cache_misses_total', 'Cache misses', ['model'])
active_explanations = Gauge('explainability_active_computations', 'Active explanation computations')

app = FastAPI(title="PolicyCortex Explainability Sidecar", version="1.0.0")

class ExplanationMethod(str, Enum):
    """Available explanation methods"""
    SHAP = "shap"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    DEEP_LIFT = "deep_lift"
    GRADIENT_SHAP = "gradient_shap"
    FEATURE_ABLATION = "feature_ablation"
    SALIENCY = "saliency"
    LIME = "lime"

class ModelType(str, Enum):
    """Supported model types"""
    COMPLIANCE_PREDICTOR = "compliance_predictor"
    COST_OPTIMIZER = "cost_optimizer"
    RISK_SCORER = "risk_scorer"
    ANOMALY_DETECTOR = "anomaly_detector"
    CORRELATION_ENGINE = "correlation_engine"

@dataclass
class ExplanationConfig:
    """Configuration for explainability service"""
    redis_url: str = "redis://localhost:6379"
    model_dir: str = "/models"
    cache_ttl: int = 3600  # 1 hour
    max_batch_size: int = 100
    shap_samples: int = 100
    enable_gpu: bool = False

class ExplanationRequest(BaseModel):
    """Request for model explanation"""
    model_type: ModelType
    input_data: Dict[str, Any]
    method: ExplanationMethod = ExplanationMethod.SHAP
    options: Dict[str, Any] = Field(default_factory=dict)
    return_visualization: bool = False

class ExplanationResponse(BaseModel):
    """Explanation response"""
    request_id: str
    model_type: str
    method: str
    prediction: Dict[str, Any]
    explanations: Dict[str, Any]
    feature_importance: List[Dict[str, float]]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]]
    visualization_url: Optional[str]
    metadata: Dict[str, Any]

class ModelWrapper:
    """Wrapper for ML models to provide uniform interface"""
    
    def __init__(self, model_type: ModelType, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load(self):
        """Load model and preprocessor"""
        if self.model_type in [ModelType.COMPLIANCE_PREDICTOR, ModelType.RISK_SCORER]:
            # Load PyTorch model
            self.model = torch.load(
                os.path.join(self.model_path, f"{self.model_type.value}.pt"),
                map_location=self.device
            )
            self.model.eval()
        else:
            # Load sklearn/xgboost model
            with open(os.path.join(self.model_path, f"{self.model_type.value}.pkl"), 'rb') as f:
                self.model = pickle.load(f)
        
        # Load preprocessor
        preprocessor_path = os.path.join(self.model_path, f"{self.model_type.value}_preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        
        # Load feature names
        features_path = os.path.join(self.model_path, f"{self.model_type.value}_features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if isinstance(self.model, nn.Module):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                outputs = self.model(X_tensor)
                return outputs.cpu().numpy()
        else:
            return self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else self.model.predict(X)
    
    def preprocess(self, data: Dict[str, Any]) -> np.ndarray:
        """Preprocess input data"""
        # Convert to DataFrame for easier preprocessing
        df = pd.DataFrame([data])
        
        if self.preprocessor:
            X = self.preprocessor.transform(df)
        else:
            # Simple preprocessing if no preprocessor available
            X = df.select_dtypes(include=[np.number]).values
        
        return X

class ExplainabilityService:
    """Main explainability service"""
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.redis_client = None
        self.models: Dict[ModelType, ModelWrapper] = {}
        self.shap_explainers: Dict[ModelType, Any] = {}
        
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing Explainability Service")
        
        # Connect to Redis
        self.redis_client = await redis.from_url(
            self.config.redis_url,
            decode_responses=False  # For binary data
        )
        
        # Load models
        await self._load_models()
        
        # Initialize SHAP explainers
        await self._init_shap_explainers()
        
        logger.info("Explainability Service initialized")
    
    async def _load_models(self):
        """Load all available models"""
        model_dir = Path(self.config.model_dir)
        
        for model_type in ModelType:
            model_path = model_dir / model_type.value
            if model_path.exists():
                try:
                    wrapper = ModelWrapper(model_type, str(model_dir))
                    wrapper.load()
                    self.models[model_type] = wrapper
                    logger.info(f"Loaded model: {model_type.value}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_type.value}: {e}")
    
    async def _init_shap_explainers(self):
        """Initialize SHAP explainers for loaded models"""
        for model_type, wrapper in self.models.items():
            try:
                # Create background dataset for SHAP
                # In production, load from actual data
                background_data = np.random.randn(self.config.shap_samples, len(wrapper.feature_names))
                
                if isinstance(wrapper.model, nn.Module):
                    # Deep learning model - use DeepExplainer
                    self.shap_explainers[model_type] = shap.DeepExplainer(
                        wrapper.model,
                        torch.FloatTensor(background_data).to(wrapper.device)
                    )
                else:
                    # Tree-based or linear model
                    if hasattr(wrapper.model, 'predict_proba'):
                        self.shap_explainers[model_type] = shap.Explainer(
                            wrapper.model.predict_proba,
                            background_data,
                            feature_names=wrapper.feature_names
                        )
                    else:
                        self.shap_explainers[model_type] = shap.Explainer(
                            wrapper.model,
                            background_data,
                            feature_names=wrapper.feature_names
                        )
                
                logger.info(f"Initialized SHAP explainer for {model_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize SHAP for {model_type.value}: {e}")
    
    async def get_cached_explanation(self, cache_key: str) -> Optional[Dict]:
        """Get cached explanation"""
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                cache_hits.labels(model="all").inc()
                return pickle.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        
        cache_misses.labels(model="all").inc()
        return None
    
    async def cache_explanation(self, cache_key: str, explanation: Dict):
        """Cache explanation"""
        try:
            await self.redis_client.setex(
                cache_key,
                self.config.cache_ttl,
                pickle.dumps(explanation)
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def explain_with_shap(
        self,
        model_wrapper: ModelWrapper,
        X: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        explainer = self.shap_explainers.get(model_wrapper.model_type)
        
        if not explainer:
            raise ValueError(f"No SHAP explainer for {model_wrapper.model_type}")
        
        # Calculate SHAP values
        if isinstance(model_wrapper.model, nn.Module):
            X_tensor = torch.FloatTensor(X).to(model_wrapper.device)
            shap_values = explainer.shap_values(X_tensor)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values.cpu().numpy()
        else:
            shap_values = explainer(X)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dict
        importance_dict = {
            model_wrapper.feature_names[i] if i < len(model_wrapper.feature_names) else f"feature_{i}": 
            float(importance_dict[i])
            for i in range(len(feature_importance))
        }
        
        return {
            "shap_values": shap_values.tolist(),
            "base_value": float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0,
            "feature_importance": importance_dict,
            "feature_names": model_wrapper.feature_names
        }
    
    async def explain_with_captum(
        self,
        model_wrapper: ModelWrapper,
        X: np.ndarray,
        method: ExplanationMethod,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate explanations using Captum"""
        if not isinstance(model_wrapper.model, nn.Module):
            raise ValueError("Captum methods only work with PyTorch models")
        
        X_tensor = torch.FloatTensor(X).to(model_wrapper.device)
        X_tensor.requires_grad = True
        
        # Select attribution method
        if method == ExplanationMethod.INTEGRATED_GRADIENTS:
            attributor = IntegratedGradients(model_wrapper.model)
            attributions = attributor.attribute(X_tensor, n_steps=50)
            
        elif method == ExplanationMethod.DEEP_LIFT:
            attributor = DeepLift(model_wrapper.model)
            baseline = torch.zeros_like(X_tensor)
            attributions = attributor.attribute(X_tensor, baseline)
            
        elif method == ExplanationMethod.GRADIENT_SHAP:
            attributor = GradientShap(model_wrapper.model)
            baseline_dist = torch.randn(10, X_tensor.shape[1]).to(model_wrapper.device)
            attributions = attributor.attribute(X_tensor, baseline_dist)
            
        elif method == ExplanationMethod.FEATURE_ABLATION:
            attributor = FeatureAblation(model_wrapper.model)
            attributions = attributor.attribute(X_tensor)
            
        elif method == ExplanationMethod.SALIENCY:
            attributor = Saliency(model_wrapper.model)
            attributions = attributor.attribute(X_tensor)
            
        else:
            raise ValueError(f"Unsupported Captum method: {method}")
        
        # Add noise tunnel for smoothing
        if kwargs.get("smooth", False):
            noise_tunnel = NoiseTunnel(attributor)
            attributions = noise_tunnel.attribute(
                X_tensor,
                nt_samples=10,
                nt_type='smoothgrad'
            )
        
        attributions = attributions.cpu().numpy()
        
        # Calculate feature importance
        feature_importance = np.abs(attributions).mean(axis=0)
        
        importance_dict = {
            model_wrapper.feature_names[i] if i < len(model_wrapper.feature_names) else f"feature_{i}": 
            float(feature_importance[i])
            for i in range(len(feature_importance))
        }
        
        return {
            "attributions": attributions.tolist(),
            "feature_importance": importance_dict,
            "method": method.value,
            "feature_names": model_wrapper.feature_names
        }
    
    async def generate_explanation(
        self,
        request: ExplanationRequest
    ) -> ExplanationResponse:
        """Generate model explanation"""
        request_id = f"{request.model_type.value}_{datetime.utcnow().timestamp()}"
        
        # Check cache
        cache_key = f"explanation:{request.model_type.value}:{hash(json.dumps(request.input_data, sort_keys=True))}"
        cached = await self.get_cached_explanation(cache_key)
        if cached:
            logger.info(f"Returning cached explanation for {request_id}")
            return ExplanationResponse(**cached)
        
        active_explanations.inc()
        start_time = datetime.utcnow()
        
        try:
            # Get model wrapper
            model_wrapper = self.models.get(request.model_type)
            if not model_wrapper:
                raise ValueError(f"Model {request.model_type} not loaded")
            
            # Preprocess input
            X = model_wrapper.preprocess(request.input_data)
            
            # Make prediction
            prediction = model_wrapper.predict(X)
            
            # Generate explanation based on method
            if request.method == ExplanationMethod.SHAP:
                explanation = await self.explain_with_shap(model_wrapper, X, **request.options)
            elif request.method in [
                ExplanationMethod.INTEGRATED_GRADIENTS,
                ExplanationMethod.DEEP_LIFT,
                ExplanationMethod.GRADIENT_SHAP,
                ExplanationMethod.FEATURE_ABLATION,
                ExplanationMethod.SALIENCY
            ]:
                explanation = await self.explain_with_captum(
                    model_wrapper, X, request.method, **request.options
                )
            else:
                raise ValueError(f"Unsupported explanation method: {request.method}")
            
            # Calculate confidence intervals using bootstrap
            if request.options.get("calculate_confidence", False):
                confidence_intervals = await self._calculate_confidence_intervals(
                    model_wrapper, X, n_bootstrap=100
                )
            else:
                confidence_intervals = None
            
            # Generate visualization if requested
            visualization_url = None
            if request.return_visualization:
                visualization_url = await self._generate_visualization(
                    explanation, request.model_type, request_id
                )
            
            # Create response
            response_data = {
                "request_id": request_id,
                "model_type": request.model_type.value,
                "method": request.method.value,
                "prediction": {
                    "value": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
                    "shape": prediction.shape if hasattr(prediction, 'shape') else None
                },
                "explanations": explanation,
                "feature_importance": sorted(
                    [{"feature": k, "importance": v} for k, v in explanation["feature_importance"].items()],
                    key=lambda x: abs(x["importance"]),
                    reverse=True
                )[:20],  # Top 20 features
                "confidence_intervals": confidence_intervals,
                "visualization_url": visualization_url,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                    "model_version": "1.0.0",
                    "options": request.options
                }
            }
            
            response = ExplanationResponse(**response_data)
            
            # Cache the response
            await self.cache_explanation(cache_key, response_data)
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            explanation_duration.labels(method=request.method.value).observe(duration)
            explanation_requests.labels(model=request.model_type.value, method=request.method.value).inc()
            
            logger.info(f"Generated explanation for {request_id} in {duration:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        
        finally:
            active_explanations.dec()
    
    async def _calculate_confidence_intervals(
        self,
        model_wrapper: ModelWrapper,
        X: np.ndarray,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate bootstrap confidence intervals"""
        predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx] if len(X.shape) > 1 else X.reshape(1, -1)
            
            pred = model_wrapper.predict(X_boot)
            predictions.append(pred.mean())
        
        predictions = np.array(predictions)
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, alpha/2 * 100)
        upper = np.percentile(predictions, (1 - alpha/2) * 100)
        
        return {
            "mean": float(predictions.mean()),
            "std": float(predictions.std()),
            "confidence_interval": (float(lower), float(upper)),
            "confidence_level": confidence_level
        }
    
    async def _generate_visualization(
        self,
        explanation: Dict[str, Any],
        model_type: ModelType,
        request_id: str
    ) -> str:
        """Generate and store visualization"""
        # In production, generate actual visualizations using matplotlib/plotly
        # and store in object storage, returning the URL
        
        # For now, return a placeholder URL
        visualization_path = f"/visualizations/{model_type.value}/{request_id}.html"
        
        # Store visualization metadata in Redis
        await self.redis_client.setex(
            f"visualization:{request_id}",
            3600,  # 1 hour
            json.dumps({
                "path": visualization_path,
                "explanation": explanation,
                "created_at": datetime.utcnow().isoformat()
            })
        )
        
        return visualization_path
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()


# Global service instance
service: Optional[ExplainabilityService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service
    config = ExplanationConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        model_dir=os.getenv("MODEL_DIR", "/models"),
        cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        enable_gpu=os.getenv("ENABLE_GPU", "false").lower() == "true"
    )
    service = ExplainabilityService(config)
    await service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global service
    if service:
        await service.cleanup()

@app.post("/explain", response_model=ExplanationResponse)
async def explain(
    request: ExplanationRequest,
    background_tasks: BackgroundTasks
):
    """Generate model explanation"""
    if not service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await service.generate_explanation(request)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(service.models.keys()) if service else []
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

if __name__ == "__main__":
    uvicorn.run(
        "explainability_sidecar:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )