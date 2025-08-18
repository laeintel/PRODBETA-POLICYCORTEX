#!/usr/bin/env python3
"""
SHAP/Captum Explainability Sidecar Service
Provides model interpretability for compliance predictions
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    GradientShap,
    LayerConductance,
    NeuronConductance
)
import shap
import structlog
from prometheus_client import Counter, Histogram, generate_latest
import redis.asyncio as redis

# Configure logging
logger = structlog.get_logger()

# Metrics
explanation_requests = Counter('explainability_requests_total', 'Total explanation requests')
explanation_latency = Histogram('explainability_latency_seconds', 'Explanation generation latency')
explanation_errors = Counter('explainability_errors_total', 'Total explanation errors')

app = FastAPI(title="PolicyCortex Explainability Service", version="1.0.0")

# Request/Response models
class PredictionInput(BaseModel):
    model_name: str = Field(..., description="Name of the model to explain")
    input_features: Dict[str, Any] = Field(..., description="Input features for prediction")
    prediction_output: float = Field(..., description="Model's prediction output")
    explanation_method: str = Field(default="shap", description="Method: shap, integrated_gradients, deeplift")
    top_k_features: int = Field(default=10, description="Number of top features to return")

class ExplanationResponse(BaseModel):
    model_name: str
    method: str
    prediction: float
    feature_importances: Dict[str, float]
    top_features: List[Dict[str, Any]]
    global_importance: Optional[Dict[str, float]]
    confidence_intervals: Optional[Dict[str, List[float]]]
    counterfactuals: Optional[List[Dict[str, Any]]]

class ExplainabilityService:
    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.redis_client = None
        self.model_dir = os.getenv("MODEL_DIR", "/models")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        
    async def initialize(self):
        """Initialize service connections and load models"""
        # Connect to Redis for caching
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = await redis.from_url(redis_url)
        
        # Load models from shared volume
        await self.load_models()
        
        # Initialize explainers
        self.initialize_explainers()
        
        logger.info("Explainability service initialized")
    
    async def load_models(self):
        """Load models from shared volume"""
        model_files = os.listdir(self.model_dir)
        
        for model_file in model_files:
            if model_file.endswith('.pt') or model_file.endswith('.pkl'):
                model_name = model_file.split('.')[0]
                model_path = os.path.join(self.model_dir, model_file)
                
                try:
                    if model_file.endswith('.pt'):
                        # PyTorch model
                        model = torch.load(model_path, map_location='cpu')
                        model.eval()
                    else:
                        # Scikit-learn or other pickle model
                        import pickle
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                    
                    self.models[model_name] = model
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
    
    def initialize_explainers(self):
        """Initialize SHAP and Captum explainers"""
        for model_name, model in self.models.items():
            try:
                if isinstance(model, nn.Module):
                    # PyTorch model - use Captum
                    self.explainers[model_name] = {
                        'integrated_gradients': IntegratedGradients(model),
                        'deeplift': DeepLift(model),
                        'gradient_shap': GradientShap(model)
                    }
                else:
                    # Scikit-learn model - use SHAP
                    self.explainers[model_name] = {
                        'shap': shap.Explainer(model)
                    }
                logger.info(f"Initialized explainers for model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize explainer for {model_name}: {e}")
    
    async def generate_shap_explanation(
        self,
        model_name: str,
        input_features: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        explainer = self.explainers[model_name]['shap']
        shap_values = explainer(input_features)
        
        # Get feature importance
        feature_importance = {}
        for i, name in enumerate(feature_names):
            feature_importance[name] = float(np.abs(shap_values.values[0][i]))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'feature_importances': feature_importance,
            'top_features': [
                {
                    'name': name,
                    'importance': value,
                    'contribution': float(shap_values.values[0][feature_names.index(name)])
                }
                for name, value in sorted_features[:10]
            ],
            'base_value': float(shap_values.base_values[0])
        }
    
    async def generate_captum_explanation(
        self,
        model_name: str,
        input_tensor: torch.Tensor,
        method: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate Captum explanations for PyTorch models"""
        model = self.models[model_name]
        explainer = self.explainers[model_name][method]
        
        # Generate attributions
        attributions = explainer.attribute(input_tensor)
        
        # Convert to feature importance
        feature_importance = {}
        attr_values = attributions.squeeze().detach().numpy()
        
        for i, name in enumerate(feature_names):
            feature_importance[name] = float(np.abs(attr_values[i]))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'feature_importances': feature_importance,
            'top_features': [
                {
                    'name': name,
                    'importance': value,
                    'attribution': float(attr_values[feature_names.index(name)])
                }
                for name, value in sorted_features[:10]
            ]
        }
    
    async def generate_counterfactuals(
        self,
        model_name: str,
        input_features: Dict[str, Any],
        desired_outcome: float,
        n_counterfactuals: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate counterfactual explanations"""
        model = self.models[model_name]
        counterfactuals = []
        
        # Simple counterfactual generation (would use dice-ml in production)
        original_features = np.array(list(input_features.values())).reshape(1, -1)
        
        for i in range(n_counterfactuals):
            cf_features = original_features.copy()
            # Perturb features slightly
            perturbation = np.random.normal(0, 0.1, cf_features.shape)
            cf_features += perturbation
            
            # Get prediction for counterfactual
            if isinstance(model, nn.Module):
                cf_tensor = torch.FloatTensor(cf_features)
                with torch.no_grad():
                    cf_prediction = model(cf_tensor).item()
            else:
                cf_prediction = model.predict(cf_features)[0]
            
            # Calculate changes
            changes = {}
            feature_names = list(input_features.keys())
            for j, name in enumerate(feature_names):
                original_val = original_features[0][j]
                cf_val = cf_features[0][j]
                if abs(cf_val - original_val) > 0.01:
                    changes[name] = {
                        'from': float(original_val),
                        'to': float(cf_val),
                        'change': float(cf_val - original_val)
                    }
            
            counterfactuals.append({
                'prediction': float(cf_prediction),
                'changes': changes,
                'distance': float(np.linalg.norm(cf_features - original_features))
            })
        
        return counterfactuals
    
    async def get_global_importance(self, model_name: str) -> Dict[str, float]:
        """Get global feature importance from cache or compute"""
        cache_key = f"global_importance:{model_name}"
        
        # Check cache
        cached = await self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Compute global importance (simplified - would aggregate over dataset in production)
        model = self.models.get(model_name)
        if not model:
            return {}
        
        # For demonstration, return pre-computed values
        global_importance = {
            "policy_violations": 0.25,
            "resource_drift": 0.20,
            "cost_anomaly": 0.15,
            "security_score": 0.15,
            "compliance_history": 0.10,
            "resource_age": 0.08,
            "tag_compliance": 0.07
        }
        
        # Cache result
        await self.redis_client.set(
            cache_key,
            json.dumps(global_importance),
            ex=self.cache_ttl
        )
        
        return global_importance
    
    async def explain(self, request: PredictionInput) -> ExplanationResponse:
        """Main explanation generation method"""
        with explanation_latency.time():
            explanation_requests.inc()
            
            try:
                # Validate model exists
                if request.model_name not in self.models:
                    raise ValueError(f"Model {request.model_name} not found")
                
                # Prepare input features
                feature_names = list(request.input_features.keys())
                feature_values = np.array(list(request.input_features.values())).reshape(1, -1)
                
                # Generate explanation based on method
                if request.explanation_method == "shap":
                    explanation = await self.generate_shap_explanation(
                        request.model_name,
                        feature_values,
                        feature_names
                    )
                elif request.explanation_method in ["integrated_gradients", "deeplift", "gradient_shap"]:
                    input_tensor = torch.FloatTensor(feature_values)
                    explanation = await self.generate_captum_explanation(
                        request.model_name,
                        input_tensor,
                        request.explanation_method,
                        feature_names
                    )
                else:
                    raise ValueError(f"Unknown explanation method: {request.explanation_method}")
                
                # Get global importance
                global_importance = await self.get_global_importance(request.model_name)
                
                # Generate counterfactuals
                counterfactuals = await self.generate_counterfactuals(
                    request.model_name,
                    request.input_features,
                    1.0 - request.prediction_output  # Flip the prediction
                )
                
                # Build response
                return ExplanationResponse(
                    model_name=request.model_name,
                    method=request.explanation_method,
                    prediction=request.prediction_output,
                    feature_importances=explanation['feature_importances'],
                    top_features=explanation['top_features'][:request.top_k_features],
                    global_importance=global_importance,
                    confidence_intervals=None,  # Would compute in production
                    counterfactuals=counterfactuals
                )
                
            except Exception as e:
                explanation_errors.inc()
                logger.error(f"Explanation generation failed: {e}")
                raise

# Global service instance
service = ExplainabilityService()

@app.on_event("startup")
async def startup_event():
    await service.initialize()

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: PredictionInput):
    """Generate explanation for a model prediction"""
    try:
        response = await service.explain(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": list(service.models.keys()),
        "explainers": {
            name: list(explainer.keys())
            for name, explainer in service.explainers.items()
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(service.models)}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)