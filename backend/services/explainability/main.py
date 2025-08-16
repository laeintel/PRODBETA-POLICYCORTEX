#!/usr/bin/env python3
"""
PolicyCortex Explainability Service
SHAP/LIME-based ML explainability for governance predictions
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
import shap
import lime
import lime.lime_image
import lime.lime_text
import lime.lime_tabular
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import joblib
import asyncpg
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PolicyCortex Explainability Service", version="1.0.0")

class ExplanationRequest(BaseModel):
    model_id: str
    prediction_id: str
    instance_data: Dict[str, Any]
    explanation_type: str = "shap"  # shap, lime, or both
    feature_names: Optional[List[str]] = None

class ExplanationResponse(BaseModel):
    prediction_id: str
    model_id: str
    explanation_type: str
    feature_importances: Dict[str, float]
    local_explanations: Dict[str, Any]
    global_explanations: Dict[str, Any]
    confidence_score: float
    explanation_metadata: Dict[str, Any]

class ModelCard(BaseModel):
    model_id: str
    model_name: str
    model_type: str
    training_data_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    fairness_metrics: Dict[str, float]
    bias_assessment: Dict[str, Any]
    ethical_considerations: List[str]
    use_cases: List[str]
    limitations: List[str]
    created_at: datetime
    updated_at: datetime

class ExplainabilityService:
    """Main explainability service"""
    
    def __init__(self):
        self.config = self._load_config()
        self.db_pool = None
        self.explainers = {}
        self.model_metadata = {}
        
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
            'models': {
                'storage_path': os.getenv('MODEL_STORAGE_PATH', './models'),
                'explainer_cache_size': int(os.getenv('EXPLAINER_CACHE_SIZE', 100)),
            },
            'shap': {
                'max_samples': int(os.getenv('SHAP_MAX_SAMPLES', 1000)),
                'background_samples': int(os.getenv('SHAP_BACKGROUND_SAMPLES', 100)),
            },
            'lime': {
                'num_features': int(os.getenv('LIME_NUM_FEATURES', 10)),
                'num_samples': int(os.getenv('LIME_NUM_SAMPLES', 5000)),
            }
        }
    
    async def initialize(self):
        """Initialize service"""
        logger.info("Initializing explainability service...")
        
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
        
        # Load existing explainers
        await self._load_explainers()
        
        logger.info("Explainability service initialized")
    
    async def _load_explainers(self):
        """Load pre-trained explainers"""
        try:
            models_path = self.config['models']['storage_path']
            if os.path.exists(models_path):
                for model_file in os.listdir(models_path):
                    if model_file.endswith('_explainer.pkl'):
                        model_id = model_file.replace('_explainer.pkl', '')
                        explainer_path = os.path.join(models_path, model_file)
                        
                        with open(explainer_path, 'rb') as f:
                            self.explainers[model_id] = pickle.load(f)
                        
                        logger.info(f"Loaded explainer for model: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load explainers: {e}")
    
    async def create_explainer(self, model_id: str, model_type: str, training_data: np.ndarray, 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Create explainers for a model"""
        logger.info(f"Creating explainer for model: {model_id}")
        
        explainer_info = {
            'model_id': model_id,
            'model_type': model_type,
            'feature_names': feature_names,
            'created_at': datetime.utcnow()
        }
        
        try:
            # Create SHAP explainer
            if model_type in ['tree', 'ensemble', 'xgboost', 'lightgbm']:
                # For tree-based models
                shap_explainer = shap.TreeExplainer(self._load_model(model_id))
            elif model_type in ['linear', 'logistic']:
                # For linear models
                shap_explainer = shap.LinearExplainer(self._load_model(model_id), training_data)
            else:
                # For other models (deep learning, etc.)
                background = shap.sample(training_data, self.config['shap']['background_samples'])
                shap_explainer = shap.KernelExplainer(self._predict_function(model_id), background)
            
            # Create LIME explainer
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['Non-Compliant', 'Compliant'],
                mode='classification'
            )
            
            # Store explainers
            self.explainers[model_id] = {
                'shap': shap_explainer,
                'lime': lime_explainer,
                'metadata': explainer_info
            }
            
            # Save to disk
            await self._save_explainer(model_id)
            
            # Store in database
            await self._store_explainer_metadata(explainer_info)
            
            logger.info(f"Explainer created for model: {model_id}")
            return explainer_info
            
        except Exception as e:
            logger.error(f"Failed to create explainer for {model_id}: {e}")
            raise
    
    async def explain_prediction(self, request: ExplanationRequest) -> ExplanationResponse:
        """Generate explanation for a prediction"""
        logger.info(f"Explaining prediction {request.prediction_id} for model {request.model_id}")
        
        if request.model_id not in self.explainers:
            raise HTTPException(status_code=404, f"Explainer not found for model: {request.model_id}")
        
        explainer_set = self.explainers[request.model_id]
        instance = self._prepare_instance(request.instance_data, request.feature_names)
        
        explanations = {}
        
        try:
            # Generate SHAP explanation
            if request.explanation_type in ['shap', 'both']:
                shap_explanation = await self._generate_shap_explanation(
                    explainer_set['shap'], instance, request.model_id
                )
                explanations['shap'] = shap_explanation
            
            # Generate LIME explanation
            if request.explanation_type in ['lime', 'both']:
                lime_explanation = await self._generate_lime_explanation(
                    explainer_set['lime'], instance, request.model_id, request.feature_names
                )
                explanations['lime'] = lime_explanation
            
            # Combine explanations
            feature_importances = self._combine_feature_importances(explanations)
            confidence_score = self._calculate_explanation_confidence(explanations)
            
            response = ExplanationResponse(
                prediction_id=request.prediction_id,
                model_id=request.model_id,
                explanation_type=request.explanation_type,
                feature_importances=feature_importances,
                local_explanations=explanations,
                global_explanations=await self._get_global_explanations(request.model_id),
                confidence_score=confidence_score,
                explanation_metadata={
                    'generated_at': datetime.utcnow().isoformat(),
                    'method': request.explanation_type,
                    'model_type': explainer_set['metadata']['model_type']
                }
            )
            
            # Store explanation
            await self._store_explanation(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to explain prediction {request.prediction_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_shap_explanation(self, explainer, instance: np.ndarray, model_id: str) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        try:
            shap_values = explainer.shap_values(instance.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class
            
            return {
                'shap_values': shap_values[0].tolist(),
                'expected_value': float(explainer.expected_value),
                'feature_names': self.explainers[model_id]['metadata']['feature_names'],
                'explanation_type': 'shap'
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {}
    
    async def _generate_lime_explanation(self, explainer, instance: np.ndarray, model_id: str, 
                                       feature_names: List[str]) -> Dict[str, Any]:
        """Generate LIME explanation"""
        try:
            predict_fn = self._predict_function(model_id)
            explanation = explainer.explain_instance(
                instance, 
                predict_fn,
                num_features=self.config['lime']['num_features'],
                num_samples=self.config['lime']['num_samples']
            )
            
            feature_importance = dict(explanation.as_list())
            
            return {
                'feature_importance': feature_importance,
                'score': explanation.score,
                'intercept': explanation.intercept[1],  # Positive class
                'explanation_type': 'lime'
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {}
    
    def _combine_feature_importances(self, explanations: Dict[str, Any]) -> Dict[str, float]:
        """Combine feature importances from multiple explanation methods"""
        combined = {}
        
        if 'shap' in explanations and explanations['shap']:
            shap_values = explanations['shap']['shap_values']
            feature_names = explanations['shap']['feature_names']
            
            for i, name in enumerate(feature_names):
                combined[name] = float(shap_values[i])
        
        if 'lime' in explanations and explanations['lime']:
            lime_importance = explanations['lime']['feature_importance']
            
            for feature, importance in lime_importance.items():
                if feature in combined:
                    # Average SHAP and LIME values
                    combined[feature] = (combined[feature] + importance) / 2
                else:
                    combined[feature] = importance
        
        return combined
    
    def _calculate_explanation_confidence(self, explanations: Dict[str, Any]) -> float:
        """Calculate confidence in the explanation"""
        confidences = []
        
        if 'shap' in explanations and explanations['shap']:
            # SHAP confidence based on magnitude of values
            shap_values = explanations['shap']['shap_values']
            shap_confidence = min(1.0, np.sum(np.abs(shap_values)) / 10.0)
            confidences.append(shap_confidence)
        
        if 'lime' in explanations and explanations['lime']:
            # LIME confidence based on score
            lime_confidence = min(1.0, explanations['lime']['score'])
            confidences.append(lime_confidence)
        
        return float(np.mean(confidences)) if confidences else 0.0
    
    async def _get_global_explanations(self, model_id: str) -> Dict[str, Any]:
        """Get global model explanations"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT feature_name, avg_importance, std_importance
                    FROM model_feature_importance 
                    WHERE model_id = $1
                    ORDER BY avg_importance DESC
                """
                rows = await conn.fetch(query, model_id)
                
                global_importance = {}
                for row in rows:
                    global_importance[row['feature_name']] = {
                        'average_importance': float(row['avg_importance']),
                        'std_importance': float(row['std_importance'])
                    }
                
                return {
                    'global_feature_importance': global_importance,
                    'model_id': model_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get global explanations: {e}")
            return {}
    
    def _load_model(self, model_id: str):
        """Load model for explanation"""
        model_path = os.path.join(self.config['models']['storage_path'], f"{model_id}.pkl")
        return joblib.load(model_path)
    
    def _predict_function(self, model_id: str):
        """Create prediction function for model"""
        model = self._load_model(model_id)
        
        def predict(X):
            return model.predict_proba(X)
        
        return predict
    
    def _prepare_instance(self, instance_data: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
        """Prepare instance for explanation"""
        if feature_names:
            values = [instance_data.get(name, 0) for name in feature_names]
        else:
            values = list(instance_data.values())
        
        return np.array(values, dtype=float)
    
    async def _save_explainer(self, model_id: str):
        """Save explainer to disk"""
        explainer_path = os.path.join(
            self.config['models']['storage_path'], 
            f"{model_id}_explainer.pkl"
        )
        
        os.makedirs(os.path.dirname(explainer_path), exist_ok=True)
        
        with open(explainer_path, 'wb') as f:
            pickle.dump(self.explainers[model_id], f)
    
    async def _store_explainer_metadata(self, info: Dict[str, Any]):
        """Store explainer metadata in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_explainers (model_id, model_type, feature_names, created_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (model_id) DO UPDATE SET
                    model_type = $2,
                    feature_names = $3,
                    updated_at = $4
                """,
                info['model_id'],
                info['model_type'],
                json.dumps(info['feature_names']),
                info['created_at']
            )
    
    async def _store_explanation(self, explanation: ExplanationResponse):
        """Store explanation in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO prediction_explanations 
                (prediction_id, model_id, explanation_type, feature_importances, 
                 local_explanations, global_explanations, confidence_score, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                explanation.prediction_id,
                explanation.model_id,
                explanation.explanation_type,
                json.dumps(explanation.feature_importances),
                json.dumps(explanation.local_explanations),
                json.dumps(explanation.global_explanations),
                explanation.confidence_score,
                datetime.utcnow()
            )
    
    async def generate_model_card(self, model_id: str) -> ModelCard:
        """Generate comprehensive model card"""
        logger.info(f"Generating model card for {model_id}")
        
        try:
            # Get model metadata
            async with self.db_pool.acquire() as conn:
                model_info = await conn.fetchrow(
                    "SELECT * FROM ml_models WHERE model_id = $1", model_id
                )
                
                if not model_info:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
            # Get performance metrics
            performance_metrics = await self._calculate_performance_metrics(model_id)
            
            # Get fairness metrics
            fairness_metrics = await self._calculate_fairness_metrics(model_id)
            
            # Get bias assessment
            bias_assessment = await self._assess_bias(model_id)
            
            model_card = ModelCard(
                model_id=model_id,
                model_name=model_info['name'],
                model_type=model_info['type'],
                training_data_info={
                    'dataset_size': model_info['training_samples'],
                    'feature_count': model_info['feature_count'],
                    'data_sources': json.loads(model_info['data_sources'] or '[]'),
                    'training_period': {
                        'start': model_info['training_start_date'].isoformat(),
                        'end': model_info['training_end_date'].isoformat()
                    }
                },
                performance_metrics=performance_metrics,
                fairness_metrics=fairness_metrics,
                bias_assessment=bias_assessment,
                ethical_considerations=[
                    "Model decisions may impact compliance assessments",
                    "Automated governance decisions should be auditable",
                    "Bias in training data may affect certain resource types",
                    "Model explanations should be provided for high-impact decisions"
                ],
                use_cases=[
                    "Azure resource compliance prediction",
                    "Policy violation risk assessment",
                    "Governance drift detection",
                    "Automated remediation prioritization"
                ],
                limitations=[
                    "Model performance may degrade with new Azure services",
                    "Training data may not represent all organizational contexts",
                    "Explanations are approximations and may not capture all factors",
                    "Model should be retrained periodically"
                ],
                created_at=model_info['created_at'],
                updated_at=datetime.utcnow()
            )
            
            # Store model card
            await self._store_model_card(model_card)
            
            return model_card
            
        except Exception as e:
            logger.error(f"Failed to generate model card for {model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _calculate_performance_metrics(self, model_id: str) -> Dict[str, float]:
        """Calculate model performance metrics"""
        async with self.db_pool.acquire() as conn:
            # Get recent predictions and actual outcomes
            query = """
                SELECT predicted_value, actual_value
                FROM model_predictions p
                JOIN prediction_outcomes o ON p.prediction_id = o.prediction_id
                WHERE p.model_id = $1 AND o.verified_at > NOW() - INTERVAL '30 days'
            """
            rows = await conn.fetch(query, model_id)
            
            if not rows:
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            y_pred = [row['predicted_value'] for row in rows]
            y_true = [row['actual_value'] for row in rows]
            
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                'sample_size': len(rows)
            }
    
    async def _calculate_fairness_metrics(self, model_id: str) -> Dict[str, float]:
        """Calculate fairness metrics"""
        # Placeholder implementation
        return {
            'demographic_parity': 0.95,
            'equalized_odds': 0.92,
            'calibration': 0.88
        }
    
    async def _assess_bias(self, model_id: str) -> Dict[str, Any]:
        """Assess model bias"""
        # Placeholder implementation
        return {
            'bias_detected': False,
            'affected_groups': [],
            'bias_score': 0.1,
            'mitigation_applied': True
        }
    
    async def _store_model_card(self, model_card: ModelCard):
        """Store model card in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO model_cards 
                (model_id, model_name, model_type, training_data_info, performance_metrics,
                 fairness_metrics, bias_assessment, ethical_considerations, use_cases, 
                 limitations, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (model_id) DO UPDATE SET
                    model_name = $2,
                    performance_metrics = $5,
                    fairness_metrics = $6,
                    bias_assessment = $7,
                    updated_at = $12
                """,
                model_card.model_id,
                model_card.model_name,
                model_card.model_type,
                json.dumps(model_card.training_data_info),
                json.dumps(model_card.performance_metrics),
                json.dumps(model_card.fairness_metrics),
                json.dumps(model_card.bias_assessment),
                json.dumps(model_card.ethical_considerations),
                json.dumps(model_card.use_cases),
                json.dumps(model_card.limitations),
                model_card.created_at,
                model_card.updated_at
            )

# Global service instance
explainability_service = ExplainabilityService()

@app.on_event("startup")
async def startup_event():
    await explainability_service.initialize()

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Generate explanation for a prediction"""
    return await explainability_service.explain_prediction(request)

@app.post("/models/{model_id}/explainer")
async def create_model_explainer(model_id: str, model_type: str, feature_names: List[str]):
    """Create explainer for a model"""
    # This would need training data - simplified for demo
    training_data = np.random.rand(1000, len(feature_names))
    return await explainability_service.create_explainer(model_id, model_type, training_data, feature_names)

@app.get("/models/{model_id}/card", response_model=ModelCard)
async def get_model_card(model_id: str):
    """Get model card"""
    return await explainability_service.generate_model_card(model_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "explainability"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)