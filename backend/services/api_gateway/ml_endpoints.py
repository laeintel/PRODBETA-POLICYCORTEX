"""
ML Endpoints for PolicyCortex API Gateway
Provides real ML predictions using the simple ML service
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import the simple ML service
try:
    from ..ai_engine.simple_ml_service import simple_ml_service
    ML_SERVICE_AVAILABLE = True
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    try:
        from backend.services.ai_engine.simple_ml_service import simple_ml_service
        ML_SERVICE_AVAILABLE = True
    except ImportError:
        ML_SERVICE_AVAILABLE = False
        simple_ml_service = None

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["Machine Learning"])

# Pydantic models for requests and responses

class ResourceData(BaseModel):
    """Resource data for compliance prediction"""
    id: str = Field(..., description="Resource identifier")
    type: str = Field("VM", description="Resource type")
    encryption_enabled: bool = Field(False, description="Is encryption enabled")
    backup_enabled: bool = Field(False, description="Are backups enabled")
    monitoring_enabled: bool = Field(False, description="Is monitoring enabled")
    public_access: bool = Field(False, description="Is public access allowed")
    tags: Dict[str, str] = Field(default_factory=dict, description="Resource tags")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Resource configuration")
    age_days: int = Field(30, description="Resource age in days")
    modifications_last_30_days: int = Field(0, description="Number of modifications in last 30 days")

class MetricData(BaseModel):
    """Metric data point for anomaly detection"""
    timestamp: str = Field(..., description="ISO format timestamp")
    value: float = Field(..., description="Metric value")
    resource_count: Optional[int] = Field(1, description="Number of resources")
    alert_count: Optional[int] = Field(0, description="Number of alerts")

class UsageData(BaseModel):
    """Usage data for cost optimization"""
    cpu_utilization: float = Field(50, description="CPU utilization percentage")
    memory_utilization: float = Field(50, description="Memory utilization percentage")
    storage_utilization: float = Field(50, description="Storage utilization percentage")
    network_utilization: float = Field(50, description="Network utilization percentage")
    monthly_cost: float = Field(..., description="Current monthly cost")
    compute_cost: Optional[float] = Field(0, description="Compute cost component")
    storage_cost: Optional[float] = Field(0, description="Storage cost component")
    network_cost: Optional[float] = Field(0, description="Network cost component")
    instance_count: int = Field(1, description="Number of instances")
    average_instance_age_days: int = Field(30, description="Average instance age")
    reserved_instances: bool = Field(False, description="Using reserved instances")
    spot_instances: bool = Field(False, description="Using spot instances")

class CompliancePredictionResponse(BaseModel):
    """Response for compliance prediction"""
    resource_id: str
    status: str
    confidence: float
    risk_level: str
    recommendations: List[str]
    predicted_at: str
    model_version: str

class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection"""
    anomalies_detected: int
    total_points: int
    anomaly_rate: float
    anomalies: List[Dict[str, Any]]
    summary: str
    analyzed_at: str

class CostOptimizationResponse(BaseModel):
    """Response for cost optimization"""
    current_monthly_cost: float
    predicted_monthly_cost: float
    estimated_savings: float
    savings_percentage: float
    recommendations: List[Dict[str, Any]]
    confidence: float
    analyzed_at: str

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""
    resources: List[ResourceData]

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""
    total_resources: int
    compliant: int
    non_compliant: int
    needs_review: int
    predictions: List[CompliancePredictionResponse]

# Endpoints

@router.get("/health")
async def ml_health_check():
    """Check ML service health"""
    return {
        "status": "healthy" if ML_SERVICE_AVAILABLE else "degraded",
        "ml_service_available": ML_SERVICE_AVAILABLE,
        "models_loaded": len(simple_ml_service.models) if ML_SERVICE_AVAILABLE else 0,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/predict/compliance", response_model=CompliancePredictionResponse)
async def predict_compliance(resource: ResourceData):
    """
    Predict compliance status for a resource
    
    Uses machine learning to analyze resource configuration and predict compliance status.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Convert Pydantic model to dict
        resource_dict = resource.dict()
        
        # Get prediction
        result = simple_ml_service.predict_compliance(resource_dict)
        
        return CompliancePredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Compliance prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/compliance/batch", response_model=BatchPredictionResponse)
async def predict_compliance_batch(request: BatchPredictionRequest):
    """
    Predict compliance for multiple resources
    
    Batch processing for efficiency when analyzing multiple resources.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        predictions = []
        status_counts = {"Compliant": 0, "Non-Compliant": 0, "Needs Review": 0}
        
        for resource in request.resources:
            resource_dict = resource.dict()
            result = simple_ml_service.predict_compliance(resource_dict)
            predictions.append(CompliancePredictionResponse(**result))
            
            # Count statuses
            status = result.get("status", "Unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        return BatchPredictionResponse(
            total_resources=len(request.resources),
            compliant=status_counts["Compliant"],
            non_compliant=status_counts["Non-Compliant"],
            needs_review=status_counts["Needs Review"],
            predictions=predictions
        )
    
    except Exception as e:
        logger.error(f"Batch compliance prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(metrics: List[MetricData]):
    """
    Detect anomalies in metrics data
    
    Analyzes time series data to identify unusual patterns or outliers.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Convert to list of dicts
        metrics_list = [m.dict() for m in metrics]
        
        # Detect anomalies
        result = simple_ml_service.detect_anomalies(metrics_list)
        
        return AnomalyDetectionResponse(**result)
    
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize/costs", response_model=CostOptimizationResponse)
async def optimize_costs(usage: UsageData):
    """
    Generate cost optimization recommendations
    
    Analyzes resource usage patterns to identify cost-saving opportunities.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Convert to dict
        usage_dict = usage.dict()
        
        # Get optimization recommendations
        result = simple_ml_service.optimize_costs(usage_dict)
        
        return CostOptimizationResponse(**result)
    
    except Exception as e:
        logger.error(f"Cost optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/info")
async def get_models_info():
    """
    Get information about loaded ML models
    
    Returns metadata about the currently loaded models including accuracy metrics.
    """
    if not ML_SERVICE_AVAILABLE:
        return {"error": "ML service not available"}
    
    models_info = {}
    
    for model_name in simple_ml_service.models.keys():
        models_info[model_name] = {
            "loaded": True,
            "metadata": simple_ml_service.model_metadata.get(model_name, {})
        }
    
    return {
        "models": models_info,
        "total_models": len(simple_ml_service.models),
        "config": {
            "auto_train": simple_ml_service.config.auto_train,
            "model_dir": simple_ml_service.config.model_dir,
            "retrain_interval_days": simple_ml_service.config.retrain_interval_days
        }
    }

@router.post("/models/retrain/{model_name}")
async def retrain_model(model_name: str):
    """
    Trigger retraining of a specific model
    
    Forces retraining of the specified model with latest data.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    valid_models = ["compliance", "anomaly", "cost"]
    if model_name not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Must be one of: {valid_models}")
    
    try:
        # Trigger retraining based on model type
        if model_name == "compliance":
            simple_ml_service._train_compliance_model()
        elif model_name == "anomaly":
            simple_ml_service._train_anomaly_model()
        elif model_name == "cost":
            simple_ml_service._train_cost_model()
        
        return {
            "status": "success",
            "model": model_name,
            "message": f"Model {model_name} retrained successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict/risk-score/{resource_id}")
async def get_risk_score(
    resource_id: str,
    include_details: bool = Query(False, description="Include detailed analysis")
):
    """
    Get comprehensive risk score for a resource
    
    Combines multiple ML models to provide an overall risk assessment.
    """
    if not ML_SERVICE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML service not available")
    
    try:
        # Create sample resource data (in production, fetch from database)
        resource_data = {
            "id": resource_id,
            "type": "VM",
            "encryption_enabled": True,
            "backup_enabled": False,
            "monitoring_enabled": True,
            "public_access": False,
            "tags": {"Environment": "Production"},
            "configuration": {"tls_version": "1.2"},
            "age_days": 45,
            "modifications_last_30_days": 2
        }
        
        # Get compliance prediction
        compliance_result = simple_ml_service.predict_compliance(resource_data)
        
        # Calculate overall risk score
        risk_factors = []
        
        # Factor in compliance status
        if compliance_result["status"] == "Non-Compliant":
            risk_factors.append(0.8)
        elif compliance_result["status"] == "Needs Review":
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        # Factor in confidence
        risk_factors.append(1 - compliance_result["confidence"])
        
        # Calculate overall risk
        overall_risk = sum(risk_factors) / len(risk_factors)
        
        response = {
            "resource_id": resource_id,
            "overall_risk_score": overall_risk,
            "risk_level": "High" if overall_risk > 0.7 else "Medium" if overall_risk > 0.4 else "Low",
            "compliance_status": compliance_result["status"],
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
        if include_details:
            response["details"] = {
                "compliance": compliance_result,
                "risk_factors": {
                    "compliance_risk": risk_factors[0],
                    "confidence_risk": risk_factors[1]
                }
            }
        
        return response
    
    except Exception as e:
        logger.error(f"Risk score calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export router
__all__ = ["router"]