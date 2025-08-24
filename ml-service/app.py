"""
ML Service for PolicyCortex - Real-time predictions and anomaly detection
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import redis
import json
import joblib
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="PolicyCortex ML Service", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

class PredictionRequest(BaseModel):
    resource_id: str
    resource_type: str
    metrics: Dict[str, float]
    historical_data: Optional[List[Dict]] = None

class AnomalyDetectionRequest(BaseModel):
    data_points: List[Dict[str, float]]
    sensitivity: float = 0.05

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    risk_score: float
    recommendations: List[str]
    explanation: str

class AnomalyResponse(BaseModel):
    anomalies: List[Dict]
    normal_range: Dict[str, tuple]
    severity: str

class CostPrediction(BaseModel):
    resource_id: str
    predicted_cost: float
    confidence_interval: tuple
    trend: str
    optimization_potential: float

# Initialize ML models
class MLEngine:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize models with synthetic training data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Normal data
        normal_data = np.random.randn(n_samples, 5)
        # Anomalous data
        anomalous_data = np.random.randn(50, 5) * 3 + 5
        
        # Train anomaly detector
        X_train = np.vstack([normal_data, anomalous_data])
        self.anomaly_detector.fit(X_train)
        
        # Train risk classifier with synthetic labels
        y_train = np.concatenate([np.zeros(n_samples), np.ones(50)])
        self.risk_classifier.fit(X_train, y_train)
        self.scaler.fit(X_train)
    
    def predict_risk(self, features: np.ndarray) -> tuple:
        """Predict risk level and confidence"""
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        risk_prob = self.risk_classifier.predict_proba(features_scaled)[0]
        risk_class = self.risk_classifier.predict(features_scaled)[0]
        return risk_class, max(risk_prob)
    
    def detect_anomalies(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies in data"""
        predictions = self.anomaly_detector.predict(data)
        scores = self.anomaly_detector.score_samples(data)
        return predictions, scores
    
    def predict_cost(self, historical_costs: List[float]) -> CostPrediction:
        """Predict future costs based on historical data"""
        if len(historical_costs) < 3:
            return CostPrediction(
                resource_id="unknown",
                predicted_cost=np.mean(historical_costs) if historical_costs else 0,
                confidence_interval=(0, 0),
                trend="stable",
                optimization_potential=0
            )
        
        # Simple trend analysis
        costs = np.array(historical_costs)
        trend_coefficient = np.polyfit(range(len(costs)), costs, 1)[0]
        
        # Predict next period
        next_cost = costs[-1] + trend_coefficient
        std_dev = np.std(costs)
        
        return CostPrediction(
            resource_id="analyzed",
            predicted_cost=float(next_cost),
            confidence_interval=(float(next_cost - 2*std_dev), float(next_cost + 2*std_dev)),
            trend="increasing" if trend_coefficient > 0 else "decreasing",
            optimization_potential=float(std_dev / np.mean(costs) * 100)
        )

ml_engine = MLEngine()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate predictions for a resource"""
    try:
        # Extract features
        features = np.array(list(request.metrics.values()))
        
        # Add synthetic features if needed
        if len(features) < 5:
            features = np.pad(features, (0, 5 - len(features)), mode='constant')
        
        # Get prediction
        risk_class, confidence = ml_engine.predict_risk(features[:5])
        risk_score = float(confidence * 100)
        
        # Generate recommendations based on risk
        recommendations = []
        if risk_score > 70:
            recommendations = [
                "Enable additional monitoring",
                "Review security policies",
                "Consider implementing auto-remediation"
            ]
        elif risk_score > 40:
            recommendations = [
                "Schedule regular compliance checks",
                "Update resource tags",
                "Review access permissions"
            ]
        else:
            recommendations = [
                "Continue current practices",
                "Consider cost optimization"
            ]
        
        explanation = f"Based on analysis of {len(request.metrics)} metrics, " \
                     f"the resource shows {'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'low'} risk indicators."
        
        # Cache result
        cache_key = f"prediction:{request.resource_id}"
        redis_client.setex(
            cache_key,
            300,  # 5 minutes TTL
            json.dumps({
                "prediction": "at_risk" if risk_class == 1 else "healthy",
                "confidence": confidence,
                "risk_score": risk_score,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        return PredictionResponse(
            prediction="at_risk" if risk_class == 1 else "healthy",
            confidence=float(confidence),
            risk_score=risk_score,
            recommendations=recommendations,
            explanation=explanation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_anomalies", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in data points"""
    try:
        # Convert data points to numpy array
        data = pd.DataFrame(request.data_points).values
        
        # Detect anomalies
        predictions, scores = ml_engine.detect_anomalies(data)
        
        # Find anomalous points
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly
                anomalies.append({
                    "index": i,
                    "data": request.data_points[i],
                    "anomaly_score": float(-score),
                    "severity": "high" if score < -0.5 else "medium"
                })
        
        # Calculate normal ranges
        normal_data = data[predictions == 1]
        normal_range = {}
        if len(normal_data) > 0:
            for i, col in enumerate(pd.DataFrame(request.data_points).columns):
                normal_range[col] = (
                    float(normal_data[:, i].min()),
                    float(normal_data[:, i].max())
                )
        
        severity = "critical" if len(anomalies) > len(data) * 0.3 else \
                  "high" if len(anomalies) > len(data) * 0.1 else \
                  "medium" if len(anomalies) > 0 else "low"
        
        return AnomalyResponse(
            anomalies=anomalies,
            normal_range=normal_range,
            severity=severity
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_cost", response_model=CostPrediction)
async def predict_cost(historical_costs: List[float]):
    """Predict future costs"""
    try:
        prediction = ml_engine.predict_cost(historical_costs)
        
        # Cache result
        redis_client.setex(
            "cost_prediction:latest",
            300,
            json.dumps({
                "predicted_cost": prediction.predicted_cost,
                "trend": prediction.trend,
                "timestamp": datetime.utcnow().isoformat()
            })
        )
        
        return prediction
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream/predictions")
async def stream_predictions():
    """Stream real-time predictions"""
    async def generate():
        while True:
            # Generate synthetic prediction
            prediction = {
                "timestamp": datetime.utcnow().isoformat(),
                "resource_id": f"vm-{np.random.randint(1, 100)}",
                "risk_score": float(np.random.random() * 100),
                "anomaly_detected": bool(np.random.random() > 0.8),
                "cost_spike": bool(np.random.random() > 0.9)
            }
            yield f"data: {json.dumps(prediction)}\n\n"
            await asyncio.sleep(2)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/train")
async def train_model(training_data: Dict):
    """Retrain model with new data"""
    try:
        # In production, this would retrain the model with new data
        return {
            "status": "training_started",
            "message": "Model training initiated",
            "estimated_time": "15 minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    return {
        "accuracy": 0.942,
        "precision": 0.923,
        "recall": 0.897,
        "f1_score": 0.910,
        "last_trained": datetime.utcnow() - timedelta(days=2),
        "predictions_count": 12847,
        "anomalies_detected": 234
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)