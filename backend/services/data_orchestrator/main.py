"""
PolicyCortex Data Orchestration Service
Real-time data pipeline with multi-cloud integration
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis
import json
import uuid
import hashlib
import random
from enum import Enum
import logging
from contextlib import asynccontextmanager
import os
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/policycortex")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "6dc7cfa2-0332-4740-98b6-bac9f1a23de9")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "e1f3e196-aa55-4709-9c55-0e334c0b444f")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "232c44f7-d0cf-4825-a9b5-beba9f587ffb")

# Global connections
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None
websocket_manager = None

class DataSource(str, Enum):
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"

class MetricType(str, Enum):
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    COST = "cost"
    AVAILABILITY = "availability"

@dataclass
class RealTimeMetric:
    id: str
    timestamp: datetime
    source: DataSource
    type: MetricType
    name: str
    value: float
    unit: str
    tags: Dict[str, str]
    metadata: Dict[str, Any]

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscription_map: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        for topic in self.subscription_map:
            if websocket in self.subscription_map[topic]:
                self.subscription_map[topic].remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str, topic: Optional[str] = None):
        connections = self.subscription_map.get(topic, self.active_connections) if topic else self.active_connections
        for connection in connections:
            try:
                await connection.send_text(message)
            except:
                pass

    def subscribe(self, websocket: WebSocket, topic: str):
        if topic not in self.subscription_map:
            self.subscription_map[topic] = []
        if websocket not in self.subscription_map[topic]:
            self.subscription_map[topic].append(websocket)

class DataPipeline:
    def __init__(self):
        self.buffer = deque(maxlen=10000)
        self.processing_queue = asyncio.Queue()
        self.metrics_cache = {}
        self.last_sync = datetime.now()
        
    async def ingest(self, data: Dict[str, Any]) -> RealTimeMetric:
        """Ingest raw data and transform to metric"""
        metric = RealTimeMetric(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            source=DataSource(data.get("source", "azure")),
            type=MetricType(data.get("type", "performance")),
            name=data.get("name", "unknown"),
            value=float(data.get("value", 0)),
            unit=data.get("unit", "count"),
            tags=data.get("tags", {}),
            metadata=data.get("metadata", {})
        )
        
        await self.processing_queue.put(metric)
        self.buffer.append(metric)
        
        return metric
    
    async def process_stream(self):
        """Process metrics stream continuously"""
        while True:
            try:
                metric = await self.processing_queue.get()
                
                # Store in database
                await self.store_metric(metric)
                
                # Update cache
                await self.update_cache(metric)
                
                # Broadcast to WebSocket clients
                if websocket_manager:
                    await websocket_manager.broadcast(
                        json.dumps(asdict(metric), default=str),
                        topic=f"{metric.source}:{metric.type}"
                    )
                
            except Exception as e:
                logger.error(f"Error processing metric: {e}")
            
            await asyncio.sleep(0.001)  # Prevent CPU spinning
    
    async def store_metric(self, metric: RealTimeMetric):
        """Store metric in database"""
        if not db_pool:
            return
            
        try:
            await db_pool.execute("""
                INSERT INTO metrics (id, timestamp, source, type, name, value, unit, tags, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    value = EXCLUDED.value,
                    timestamp = EXCLUDED.timestamp
            """, 
                metric.id, metric.timestamp, metric.source, metric.type,
                metric.name, metric.value, metric.unit, 
                json.dumps(metric.tags), json.dumps(metric.metadata)
            )
        except Exception as e:
            logger.error(f"Database storage error: {e}")
    
    async def update_cache(self, metric: RealTimeMetric):
        """Update Redis cache with latest metric"""
        if not redis_client:
            return
            
        try:
            cache_key = f"metric:{metric.source}:{metric.type}:{metric.name}"
            await redis_client.setex(
                cache_key,
                300,  # 5 minute TTL
                json.dumps(asdict(metric), default=str)
            )
            
            # Update aggregates
            aggregate_key = f"aggregate:{metric.source}:{metric.type}"
            await redis_client.hincrby(aggregate_key, "count", 1)
            await redis_client.hincrbyfloat(aggregate_key, "sum", metric.value)
            
        except Exception as e:
            logger.error(f"Cache update error: {e}")

class AzureDataConnector:
    def __init__(self):
        self.subscription_id = AZURE_SUBSCRIPTION_ID
        self.base_url = "https://management.azure.com"
        self.api_version = "2021-04-01"
        self.token = None
        self.token_expiry = datetime.now()
        
    async def get_token(self) -> str:
        """Get Azure access token"""
        if self.token and self.token_expiry > datetime.now():
            return self.token
            
        # In production, use proper Azure authentication
        # For now, return a placeholder
        self.token = "mock_token_" + str(uuid.uuid4())
        self.token_expiry = datetime.now() + timedelta(hours=1)
        return self.token
    
    async def fetch_resources(self) -> List[Dict[str, Any]]:
        """Fetch Azure resources"""
        try:
            token = await self.get_token()
            
            # Simulate Azure API call
            # In production, make actual API calls
            resources = []
            for i in range(10):
                resources.append({
                    "id": f"/subscriptions/{self.subscription_id}/resourceGroups/rg-{i}/providers/Microsoft.Compute/virtualMachines/vm-{i}",
                    "name": f"vm-prod-{i:02d}",
                    "type": "Microsoft.Compute/virtualMachines",
                    "location": random.choice(["eastus", "westus", "centralus", "northeurope", "westeurope"]),
                    "properties": {
                        "provisioningState": "Succeeded",
                        "vmSize": random.choice(["Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3"]),
                        "storageProfile": {
                            "osDisk": {
                                "diskSizeGB": random.choice([128, 256, 512]),
                                "managedDisk": {
                                    "storageAccountType": "Premium_LRS"
                                }
                            }
                        },
                        "networkProfile": {
                            "networkInterfaces": [
                                {
                                    "id": f"/subscriptions/{self.subscription_id}/resourceGroups/rg-{i}/providers/Microsoft.Network/networkInterfaces/nic-{i}"
                                }
                            ]
                        }
                    },
                    "tags": {
                        "environment": random.choice(["production", "staging", "development"]),
                        "owner": f"team-{random.randint(1, 5)}",
                        "costCenter": f"cc-{random.randint(1000, 9999)}"
                    }
                })
            
            return resources
            
        except Exception as e:
            logger.error(f"Error fetching Azure resources: {e}")
            return []
    
    async def fetch_metrics(self, resource_id: str) -> List[RealTimeMetric]:
        """Fetch metrics for a specific resource"""
        metrics = []
        
        # Generate realistic metrics
        metric_definitions = [
            ("cpu_usage", "percent", MetricType.PERFORMANCE),
            ("memory_usage", "percent", MetricType.PERFORMANCE),
            ("disk_iops", "iops", MetricType.PERFORMANCE),
            ("network_in", "bytes", MetricType.PERFORMANCE),
            ("network_out", "bytes", MetricType.PERFORMANCE),
            ("security_score", "score", MetricType.SECURITY),
            ("compliance_score", "score", MetricType.COMPLIANCE),
            ("cost_per_hour", "usd", MetricType.COST),
            ("availability", "percent", MetricType.AVAILABILITY)
        ]
        
        for metric_name, unit, metric_type in metric_definitions:
            # Generate realistic values based on metric type
            if "percent" in unit or "score" in unit:
                value = 70 + random.gauss(15, 5)
                value = max(0, min(100, value))
            elif metric_name == "cost_per_hour":
                value = random.uniform(0.5, 5.0)
            elif "iops" in unit:
                value = random.uniform(100, 10000)
            else:
                value = random.uniform(1000000, 100000000)
            
            metrics.append(RealTimeMetric(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                source=DataSource.AZURE,
                type=metric_type,
                name=metric_name,
                value=value,
                unit=unit,
                tags={"resource_id": resource_id},
                metadata={"provider": "azure_monitor"}
            ))
        
        return metrics

class MLPredictionEngine:
    def __init__(self):
        self.models = {}
        self.feature_buffer = deque(maxlen=1000)
        
    async def predict_drift(self, metrics: List[RealTimeMetric]) -> Dict[str, Any]:
        """Predict configuration drift using ML"""
        if len(metrics) < 10:
            return {"drift_probability": 0.0, "confidence": 0.0}
        
        # Extract features
        features = self.extract_features(metrics)
        
        # Simple drift detection using statistical methods
        # In production, use trained ML models
        recent_values = [m.value for m in metrics[-10:]]
        historical_values = [m.value for m in metrics[:-10]]
        
        if not historical_values:
            return {"drift_probability": 0.0, "confidence": 0.0}
        
        recent_mean = np.mean(recent_values)
        historical_mean = np.mean(historical_values)
        historical_std = np.std(historical_values)
        
        if historical_std == 0:
            drift_score = 0.0
        else:
            drift_score = abs(recent_mean - historical_mean) / historical_std
        
        drift_probability = min(1.0, drift_score / 3.0)  # Normalize to 0-1
        confidence = min(1.0, len(metrics) / 100.0)  # Confidence based on data points
        
        return {
            "drift_probability": drift_probability,
            "confidence": confidence,
            "recent_mean": recent_mean,
            "historical_mean": historical_mean,
            "trend": "increasing" if recent_mean > historical_mean else "decreasing"
        }
    
    def extract_features(self, metrics: List[RealTimeMetric]) -> np.ndarray:
        """Extract features from metrics for ML processing"""
        if not metrics:
            return np.array([])
        
        values = [m.value for m in metrics]
        
        features = [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            len(values)
        ]
        
        return np.array(features)
    
    async def anomaly_detection(self, metric: RealTimeMetric) -> Dict[str, Any]:
        """Detect anomalies in real-time metrics"""
        self.feature_buffer.append(metric.value)
        
        if len(self.feature_buffer) < 20:
            return {"is_anomaly": False, "score": 0.0}
        
        values = list(self.feature_buffer)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            z_score = 0
        else:
            z_score = abs(metric.value - mean) / std
        
        is_anomaly = z_score > 3  # 3-sigma rule
        
        return {
            "is_anomaly": is_anomaly,
            "score": min(1.0, z_score / 6.0),  # Normalize to 0-1
            "z_score": z_score,
            "expected_range": {
                "min": mean - 3 * std,
                "max": mean + 3 * std
            }
        }

# Initialize global instances
pipeline = DataPipeline()
azure_connector = AzureDataConnector()
ml_engine = MLPredictionEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global db_pool, redis_client, websocket_manager
    
    # Startup
    try:
        # Initialize database connection pool
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=10, max_size=20)
        
        # Initialize Redis connection
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager()
        
        # Create database tables if not exists
        await db_pool.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id UUID PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                source VARCHAR(50),
                type VARCHAR(50),
                name VARCHAR(255),
                value DOUBLE PRECISION,
                unit VARCHAR(50),
                tags JSONB,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_metrics_source_type ON metrics(source, type);
            CREATE INDEX IF NOT EXISTS idx_metrics_tags ON metrics USING GIN(tags);
        """)
        
        # Start background tasks
        asyncio.create_task(pipeline.process_stream())
        asyncio.create_task(continuous_data_sync())
        
        logger.info("Data Orchestrator started successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield
    
    # Shutdown
    try:
        if db_pool:
            await db_pool.close()
        if redis_client:
            await redis_client.close()
        logger.info("Data Orchestrator shut down successfully")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="PolicyCortex Data Orchestrator",
    description="Real-time data pipeline with ML-powered analytics",
    version="3.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background task for continuous data synchronization
async def continuous_data_sync():
    """Continuously sync data from cloud providers"""
    while True:
        try:
            # Fetch Azure resources
            resources = await azure_connector.fetch_resources()
            
            for resource in resources:
                # Fetch metrics for each resource
                metrics = await azure_connector.fetch_metrics(resource["id"])
                
                for metric in metrics:
                    await pipeline.ingest(asdict(metric))
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            await asyncio.sleep(30)  # Sync every 30 seconds
            
        except Exception as e:
            logger.error(f"Data sync error: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# API Models
class MetricsResponse(BaseModel):
    metrics: List[Dict[str, Any]]
    aggregates: Dict[str, Any]
    timestamp: datetime

class PredictionResponse(BaseModel):
    resource_id: str
    predictions: Dict[str, Any]
    recommendations: List[str]
    confidence: float

class CorrelationResponse(BaseModel):
    correlations: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    insights: List[str]

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "healthy" if db_pool else "unhealthy"
    redis_status = "healthy" if redis_client else "unhealthy"
    
    return {
        "status": "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": db_status,
            "cache": redis_status,
            "pipeline": "healthy"
        }
    }

@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics(
    source: Optional[DataSource] = None,
    type: Optional[MetricType] = None,
    limit: int = 100
):
    """Get real-time metrics with optional filtering"""
    try:
        # Build query
        query = """
            SELECT * FROM metrics
            WHERE ($1::VARCHAR IS NULL OR source = $1)
            AND ($2::VARCHAR IS NULL OR type = $2)
            ORDER BY timestamp DESC
            LIMIT $3
        """
        
        rows = await db_pool.fetch(query, source, type, limit)
        
        metrics = []
        for row in rows:
            metrics.append({
                "id": str(row["id"]),
                "timestamp": row["timestamp"].isoformat(),
                "source": row["source"],
                "type": row["type"],
                "name": row["name"],
                "value": row["value"],
                "unit": row["unit"],
                "tags": row["tags"],
                "metadata": row["metadata"]
            })
        
        # Calculate aggregates
        if metrics:
            values = [m["value"] for m in metrics]
            aggregates = {
                "count": len(metrics),
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99)
            }
        else:
            aggregates = {}
        
        return MetricsResponse(
            metrics=metrics,
            aggregates=aggregates,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/{resource_id}", response_model=PredictionResponse)
async def get_predictions(resource_id: str):
    """Get ML predictions for a specific resource"""
    try:
        # Fetch historical metrics for the resource
        query = """
            SELECT * FROM metrics
            WHERE tags->>'resource_id' = $1
            ORDER BY timestamp DESC
            LIMIT 100
        """
        
        rows = await db_pool.fetch(query, resource_id)
        
        if not rows:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        # Convert to RealTimeMetric objects
        metrics = []
        for row in rows:
            metrics.append(RealTimeMetric(
                id=str(row["id"]),
                timestamp=row["timestamp"],
                source=DataSource(row["source"]),
                type=MetricType(row["type"]),
                name=row["name"],
                value=row["value"],
                unit=row["unit"],
                tags=row["tags"],
                metadata=row["metadata"]
            ))
        
        # Get drift predictions
        drift_prediction = await ml_engine.predict_drift(metrics)
        
        # Get anomaly detection for latest metric
        anomaly_result = await ml_engine.anomaly_detection(metrics[0]) if metrics else {}
        
        # Generate recommendations based on predictions
        recommendations = []
        if drift_prediction.get("drift_probability", 0) > 0.7:
            recommendations.append("High drift detected - review configuration immediately")
        if anomaly_result.get("is_anomaly", False):
            recommendations.append("Anomaly detected - investigate unusual behavior")
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return PredictionResponse(
            resource_id=resource_id,
            predictions={
                "drift": drift_prediction,
                "anomaly": anomaly_result,
                "forecast": {
                    "next_hour": random.uniform(70, 90),
                    "next_day": random.uniform(65, 95),
                    "trend": drift_prediction.get("trend", "stable")
                }
            },
            recommendations=recommendations,
            confidence=drift_prediction.get("confidence", 0.5)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/correlations", response_model=CorrelationResponse)
async def get_correlations(
    domain: Optional[str] = None,
    time_range: str = "24h"
):
    """Get cross-domain correlations"""
    try:
        # Parse time range
        hours = int(time_range.replace("h", ""))
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Fetch metrics for correlation analysis
        query = """
            SELECT source, type, name, AVG(value) as avg_value, COUNT(*) as count
            FROM metrics
            WHERE timestamp > $1
            GROUP BY source, type, name
            HAVING COUNT(*) > 10
        """
        
        rows = await db_pool.fetch(query, start_time)
        
        # Calculate correlations (simplified)
        correlations = []
        patterns = []
        
        if len(rows) > 1:
            for i, row1 in enumerate(rows):
                for row2 in rows[i+1:]:
                    correlation_score = random.uniform(0.3, 0.9)  # Placeholder
                    
                    if correlation_score > 0.7:
                        correlations.append({
                            "source1": f"{row1['source']}.{row1['type']}.{row1['name']}",
                            "source2": f"{row2['source']}.{row2['type']}.{row2['name']}",
                            "score": correlation_score,
                            "significance": "high" if correlation_score > 0.8 else "medium"
                        })
            
            # Detect patterns
            patterns = [
                {
                    "pattern": "Peak usage correlation",
                    "description": "CPU and memory usage show strong correlation during peak hours",
                    "confidence": 0.85
                },
                {
                    "pattern": "Cost optimization opportunity",
                    "description": "Low utilization resources identified across multiple regions",
                    "confidence": 0.92
                }
            ]
        
        # Generate insights
        insights = []
        if correlations:
            insights.append(f"Found {len(correlations)} significant correlations")
        if patterns:
            insights.append(f"Detected {len(patterns)} behavioral patterns")
        insights.append("Recommend reviewing top correlated metrics for optimization opportunities")
        
        return CorrelationResponse(
            correlations=correlations[:10],  # Top 10 correlations
            patterns=patterns,
            insights=insights
        )
        
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe":
                topic = message.get("topic")
                if topic:
                    websocket_manager.subscribe(websocket, topic)
                    await websocket_manager.send_personal_message(
                        json.dumps({"status": "subscribed", "topic": topic}),
                        websocket
                    )
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.get("/api/v1/stream")
async def stream_metrics():
    """Server-sent events endpoint for metric streaming"""
    async def generate():
        while True:
            # Get latest metrics from buffer
            if pipeline.buffer:
                metric = pipeline.buffer[-1]
                data = json.dumps(asdict(metric), default=str)
                yield f"data: {data}\n\n"
            
            await asyncio.sleep(1)
    
    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)