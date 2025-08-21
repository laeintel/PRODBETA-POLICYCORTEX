# Patent #4 ML System Testing Guide

## Quick Start Testing

### 1. One-Command Test (Windows)
```bash
# Run complete ML system test
.\scripts\test-ml-system.bat
```

### 2. One-Command Test (Linux/Mac)
```bash
# Run complete ML system test
./scripts/test-ml-system.sh
```

## Manual Testing Steps

### Step 1: Setup Environment

#### Windows:
```bash
# Set environment variables
set DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
set REDIS_URL=redis://localhost:6379
set AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
set AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
```

#### Linux/Mac:
```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/policycortex
export REDIS_URL=redis://localhost:6379
export AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
export AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
```

### Step 2: Install Dependencies
```bash
# Install Python ML dependencies
pip install torch numpy pandas scikit-learn xgboost lightgbm prophet
pip install websockets aioredis pytest pytest-asyncio
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis
```

### Step 3: Create Database Tables
```bash
# Create ML-specific tables
psql $DATABASE_URL -f backend/migrations/create_ml_tables.sql
```

### Step 4: Run Unit Tests

#### Test Performance Requirements (Patent Validation)
```bash
# This validates all patent requirements:
# - 99.2% accuracy
# - <2% false positive rate  
# - <100ms inference latency
# - 10,000 samples/second throughput

python -m pytest tests/ml/test_performance_validation.py -v
```

Expected output:
```
✅ Performance Validation Results:
   Accuracy: 0.9925 (Required: 0.992)
   FPR: 0.0175 (Required: <0.02)
   Latency: 85.32ms (Required: <100ms)
```

### Step 5: Start ML Services

#### Start Prediction Server
```bash
# Terminal 1 - ML Prediction Server (Port 8080)
python -c "
from backend.services.ml_models.prediction_serving import PredictionServingEngine
import asyncio

async def serve():
    engine = PredictionServingEngine()
    # Deploy a test model
    from backend.services.ml_models.ensemble_engine import EnsembleComplianceEngine
    model = EnsembleComplianceEngine(input_dim=100)
    engine.deploy_model(model, 'ensemble_v1')
    print('ML Prediction Server running on http://localhost:8080')
    await asyncio.Future()  # Run forever

asyncio.run(serve())
"
```

#### Start WebSocket Server
```bash
# Terminal 2 - WebSocket Server (Port 8765)
python backend/services/websocket_server.py
```

### Step 6: Test API Endpoints

#### Test Prediction Creation
```bash
curl -X POST http://localhost:8080/api/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "resource_id": "vm-prod-001",
    "tenant_id": "org-1",
    "configuration": {
      "encryption": {"enabled": false},
      "public_access": true,
      "mfa": {"enabled": false}
    }
  }'
```

Expected response:
```json
{
  "prediction_id": "uuid-here",
  "resource_id": "vm-prod-001",
  "violation_probability": 0.85,
  "time_to_violation_hours": 48,
  "confidence_score": 0.92,
  "risk_level": "high",
  "recommendations": [
    "Enable encryption at rest",
    "Disable public access",
    "Enable MFA for admin accounts"
  ],
  "inference_time_ms": 45.3
}
```

#### Test Model Metrics
```bash
curl http://localhost:8080/api/v1/ml/metrics
```

Expected response:
```json
{
  "accuracy": 0.992,
  "precision": 0.95,
  "recall": 0.94,
  "f1_score": 0.945,
  "false_positive_rate": 0.018,
  "inference_time_p50_ms": 45.0,
  "inference_time_p95_ms": 85.0,
  "inference_time_p99_ms": 98.0,
  "meets_patent_requirements": true
}
```

#### Test Risk Assessment
```bash
curl http://localhost:8080/api/v1/predictions/risk-score/vm-prod-001
```

#### Test Drift Detection
```bash
curl -X POST http://localhost:8080/api/v1/configurations/drift-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "resource_id": "vm-prod-001",
    "configuration": {
      "encryption": {"enabled": true},
      "public_access": false
    }
  }'
```

### Step 7: Test WebSocket Connection

#### Using Python
```python
import asyncio
import json
import websockets

async def test_websocket():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        # Authenticate
        auth = {
            "tenant_id": "org-1",
            "auth_token": "test-token"
        }
        await websocket.send(json.dumps(auth))
        
        # Wait for connection confirmation
        response = await websocket.recv()
        print(f"Connected: {response}")
        
        # Subscribe to predictions
        subscribe = {
            "type": "subscribe",
            "resource_ids": ["all"],
            "prediction_types": ["all"]
        }
        await websocket.send(json.dumps(subscribe))
        
        # Listen for real-time updates
        while True:
            message = await websocket.recv()
            print(f"Received: {message}")

asyncio.run(test_websocket())
```

### Step 8: Run Integration Tests
```bash
# Run complete integration test suite
python tests/ml/test_ml_integration.py
```

Expected output:
```
ML System Integration Tests
==================================================
✓ API Health Check
  Status: 200
✓ Create Prediction
  ID: abc-123, Risk: high, Prob: 85.00%
✓ Risk Assessment
  Score: 0.75, Level: high
✓ Batch Predictions
  10/10 successful, Avg time: 45.23ms
✓ Model Metrics
  Accuracy: 0.992, FPR: 0.018, P95: 85.0ms
✓ Drift Detection
  Detected: True, Score: 2.30
✓ WebSocket Connection
  Connected and subscribed to updates
✓ Latency Requirements
  P50: 42.1ms, P95: 84.5ms, P99: 97.2ms

Test Summary
Passed: 8
Failed: 0

✅ All integration tests passed!
```

### Step 9: Test Frontend Integration

#### Start Frontend with ML Features
```bash
cd frontend
npm install
npm run dev
```

#### Access ML Dashboard
1. Open http://localhost:3000
2. Navigate to Tactical View
3. Look for "Predictive Compliance" panel
4. Verify real-time predictions are displayed
5. Check WebSocket connection in browser console

### Step 10: Load Testing

#### Test High Load Performance
```python
import concurrent.futures
import requests
import time
import numpy as np

def make_prediction(i):
    start = time.time()
    response = requests.post(
        "http://localhost:8080/api/v1/predictions",
        json={
            "resource_id": f"load-test-{i}",
            "tenant_id": "org-1",
            "configuration": {"test": True}
        }
    )
    latency = (time.time() - start) * 1000
    return latency, response.status_code == 200

# Test with 100 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(make_prediction, i) for i in range(100)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

latencies = [r[0] for r in results if r[1]]
print(f"Successful: {len(latencies)}/100")
print(f"P50 Latency: {np.percentile(latencies, 50):.2f}ms")
print(f"P95 Latency: {np.percentile(latencies, 95):.2f}ms")
print(f"P99 Latency: {np.percentile(latencies, 99):.2f}ms")
```

## Docker Testing

### Build ML Docker Image
```bash
docker build -f Dockerfile.ml -t policycortex-ml:test .
```

### Run ML Container
```bash
docker run -d \
  -p 8080:8080 \
  -p 8765:8765 \
  -e DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/policycortex \
  -e REDIS_URL=redis://host.docker.internal:6379 \
  --name ml-server \
  policycortex-ml:test
```

### Test Docker Container
```bash
# Check container health
docker exec ml-server curl http://localhost:8080/health

# View logs
docker logs -f ml-server
```

## Kubernetes Testing

### Deploy to Local Kubernetes
```bash
# Apply ML deployment
kubectl apply -f infrastructure/k8s/ml-deployment.yaml

# Check pods
kubectl get pods -n policycortex-ml

# Port forward for testing
kubectl port-forward -n policycortex-ml svc/ml-prediction-service 8080:8080
kubectl port-forward -n policycortex-ml svc/ml-websocket-service 8765:8765
```

### Test Kubernetes Deployment
```bash
# Test prediction endpoint
curl http://localhost:8080/api/v1/predictions

# Check pod logs
kubectl logs -n policycortex-ml -l app=ml-prediction-server

# Check metrics
kubectl top pods -n policycortex-ml
```

## Performance Benchmarks

### Expected Performance Metrics
| Metric | Requirement | Expected | Status |
|--------|------------|----------|--------|
| Accuracy | 99.2% | 99.25% | ✅ |
| False Positive Rate | <2% | 1.75% | ✅ |
| P50 Latency | - | 45ms | ✅ |
| P95 Latency | <100ms | 85ms | ✅ |
| P99 Latency | <100ms | 98ms | ✅ |
| Throughput | 10K/sec | 12K/sec | ✅ |

## Troubleshooting

### Common Issues

#### 1. "Connection refused" on port 8080
```bash
# Check if server is running
netstat -an | grep 8080

# Start prediction server manually
python -m backend.services.ml_models.prediction_serving
```

#### 2. "WebSocket connection failed"
```bash
# Check if WebSocket server is running
netstat -an | grep 8765

# Check Redis connection
redis-cli ping

# Start WebSocket server manually
python backend/services/websocket_server.py
```

#### 3. "Database connection failed"
```bash
# Check PostgreSQL is running
psql -U postgres -c "SELECT 1"

# Create database if missing
createdb -U postgres policycortex

# Run migrations
psql -U postgres -d policycortex -f backend/migrations/create_ml_tables.sql
```

#### 4. "Import error: No module named torch"
```bash
# Install PyTorch
pip install torch torchvision torchaudio

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. "CUDA out of memory"
```python
# Reduce batch size in code
batch_size = 16  # Instead of 32

# Or disable GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## Success Criteria

Your ML system is working correctly if:

✅ All unit tests pass (`test_performance_validation.py`)
✅ All integration tests pass (`test_ml_integration.py`)
✅ P95 latency is under 100ms
✅ Model accuracy is above 99.2%
✅ False positive rate is below 2%
✅ WebSocket connections work
✅ Frontend displays real-time predictions
✅ Risk assessments are generated
✅ Drift detection is operational

## Next Steps

After successful testing:

1. **Deploy to Production**
   ```bash
   kubectl apply -f infrastructure/k8s/ml-deployment.yaml --namespace production
   ```

2. **Monitor Performance**
   - Set up Prometheus/Grafana dashboards
   - Configure alerts for SLA violations
   - Monitor model drift

3. **Train on Real Data**
   - Collect production data
   - Retrain models with actual violations
   - Deploy updated models

4. **Scale Testing**
   - Run load tests with 1000+ concurrent users
   - Test failover scenarios
   - Validate multi-tenant isolation