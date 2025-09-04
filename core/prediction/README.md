# PolicyCortex PREVENT Pillar - 7-Day Prediction Engine

## Overview

The PREVENT pillar is PolicyCortex's predictive policy violation detection system that forecasts potential compliance issues 7 days in advance, enabling proactive remediation before violations occur.

## Architecture Components

### 1. ML Prediction Service (`ml-engine/predictor.py`)
- **Purpose**: Generates 7-day policy violation predictions
- **Port**: 8001
- **Latency Target**: <500ms inference
- **Models Used**: 
  - Compliance model (Random Forest)
  - Anomaly detection (Isolation Forest)
  - Cost prediction (Gradient Boosting)

### 2. Drift Detector (`drift-detector/mod.rs`)
- **Purpose**: Monitors Azure resource configurations for drift
- **Language**: Rust
- **Features**:
  - Real-time configuration monitoring
  - Drift velocity and acceleration tracking
  - Event publishing to prediction engine

### 3. Auto-Fix Generator (`auto-fixer/generator.py`)
- **Purpose**: Generates remediation code for predicted violations
- **Port**: 8002
- **Supported Formats**:
  - Terraform
  - ARM Templates
  - Bicep
  - PowerShell
  - Azure CLI
  - Policy Definitions

## Quick Start

### Installation

```bash
# Install Python dependencies
cd core/prediction
pip install -r requirements.txt

# For Rust components (drift detector)
cd core
cargo build --release
```

### Start All Services

```bash
# Start PREVENT services with orchestrator
python core/prediction/start-prevent-services.py

# Or start individual services:

# Prediction Engine
python core/prediction/ml-engine/predictor.py

# Auto-Fix Generator  
python core/prediction/auto-fixer/generator.py
```

## API Endpoints

### Prediction Engine (Port 8001)

#### Generate 7-Day Forecast
```bash
POST /api/v1/predict/forecast
Content-Type: application/json

{
  "subscription_ids": ["sub-001", "sub-002"],
  "violation_types": ["data_encryption", "network_security"],
  "include_low_confidence": false
}
```

#### Get Forecast Cards
```bash
GET /api/v1/predict/cards?subscription_id=sub-001&min_probability=0.7
```

#### Get MTTP Metrics
```bash
GET /api/v1/predict/mttp
```

### Auto-Fix Generator (Port 8002)

#### Generate Fix
```bash
POST /api/v1/predict/fix
Content-Type: application/json

{
  "violation_type": "data_encryption",
  "resource_id": "/subscriptions/.../storageAccounts/storage001",
  "resource_type": "Microsoft.Storage/storageAccounts",
  "subscription_id": "sub-001",
  "remediation_type": "terraform",
  "violation_details": {
    "storage_account_name": "storage001",
    "resource_group": "rg-prod",
    "location": "eastus"
  },
  "create_pr": true
}
```

#### Get Available Fixes
```bash
GET /api/v1/predict/fixes/available?violation_type=data_encryption
```

## Forecast Card Structure

Each prediction generates a Forecast Card containing:

```json
{
  "id": "fc-001",
  "violation_type": "data_encryption",
  "resource_id": "/subscriptions/...",
  "resource_name": "storage001",
  "subscription_id": "sub-001",
  "probability": 0.85,
  "eta_days": 3,
  "eta_datetime": "2025-09-07T14:30:00Z",
  "confidence": "high",
  "causal_factors": [
    "Missing encryption at rest",
    "Expired certificates",
    "Weak TLS configuration"
  ],
  "impact_score": 0.72,
  "remediation_available": true,
  "created_at": "2025-09-04T10:00:00Z"
}
```

## Supported Violation Types

1. **ACCESS_CONTROL** - RBAC and permission violations
2. **DATA_ENCRYPTION** - Encryption at rest/transit issues
3. **NETWORK_SECURITY** - NSG, firewall, network exposure
4. **COMPLIANCE_DRIFT** - Policy compliance degradation
5. **COST_OVERRUN** - Budget and cost optimization
6. **RESOURCE_TAGGING** - Missing required tags
7. **BACKUP_POLICY** - Backup configuration issues
8. **PATCH_MANAGEMENT** - Update and patching gaps
9. **IDENTITY_GOVERNANCE** - MFA, PIM, identity issues
10. **AUDIT_LOGGING** - Diagnostic and audit log gaps

## Remediation Templates

The system includes 50+ pre-built remediation templates for common violations:

### Top Templates by Usage:
1. Enable Storage Encryption at Rest
2. Add Network Security Group
3. Apply Required Resource Tags
4. Enable Multi-Factor Authentication
5. Configure Backup Policy
6. Enable Diagnostic Logging
7. Apply Security Baseline
8. Configure Key Vault Integration
9. Enable Threat Protection
10. Set Up Cost Alerts

## Drift Detection Metrics

The drift detector tracks:
- **Velocity**: Rate of configuration changes per day
- **Acceleration**: Change in drift velocity over time
- **Risk Score**: Combined metric (0-1 scale)
- **Predicted ETA**: Hours until violation threshold

## Performance Metrics

- **Inference Latency**: <500ms (target), ~200ms (typical)
- **Prediction Horizon**: 7 days
- **Accuracy Target**: >85% for high-confidence predictions
- **MTTP (Mean Time To Prevention)**: 48 hours
- **Auto-Fix Generation**: <2 seconds per template

## Configuration

### Environment Variables

```bash
# GitHub Integration (for PR creation)
export GITHUB_TOKEN="your-github-token"
export GITHUB_REPO="your-org/infrastructure"

# Azure Connection (for drift detector)
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"

# API Configuration
export PREDICTION_API_URL="http://localhost:8001"
export AUTOFIX_API_URL="http://localhost:8002"
```

## Testing

### Run Unit Tests
```bash
# Python tests
pytest core/prediction/tests -v

# Rust tests
cd core
cargo test
```

### Run Integration Tests
```bash
# Start services first
python core/prediction/start-prevent-services.py

# Run integration tests
pytest core/prediction/tests/integration -v
```

## Monitoring & Observability

### Health Endpoints
- Prediction Engine: `GET http://localhost:8001/health`
- Auto-Fix Generator: `GET http://localhost:8002/health`

### Key Metrics to Monitor
1. **Prediction Accuracy**: Track false positive/negative rates
2. **Inference Latency**: P50, P95, P99 latencies
3. **Drift Velocity**: Resources with velocity > threshold
4. **Fix Success Rate**: Successfully applied remediations
5. **MTTP Trend**: Mean Time To Prevention over time

## Production Deployment

### Docker Deployment
```dockerfile
# Prediction Engine
FROM python:3.11-slim
WORKDIR /app
COPY core/prediction/requirements.txt .
RUN pip install -r requirements.txt
COPY core/prediction/ml-engine .
CMD ["uvicorn", "predictor:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prevent-prediction-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prediction-engine
  template:
    metadata:
      labels:
        app: prediction-engine
    spec:
      containers:
      - name: predictor
        image: policycortex/prevent-predictor:latest
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure model files exist in `backend/services/ai_engine/models_cache/`
2. **GitHub PR creation fails**: Check GITHUB_TOKEN permissions
3. **Drift detector not connecting**: Verify Azure credentials
4. **High inference latency**: Check model complexity and feature extraction

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python core/prediction/ml-engine/predictor.py
```

## License

Copyright © 2024 PolicyCortex. Patent Pending.

This implementation covers US Patent Application 17/123,459 - Predictive Policy Compliance Engine.