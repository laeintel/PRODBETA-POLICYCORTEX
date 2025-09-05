# Real Mode Configuration Guide

## Overview
This guide provides step-by-step instructions for enabling Real Mode in PolicyCortex PCG Platform. Real Mode connects to live Azure services and provides actual predictive compliance data.

## Phase 2: Real Mode Integration Status

### ✅ Completed
- Fail-fast pattern implemented in core API
- Health endpoint with comprehensive sub-checks  
- ServiceUnavailable errors with configuration hints
- Real mode detection based on USE_REAL_DATA environment variable

### 🚧 In Progress
- Azure PolicyInsights API integration
- Azure Cost Management API connection
- ML model deployment automation

## Quick Start

### 1. Set Environment Variables

Create a `.env` file in the project root:

```env
# Enable Real Mode
USE_REAL_DATA=true

# Azure Service Principal (Required)
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_SUBSCRIPTION_ID=your-subscription-id

# Azure AD Authentication (Required for SSO)
AZURE_AD_TENANT_ID=your-ad-tenant-id
AZURE_AD_CLIENT_ID=your-ad-client-id

# Database (Required for persistence)
DATABASE_URL=postgresql://user:password@localhost:5432/policycortex

# ML Service (Required for predictions)
PREDICTIONS_URL=http://localhost:8000

# Cache (Optional but recommended)
REDIS_URL=redis://localhost:6379

# Optional: Azure Key Vault for secrets
AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net
```

### 2. Create Azure Service Principal

```bash
# Login to Azure CLI
az login

# Create service principal with required permissions
az ad sp create-for-rbac \
  --name "PolicyCortex-PCG" \
  --role "Policy Insights Data Reader" \
  --scopes "/subscriptions/{subscription-id}"

# Save the output - you'll need:
# - appId (AZURE_CLIENT_ID)
# - password (AZURE_CLIENT_SECRET)  
# - tenant (AZURE_TENANT_ID)
```

### 3. Grant Required Permissions

The service principal needs these Azure permissions:

#### Minimum Required Roles
- `Policy Insights Data Reader` - Read policy compliance data
- `Cost Management Reader` - Access cost and billing data
- `Reader` - View resources and configurations

#### Additional Recommended Roles
- `Log Analytics Reader` - Query Azure Monitor logs
- `Security Reader` - Access security recommendations

```bash
# Grant additional roles
az role assignment create \
  --assignee {client-id} \
  --role "Cost Management Reader" \
  --scope "/subscriptions/{subscription-id}"

az role assignment create \
  --assignee {client-id} \
  --role "Log Analytics Reader" \
  --scope "/subscriptions/{subscription-id}"
```

### 4. Deploy ML Models

```bash
# Navigate to ML service directory
cd backend/services/ai_engine

# Install dependencies
pip install -r requirements.txt

# Deploy pre-trained models
python deploy_models.py

# Start ML service
uvicorn simple_ml_service:app --port 8000
```

### 5. Set Up Database

```bash
# Start PostgreSQL (using Docker)
docker run -d \
  --name policycortex-db \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=policycortex \
  -p 5432:5432 \
  postgres:15

# Run migrations
cd core
sqlx migrate run
```

### 6. Optional: Set Up Cache

```bash
# Using Redis
docker run -d \
  --name policycortex-cache \
  -p 6379:6379 \
  redis:7

# OR using DragonflyDB (faster)
docker run -d \
  --name policycortex-cache \
  -p 6379:6379 \
  docker.dragonflydb.io/dragonflydb/dragonfly
```

## Health Check Validation

Once configured, verify your setup:

```bash
# Check health endpoint
curl http://localhost:8080/api/v1/health

# Expected response when properly configured:
{
  "status": "healthy",
  "version": "2.27.4",
  "mode": "real",
  "checks": {
    "azure_connection": {
      "status": "healthy",
      "latency_ms": 250
    },
    "database": {
      "status": "healthy",
      "latency_ms": 5
    },
    "ml_service": {
      "status": "healthy", 
      "latency_ms": 15
    },
    "cache": {
      "status": "healthy",
      "latency_ms": 2
    },
    "authentication": {
      "status": "healthy",
      "latency_ms": 0
    }
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Azure Connection Failed
**Error**: "Azure client not configured"
**Solution**: 
- Verify all Azure environment variables are set
- Check service principal has correct permissions
- Ensure credentials are not expired

#### 2. ML Service Unavailable  
**Error**: "ML service unreachable"
**Solution**:
- Start the ML service: `python simple_ml_service.py`
- Check PREDICTIONS_URL points to correct address
- Verify models are deployed in `models_cache/` directory

#### 3. Database Connection Failed
**Error**: "Database error: connection refused"
**Solution**:
- Ensure PostgreSQL is running
- Verify DATABASE_URL format is correct
- Check network connectivity to database host

#### 4. Authentication Incomplete
**Error**: "Authentication configuration incomplete"  
**Solution**:
- Set both AZURE_AD_TENANT_ID and AZURE_AD_CLIENT_ID
- Register app in Azure AD if not already done

### Fail-Fast Behavior

When `USE_REAL_DATA=true` but services are not configured:

1. **API returns 503 Service Unavailable** with helpful configuration hints
2. **Health endpoint shows "degraded" status** with specific issues
3. **Frontend displays configuration instructions** instead of mock data

Example error response:
```json
{
  "error": "service_unavailable",
  "message": "Service 'Predictive Compliance Engine' unavailable. Real mode is disabled. Set USE_REAL_DATA=true and configure:\n- AZURE_SUBSCRIPTION_ID\n- AZURE_TENANT_ID\n- AZURE_CLIENT_ID\n- AZURE_CLIENT_SECRET\nSee docs/REVAMP/REAL_MODE_SETUP.md for configuration details.",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## API Endpoints Affected by Real Mode

These endpoints behave differently in real vs mock mode:

| Endpoint | Mock Mode | Real Mode |
|----------|-----------|-----------|
| `/api/v1/predictions` | Returns demo predictions | Queries Azure PolicyInsights |
| `/api/v1/evidence` | Generates fake evidence | Creates real compliance records |
| `/api/v1/correlations` | Shows sample correlations | Analyzes actual cost/policy data |
| `/api/v1/resources` | Lists mock resources | Fetches from Azure Resource Graph |
| `/api/v1/health` | Always healthy | Validates all connections |

## Next Steps

After completing Real Mode setup:

1. **Test Predictive Compliance**: Navigate to PREVENT tab, should see real violations
2. **Verify Evidence Chain**: PROVE tab should show actual compliance records
3. **Check ROI Calculations**: PAYBACK tab displays real cost savings
4. **Monitor Performance**: Ensure <500ms latency for predictions
5. **Set Up Alerts**: Configure monitoring for service degradation

## Security Considerations

1. **Never commit credentials** to version control
2. **Use Azure Key Vault** for production secrets
3. **Rotate service principal keys** regularly (every 90 days)
4. **Enable audit logging** for compliance tracking
5. **Use managed identities** when running in Azure

## Support

If you encounter issues not covered here:

1. Check health endpoint for diagnostic hints: `GET /api/v1/health`
2. Review logs: `tail -f logs/policycortex.log`
3. Consult TD.MD for acceptance criteria
4. File issue at: https://github.com/policycortex/pcg/issues