# Container Startup Troubleshooting Guide

## Problem Summary
All containers are failing to start with exit code '1' and "ProcessExited" reasons. The logs show:
- Container termination with exit code '1'
- "ScaledObject doesn't have correct triggers specification"
- "Persistent Failure to start container"

## Root Causes Identified

### 1. Missing Environment Variables
**Problem**: Containers can't access required configuration from Key Vault
**Symptoms**: 
- Import errors in startup
- Configuration initialization failures
- Missing JWT_SECRET_KEY and other required variables

**Fix Applied**:
- Added proper environment variable mapping in Bicep template
- Added startup scripts with environment validation
- Fixed Python path issues for shared module imports

### 2. Health Probe Issues
**Problem**: Health probes were disabled, causing container startup failures
**Symptoms**:
- "Persistent Failure to start container"
- Containers terminating immediately after startup

**Fix Applied**:
- Added proper startup, readiness, and liveness probes
- Configured appropriate timeouts and failure thresholds
- Added health check endpoints to all services

### 3. ScaledObject Configuration Issues
**Problem**: KEDA ScaledObject triggers not properly configured
**Symptoms**:
- "ScaledObject doesn't have correct triggers specification"
- Scaling rules not working properly

**Fix Applied**:
- Updated scaling rules in Bicep template
- Fixed HTTP scaling configuration
- Added proper concurrent request limits

### 4. Python Module Import Issues
**Problem**: Services can't import shared modules
**Symptoms**:
- ImportError exceptions during startup
- ModuleNotFoundError for shared.config, shared.database

**Fix Applied**:
- Added Python path configuration in startup scripts
- Created proper module import error handling
- Added startup validation scripts

## Fixes Implemented

### 1. Startup Scripts
Created `startup.py` scripts for each service that:
- Validate environment variables
- Check Python module availability
- Provide detailed error logging
- Handle import errors gracefully

### 2. Updated Dockerfiles
Modified Dockerfiles to:
- Use startup scripts instead of direct uvicorn calls
- Add proper environment variable configuration
- Include better error handling

### 3. Infrastructure Fixes
Updated Bicep template to:
- Add proper health probes (startup, readiness, liveness)
- Fix environment variable mapping
- Configure proper scaling rules
- Add SERVICE_HOST environment variable

### 4. Application Code Fixes
Updated main.py files to:
- Add proper error handling for imports
- Include startup logging
- Handle configuration initialization errors

## Deployment Steps

### Option 1: Use the Fix Script (Recommended)
```bash
# For Linux/Mac
chmod +x fix-container-startup.sh
./fix-container-startup.sh dev

# For Windows PowerShell
.\fix-container-startup.ps1 -Environment dev
```

### Option 2: Manual Deployment
1. Build and push containers:
```bash
# Build all services
docker build -t crpolicortex001dev.azurecr.io/policortex001-api-gateway:latest -f backend/services/api_gateway/Dockerfile backend/
docker build -t crpolicortex001dev.azurecr.io/policortex001-notification:latest -f backend/services/notification/Dockerfile backend/
# ... repeat for all services

# Push to registry
docker push crpolicortex001dev.azurecr.io/policortex001-api-gateway:latest
docker push crpolicortex001dev.azurecr.io/policortex001-notification:latest
# ... repeat for all services
```

2. Deploy infrastructure:
```bash
cd infrastructure/bicep
az deployment group create \
  --resource-group rg-policortex001-dev \
  --template-file main.bicep \
  --parameters environment=dev \
  --verbose
```

## Verification Steps

### 1. Check Container Status
```bash
az containerapp revision list \
  --name ca-api-gateway-dev \
  --resource-group rg-policortex001-dev
```

### 2. View Container Logs
```bash
az containerapp logs show \
  --name ca-api-gateway-dev \
  --resource-group rg-policortex001-dev \
  --follow
```

### 3. Test Health Endpoints
```bash
# Test API Gateway health
curl https://ca-api-gateway-dev.azurecontainerapps.io/health

# Test Notification service health
curl https://ca-notification-dev.azurecontainerapps.io/health
```

### 4. Check Environment Variables
```bash
az containerapp show \
  --name ca-api-gateway-dev \
  --resource-group rg-policortex001-dev \
  --query "properties.template.containers[0].env"
```

## Expected Results After Fix

### Successful Startup Logs
```
2025-07-24 11:04:29 - __main__ - INFO - Starting API Gateway service...
2025-07-24 11:04:29 - __main__ - INFO - All required environment variables are set
2025-07-24 11:04:29 - __main__ - INFO - Python path: ['/app', '/app/services/api_gateway', ...]
2025-07-24 11:04:29 - __main__ - INFO - FastAPI version: 0.104.1
2025-07-24 11:04:29 - __main__ - INFO - Uvicorn version: 0.24.0
2025-07-24 11:04:29 - __main__ - INFO - Startup checks passed, starting application...
2025-07-24 11:04:29 - __main__ - INFO - Starting server on 0.0.0.0:80
```

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2025-07-24T11:04:29Z",
  "service": "api-gateway",
  "version": "1.0.0"
}
```

## Troubleshooting Commands

### Check Container App Status
```bash
az containerapp show \
  --name ca-api-gateway-dev \
  --resource-group rg-policortex001-dev \
  --query "properties.latestRevisionName,properties.latestReadyRevisionName,properties.runningStatus"
```

### Check Revision Details
```bash
az containerapp revision show \
  --name ca-api-gateway-dev \
  --resource-group rg-policortex001-dev \
  --revision <revision-name>
```

### Check Key Vault Access
```bash
az keyvault secret list \
  --vault-name kv-policortex001-dev \
  --query "[].name"
```

### Test Container Locally
```bash
# Build and test locally
docker build -t test-api-gateway -f backend/services/api_gateway/Dockerfile backend/
docker run -p 8000:8000 -e ENVIRONMENT=development -e SERVICE_NAME=api-gateway -e PORT=8000 test-api-gateway
```

## Common Issues and Solutions

### Issue: "ModuleNotFoundError: No module named 'shared'"
**Solution**: The startup script now adds the backend directory to Python path

### Issue: "Missing required environment variables"
**Solution**: Check Key Vault permissions and ensure managed identity has access

### Issue: "Health check failed"
**Solution**: Verify the health endpoint is responding correctly and check probe configuration

### Issue: "Container terminated with exit code 1"
**Solution**: Check the startup logs for specific error messages using the new startup scripts

## Monitoring and Alerts

### Set up Log Analytics Queries
```kusto
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "ca-api-gateway-dev"
| where TimeGenerated > ago(1h)
| where Log_s contains "ERROR" or Log_s contains "Failed"
| order by TimeGenerated desc
```

### Set up Alerts
```bash
# Create alert for container failures
az monitor metrics alert create \
  --name "Container-Failure-Alert" \
  --resource-group rg-policortex001-dev \
  --scopes /subscriptions/<subscription-id>/resourceGroups/rg-policortex001-dev/providers/Microsoft.App/containerApps/ca-api-gateway-dev \
  --condition "avg Percentage CPU > 80" \
  --description "Alert when container CPU usage is high"
```

## Next Steps

1. **Deploy the fixes** using the provided scripts
2. **Monitor the logs** for successful startup
3. **Test the health endpoints** to verify functionality
4. **Set up monitoring** for ongoing health checks
5. **Document any remaining issues** for further troubleshooting

## Support

If issues persist after applying these fixes:
1. Check the detailed logs using the new startup scripts
2. Verify Key Vault permissions and secrets
3. Test containers locally to isolate issues
4. Review the infrastructure configuration for any remaining issues 