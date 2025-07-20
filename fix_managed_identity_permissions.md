# Fix Managed Identity Permissions for PolicyCortex Container Apps

## Root Cause Identified ✅
The container apps are failing because the managed identity `id-policycortex-dev` (Principal ID: `5389fdab-721b-4aa9-b9c1-5e33aa9dde3d`) has **NO role assignments** and cannot access any Azure resources.

## Required Role Assignments

The managed identity needs access to these resources in `rg-policycortex-dev`:

### 1. Key Vault Access
- **Resource**: `kvpolicycortexdev`
- **Role**: `Key Vault Secrets User`
- **Purpose**: Access secrets (SQL password, Cosmos keys, etc.)

### 2. Cosmos DB Access  
- **Resource**: `policycortex-cosmos-dev`
- **Role**: `Cosmos DB Built-in Data Contributor`
- **Purpose**: Read/write data to Cosmos DB

### 3. Redis Cache Access
- **Resource**: `policycortex-redis-dev` 
- **Role**: `Redis Cache Contributor`
- **Purpose**: Access Redis cache for session storage

### 4. Storage Account Access
- **Resource**: `stpolicycortexdevstg`
- **Role**: `Storage Blob Data Contributor`
- **Purpose**: Read/write blob storage

### 5. Cognitive Services Access
- **Resource**: `policycortex-cognitive-dev`
- **Role**: `Cognitive Services User`
- **Purpose**: Use AI services for NLP/ML operations

### 6. Application Insights Access
- **Resource**: `ai-policycortex-dev`
- **Role**: `Monitoring Contributor`
- **Purpose**: Send telemetry and monitoring data

### 7. Resource Group Read Access
- **Resource**: `rg-policycortex-dev`
- **Role**: `Reader`
- **Purpose**: Discover and enumerate resources

## Quick Fix - Azure Portal Method

Since Azure CLI is having authentication issues, assign these roles through the Azure Portal:

1. **Go to Azure Portal** → Resource Groups → `rg-policycortex-dev`
2. **Click Access Control (IAM)** → Add → Add role assignment
3. **For each resource above:**
   - Select the role (e.g., "Key Vault Secrets User")
   - Assign access to: "Managed Identity"
   - Select: `id-policycortex-dev`
   - Click "Save"

## Alternative - Azure CLI Commands (Run when CLI is working)

```bash
# Set subscription context
az account set --subscription 9f16cc88-89ce-49ba-a96d-308ed3169595

# Managed Identity Principal ID
PRINCIPAL_ID="5389fdab-721b-4aa9-b9c1-5e33aa9dde3d"

# Key Vault Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Key Vault Secrets User" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.KeyVault/vaults/kvpolicycortexdev"

# Cosmos DB Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cosmos DB Built-in Data Contributor" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.DocumentDB/databaseAccounts/policycortex-cosmos-dev"

# Redis Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Redis Cache Contributor" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.Cache/redis/policycortex-redis-dev"

# Storage Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.Storage/storageAccounts/stpolicycortexdevstg"

# Cognitive Services Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Cognitive Services User" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.CognitiveServices/accounts/policycortex-cognitive-dev"

# Application Insights Access  
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Monitoring Contributor" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev/providers/Microsoft.Insights/components/ai-policycortex-dev"

# Resource Group Reader Access
az role assignment create \
  --assignee $PRINCIPAL_ID \
  --role "Reader" \
  --scope "/subscriptions/9f16cc88-89ce-49ba-a96d-308ed3169595/resourceGroups/rg-policycortex-dev"
```

## After Role Assignment

Once the managed identity has proper permissions:

1. **Container apps will automatically restart** and should start successfully
2. **Test the health endpoints:**
   ```bash
   curl https://ca-api-gateway-dev.ambitiousriver-40e95c18.eastus.azurecontainerapps.io/health
   ```
3. **Check container logs** to verify no more permission errors

## Expected Result

✅ All container apps should start successfully  
✅ Health endpoints should respond  
✅ No more "Field required" or authentication errors in logs  
✅ Full PolicyCortex application stack functional  

This single fix should resolve all the container app startup issues!