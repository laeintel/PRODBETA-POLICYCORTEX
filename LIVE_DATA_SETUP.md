# PolicyCortex v2 - Live Data Configuration Guide

## Current Status
Your application is running in **SIMULATED MODE** because it lacks proper Azure authentication. Here's how to enable live data:

## Why You Don't Have Live Data

1. **No Azure Credentials**: The Container Apps don't have Azure AD authentication configured
2. **No Managed Identity**: System-assigned managed identity wasn't enabled
3. **No Service Principal**: No client secret is configured for authentication
4. **Default to Simulated**: The code defaults to simulated mode when Azure connection fails

## Option 1: Service Principal with Client Secret (Easiest)

### Step 1: Create Service Principal
```bash
# Create service principal and save credentials
az ad sp create-for-rbac --name "PolicyCortex-Dev" \
  --role "Reader" \
  --scopes /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78 \
  --sdk-auth
```

This will output:
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "205b477d-17e7-4b3b-92c1-32cf02626b78",
  "tenantId": "9ef5b184-d371-462a-bc75-5024ce8baff7"
}
```

### Step 2: Add Permissions to Service Principal
```bash
# Get the clientId from above output
CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Grant additional permissions
az role assignment create --assignee $CLIENT_ID --role "Policy Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
az role assignment create --assignee $CLIENT_ID --role "Cost Management Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
az role assignment create --assignee $CLIENT_ID --role "Security Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
```

### Step 3: Store Client Secret in Key Vault
```bash
# Create Key Vault if not exists
az keyvault create --name kv-policycortex-dev --resource-group rg-cortex-dev --location eastus

# Store secret
az keyvault secret set --vault-name kv-policycortex-dev --name azure-client-secret --value "YOUR_CLIENT_SECRET_HERE"
```

### Step 4: Update Container Apps Environment Variables
```bash
# Update Core API
az containerapp update -n ca-cortex-core-dev -g rg-cortex-dev \
  --set-env-vars \
    USE_REAL_DATA=true \
    AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78 \
    AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 \
    AZURE_CLIENT_ID=YOUR_CLIENT_ID_HERE \
    AZURE_CLIENT_SECRET=YOUR_CLIENT_SECRET_HERE

# Update Frontend (for Graph API calls)
az containerapp update -n ca-cortex-frontend-dev -g rg-cortex-dev \
  --set-env-vars \
    NEXT_PUBLIC_USE_REAL_DATA=true \
    AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78
```

## Option 2: Managed Identity (More Secure)

### Step 1: Enable Managed Identity
```bash
# Already done - your identity: 984a669b-37ef-4890-8c32-1ef98f7c3a8d
az containerapp identity assign -n ca-cortex-core-dev -g rg-cortex-dev --system-assigned
```

### Step 2: Grant Permissions
```bash
# The managed identity needs permissions at subscription level
IDENTITY_ID=984a669b-37ef-4890-8c32-1ef98f7c3a8d

# Grant roles (run these from Azure Portal or with correct context)
az role assignment create --assignee $IDENTITY_ID --role "Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
az role assignment create --assignee $IDENTITY_ID --role "Policy Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
az role assignment create --assignee $IDENTITY_ID --role "Cost Management Reader" --scope /subscriptions/205b477d-17e7-4b3b-92c1-32cf02626b78
```

### Step 3: Update Container App to Use Managed Identity
```bash
az containerapp update -n ca-cortex-core-dev -g rg-cortex-dev \
  --set-env-vars \
    USE_REAL_DATA=true \
    AZURE_USE_MANAGED_IDENTITY=true \
    AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78 \
    AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
```

## Option 3: Use Existing Service Principal (If Available)

If you already have a service principal with appropriate permissions:

```bash
# Set the credentials
az containerapp update -n ca-cortex-core-dev -g rg-cortex-dev \
  --set-env-vars \
    USE_REAL_DATA=true \
    AZURE_SUBSCRIPTION_ID=205b477d-17e7-4b3b-92c1-32cf02626b78 \
    AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 \
    AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c \
    AZURE_CLIENT_SECRET=YOUR_SECRET_HERE
```

## Verification

After configuration, test if live data is working:

```bash
# 1. Restart the Container App
az containerapp revision restart -n ca-cortex-core-dev -g rg-cortex-dev

# 2. Check health endpoint
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/health

# 3. Check metrics - should show real Azure data
curl https://ca-cortex-core-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io/api/v1/metrics

# 4. Look for "mode": "real" in response instead of "simulated"
```

## Required Azure Permissions

For full functionality, the service principal or managed identity needs:

### Subscription Level
- Reader
- Policy Reader  
- Cost Management Reader
- Security Reader
- Log Analytics Reader

### Resource Group Level (optional, for write operations)
- Contributor (for remediation actions)
- User Access Administrator (for RBAC changes)

### Azure AD Permissions (for RBAC analysis)
- Directory.Read.All
- Policy.Read.All
- SecurityEvents.Read.All

## Code Changes Needed

Your code already supports live data! The switch happens automatically when:

1. `USE_REAL_DATA=true` is set
2. Valid Azure credentials are available
3. The Azure client can authenticate successfully

The relevant code is in:
- `core/src/data_mode.rs` - Checks USE_REAL_DATA environment variable
- `core/src/azure_client_async.rs` - Uses DefaultAzureCredential which supports:
  - Environment variables (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET)
  - Managed Identity
  - Azure CLI credentials

## Troubleshooting

### Still Getting Simulated Data?
1. Check Container App logs:
```bash
az containerapp logs show -n ca-cortex-core-dev -g rg-cortex-dev --follow
```

2. Verify environment variables:
```bash
az containerapp show -n ca-cortex-core-dev -g rg-cortex-dev --query "properties.template.containers[0].env[?contains(name,'AZURE') || name=='USE_REAL_DATA']"
```

3. Test Azure connection directly:
```bash
# SSH into container (if enabled) or run locally with same env vars
curl -H "Metadata: true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/"
```

### Common Issues
- **"Azure connection not available"** - Missing or invalid credentials
- **"Unauthorized"** - Service principal lacks required permissions
- **Still showing simulated** - USE_REAL_DATA not set or credentials invalid
- **Timeout errors** - Network restrictions or firewall rules

## Security Notes

⚠️ **NEVER** commit client secrets to git
⚠️ Use Key Vault or Container App secrets for production
⚠️ Rotate secrets regularly
⚠️ Use Managed Identity when possible (no secrets needed)
⚠️ Apply principle of least privilege for permissions

## Next Steps

1. Choose authentication method (Service Principal recommended for dev)
2. Create and configure credentials
3. Update Container App environment variables
4. Restart Container App
5. Verify live data is flowing

Once configured, your application will automatically:
- Fetch real Azure policies
- Show actual resource counts
- Display real cost data
- Analyze actual RBAC assignments
- Detect real compliance violations

The UI will show "REAL DATA" indicator instead of "SIMULATED" when properly configured.