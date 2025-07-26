# Azure Container Registry Authentication Setup

## Overview

The PolicyCortex deployment uses **managed identity authentication** for seamless, secure access to Azure Container Registry (ACR) without managing passwords or connection strings.

## Configuration Summary

### ✅ Current Setup

1. **User-Assigned Managed Identity**
   - Name: `id-policortex001-{environment}`
   - Assigned to all Container Apps
   - Used for ACR authentication and Key Vault access

2. **ACR Permissions (Automatic)**
   - **AcrPull**: Container Apps can pull images
   - **AcrPush**: CI/CD pipelines can push images
   - Permissions assigned at ACR resource level

3. **Container Apps Configuration**
   ```bicep
   registries: [
     {
       server: containerRegistryLoginServer
       identity: userAssignedIdentityId  // No username/password needed!
     }
   ]
   ```

### 🔧 How It Works

1. **Infrastructure Deployment**:
   - Creates User-Assigned Managed Identity
   - Creates ACR with Premium SKU
   - Automatically assigns `AcrPull` and `AcrPush` roles to the managed identity

2. **Container Apps Authentication**:
   - Container Apps use the assigned managed identity
   - Azure handles token exchange automatically
   - No credentials to manage or rotate

3. **Image Pull Process**:
   ```
   Container App → Managed Identity → Azure AD Token → ACR → Image Pull
   ```

## Files Involved

### Bicep Configuration
- `modules/user-identity.bicep` - Creates managed identity
- `modules/container-registry.bicep` - Creates ACR + role assignments
- `modules/container-apps.bicep` - Configures managed identity auth

### Terraform Configuration  
- `main.tf` lines 452-472 - User-assigned identity and ACR role assignments

## Verification Commands

### Check Managed Identity
```bash
az identity show --name id-policortex001-dev --resource-group rg-policortex001-app-dev
```

### Check ACR Role Assignments
```bash
# Get identity principal ID
PRINCIPAL_ID=$(az identity show --name id-policortex001-dev --resource-group rg-policortex001-app-dev --query principalId -o tsv)

# Check AcrPull permission
az role assignment list --assignee $PRINCIPAL_ID --scope "/subscriptions/{sub-id}/resourceGroups/rg-policortex001-app-dev/providers/Microsoft.ContainerRegistry/registries/crpolicortex001dev" --role "AcrPull"

# Check AcrPush permission  
az role assignment list --assignee $PRINCIPAL_ID --scope "/subscriptions/{sub-id}/resourceGroups/rg-policortex001-app-dev/providers/Microsoft.ContainerRegistry/registries/crpolicortex001dev" --role "AcrPush"
```

### Automated Verification
```powershell
./verify-acr-permissions.ps1 -Environment dev -SubscriptionId "your-subscription-id"
```

## Role Definitions

| Role | GUID | Permissions | Purpose |
|------|------|-------------|---------|
| AcrPull | `7f951dda-4ed3-4680-a7ca-43fe172d538d` | Pull images | Container Apps |
| AcrPush | `8311e382-0749-4cb8-b61a-304f252e45ec` | Push/Pull images | CI/CD Pipelines |

## Troubleshooting

### Common Issues

1. **"UNAUTHORIZED: authentication required"**
   - ✅ **Solution**: Managed identity lacks AcrPull permission
   - Run: `az role assignment create --assignee {principal-id} --scope {acr-id} --role AcrPull`

2. **"Identity not found"**
   - ✅ **Solution**: Container App not assigned managed identity
   - Check Container App identity configuration

3. **"Image not found"**
   - ✅ **Solution**: Image doesn't exist in ACR
   - Build and push images first: `./build-and-push-images.ps1`

### Debugging Steps

1. **Check Container App Logs**:
   ```bash
   az containerapp logs show --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --follow
   ```

2. **Verify Identity Assignment**:
   ```bash
   az containerapp show --name ca-api-gateway-dev --resource-group rg-policortex001-app-dev --query identity
   ```

3. **Test ACR Access**:
   ```bash
   az acr repository list --name crpolicortex001dev
   ```

## Security Benefits

### ✅ Advantages of Managed Identity Auth
- **No credential management** - Azure handles tokens automatically
- **Automatic token rotation** - No expired credentials
- **Least privilege access** - Scoped to specific ACR
- **Audit trail** - All access logged in Azure AD
- **Zero secrets in code** - No connection strings or passwords

### ❌ Avoided Issues  
- No hardcoded ACR passwords
- No expired authentication tokens  
- No credential rotation overhead
- No secrets in Key Vault for ACR auth

## Best Practices

1. **Use Premium ACR SKU** - Better performance and security features
2. **Scope permissions minimally** - Only grant necessary roles  
3. **Monitor access logs** - Track image pull/push activities
4. **Regular access reviews** - Audit role assignments periodically
5. **Network security** - Consider private endpoints for production

## Related Documentation

- [Azure Container Registry authentication with managed identities](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)
- [Container Apps managed identity](https://docs.microsoft.com/en-us/azure/container-apps/managed-identity)
- [Azure RBAC built-in roles](https://docs.microsoft.com/en-us/azure/role-based-access-control/built-in-roles)