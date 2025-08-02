# Azure AD Configuration for PolicyCortex

## Overview
This guide walks through setting up Azure Active Directory authentication for PolicyCortex frontend and backend services.

## Prerequisites
- Azure subscription with admin privileges
- Azure CLI installed (`az login` completed)
- PowerShell or Bash terminal

## Step 1: Create Azure AD App Registration

### Option A: Using Azure Portal (Recommended for beginners)

1. **Go to Azure Portal**
   - Navigate to https://portal.azure.com
   - Search for "Azure Active Directory"
   - Click "App registrations" → "New registration"

2. **Configure App Registration**
   ```
   Name: PolicyCortex
   Supported account types: Accounts in this organizational directory only
   Redirect URI (optional): 
     - Platform: Single-page application (SPA)
     - URL: http://localhost:3000
   ```

3. **Add Additional Redirect URIs**
   - After creation, go to "Authentication"
   - Add these URIs under "Single-page application":
     ```
     http://localhost:3000
     http://localhost:5173
     https://yourdomain.com (for production)
     ```

4. **Configure API Permissions**
   - Go to "API permissions" → "Add a permission"
   - Microsoft Graph → Delegated permissions
   - Add these permissions:
     ```
     ✓ User.Read
     ✓ User.ReadBasic.All
     ✓ Directory.Read.All
     ✓ openid
     ✓ profile
     ✓ email
     ```

5. **Get Configuration Values**
   - Go to "Overview" tab
   - Copy these values:
     ```
     Application (client) ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
     Directory (tenant) ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
     ```

### Option B: Using Azure CLI (Automated)

```bash
# Create the app registration
az ad app create \
  --display-name "PolicyCortex" \
  --sign-in-audience "AzureADMyOrg" \
  --web-redirect-uris "http://localhost:3000" \
  --enable-id-token-issuance \
  --enable-access-token-issuance

# Get the app ID
APP_ID=$(az ad app list --display-name "PolicyCortex" --query "[0].appId" -o tsv)
echo "App ID: $APP_ID"

# Get tenant ID
TENANT_ID=$(az account show --query "tenantId" -o tsv)
echo "Tenant ID: $TENANT_ID"

# Add additional redirect URIs
az ad app update --id $APP_ID \
  --web-redirect-uris "http://localhost:3000" "http://localhost:5173"

# Add required API permissions
az ad app permission add --id $APP_ID --api 00000003-0000-0000-c000-000000000000 --api-permissions e1fe6dd8-ba31-4d61-89e7-88639da4683d=Scope
```

## Step 2: Configure Frontend Environment

Create `.env.local` file in frontend directory:

```bash
# Frontend Environment Configuration
VITE_AZURE_CLIENT_ID=your-client-id-from-step-1
VITE_AZURE_TENANT_ID=your-tenant-id-from-step-1
VITE_AZURE_REDIRECT_URI=http://localhost:3000
VITE_API_BASE_URL=http://localhost:8000/api
```

## Step 3: Configure Backend Authentication

Update your backend API Gateway service to validate JWT tokens from Azure AD:

### Backend Environment Variables:
```bash
# API Gateway Authentication
AZURE_CLIENT_ID=your-client-id-from-step-1
AZURE_TENANT_ID=your-tenant-id-from-step-1
JWT_ISSUER=https://login.microsoftonline.com/your-tenant-id/v2.0
JWT_AUDIENCE=your-client-id-from-step-1
```

## Step 4: Test Authentication

1. **Start your services:**
   ```bash
   # Backend services
   docker-compose -f docker-compose.local.yml up -d
   
   # Frontend
   cd frontend
   npm run dev
   ```

2. **Test login flow:**
   - Navigate to http://localhost:3000
   - Click sign-in button
   - Should redirect to Microsoft login
   - After successful login, should redirect back to app

## Step 5: Production Configuration

For production deployment:

1. **Update Redirect URIs:**
   - Add your production domain to redirect URIs
   - Remove localhost URIs

2. **Configure Custom Domain:**
   ```
   Redirect URI: https://policycortex.yourdomain.com
   Logout URI: https://policycortex.yourdomain.com
   ```

3. **Add to Azure Key Vault:**
   ```bash
   az keyvault secret set --vault-name "your-keyvault" --name "azure-client-id" --value "your-client-id"
   az keyvault secret set --vault-name "your-keyvault" --name "azure-tenant-id" --value "your-tenant-id"
   ```

## Step 6: Advanced Configuration (Optional)

### Multi-tenant Support:
```bash
# For multi-tenant scenarios
az ad app update --id $APP_ID --sign-in-audience "AzureADMultipleOrgs"
```

### Custom Scopes:
```bash
# Add custom API scopes for your backend
az ad app update --id $APP_ID \
  --identifier-uris "api://policycortex" \
  --app-roles '[{"allowedMemberTypes":["User"],"description":"Admin access","displayName":"Admin","id":"'$(uuidgen)'","isEnabled":true,"value":"Admin"}]'
```

### Group Claims:
- In Azure Portal → App Registration → Token configuration
- Add "groups" claim for ID and Access tokens

## Troubleshooting

### Common Issues:

1. **"AADSTS50011: Redirect URI mismatch"**
   - Ensure redirect URI in code matches Azure AD configuration exactly
   - Check for trailing slashes and http vs https

2. **"AADSTS700016: Application not found"**
   - Verify client ID is correct
   - Ensure app registration exists in correct tenant

3. **"AADSTS65001: User or administrator has not consented"**
   - Grant admin consent for API permissions
   - Or implement incremental consent in your app

4. **Token validation errors in backend:**
   - Verify JWT issuer and audience configuration
   - Check token signing key validation

### Debug Commands:
```bash
# Verify app registration
az ad app show --id $APP_ID

# Check permissions
az ad app permission list --id $APP_ID

# Verify service principal
az ad sp show --id $APP_ID
```

## Security Best Practices

1. **Use HTTPS in production**
2. **Implement proper token refresh**
3. **Store secrets in Key Vault**
4. **Enable conditional access policies**
5. **Monitor authentication logs**
6. **Implement role-based access control**

## Sample Configuration Files

### Frontend (.env.production):
```
VITE_AZURE_CLIENT_ID=12345678-1234-1234-1234-123456789012
VITE_AZURE_TENANT_ID=87654321-4321-4321-4321-210987654321
VITE_AZURE_REDIRECT_URI=https://policycortex.yourdomain.com
VITE_API_BASE_URL=https://api.policycortex.yourdomain.com/api
```

### Backend (docker-compose.yml):
```yaml
environment:
  - AZURE_CLIENT_ID=${AZURE_CLIENT_ID}
  - AZURE_TENANT_ID=${AZURE_TENANT_ID}
  - JWT_ISSUER=https://login.microsoftonline.com/${AZURE_TENANT_ID}/v2.0
  - JWT_AUDIENCE=${AZURE_CLIENT_ID}
```

After completing these steps, your PolicyCortex application will have proper Azure AD authentication configured!