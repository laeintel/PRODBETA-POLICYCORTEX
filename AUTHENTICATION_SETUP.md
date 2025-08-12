# PolicyCortex v2 Authentication Setup Guide

## Overview
PolicyCortex v2 uses Azure Active Directory (Azure AD) for authentication. This guide will help you properly configure authentication to prevent unauthorized access.

## Current Issue Resolution
The application currently allows users to bypass the login screen. This guide addresses that security issue by implementing proper authentication enforcement.

## Prerequisites
- Azure subscription with Azure AD access
- Admin permissions to create App Registrations in Azure AD
- Access to modify environment variables

## Step 1: Azure AD App Registration

1. **Navigate to Azure Portal**
   - Go to [Azure Portal](https://portal.azure.com)
   - Navigate to `Azure Active Directory` > `App registrations`

2. **Create New Registration**
   - Click `New registration`
   - Name: `PolicyCortex v2`
   - Supported account types: `Single tenant` (recommended for organization use)
   - Redirect URI:
     - Platform: `Single-page application`
     - URI: `http://localhost:3000` (for development)
     - Add production URI later: `https://your-domain.com`

3. **Note Important IDs**
   After creation, note these values:
   - Application (client) ID: `1ecc95d1-e5bb-43e2-9324-30a17cb6b01c` (example)
   - Directory (tenant) ID: `9ef5b184-d371-462a-bc75-5024ce8baff7` (example)

## Step 2: Configure API Permissions

1. **Navigate to API Permissions**
   In your App Registration, go to `API permissions`

2. **Add Microsoft Graph Permissions**
   Click `Add a permission` > `Microsoft Graph` > `Delegated permissions`:
   - `User.Read` - Sign in and read user profile
   - `Directory.Read.All` - Read directory data
   - `Policy.Read.All` - Read organization policies
   - `GroupMember.Read.All` - Read group memberships
   - `Organization.Read.All` - Read organization information

3. **Add Azure Service Management**
   Click `Add a permission` > `Azure Service Management` > `Delegated permissions`:
   - `user_impersonation` - Access Azure resources

4. **Grant Admin Consent**
   Click `Grant admin consent for [Your Organization]`

## Step 3: Create Client Secret (for Backend)

1. **Navigate to Certificates & Secrets**
   In your App Registration, go to `Certificates & secrets`

2. **Create New Secret**
   - Click `New client secret`
   - Description: `PolicyCortex Backend`
   - Expires: Choose appropriate expiration
   - **IMPORTANT**: Copy the secret value immediately (you won't see it again)

## Step 4: Expose an API (for Backend Authentication)

1. **Navigate to Expose an API**
   In your App Registration, go to `Expose an API`

2. **Set Application ID URI**
   - Click `Set` next to Application ID URI
   - Accept the default: `api://[your-client-id]`

3. **Add a Scope**
   - Click `Add a scope`
   - Scope name: `access_as_user`
   - Who can consent: `Admins and users`
   - Admin consent display name: `Access PolicyCortex API`
   - Admin consent description: `Allows the app to access PolicyCortex API on behalf of the signed-in user`
   - State: `Enabled`

## Step 5: Configure Environment Variables

1. **Copy the Example File**
   ```bash
   cp .env.example .env.local  # For frontend
   cp .env.example .env        # For backend
   ```

2. **Update Frontend Configuration** (`.env.local` or `frontend/.env.local`)
   ```env
   # Required for authentication
   NEXT_PUBLIC_AZURE_CLIENT_ID=your-client-id-here
   NEXT_PUBLIC_AZURE_TENANT_ID=your-tenant-id-here
   NEXT_PUBLIC_MSAL_REDIRECT_URI=http://localhost:3000
   NEXT_PUBLIC_MSAL_POST_LOGOUT_REDIRECT_URI=http://localhost:3000
   NEXT_PUBLIC_CORE_API_SCOPE=api://your-client-id-here/access_as_user
   ```

3. **Update Backend Configuration** (`.env` or `core/.env`)
   ```env
   # Required for JWT validation
   AZURE_TENANT_ID=your-tenant-id-here
   AZURE_CLIENT_ID=your-client-id-here
   AZURE_CLIENT_SECRET=your-client-secret-here
   AZURE_SUBSCRIPTION_ID=your-subscription-id-here
   
   # Authentication enforcement
   REQUIRE_AUTH=true
   ALLOW_ANY_AUDIENCE=false
   ```

## Step 6: Authentication Enforcement

The application now includes several layers of authentication enforcement:

### Frontend Protection
1. **Middleware** (`frontend/middleware.ts`)
   - Protects routes at the Next.js middleware level
   - Redirects unauthenticated users to login

2. **AuthGuard Component** (`frontend/components/AuthGuard.tsx`)
   - Wraps protected components
   - Shows login prompt for unauthenticated users
   - Prevents access to dashboard without authentication

3. **Context Provider** (`frontend/contexts/AuthContext.tsx`)
   - Manages authentication state
   - Handles login/logout flows
   - Provides access tokens for API calls

### Backend Protection
1. **JWT Validation** (`core/src/auth.rs`)
   - Validates Azure AD tokens
   - Extracts user claims
   - Enforces authentication when `REQUIRE_AUTH=true`

2. **API Endpoints** (`core/src/api/mod.rs`)
   - Uses `AuthUser` extractor for protected endpoints
   - Returns 401 Unauthorized for invalid/missing tokens

## Step 7: Testing Authentication

1. **Start the Application**
   ```bash
   # Terminal 1: Backend
   cd core
   cargo run

   # Terminal 2: Frontend
   cd frontend
   npm run dev
   ```

2. **Verify Authentication Flow**
   - Navigate to http://localhost:3000
   - Click "Get Started" or try to access `/dashboard`
   - You should see the authentication prompt
   - Login with your Azure AD credentials
   - Verify you can access the dashboard after authentication
   - Try closing the login popup - you should NOT be able to access the dashboard

3. **Test Protected APIs**
   ```bash
   # Without token (should fail)
   curl http://localhost:8080/api/v1/metrics
   
   # With token (should succeed)
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8080/api/v1/metrics
   ```

## Step 8: Production Deployment

1. **Update Redirect URIs**
   In Azure AD App Registration, add your production URLs:
   - `https://your-domain.com`
   - `https://your-domain.com/auth/callback` (if using callback)

2. **Use Azure Key Vault**
   Store sensitive values in Azure Key Vault:
   ```bash
   az keyvault secret set --vault-name your-vault --name azure-client-secret --value "your-secret"
   ```

3. **Enable Managed Identity**
   When running in Azure, use Managed Identity instead of client secrets

4. **Configure CORS**
   Ensure your backend allows requests from your frontend domain

## Troubleshooting

### Issue: "AADSTS50011: Reply URL mismatch"
**Solution**: Ensure redirect URI in Azure AD matches exactly with your application URL

### Issue: "AADSTS700016: Application not found"
**Solution**: Check that AZURE_CLIENT_ID and AZURE_TENANT_ID are correct

### Issue: "401 Unauthorized on API calls"
**Solution**: 
- Ensure token is being sent in Authorization header
- Check token expiration
- Verify REQUIRE_AUTH setting in backend

### Issue: "Login popup blocked"
**Solution**: 
- Allow popups for your domain
- Consider using redirect flow instead of popup

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use environment-specific** configurations
3. **Enable MFA** for all admin accounts
4. **Regularly rotate** client secrets
5. **Monitor sign-in** logs in Azure AD
6. **Use Conditional Access** policies for additional security
7. **Implement proper RBAC** within the application
8. **Enable audit logging** for all authentication events

## Additional Resources

- [Azure AD Documentation](https://docs.microsoft.com/en-us/azure/active-directory/)
- [MSAL.js Documentation](https://github.com/AzureAD/microsoft-authentication-library-for-js)
- [Next.js Authentication Patterns](https://nextjs.org/docs/authentication)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)

## Support

If you encounter issues with authentication setup:
1. Check the browser console for error messages
2. Review Azure AD sign-in logs
3. Verify all environment variables are set correctly
4. Ensure Azure AD permissions are granted with admin consent