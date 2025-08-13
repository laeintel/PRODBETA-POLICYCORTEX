# Azure AD Single-Page Application (SPA) Configuration

## Error Fix: AADSTS9002326
The error "Cross-origin token redemption is permitted only for the 'Single-Page Application' client-type" means your Azure AD app registration needs to be configured for SPA.

## Steps to Fix:

### 1. Azure Portal Configuration
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Azure Active Directory** → **App registrations**
3. Find your app: `PolicyCortex-Dev` (Client ID: 1ecc95d1-e5bb-43e2-9324-30a17cb6b01c)
4. Click on **Authentication** in the left menu

### 2. Configure as Single-Page Application
1. Under **Platform configurations**, click **+ Add a platform**
2. Select **Single-page application**
3. Add Redirect URIs:
   - `http://localhost:3000`
   - `http://localhost:3001`
   - `https://ca-cortex-frontend-dev.agreeableocean-dbcff600.eastus.azurecontainerapps.io`
4. Click **Configure**

### 3. Remove Web Platform (if exists)
1. If you have a "Web" platform configured, remove it
2. SPA and Web platforms cannot coexist for the same redirect URIs

### 4. Enable Required Settings
Under the SPA configuration, ensure:
- ✅ Access tokens (used for implicit flows)
- ✅ ID tokens (used for implicit and hybrid flows)

### 5. CORS Configuration (if needed)
In **Expose an API** section:
1. Add authorized client applications
2. Add your frontend URLs to allowed origins

### 6. API Permissions
Ensure these permissions are granted:
- Microsoft Graph:
  - User.Read (Delegated)
  - Directory.Read.All (Delegated)
  - Policy.Read.All (Delegated)
- Azure Service Management:
  - user_impersonation (Delegated)

## Local Development Configuration

### Environment Variables (.env.local)
```env
NEXT_PUBLIC_AZURE_CLIENT_ID=1ecc95d1-e5bb-43e2-9324-30a17cb6b01c
NEXT_PUBLIC_AZURE_TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7
NEXT_PUBLIC_MSAL_REDIRECT_URI=http://localhost:3000
```

## Testing the Fix
1. Clear browser cache and cookies
2. Open Developer Tools → Application → Clear Storage
3. Try logging in again

## Alternative: Use Device Code Flow (for CLI/Backend)
If you need backend authentication, use device code flow instead:
```bash
az login --tenant 9ef5b184-d371-462a-bc75-5024ce8baff7
```

## Common Issues
- **Still getting AADSTS9002326**: Make sure to remove Web platform and only use SPA
- **Token not working**: Clear all browser storage and re-authenticate
- **CORS errors**: Add your domain to Azure AD app's exposed API settings