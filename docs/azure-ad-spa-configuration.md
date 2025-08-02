# Azure AD Single-Page Application Configuration

## Problem
Your Azure AD app registration is currently configured for a different client type (likely "Web" with a client secret), but Single-Page Applications require a specific configuration without secrets.

## Solution Steps

### 1. Go to Azure Portal
Navigate to: https://portal.azure.com

### 2. Find Your App Registration
- Go to "Azure Active Directory" → "App registrations"
- Find your app: **PolicyCortex** (Application ID: 9ff9bdf6-a2b2-4b39-a270-71a9b8b9178d)

### 3. Configure Authentication for SPA

1. Click on **Authentication** in the left menu

2. Under **Platform configurations**, you need to:
   - Delete any existing "Web" platform configuration
   - Click **"+ Add a platform"**
   - Choose **"Single-page application"**
   - Add redirect URIs:
     - `http://localhost:3000`
     - `http://localhost:3001`
     - `http://localhost:3002`
     - `http://localhost:5173` (for Vite dev server)

3. Under **Implicit grant and hybrid flows**:
   - ✅ Check "Access tokens"
   - ✅ Check "ID tokens"

4. Under **Supported account types**:
   - Keep as "Accounts in this organizational directory only"

5. Click **Save**

### 4. Verify Configuration

After saving, verify:
- Platform type shows as "Single-page application"
- No client secrets are configured (SPAs don't use secrets)
- Redirect URIs are properly set
- Token configuration is enabled

### 5. Clear Browser Cache

After making these changes:
1. Clear your browser cache and cookies for localhost
2. Close all browser tabs
3. Restart the frontend application

## Why This Error Occurs

SPAs use the Authorization Code Flow with PKCE (Proof Key for Code Exchange) which:
- Doesn't require a client secret
- Uses a different token endpoint configuration
- Requires specific CORS settings for cross-origin requests

The error indicates your app is currently configured as a traditional web app that expects server-side token exchange, not a browser-based SPA.

## Alternative: Using Azure CLI

If you prefer command line, you can update the app registration:

```bash
# Update to SPA type
az ad app update --id 9ff9bdf6-a2b2-4b39-a270-71a9b8b9178d \
  --spa-redirect-uris "http://localhost:3000" "http://localhost:3001" "http://localhost:3002" "http://localhost:5173"

# Remove web redirect URIs if any exist
az ad app update --id 9ff9bdf6-a2b2-4b39-a270-71a9b8b9178d \
  --web-redirect-uris []

# Enable token issuance
az ad app update --id 9ff9bdf6-a2b2-4b39-a270-71a9b8b9178d \
  --enable-id-token-issuance true \
  --enable-access-token-issuance true
```

## Testing After Configuration

Once configured correctly, the authentication flow should:
1. Open Azure AD login page
2. Allow you to sign in with your organizational account
3. Redirect back to http://localhost:3000 with tokens
4. Successfully authenticate in the PolicyCortex app