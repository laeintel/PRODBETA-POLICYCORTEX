# GitHub Actions Secrets Setup Guide

This guide explains how to configure the required secrets for the PolicyCortex CI/CD pipeline.

## Required Secrets

### 1. AZURE_CREDENTIALS (Required)
This is the main authentication secret for Azure. It should contain a JSON object with service principal credentials.

**Format:**
```json
{
  "clientId": "YOUR_CLIENT_ID",
  "clientSecret": "YOUR_CLIENT_SECRET",
  "subscriptionId": "YOUR_SUBSCRIPTION_ID",
  "tenantId": "YOUR_TENANT_ID"
}
```

**How to create:**
```bash
# Create a service principal
az ad sp create-for-rbac --name "github-actions-policycortex" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID \
  --sdk-auth
```

Copy the entire JSON output and add it as the `AZURE_CREDENTIALS` secret.

### 2. Optional Environment-Specific Secrets

These are optional and will use defaults if not provided:

- `AZURE_SUBSCRIPTION_ID_DEV` - Development subscription ID (defaults to main subscription)
- `AZURE_SUBSCRIPTION_ID_PROD` - Production subscription ID (defaults to main subscription)
- `NEXT_PUBLIC_GRAPHQL_ENDPOINT_DEV` - GraphQL endpoint for dev environment
- `NEXT_PUBLIC_GRAPHQL_ENDPOINT_PROD` - GraphQL endpoint for prod environment
- `NEXT_PUBLIC_API_URL_DEV` - API URL for dev environment
- `NEXT_PUBLIC_API_URL_PROD` - API URL for prod environment
- `NEXT_PUBLIC_AZURE_CLIENT_ID_DEV` - Azure AD app client ID for dev
- `NEXT_PUBLIC_AZURE_CLIENT_ID_PROD` - Azure AD app client ID for prod
- `NEXT_PUBLIC_AZURE_TENANT_ID_DEV` - Azure AD tenant ID for dev
- `NEXT_PUBLIC_AZURE_TENANT_ID_PROD` - Azure AD tenant ID for prod
- `NEXT_PUBLIC_AZURE_REDIRECT_URI_DEV` - OAuth redirect URI for dev
- `NEXT_PUBLIC_AZURE_REDIRECT_URI_PROD` - OAuth redirect URI for prod
- `NEXT_PUBLIC_AZURE_SCOPES_DEV` - OAuth scopes for dev
- `NEXT_PUBLIC_AZURE_SCOPES_PROD` - OAuth scopes for prod

## How to Add Secrets to GitHub

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add the secret name and value
6. Click **Add secret**

## Verifying Secrets

After adding the secrets, you can verify they're working by:

1. Triggering a workflow run manually:
   ```bash
   gh workflow run application.yml
   ```

2. Checking the workflow logs for successful Azure login:
   - The "Azure Login" step should show "Login successful"
   - If it fails, check that the AZURE_CREDENTIALS JSON is properly formatted

## Troubleshooting

### Error: "Resource not accessible by integration"
This occurs when the GitHub token doesn't have sufficient permissions. The workflow has been updated to handle this.

### Error: "Login failed with Error: Using auth-type: SERVICE_PRINCIPAL"
This means the AZURE_CREDENTIALS secret is missing or improperly formatted. Double-check the JSON structure.

### Error: "Docker is not available on this self-hosted runner"
This is expected if Docker isn't installed on the self-hosted runner. The workflow will skip Docker-related steps.

## Security Best Practices

1. **Limit Service Principal Scope**: Only grant the minimum required permissions
2. **Rotate Secrets Regularly**: Update the service principal secret every 90 days
3. **Use Environment Protection**: For production deployments, enable environment protection rules
4. **Audit Access**: Regularly review who has access to secrets

## Contact

For issues with the CI/CD pipeline, please open an issue in the repository.