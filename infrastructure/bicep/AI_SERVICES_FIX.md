# AI Services Deployment Issue Fix

## Problem

The deployment is failing with the error:
```
"UpdatingCustomDomainNotAllowed","message":"Updating or disabling sub domain is not supported."
```

This occurs when:
1. An AI Services (Cognitive Services or OpenAI) resource already exists in Azure
2. The resource has a custom domain configured
3. The Bicep deployment tries to update the resource, which isn't allowed for custom domains

## Solutions

### Option 1: Check and Clean Existing Resources (Recommended)

Run the diagnostic script to check for existing resources:

```powershell
# Check existing resources
.\infrastructure\bicep\scripts\fix-ai-services-deployment.ps1 -Environment dev -ListOnly

# If resources exist with custom domains, delete them
.\infrastructure\bicep\scripts\fix-ai-services-deployment.ps1 -Environment dev -DeleteExisting
```

Then redeploy the infrastructure through GitHub Actions or manually.

### Option 2: Use the Safe AI Services Module

Replace the AI Services module reference in `main.bicep` to use the safe version that handles existing resources:

```bicep
// In main.bicep, replace:
module aiServices 'modules/ai-services.bicep' = {
  // ...
}

// With:
module aiServices 'modules/ai-services-safe.bicep' = {
  name: 'aiServices'
  scope: appResourceGroup
  params: {
    // ... existing params ...
    useExistingCognitiveServices: true  // Set to true if resource exists
    useExistingOpenAI: true             // Set to true if resource exists
  }
}
```

### Option 3: Manual Cleanup via Azure Portal

1. Navigate to the Azure Portal
2. Go to Resource Groups > `rg-policycortex-app-dev`
3. Find and delete:
   - `policycortex-cognitive-dev` (Cognitive Services)
   - `policycortex-openai-dev` (OpenAI Service)
4. Wait for deletion to complete
5. Redeploy the infrastructure

## Prevention

To prevent this issue in the future:

1. **Don't manually configure custom domains** on AI Services resources managed by Bicep
2. **Use consistent deployment methods** - either always use Bicep or always use Portal/CLI
3. **Document any manual changes** made to resources outside of Infrastructure as Code

## GitHub Actions Environment Setup

The workflow also requires GitHub environments to be configured. In your repository settings:

1. Go to Settings > Environments
2. Create three environments:
   - `dev`
   - `staging`
   - `prod`
3. Add required secrets to each environment:
   - `JWT_SECRET_KEY_DEV` (for dev)
   - `JWT_SECRET_KEY_STAGING` (for staging)
   - `JWT_SECRET_KEY_PROD` (for prod)

## Additional Notes

- The error tracking ID `9f67a715-4ac4-4f65-b673-f7771e1d4aa4` can be used to find more details in Azure Activity Logs
- Custom domains on AI Services are immutable once set
- Private endpoints may also need to be recreated if the AI Services resources are deleted 