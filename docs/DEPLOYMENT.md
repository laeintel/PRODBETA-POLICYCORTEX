# PolicyCortex Deployment Guide

## Prerequisites

1. Azure CLI installed and configured
2. Azure subscription with appropriate permissions
3. GitHub repository with Actions enabled

## Azure Resource Setup

### Step 1: Create Azure Resources

Run the setup script to create all required Azure resources:

```bash
# Make script executable
chmod +x scripts/create-azure-resources.sh

# Run the script
./scripts/create-azure-resources.sh
```

This script will create:
- Resource Group: `policycortex-rg`
- App Service Plan: `policycortex-plan`
- Frontend Web App: `policycortex`
- Backend API Web App: `policycortex-api`

### Step 2: Configure GitHub Secrets

1. Create a service principal for GitHub Actions:
```bash
az ad sp create-for-rbac \
  --name "policycortex-github" \
  --role contributor \
  --scopes /subscriptions/{your-subscription-id}/resourceGroups/policycortex-rg \
  --sdk-auth
```

2. Copy the JSON output and add it as a GitHub secret:
   - Go to your GitHub repository
   - Navigate to Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `AZURE_CREDENTIALS`
   - Value: Paste the JSON output from the command above

3. Add your Azure Subscription ID as another secret:
   - Name: `AZURE_SUBSCRIPTION_ID`
   - Value: Your Azure subscription ID (get it with `az account show --query id -o tsv`)

## Deployment Process

### Automatic Deployment (CI/CD)

The GitHub Actions workflow (`azure-deploy.yml`) will automatically deploy on:
- Push to the `main` branch
- Manual workflow dispatch

The deployment process:
1. Builds frontend (Next.js)
2. Builds backend services
3. Runs tests
4. Creates deployment package
5. Deploys to Azure Web Apps
6. Configures app settings

### Manual Deployment

If you need to deploy manually:

```bash
# Build frontend
cd frontend
npm ci
npm run build

# Deploy to Azure
az webapp deploy \
  --name policycortex \
  --resource-group policycortex-rg \
  --src-path .next \
  --type zip
```

## Environment Configuration

### Required Environment Variables

#### Frontend (`policycortex` Web App):
- `NEXT_PUBLIC_DEMO_MODE`: Set to `false` for production
- `NEXT_PUBLIC_API_URL`: https://policycortex.azurewebsites.net
- `NEXT_PUBLIC_REAL_API_BASE`: https://policycortex-api.azurewebsites.net
- `USE_REAL_DATA`: Set to `true` for production

#### Backend API (`policycortex-api` Web App):
- `NODE_ENV`: production
- `PORT`: 8080
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID
- `DATABASE_URL`: PostgreSQL connection string (if using database)
- `REDIS_URL`: Redis connection string (if using cache)

## Monitoring and Troubleshooting

### View Logs

```bash
# View frontend logs
az webapp log tail --name policycortex --resource-group policycortex-rg

# View API logs  
az webapp log tail --name policycortex-api --resource-group policycortex-rg
```

### Check Deployment Status

```bash
# Check frontend deployment
az webapp deployment list --name policycortex --resource-group policycortex-rg

# Check API deployment
az webapp deployment list --name policycortex-api --resource-group policycortex-rg
```

### Common Issues

1. **"Resource doesn't exist" error**: Run the `create-azure-resources.sh` script first
2. **Authentication failures**: Verify AZURE_CREDENTIALS secret is correctly formatted
3. **Build failures**: Check that all dependencies are listed in package.json
4. **Runtime errors**: Check application logs using `az webapp log tail`

## Production Checklist

- [ ] Azure resources created
- [ ] GitHub secrets configured
- [ ] DEMO_MODE set to false
- [ ] USE_REAL_DATA set to true  
- [ ] SSL certificates configured
- [ ] Custom domain configured (optional)
- [ ] Application Insights enabled
- [ ] Database backup configured
- [ ] Scaling rules configured
- [ ] Security headers configured

## URLs

After deployment, your application will be available at:
- Frontend: https://policycortex.azurewebsites.net
- API: https://policycortex-api.azurewebsites.net

## Support

For issues with deployment, check:
1. GitHub Actions logs
2. Azure Web App logs
3. Application Insights (if configured)