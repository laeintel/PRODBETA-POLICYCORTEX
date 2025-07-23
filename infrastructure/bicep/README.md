# Policortex Bicep Infrastructure

This directory contains the complete Bicep infrastructure for Policortex, providing an alternative to the Terraform implementation with identical resource configurations.

## Overview

The Bicep templates deploy a comprehensive Azure infrastructure including:

- **Core Infrastructure**: Resource Groups, Storage Account, Key Vault, Container Registry
- **Networking**: Virtual Network with subnets, NSGs, Route Tables, Private DNS Zones
- **Data Services**: Cosmos DB, Redis Cache, SQL Server (optional), Private Endpoints
- **AI Services**: Cognitive Services, Azure OpenAI, ML Workspace (optional), EventGrid
- **Container Platform**: Container Apps Environment with auto-scaling capabilities
- **Monitoring**: Log Analytics, Application Insights, Alerts, Dashboards, Budgets

## Structure

```
bicep/
├── main.bicep                 # Main template (subscription scope)
├── modules/                   # Reusable Bicep modules
│   ├── storage.bicep
│   ├── container-registry.bicep
│   ├── key-vault.bicep
│   ├── log-analytics.bicep
│   ├── application-insights.bicep
│   ├── user-identity.bicep
│   ├── networking.bicep
│   ├── data-services.bicep
│   ├── ai-services.bicep
│   ├── container-apps-environment.bicep
│   ├── container-apps.bicep
│   ├── key-vault-secrets.bicep
│   └── monitoring.bicep
├── environments/              # Environment-specific parameters
│   ├── dev.bicepparam
│   ├── staging.bicepparam
│   └── prod.bicepparam
└── README.md
```

## Key Features

### 🔄 **Terraform Compatibility**
- **Identical Resources**: Same resource names, SKUs, and configurations as Terraform
- **Same Naming Convention**: `policycortex-{service}-{environment}` pattern
- **Matching Outputs**: Compatible with existing automation and scripts

### 🏗️ **Modular Architecture**
- **Reusable Modules**: Each service is a separate, testable module
- **Environment Separation**: Dev, staging, and production parameter files
- **Conditional Deployment**: Optional components based on environment needs

### 🔐 **Security First**
- **Private Endpoints**: All data services use private connectivity
- **Key Vault Integration**: Secrets management with managed identity access
- **Network Isolation**: VNet integration with proper subnet segmentation
- **RBAC**: Role-based access control throughout

### 📊 **Comprehensive Monitoring**
- **Observability**: Log Analytics, Application Insights, custom dashboards
- **Alerting**: Email notifications for critical and warning events
- **Cost Management**: Budget alerts and spending monitoring
- **Health Checks**: Container app health probes and scaling rules

## Quick Start

### Prerequisites

1. **Azure CLI** with Bicep extension:
   ```bash
   az bicep install
   ```

2. **Azure Subscription** with appropriate permissions

3. **Service Principal** for GitHub Actions (same as Terraform)

### Local Deployment

1. **Validate Templates**:
   ```bash
   az bicep build --file main.bicep
   ```

2. **Deploy to Dev**:
   ```bash
   az deployment sub create \
     --location "East US" \
     --template-file main.bicep \
     --parameters environments/dev.bicepparam \
     --name "policycortex-dev-$(date +%Y%m%d-%H%M%S)"
   ```

3. **Check Deployment**:
   ```bash
   az deployment sub show --name "policycortex-dev-YYYYMMDD-HHMMSS"
   ```

### GitHub Actions Deployment

The Bicep infrastructure includes its own GitHub Actions workflow (`.github/workflows/bicep-deploy.yml`):

1. **Automatic Validation**: On every push and PR
2. **Dev Deployment**: On main branch pushes
3. **Manual Deployment**: Workflow dispatch for staging/prod
4. **Environment Protection**: Required approvals for production

#### Workflow Triggers

- **Push to main**: Deploys to dev environment
- **Pull Request**: Validates templates
- **Manual Dispatch**: Deploy to any environment with approval

#### Required Secrets

Same as Terraform workflow:
- `AZURE_CREDENTIALS`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`
- `AZURE_SUBSCRIPTION_ID`
- `AZURE_TENANT_ID`
- `JWT_SECRET_KEY_DEV`
- `JWT_SECRET_KEY_STAGING`
- `JWT_SECRET_KEY_PROD`

## Environment Configuration

### Development (`dev.bicepparam`)
- **Minimal Resources**: Basic SKUs for cost optimization
- **No ML Workspace**: Disabled to avoid conflicts
- **No OpenAI**: Cost savings for development
- **Basic Redis**: Single node configuration

### Staging (`staging.bicepparam`)
- **Production-like**: Similar to prod but smaller scale
- **Full Feature Set**: All services enabled
- **Moderate Scale**: 2-3 replicas, smaller VM sizes

### Production (`prod.bicepparam`)
- **High Availability**: Premium SKUs and redundancy
- **Auto-scaling**: Higher replica counts and thresholds
- **Enhanced Security**: Stronger consistency, premium caching
- **Comprehensive Monitoring**: Multiple alert recipients

## Resource Mapping

| Service | Terraform Resource | Bicep Resource | Notes |
|---------|-------------------|----------------|-------|
| Resource Groups | `azurerm_resource_group` | `Microsoft.Resources/resourceGroups` | ✅ Identical |
| Storage Account | `azurerm_storage_account` | `Microsoft.Storage/storageAccounts` | ✅ Same configuration |
| Container Registry | `azurerm_container_registry` | `Microsoft.ContainerRegistry/registries` | ✅ Premium SKU |
| Key Vault | `azurerm_key_vault` | `Microsoft.KeyVault/vaults` | ✅ Same access policies |
| Virtual Network | `azurerm_virtual_network` | `Microsoft.Network/virtualNetworks` | ✅ Same subnets |
| Cosmos DB | `azurerm_cosmosdb_account` | `Microsoft.DocumentDB/databaseAccounts` | ✅ Same consistency |
| Redis Cache | `azurerm_redis_cache` | `Microsoft.Cache/redis` | ✅ Same SKU/capacity |
| Cognitive Services | `azurerm_cognitive_account` | `Microsoft.CognitiveServices/accounts` | ✅ Same endpoints |
| Container Apps | `azurerm_container_app` | `Microsoft.App/containerApps` | ✅ Same scaling rules |

## Advantages Over Terraform

### 🚀 **Native Azure Integration**
- **First-party Support**: Direct Microsoft support and updates
- **Latest Features**: Immediate access to new Azure capabilities
- **ARM Template Compatibility**: Leverages proven ARM foundation

### 📝 **Developer Experience**
- **Intellisense**: Full VS Code integration with auto-completion
- **Type Safety**: Strong typing prevents configuration errors
- **Validation**: Built-in linting and validation tools

### 🔄 **Deployment Reliability**
- **Idempotent**: True idempotency without state file complexities
- **Incremental**: Only deploys what changed
- **Rollback**: Native Azure rollback capabilities

### 🎯 **Simplified State Management**
- **No State Files**: Azure Resource Manager handles state
- **No Import Issues**: Resources are naturally managed
- **Team Collaboration**: No state file conflicts

## Migration from Terraform

If you're migrating from Terraform:

1. **Deploy Bicep in Parallel**: Test in a separate subscription first
2. **Validate Resources**: Ensure all resources match exactly
3. **Export Data**: Backup any stateful data before switching
4. **Update Pipelines**: Switch workflow triggers when ready
5. **Clean Up**: Remove Terraform state and files

## Troubleshooting

### Common Issues

**Deployment Fails with "Resource Already Exists"**
- Check if resources exist from previous deployments
- Use `az resource list` to check existing resources
- Delete conflicting resources or use different names

**Key Vault Access Denied**
- Verify service principal has Key Vault permissions
- Check if access policies are correctly configured
- Ensure managed identity is properly assigned

**Container Apps Won't Start**
- Verify container images exist in registry
- Check Key Vault secret references
- Review application logs in Log Analytics

### Useful Commands

```bash
# Validate all templates
for file in modules/*.bicep; do az bicep build --file "$file"; done

# Check deployment status
az deployment sub list --query "[].{name:name, state:properties.provisioningState}"

# Get deployment outputs
az deployment sub show --name "deployment-name" --query "properties.outputs"

# Delete all resources
az group delete --name "rg-policycortex-app-dev" --yes
az group delete --name "rg-policycortex-network-dev" --yes
```

## Support

For issues with the Bicep infrastructure:

1. **Check Azure Activity Log**: Review deployment errors
2. **Validate Templates**: Run `az bicep build` locally
3. **Review Parameters**: Ensure environment files are correct
4. **Check Permissions**: Verify service principal access

The Bicep templates are designed to be a drop-in replacement for Terraform while providing better integration with Azure and simplified state management.