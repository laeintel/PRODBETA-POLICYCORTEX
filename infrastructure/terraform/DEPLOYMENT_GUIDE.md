# Azure Free Tier Infrastructure Deployment Guide

## Overview
This Terraform configuration is optimized to maximize usage of Azure's free tier resources, minimizing costs while providing a complete PolicyCortex infrastructure.

## Free Tier Resources Utilized

### 🆓 12-Month Free Services Used:
1. **Virtual Machine (B1s)** - 750 hours/month
   - 1 vCPU, 1GB RAM
   - Only deployed in DEV environment to stay within limits
   
2. **PostgreSQL Flexible Server (B1MS)** - 750 hours/month
   - 1 vCore, 2GB RAM, 32GB storage
   - Used as primary database
   
3. **Cosmos DB** - Free tier
   - 25GB storage, 1000 RU/s
   - Used for document storage and caching
   
4. **Storage Account**
   - 5GB Hot Blob storage
   - 100GB File storage
   - LRS replication (cheapest)
   
5. **Public IP** - 1500 hours/month
   - Basic SKU (cheaper than Standard)
   
6. **Application Insights** - 5GB/month
   - Configured with 50% sampling to stay under limit

### 💰 Cost-Optimized Resources:
- **Container Registry**: Basic tier (cheapest)
- **Key Vault**: Standard tier (minimal cost)
- **Virtual Network & Subnets**: Free
- **Service Bus**: Basic tier (prod only)

## Prerequisites

1. **Azure CLI** installed and configured
2. **Terraform** >= 1.6.0
3. **Service Principal** with Contributor role
4. **GitHub Secrets** configured:
   - AZURE_CLIENT_ID
   - AZURE_TENANT_ID
   - AZURE_SUBSCRIPTION_ID

## Deployment Steps

### 1. Initialize Terraform Backend (separate bootstrap)

```bash
cd infrastructure/terraform

# For Development
terraform init \
  -backend-config="resource_group_name=tfstate-pcx-dev" \
  -backend-config="storage_account_name=pcxtfdev<hash>" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=dev.tfstate" \
  -backend-config="use_azuread_auth=true"

# For Production
terraform init \
  -backend-config="resource_group_name=tfstate-pcx-prod" \
  -backend-config="storage_account_name=pcxtfprod<hash>" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=prod.tfstate" \
  -backend-config="use_azuread_auth=true"
```

### 2. Plan Deployment

```bash
# Development
terraform plan -var-file=environments/dev/terraform.tfvars -out=dev.tfplan

# Production
terraform plan -var-file=environments/prod/terraform.tfvars -out=prod.tfplan
```

### 3. Apply Configuration

```bash
# Development
terraform apply dev.tfplan

# Production
terraform apply prod.tfplan
```

### 4. Generate SSH Key (for VM access)

```bash
# Generate SSH key if not exists
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

## Resource Limits & Cost Management

### Monthly Free Tier Limits:
| Resource | Free Limit | Our Usage | Status |
|----------|------------|-----------|--------|
| B1s VM | 750 hours | Dev only | ✅ Within limit |
| PostgreSQL B1MS | 750 hours | 24/7 | ✅ Within limit |
| Cosmos DB | 1000 RU/s | 400 RU/s | ✅ Within limit |
| Storage Hot | 5GB | < 1GB | ✅ Within limit |
| Storage Files | 100GB | < 10GB | ✅ Within limit |
| Public IP | 1500 hours | Dev only | ✅ Within limit |
| App Insights | 5GB | 0.5GB cap | ✅ Within limit |

### Cost Optimization Strategies:
1. **VM deployed only in DEV** - Saves 750 hours for testing
2. **PostgreSQL B1MS** - Free tier with burst performance
3. **Cosmos DB free tier** - 25GB storage included
4. **Storage LRS** - Locally redundant (cheapest option)
5. **Application Insights sampling** - 50% to reduce data ingestion
6. **Container Registry Basic** - Sufficient for our needs
7. **No Load Balancer** - Use Public IP directly to avoid charges

## Environment Variables

After deployment, configure your applications with:

```bash
# Database
export POSTGRES_HOST=$(terraform output -raw postgresql_fqdn)
export POSTGRES_DB=policycortex
export POSTGRES_USER=pcxadmin
export POSTGRES_PASSWORD=$(az keyvault secret show --vault-name <kv-name> --name postgres-password --query value -o tsv)

# Cosmos DB
export COSMOS_ENDPOINT=$(terraform output -raw cosmosdb_endpoint)
export COSMOS_KEY=$(az cosmosdb keys list --name <cosmos-name> --resource-group <rg-name> --query primaryMasterKey -o tsv)

# Container Registry
export ACR_SERVER=$(terraform output -raw container_registry_login_server)
export ACR_USERNAME=$(az acr credential show --name <acr-name> --query username -o tsv)
export ACR_PASSWORD=$(az acr credential show --name <acr-name> --query passwords[0].value -o tsv)

# Application Insights
export APPINSIGHTS_KEY=$(terraform output -raw application_insights_instrumentation_key)
```

## Monitoring Costs

### Check Current Usage:
```bash
# Check current month's cost
az consumption usage list \
  --subscription $(az account show --query id -o tsv) \
  --start-date $(date -u -d "$(date +%Y-%m-01)" '+%Y-%m-%d') \
  --end-date $(date -u '+%Y-%m-%d') \
  --query "[?contains(instanceName, 'pcx')].{Resource:instanceName, Cost:pretaxCost}" \
  -o table
```

### Set Up Cost Alerts:
```bash
# Create budget with alert
az consumption budget create \
  --amount 10 \
  --budget-name "PolicyCortex-FreeTier" \
  --category Cost \
  --time-grain Monthly \
  --start-date $(date -u -d "$(date +%Y-%m-01)" '+%Y-%m-%d') \
  --end-date $(date -u -d "+1 year" '+%Y-%m-%d') \
  --subscription $(az account show --query id -o tsv)
```

## Cleanup Resources

To avoid any charges:

```bash
# Destroy specific environment
terraform destroy -var-file=environments/dev/terraform.tfvars -auto-approve

# Or destroy everything
terraform destroy -auto-approve
```

## Troubleshooting

### Common Issues:

1. **"Subscription not found" error**
   - Ensure Service Principal has Contributor role
   - Run: `az role assignment create --assignee <CLIENT_ID> --role Contributor --scope /subscriptions/<SUBSCRIPTION_ID>`

2. **"Free tier already in use" for Cosmos DB**
   - Only one free Cosmos DB account per subscription
   - Check existing resources: `az cosmosdb list --query "[?enableFreeTier==true].name"`

3. **VM quota exceeded**
   - Check quota: `az vm list-usage --location eastus --query "[?name.value=='standardBSFamily'].{Name:name.localizedValue, Current:currentValue, Limit:limit}"`
   - Request increase if needed

## GitHub Actions Deployment

The infrastructure can also be deployed via GitHub Actions:

```yaml
# Trigger deployment
gh workflow run azure-infra.yml -f environment=dev -f apply=true
```

## Security Best Practices

1. **Never commit secrets** - Use Key Vault for all sensitive data
2. **Enable RBAC** - All resources use Azure RBAC for access control
3. **Network isolation** - Resources are in private subnets where possible
4. **Firewall rules** - PostgreSQL only allows Azure services by default
5. **Managed identities** - Use for service-to-service authentication

## Support

For issues or questions:
- Check Azure Portal for resource status
- Review Terraform state: `terraform show`
- Check logs: `terraform output -json | jq`
- GitHub Issues: https://github.com/laeintel/policycortex/issues