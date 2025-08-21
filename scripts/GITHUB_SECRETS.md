# GitHub Secrets Documentation

## Current Active Secrets

### Azure Authentication (OIDC - No passwords needed!)
- `AZURE_TENANT_ID`: 9ef5b184-d371-462a-bc75-5024ce8baff7
- `AZURE_CLIENT_ID_DEV`: 1ecc95d1-e5bb-43e2-9324-30a17cb6b01c (PolicyCortex Dev App)
- `AZURE_CLIENT_ID_PROD`: 8f0208b4-82b1-47cd-b02a-75e2f7afddb5 (PolicyCortex PROD App)
- `AZURE_SUBSCRIPTION_ID_DEV`: 205b477d-17e7-4b3b-92c1-32cf02626b78
- `AZURE_SUBSCRIPTION_ID_PROD`: 9f16cc88-89ce-49ba-a96d-308ed3169595

### Container Registry & AKS
- `ACR_NAME_DEV`: crpcxdev
- `ACR_NAME_PROD`: crcortexprodvb9v2h
- `AKS_CLUSTER_NAME_DEV`: pcx42178531-aks
- `AKS_RESOURCE_GROUP_DEV`: pcx42178531-rg

### Azure OpenAI
- `AOAI_API_KEY`: API key for Azure OpenAI
- `AOAI_API_VERSION`: API version for Azure OpenAI
- `AOAI_CHAT_DEPLOYMENT`: Chat deployment name
- `AOAI_ENDPOINT`: Azure OpenAI endpoint

### Security & Tools
- `GITLEAKS_LICENSE`: License for Gitleaks security scanning
- `JWT_SECRET_KEY_DEV`: JWT secret for dev environment

### Terraform Backend
- `TERRAFORM_BACKEND_CONTAINER`: tfstate
- `TERRAFORM_BACKEND_RESOURCE_GROUP`: Terraform state resource group
- `TERRAFORM_BACKEND_STORAGE_ACCOUNT`: Terraform state storage account

## Deleted Secrets (No longer needed)
- ❌ `AZURE_CLIENT_ID` - Replaced by environment-specific versions
- ❌ `AZURE_CLIENT_SECRET` - Using OIDC instead
- ❌ `AZURE_CREDENTIALS` - Using OIDC instead
- ❌ `AZURE_SUBSCRIPTION_ID` - Replaced by environment-specific versions
- ❌ `AZURE_CONTAINER_REGISTRY` - Replaced by ACR_NAME_DEV/PROD
- ❌ `AZURE_CONTAINER_REGISTRY_PASSWORD` - Using OIDC auth
- ❌ `AZURE_CONTAINER_REGISTRY_USERNAME` - Using OIDC auth
- ❌ `AKS_CLUSTER_NAME_STAGING` - Not using staging environment
- ❌ `AZURE_RESOURCE_GROUP_STAGING` - Not using staging environment
- ❌ `AZURE_RESOURCE_GROUP_DEV` - Duplicate
- ❌ `AZURE_RESOURCE_GROUP_PROD` - Duplicate
- ❌ `AKS_CLUSTER_NAME_PROD` - Will be recreated when prod AKS is set up

## Authentication Method
We use **OIDC (OpenID Connect)** for passwordless authentication:
- No client secrets stored in GitHub
- Federated credentials configured in Azure AD
- More secure than password-based authentication
- Automatic token rotation

## How to Add New Secrets
```bash
# For organization secrets
gh secret set SECRET_NAME --body "secret_value" -R laeintel/policycortex

# For environment-specific secrets (requires environment setup in GitHub)
gh secret set SECRET_NAME --env production --body "secret_value" -R laeintel/policycortex
```

## How to Update Existing Secrets
```bash
# Same command as adding - it will overwrite
gh secret set SECRET_NAME --body "new_secret_value" -R laeintel/policycortex
```

## Required Permissions in Azure
The service principals need these roles:
- **Contributor**: On the subscription
- **AcrPush**: On the container registry
- **Azure Kubernetes Service Cluster Admin**: On the AKS cluster

## Federated Credentials Setup
Run `scripts/setup-oidc-federation.sh` to configure OIDC federation for both environments.