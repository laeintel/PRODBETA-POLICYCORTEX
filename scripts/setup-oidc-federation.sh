#!/bin/bash
# Setup OIDC Federation for GitHub Actions with Azure AD
# This enables passwordless authentication for CI/CD pipelines

echo "Setting up OIDC Federation for PolicyCortex GitHub Actions..."
echo ""

# Configuration
GITHUB_ORG="laeintel"
GITHUB_REPO="policycortex"
TENANT_ID="9ef5b184-d371-462a-bc75-5024ce8baff7"

# Dev Environment
DEV_CLIENT_ID="1ecc95d1-e5bb-43e2-9324-30a17cb6b01c"
DEV_SUBSCRIPTION_ID="205b477d-17e7-4b3b-92c1-32cf02626b78"

# Prod Environment  
PROD_CLIENT_ID="8f0208b4-82b1-47cd-b02a-75e2f7afddb5"
PROD_SUBSCRIPTION_ID="9f16cc88-89ce-49ba-a96d-308ed3169595"

echo "=== DEV Environment OIDC Setup ==="
echo "Creating federated credentials for DEV service principal..."

# Create federated credential for main branch (DEV)
az ad app federated-credential create \
    --id $DEV_CLIENT_ID \
    --parameters @- <<EOF
{
    "name": "github-deploy-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:${GITHUB_ORG}/${GITHUB_REPO}:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "Deploy from main branch to dev environment"
}
EOF

# Create federated credential for pull requests (DEV)
az ad app federated-credential create \
    --id $DEV_CLIENT_ID \
    --parameters @- <<EOF
{
    "name": "github-deploy-pr",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:${GITHUB_ORG}/${GITHUB_REPO}:pull_request",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "Deploy from pull requests to dev environment"
}
EOF

# Create federated credential for environment-specific deployments (DEV)
az ad app federated-credential create \
    --id $DEV_CLIENT_ID \
    --parameters @- <<EOF
{
    "name": "github-deploy-dev-env",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:${GITHUB_ORG}/${GITHUB_REPO}:environment:dev",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "Deploy to dev environment"
}
EOF

echo ""
echo "=== PROD Environment OIDC Setup ==="
echo "Creating federated credentials for PROD service principal..."

# Create federated credential for production deployments (PROD)
az ad app federated-credential create \
    --id $PROD_CLIENT_ID \
    --parameters @- <<EOF
{
    "name": "github-deploy-prod",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:${GITHUB_ORG}/${GITHUB_REPO}:environment:production",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "Deploy to production environment"
}
EOF

# Create federated credential for release tags (PROD)
az ad app federated-credential create \
    --id $PROD_CLIENT_ID \
    --parameters @- <<EOF
{
    "name": "github-deploy-release",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:${GITHUB_ORG}/${GITHUB_REPO}:ref:refs/tags/v*",
    "audiences": ["api://AzureADTokenExchange"],
    "description": "Deploy release tags to production"
}
EOF

echo ""
echo "=== Assigning Required Roles ==="

# Assign Contributor role to DEV service principal
echo "Assigning Contributor role to DEV service principal..."
az role assignment create \
    --assignee $DEV_CLIENT_ID \
    --role "Contributor" \
    --scope "/subscriptions/${DEV_SUBSCRIPTION_ID}"

# Assign AcrPush role for container registry access (DEV)
echo "Assigning AcrPush role for DEV ACR..."
az role assignment create \
    --assignee $DEV_CLIENT_ID \
    --role "AcrPush" \
    --scope "/subscriptions/${DEV_SUBSCRIPTION_ID}/resourceGroups/pcx42178531-rg/providers/Microsoft.ContainerRegistry/registries/crpcxdev"

# Assign Azure Kubernetes Service Cluster Admin role (DEV)
echo "Assigning AKS Cluster Admin role for DEV..."
az role assignment create \
    --assignee $DEV_CLIENT_ID \
    --role "Azure Kubernetes Service Cluster Admin" \
    --scope "/subscriptions/${DEV_SUBSCRIPTION_ID}/resourceGroups/pcx42178531-rg/providers/Microsoft.ContainerService/managedClusters/pcx42178531-aks"

# Assign Contributor role to PROD service principal (when ready)
echo "Assigning Contributor role to PROD service principal..."
az role assignment create \
    --assignee $PROD_CLIENT_ID \
    --role "Contributor" \
    --scope "/subscriptions/${PROD_SUBSCRIPTION_ID}"

echo ""
echo "=== Verification ==="
echo "Listing federated credentials for DEV app..."
az ad app federated-credential list --id $DEV_CLIENT_ID --query "[].{name:name, subject:subject}" -o table

echo ""
echo "Listing federated credentials for PROD app..."
az ad app federated-credential list --id $PROD_CLIENT_ID --query "[].{name:name, subject:subject}" -o table

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "GitHub secrets that need to be set:"
echo "  - AZURE_TENANT_ID: ${TENANT_ID}"
echo "  - AZURE_CLIENT_ID_DEV: ${DEV_CLIENT_ID}"
echo "  - AZURE_SUBSCRIPTION_ID_DEV: ${DEV_SUBSCRIPTION_ID}"
echo "  - AZURE_CLIENT_ID_PROD: ${PROD_CLIENT_ID}"
echo "  - AZURE_SUBSCRIPTION_ID_PROD: ${PROD_SUBSCRIPTION_ID}"
echo ""
echo "No client secrets needed! OIDC federation is configured for passwordless auth."