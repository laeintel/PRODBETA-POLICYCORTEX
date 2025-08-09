#!/bin/bash

# Setup federated identity credentials for GitHub Actions OIDC authentication
# This configures the Azure Service Principal to trust GitHub Actions

set -e

# Configuration
AZURE_CLIENT_ID="${AZURE_CLIENT_ID:-1ecc95d1-e5bb-43e2-9324-30a17cb6b01c}"
GITHUB_ORG="laeintel"
GITHUB_REPO="policycortex"

echo "Setting up GitHub Actions OIDC authentication for Azure..."
echo "=================================================="
echo "Service Principal ID: $AZURE_CLIENT_ID"
echo "GitHub Repository: $GITHUB_ORG/$GITHUB_REPO"
echo ""

# Check if logged in to Azure
if ! az account show &>/dev/null; then
    echo "Please login to Azure first:"
    az login
fi

# Get the Service Principal Object ID
SP_OBJECT_ID=$(az ad sp show --id "$AZURE_CLIENT_ID" --query id -o tsv)
echo "Service Principal Object ID: $SP_OBJECT_ID"

# Configure federated identity credential for main branch
echo "Creating federated identity credential for main branch..."
az ad app federated-credential create \
    --id "$AZURE_CLIENT_ID" \
    --parameters '{
        "name": "GitHub-main",
        "issuer": "https://token.actions.githubusercontent.com",
        "subject": "repo:'$GITHUB_ORG'/'$GITHUB_REPO':ref:refs/heads/main",
        "description": "GitHub Actions main branch",
        "audiences": ["api://AzureADTokenExchange"]
    }' 2>/dev/null || echo "Credential for main branch already exists"

# Configure federated identity credential for pull requests
echo "Creating federated identity credential for pull requests..."
az ad app federated-credential create \
    --id "$AZURE_CLIENT_ID" \
    --parameters '{
        "name": "GitHub-PR",
        "issuer": "https://token.actions.githubusercontent.com",
        "subject": "repo:'$GITHUB_ORG'/'$GITHUB_REPO':pull_request",
        "description": "GitHub Actions pull requests",
        "audiences": ["api://AzureADTokenExchange"]
    }' 2>/dev/null || echo "Credential for pull requests already exists"

# Configure federated identity credential for environment: dev
echo "Creating federated identity credential for dev environment..."
az ad app federated-credential create \
    --id "$AZURE_CLIENT_ID" \
    --parameters '{
        "name": "GitHub-env-dev",
        "issuer": "https://token.actions.githubusercontent.com",
        "subject": "repo:'$GITHUB_ORG'/'$GITHUB_REPO':environment:dev",
        "description": "GitHub Actions dev environment",
        "audiences": ["api://AzureADTokenExchange"]
    }' 2>/dev/null || echo "Credential for dev environment already exists"

# Configure federated identity credential for environment: prod
echo "Creating federated identity credential for prod environment..."
az ad app federated-credential create \
    --id "$AZURE_CLIENT_ID" \
    --parameters '{
        "name": "GitHub-env-prod",
        "issuer": "https://token.actions.githubusercontent.com",
        "subject": "repo:'$GITHUB_ORG'/'$GITHUB_REPO':environment:prod",
        "description": "GitHub Actions prod environment",
        "audiences": ["api://AzureADTokenExchange"]
    }' 2>/dev/null || echo "Credential for prod environment already exists"

echo ""
echo "✅ GitHub Actions OIDC setup complete!"
echo ""
echo "Required GitHub Secrets:"
echo "========================"
echo "AZURE_CLIENT_ID=$AZURE_CLIENT_ID"
echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"
echo "AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)"
echo ""
echo "Please ensure these secrets are configured in your GitHub repository settings."