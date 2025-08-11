#!/usr/bin/env bash
# CI-friendly deploy script: reconciles Terraform state, imports existing resources, then applies
# Usage: ./ci-deploy.sh <env> <subscription_id>

set -euo pipefail

ENVIRONMENT="${1:-dev}"
SUBSCRIPTION_ID="${2:-${AZURE_SUBSCRIPTION_ID:-}}"

if [[ -z "${SUBSCRIPTION_ID}" ]]; then
  echo "AZURE_SUBSCRIPTION_ID is not set and no subscription id was provided" >&2
  exit 1
fi

echo "Environment: ${ENVIRONMENT}"
echo "Subscription: ${SUBSCRIPTION_ID}"

# Ensure correct subscription context (noop if already logged in by CI)
az account set --subscription "${SUBSCRIPTION_ID}" 2>/dev/null || true

# Terraform AzureRM auth via Service Principal or OIDC (avoid CLI user-only auth)
# If CI provides AZURE_CLIENT_ID/TENANT_ID, wire them to ARM_* so both backend and provider use SP/OIDC
if [[ -n "${AZURE_CLIENT_ID:-}" && -n "${AZURE_TENANT_ID:-}" ]]; then
  export ARM_CLIENT_ID="${AZURE_CLIENT_ID}"
  export ARM_TENANT_ID="${AZURE_TENANT_ID}"
  export ARM_SUBSCRIPTION_ID="${SUBSCRIPTION_ID}"
  if [[ -n "${AZURE_CLIENT_SECRET:-}" ]]; then
    export ARM_CLIENT_SECRET="${AZURE_CLIENT_SECRET}"
  elif [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    # Enable OIDC when running under GitHub Actions with azure/login OIDC
    export ARM_USE_OIDC=true
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Backend config (match deploy.ps1 hashing approach)
REPO_NAME="${GITHUB_REPOSITORY:-laeintel/policycortex}"
HASH=$(printf "%s" "${REPO_NAME}" | sha1sum | cut -c1-6)
BACKEND_RG="rg-tfstate-cortex-${ENVIRONMENT}"
BACKEND_SA="sttfcortex${ENVIRONMENT}${HASH}"
BACKEND_CONTAINER="tfstate"
BACKEND_KEY="${ENVIRONMENT}.tfstate"

echo "Configuring Terraform backend: ${BACKEND_RG}/${BACKEND_SA}/${BACKEND_CONTAINER}/${BACKEND_KEY}"

# Idempotent create backend RG/SA/container
az group create --name "${BACKEND_RG}" --location eastus --output none 2>/dev/null || true
if ! az storage account show --name "${BACKEND_SA}" --resource-group "${BACKEND_RG}" >/dev/null 2>&1; then
  az storage account create \
    --name "${BACKEND_SA}" \
    --resource-group "${BACKEND_RG}" \
    --location eastus \
    --sku Standard_LRS \
    --encryption-services blob \
    --output none
fi
az storage container create \
  --name "${BACKEND_CONTAINER}" \
  --account-name "${BACKEND_SA}" \
  --auth-mode login \
  --output none 2>/dev/null || true

# Init Terraform
terraform init \
  -backend-config="resource_group_name=${BACKEND_RG}" \
  -backend-config="storage_account_name=${BACKEND_SA}" \
  -backend-config="container_name=${BACKEND_CONTAINER}" \
  -backend-config="key=${BACKEND_KEY}" \
  -reconfigure

# Reconcile state: import existing resources if they exist
export AZURE_SUBSCRIPTION_ID="${SUBSCRIPTION_ID}"
"${SCRIPT_DIR}/import-existing.sh" "${ENVIRONMENT}" "${SUBSCRIPTION_ID}"

# Verify resource group existence after import (create-on-missing safety)
RG_NAME="rg-cortex-${ENVIRONMENT}"
if ! az group show --name "${RG_NAME}" >/dev/null 2>&1; then
  echo "Resource group ${RG_NAME} not found. Creating minimal RG so Terraform can proceed..."
  az group create --name "${RG_NAME}" --location eastus --output none
fi

# Apply desired state
TFVARS="environments/${ENVIRONMENT}/terraform.tfvars"
terraform apply -var-file="${TFVARS}" -auto-approve

echo "Deployment complete."


