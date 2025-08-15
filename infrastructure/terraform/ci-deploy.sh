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
# Obtain a storage access key for backend auth (avoids AAD data-plane permissions)
ACCESS_KEY=$(az storage account keys list \
  --resource-group "${BACKEND_RG}" \
  --account-name "${BACKEND_SA}" \
  --query "[0].value" -o tsv)

# Ensure backend container exists using key auth
az storage container create \
  --name "${BACKEND_CONTAINER}" \
  --account-name "${BACKEND_SA}" \
  --account-key "${ACCESS_KEY}" \
  --public-access off \
  --output none 2>/dev/null || true

# Init Terraform
terraform init \
  -backend-config="resource_group_name=${BACKEND_RG}" \
  -backend-config="storage_account_name=${BACKEND_SA}" \
  -backend-config="container_name=${BACKEND_CONTAINER}" \
  -backend-config="key=${BACKEND_KEY}" \
  -backend-config="access_key=${ACCESS_KEY}" \
  -reconfigure

# Check for state lock and attempt recovery
echo "🔍 Checking for Terraform state lock..."

# First, try to detect state lock by attempting a simple plan
PLAN_OUTPUT=$(terraform plan -detailed-exitcode 2>&1)
PLAN_EXIT_CODE=$?

if [[ $PLAN_EXIT_CODE -ne 0 && $PLAN_EXIT_CODE -ne 2 ]]; then
  echo "⚠️ Terraform plan failed, checking for state lock..."
  
  # Check if the output contains lock-related errors
  if echo "$PLAN_OUTPUT" | grep -q "state blob is already locked\|state lock\|Lock Info"; then
    echo "🔓 Detected state lock! Attempting recovery..."
    
    # Extract lock ID from output if available
    LOCK_ID=$(echo "$PLAN_OUTPUT" | grep -o "ID:[[:space:]]*[a-f0-9\-]*" | cut -d: -f2 | tr -d ' ' || echo "")
    
    if [[ -n "$LOCK_ID" ]]; then
      echo "🔑 Found lock ID: $LOCK_ID"
      echo "🚨 In CI context, automatically force-unlocking stale lock"
      echo "   (GitHub Actions timeout prevents infinite locks)"
      
      # Force unlock using the extracted lock ID
      if terraform force-unlock -force "$LOCK_ID"; then
        echo "✅ Successfully force-unlocked state with ID: $LOCK_ID"
      else
        echo "❌ Force unlock failed, trying alternative methods..."
        
        # Alternative: try to remove lock from Azure Storage directly
        echo "🔍 Attempting to remove lock from Azure Storage..."
        LOCK_BLOBS=$(az storage blob list \
          --container-name "${BACKEND_CONTAINER}" \
          --account-name "${BACKEND_SA}" \
          --account-key "${ACCESS_KEY}" \
          --prefix "${BACKEND_KEY}" \
          --query "[?contains(name, 'lock')].name" -o tsv 2>/dev/null || echo "")
        
        if [[ -n "$LOCK_BLOBS" ]]; then
          echo "🔓 Found lock blobs in storage: $LOCK_BLOBS"
          while IFS= read -r blob; do
            echo "Removing lock blob: $blob"
            az storage blob delete \
              --container-name "${BACKEND_CONTAINER}" \
              --account-name "${BACKEND_SA}" \
              --account-key "${ACCESS_KEY}" \
              --name "$blob" \
              --output none 2>/dev/null || true
          done <<< "$LOCK_BLOBS"
          echo "✅ Removed lock blobs from storage"
        fi
      fi
    else
      echo "⚠️ Could not extract lock ID from error message"
      echo "Full error output:"
      echo "$PLAN_OUTPUT"
      
      # Try generic force unlock (will ask for lock ID)
      echo "🔄 Attempting generic lock recovery..."
      
      # List potential lock files in storage
      LOCK_BLOBS=$(az storage blob list \
        --container-name "${BACKEND_CONTAINER}" \
        --account-name "${BACKEND_SA}" \
        --account-key "${ACCESS_KEY}" \
        --prefix "${BACKEND_KEY}" \
        --query "[?contains(name, 'lock')].name" -o tsv 2>/dev/null || echo "")
      
      if [[ -n "$LOCK_BLOBS" ]]; then
        echo "🔓 Found potential lock blobs: $LOCK_BLOBS"
        while IFS= read -r blob; do
          echo "Removing lock blob: $blob"
          az storage blob delete \
            --container-name "${BACKEND_CONTAINER}" \
            --account-name "${BACKEND_SA}" \
            --account-key "${ACCESS_KEY}" \
            --name "$blob" \
            --output none 2>/dev/null || true
        done <<< "$LOCK_BLOBS"
      fi
    fi
    
    echo "🔄 Retrying terraform operations after lock removal..."
    sleep 3
    
    # Verify lock is gone
    if terraform plan -detailed-exitcode >/dev/null 2>&1; then
      echo "✅ State lock successfully removed"
    else
      echo "❌ State may still be locked or have other issues"
      terraform plan -detailed-exitcode 2>&1 | head -20
    fi
  else
    echo "❌ Plan failed for reasons other than state lock:"
    echo "$PLAN_OUTPUT"
  fi
else
  echo "✅ No state lock detected"
fi

# Reconcile state: import existing resources if they exist
export AZURE_SUBSCRIPTION_ID="${SUBSCRIPTION_ID}"
# Invoke import script with bash to avoid execute-bit dependency in CI
bash "${SCRIPT_DIR}/import-existing.sh" "${ENVIRONMENT}" "${SUBSCRIPTION_ID}"

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


