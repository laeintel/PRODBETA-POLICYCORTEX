#!/usr/bin/env bash
# PolicyCortex: Create PROD App Registration + Service Principal and wire SPA + API scope
# Usage:
#   export AZURE_SUBSCRIPTION_ID=<your-subscription-id>
#   TENANT_ID=9ef5b184-d371-462a-bc75-5024ce8baff7 ./scripts/setup-azure-prod.sh
# Notes:
#   - Creates a new App Registration named "PolicyCortex PROD" (separate from any dev app)
#   - Adds SPA redirect URIs for prod + dev
#   - Adds delegated perms (Graph User.Read, Azure Mgmt user_impersonation) and grants admin consent
#   - Exposes core API scope (access_as_user) under api://<APP_ID>
#   - Creates Service Principal and assigns subscription roles (Reader, Cost Management Reader)
#   - Prints env values to use in prod configs

set -euo pipefail

APP_NAME="PolicyCortex PROD"
TENANT_ID="${TENANT_ID:-9ef5b184-d371-462a-bc75-5024ce8baff7}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"

if [ -z "${SUBSCRIPTION_ID}" ]; then
  echo "ERROR: Set AZURE_SUBSCRIPTION_ID environment variable before running." >&2
  exit 1
fi

REDIRECT_PROD="https://policycortex.com"
REDIRECT_WWW="https://www.policycortex.com"
REDIRECT_DEV="http://localhost:3000"

GRAPH_APP_ID="00000003-0000-0000-c000-000000000000"      # Microsoft Graph
AZ_MGMT_APP_ID="797f4846-ba00-4fd7-ba43-dac1f8f63013"    # Azure Service Management

echo "==> Login/Context"
az account show >/dev/null 2>&1 || az login --tenant "$TENANT_ID"
az account set --subscription "$SUBSCRIPTION_ID"

# Create App Registration
echo "==> Create App Registration: $APP_NAME"
APP_ID=$(az ad app create \
  --display-name "$APP_NAME" \
  --sign-in-audience AzureADMyOrg \
  --query "appId" -o tsv)
APP_OBJ_ID=$(az ad app show --id "$APP_ID" --query id -o tsv)
echo "    AppId: $APP_ID"
echo "    ObjectId: $APP_OBJ_ID"

# Create Service Principal (no-op if exists)
echo "==> Create Service Principal"
az ad sp show --id "$APP_ID" >/dev/null 2>&1 || az ad sp create --id "$APP_ID" >/dev/null
SP_OBJ_ID=$(az ad sp show --id "$APP_ID" --query id -o tsv)
echo "    SP ObjectId: $SP_OBJ_ID"

## Configure SPA redirect URIs (fallback to MS Graph REST for older CLI)
echo "==> Configure SPA redirect URIs"
if ! az ad app update --id "$APP_ID" --set spa.redirectUris='["'$REDIRECT_PROD'","'$REDIRECT_WWW'","'$REDIRECT_DEV'"]' >/dev/null 2>&1; then
  echo "    az ad app update failed; patching via Microsoft Graph REST"
  az rest --method PATCH \
    --uri "https://graph.microsoft.com/v1.0/applications/$APP_OBJ_ID" \
    --headers "Content-Type=application/json" \
    --body '{"spa":{"redirectUris":["'$REDIRECT_PROD'","'$REDIRECT_WWW'","'$REDIRECT_DEV'"]}}'
fi

# Add delegated permissions (Graph User.Read, Azure Mgmt user_impersonation)
echo "==> Add delegated permissions (Graph User.Read, Azure Mgmt user_impersonation)"

# Robustly resolve scope IDs across CLI/Graph shapes, with safe fallbacks
GRAPH_SCOPE_ID="$(az ad sp show --id "$GRAPH_APP_ID" --query "oauth2PermissionScopes[?value=='User.Read'].id" -o tsv 2>/dev/null || true)"
if [ -z "$GRAPH_SCOPE_ID" ]; then
  GRAPH_SCOPE_ID="$(az ad sp show --id "$GRAPH_APP_ID" --query "oauth2Permissions[?value=='User.Read'].id" -o tsv 2>/dev/null || true)"
fi
# Known constant for Microsoft Graph User.Read (as last-resort fallback)
if [ -z "$GRAPH_SCOPE_ID" ]; then
  GRAPH_SCOPE_ID="311a71cc-e848-46a1-bdf8-97ff7156d8e6"
  echo "    WARNING: Falling back to known Graph User.Read scope id ($GRAPH_SCOPE_ID)"
fi

AZ_MGMT_SCOPE_ID="$(az ad sp show --id "$AZ_MGMT_APP_ID" --query "oauth2PermissionScopes[?value=='user_impersonation'].id" -o tsv 2>/dev/null || true)"
if [ -z "$AZ_MGMT_SCOPE_ID" ]; then
  AZ_MGMT_SCOPE_ID="$(az ad sp show --id "$AZ_MGMT_APP_ID" --query "oauth2Permissions[?value=='user_impersonation'].id" -o tsv 2>/dev/null || true)"
fi
if [ -z "$AZ_MGMT_SCOPE_ID" ]; then
  echo "    WARNING: Could not resolve Azure Mgmt user_impersonation scope id automatically; attempting known default"
  # Common user_impersonation scope id for Azure Service Management
  AZ_MGMT_SCOPE_ID="41094075-9dad-400e-a0bd-54e686782033"
fi

if [ -n "$GRAPH_SCOPE_ID" ]; then
  az ad app permission add --id "$APP_ID" --api "$GRAPH_APP_ID" --api-permissions "${GRAPH_SCOPE_ID}=Scope"
else
  echo "    WARNING: Skipping Graph User.Read permission because scope id is empty"
fi

if [ -n "$AZ_MGMT_SCOPE_ID" ]; then
  az ad app permission add --id "$APP_ID" --api "$AZ_MGMT_APP_ID" --api-permissions "${AZ_MGMT_SCOPE_ID}=Scope"
else
  echo "    WARNING: Skipping Azure Mgmt user_impersonation permission because scope id is empty"
fi

# Grant admin consent (requires admin)
echo "==> Grant admin consent for delegated permissions"
az ad app permission admin-consent --id "$APP_ID"

# Ensure permissions are effective by granting app permissions per API
echo "==> Grant application permissions for added scopes (make effective)"
az ad app permission grant --id "$APP_ID" --api "$GRAPH_APP_ID" >/dev/null || true
az ad app permission grant --id "$APP_ID" --api "$AZ_MGMT_APP_ID" >/dev/null || true

# Expose core API scope under api://<APP_ID>
echo "==> Expose core API scope: api://$APP_ID/access_as_user"
if ! az ad app update --id "$APP_ID" --set identifierUris='["api://'$APP_ID'"]' >/dev/null 2>&1; then
  echo "    az ad app update failed; patching identifierUris via Microsoft Graph REST"
  az rest --method PATCH \
    --uri "https://graph.microsoft.com/v1.0/applications/$APP_OBJ_ID" \
    --headers "Content-Type=application/json" \
    --body '{"identifierUris":["api://'$APP_ID'"]}'
fi

SCOPE_ID="$(cat /proc/sys/kernel/random/uuid)"
TMP_SCOPES="$(mktemp)"
cat > "$TMP_SCOPES" <<JSON
{
  "oauth2PermissionScopes": [
    {
      "adminConsentDescription": "Allow the app to access the PolicyCortex Core API on behalf of the signed-in user.",
      "adminConsentDisplayName": "Access PolicyCortex Core API",
      "id": "$SCOPE_ID",
      "isEnabled": true,
      "type": "User",
      "userConsentDescription": "Allow access to PolicyCortex Core API.",
      "userConsentDisplayName": "Access PolicyCortex Core API",
      "value": "access_as_user"
    }
  ]
}
JSON
az ad app update --id "$APP_ID" --set api=@"$TMP_SCOPES"
rm -f "$TMP_SCOPES"

# Assign subscription roles to the Service Principal
echo "==> Assign subscription roles to the Service Principal"
SUB_SCOPE="/subscriptions/$SUBSCRIPTION_ID"
# Prefer the Service Principal ObjectId as assignee for reliability across tenants
az role assignment create --assignee "$SP_OBJ_ID" --role "Reader" --scope "$SUB_SCOPE" >/dev/null || true
az role assignment create --assignee "$SP_OBJ_ID" --role "Cost Management Reader" --scope "$SUB_SCOPE" >/dev/null || true
# Optional:
# az role assignment create --assignee "$APP_ID" --role "Security Reader" --subscription "$SUBSCRIPTION_ID" >/dev/null || true
# az role assignment create --assignee "$APP_ID" --role "Policy Insights Data Reader (Preview)" --subscription "$SUBSCRIPTION_ID" >/dev/null || true

# Output values for envs
AZ_TENANT_ID=$(az account show --query tenantId -o tsv)
CORE_SCOPE="api://$APP_ID/access_as_user"

echo
echo "==> OUTPUT (use these in PROD envs):"
echo "NEXT_PUBLIC_AZURE_CLIENT_ID=$APP_ID"
echo "AZURE_CLIENT_ID=$APP_ID"
echo "NEXT_PUBLIC_AZURE_TENANT_ID=$AZ_TENANT_ID"
echo "AZURE_TENANT_ID=$AZ_TENANT_ID"
echo "NEXT_PUBLIC_CORE_API_SCOPE=$CORE_SCOPE"
echo "REQUIRE_STRICT_AUDIENCE=true"

echo
echo "==> Verify:"
echo "Redirect URIs:"; az ad app show --id "$APP_ID" --query "spa.redirectUris" -o tsv
echo "Identifier URIs:"; az ad app show --id "$APP_ID" --query "identifierUris" -o tsv
echo "Exposed scopes:"; az ad app show --id "$APP_ID" --query "api.oauth2PermissionScopes[].{value:value,id:id}" -o table
echo "Role assignments:"; az role assignment list --assignee "$SP_OBJ_ID" --scope "$SUB_SCOPE" -o table

echo
echo "==> Done. This is PROD-only configuration. Dev continues using the existing dev app."
