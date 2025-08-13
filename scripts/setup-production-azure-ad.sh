#!/bin/bash

# Setup Azure AD for Production
# Run this script to configure Azure AD app for production deployment

echo "🔐 Configuring Azure AD for Production..."

# Variables
CLIENT_ID="1ecc95d1-e5bb-43e2-9324-30a17cb6b01c"
TENANT_ID="9ef5b184-d371-462a-bc75-5024ce8baff7"
PROD_URL="https://ca-cortex-frontend-prod.azurecontainerapps.io"

# Add production redirect URIs
echo "📝 Adding production redirect URIs..."
az ad app update --id $CLIENT_ID \
  --spa-redirect-uris \
    "${PROD_URL}" \
    "${PROD_URL}/auth/callback" \
    "http://localhost:3000" \
    "http://localhost:3001"

# Set API permissions
echo "🔑 Setting API permissions..."
az ad app permission add --id $CLIENT_ID \
  --api 00000003-0000-0000-c000-000000000000 \
  --api-permissions \
    e1fe6dd8-ba31-4d61-89e7-88639da4683d=Scope \
    37f7f235-527c-4136-accd-4a02d197296e=Scope \
    14dad69e-099b-42c9-810b-d002981feec1=Scope

# Grant admin consent
echo "✅ Granting admin consent..."
az ad app permission admin-consent --id $CLIENT_ID

# Configure token settings
echo "🎫 Configuring token settings..."
az ad app update --id $CLIENT_ID \
  --optional-claims '{
    "idToken": [
      {
        "name": "email",
        "essential": false
      },
      {
        "name": "preferred_username",
        "essential": false
      }
    ],
    "accessToken": [
      {
        "name": "email",
        "essential": false
      }
    ]
  }'

echo "✅ Azure AD production configuration complete!"
echo ""
echo "Production URLs configured:"
echo "  - ${PROD_URL}"
echo ""
echo "Next steps:"
echo "1. Update your production environment variables"
echo "2. Set NEXT_PUBLIC_DEMO_MODE=false"
echo "3. Deploy to production"