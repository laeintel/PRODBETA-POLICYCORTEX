# PolicyCortex Azure AD Setup Script
Write-Host "🔐 Setting up Azure AD for PolicyCortex" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Check if logged in to Azure
Write-Host "`nChecking Azure CLI login status..." -ForegroundColor Yellow
try {
    $account = az account show --query "user.name" -o tsv 2>$null
    if ($account) {
        Write-Host "✅ Logged in as: $account" -ForegroundColor Green
    } else {
        Write-Host "❌ Not logged in to Azure. Please run 'az login' first." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Azure CLI not found or not logged in. Please install Azure CLI and run 'az login'." -ForegroundColor Red
    exit 1
}

# Get tenant information
$tenantId = az account show --query "tenantId" -o tsv
$tenantName = az account show --query "name" -o tsv
Write-Host "🏢 Current Tenant: $tenantName" -ForegroundColor Blue
Write-Host "🆔 Tenant ID: $tenantId" -ForegroundColor Blue

# Create App Registration
Write-Host "`n📱 Creating PolicyCortex App Registration..." -ForegroundColor Yellow

$appCreateResult = az ad app create `
  --display-name "PolicyCortex" `
  --sign-in-audience "AzureADMyOrg" `
  --web-redirect-uris "http://localhost:3000" "http://localhost:5173" `
  --enable-id-token-issuance `
  --enable-access-token-issuance `
  --query "{appId:appId, displayName:displayName}" `
  --output json

if ($LASTEXITCODE -eq 0) {
    $appInfo = $appCreateResult | ConvertFrom-Json
    $clientId = $appInfo.appId
    Write-Host "✅ App Registration created successfully!" -ForegroundColor Green
    Write-Host "📱 App Name: $($appInfo.displayName)" -ForegroundColor White
    Write-Host "🆔 Client ID: $clientId" -ForegroundColor White
} else {
    Write-Host "❌ Failed to create app registration. It might already exist." -ForegroundColor Red
    Write-Host "Trying to find existing app..." -ForegroundColor Yellow
    
    $existingApp = az ad app list --display-name "PolicyCortex" --query "[0].{appId:appId, displayName:displayName}" -o json
    if ($existingApp -and $existingApp -ne "null") {
        $appInfo = $existingApp | ConvertFrom-Json
        $clientId = $appInfo.appId
        Write-Host "✅ Found existing app registration!" -ForegroundColor Green
        Write-Host "🆔 Client ID: $clientId" -ForegroundColor White
    } else {
        Write-Host "❌ Could not create or find PolicyCortex app registration." -ForegroundColor Red
        exit 1
    }
}

# Add API permissions
Write-Host "`n🔑 Adding required API permissions..." -ForegroundColor Yellow
az ad app permission add --id $clientId --api 00000003-0000-0000-c000-000000000000 --api-permissions e1fe6dd8-ba31-4d61-89e7-88639da4683d=Scope

Write-Host "✅ Added Microsoft Graph User.Read permission" -ForegroundColor Green

# Create .env.local file for frontend
Write-Host "`n📝 Creating frontend environment configuration..." -ForegroundColor Yellow

$envContent = @"
# Azure AD Configuration - Generated $(Get-Date)
VITE_AZURE_CLIENT_ID=$clientId
VITE_AZURE_TENANT_ID=$tenantId
VITE_AZURE_REDIRECT_URI=http://localhost:3000

# API Configuration
VITE_API_BASE_URL=http://localhost:8000/api
VITE_WS_URL=ws://localhost:8000/ws

# Development Features
VITE_ENABLE_DEBUG=true
VITE_ENABLE_NOTIFICATIONS=true
VITE_ENABLE_WEBSOCKET=true
VITE_ENABLE_DARK_MODE=true
"@

$envContent | Out-File -FilePath "frontend\.env.local" -Encoding UTF8
Write-Host "✅ Created frontend\.env.local" -ForegroundColor Green

# Update backend environment
Write-Host "`n🔧 Backend configuration needed..." -ForegroundColor Yellow
Write-Host "Add these environment variables to your backend services:" -ForegroundColor White
Write-Host "AZURE_CLIENT_ID=$clientId" -ForegroundColor Cyan
Write-Host "AZURE_TENANT_ID=$tenantId" -ForegroundColor Cyan
Write-Host "JWT_ISSUER=https://login.microsoftonline.com/$tenantId/v2.0" -ForegroundColor Cyan

# Summary
Write-Host "`n🎉 Azure AD Setup Complete!" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green
Write-Host "Client ID: $clientId" -ForegroundColor White
Write-Host "Tenant ID: $tenantId" -ForegroundColor White
Write-Host "Redirect URIs: http://localhost:3000, http://localhost:5173" -ForegroundColor White

Write-Host "`n📋 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Restart your frontend: cd frontend && npm run dev" -ForegroundColor White
Write-Host "2. Navigate to http://localhost:3000" -ForegroundColor White
Write-Host "3. Click 'Sign In' to test Azure AD authentication" -ForegroundColor White
Write-Host "4. Grant admin consent if prompted" -ForegroundColor White

Write-Host "`n🔗 Azure Portal Links:" -ForegroundColor Blue
Write-Host "App Registration: https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationMenuBlade/~/Overview/appId/$clientId" -ForegroundColor Cyan
Write-Host "Enterprise App: https://portal.azure.com/#view/Microsoft_AAD_IAM/ManagedAppMenuBlade/~/Overview/objectId/$clientId" -ForegroundColor Cyan