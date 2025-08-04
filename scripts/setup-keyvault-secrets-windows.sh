# PowerShell script to populate Azure Key Vault with PolicyCortex secrets
# This script creates/updates all secrets needed by the Container Apps

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev"
)

$ErrorActionPreference = "Stop"

# Configuration
$ResourceGroup = "rg-pcx-app-$Environment"
$KeyVaultName = "kvpolicortex001$Environment"

Write-Host "🔐 Setting up Key Vault secrets for PolicyCortex $Environment environment" -ForegroundColor Green
Write-Host "Resource Group: $ResourceGroup"
Write-Host "Key Vault: $KeyVaultName"

# Check if Key Vault exists
try {
    $keyVault = az keyvault show --name $KeyVaultName --resource-group $ResourceGroup | ConvertFrom-Json
    Write-Host "✅ Key Vault $KeyVaultName found" -ForegroundColor Green
} catch {
    Write-Host "❌ Key Vault $KeyVaultName not found in resource group $ResourceGroup" -ForegroundColor Red
    Write-Host "Please ensure the infrastructure is deployed first."
    exit 1
}

# Function to set or update a secret
function Set-Secret {
    param($SecretName, $SecretValue, $Description)
    
    Write-Host "Setting secret: $SecretName ($Description)" -ForegroundColor Yellow
    az keyvault secret set --vault-name $KeyVaultName --name $SecretName --value $SecretValue --output none
}

# Function to generate a random secret
function New-RandomSecret {
    $bytes = New-Object Byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    return [Convert]::ToBase64String($bytes)
}

# Function to get existing secret or generate new one
function Get-OrGenerateSecret {
    param($SecretName)
    
    try {
        $existingSecret = az keyvault secret show --vault-name $KeyVaultName --name $SecretName --query "value" -o tsv 2>$null
        if ($existingSecret) {
            return $existingSecret
        }
    } catch {
        # Secret doesn't exist
    }
    return New-RandomSecret
}

Write-Host "🔑 Setting up authentication secrets..." -ForegroundColor Cyan

# JWT Secret Key
$jwtSecret = Get-OrGenerateSecret "jwt-secret"
Set-Secret "jwt-secret" $jwtSecret "JWT token signing key"

# Encryption Key
$encryptionKey = Get-OrGenerateSecret "encryption-key"
Set-Secret "encryption-key" $encryptionKey "Data encryption key"

Write-Host "🌐 Setting up Azure AD secrets..." -ForegroundColor Cyan

# Azure AD Configuration (replace with real values)
$azureClientId = if ($env:AZURE_CLIENT_ID) { $env:AZURE_CLIENT_ID } else { "e8c5b8a0-123e-4567-8901-234567890123" }
$azureTenantId = if ($env:AZURE_TENANT_ID) { $env:AZURE_TENANT_ID } else { "9ef5b184-d371-462a-bc75-5024ce8baff7" }

Set-Secret "azure-client-id" $azureClientId "Azure AD Application Client ID"
Set-Secret "azure-tenant-id" $azureTenantId "Azure AD Tenant ID"

Write-Host "🗄️ Setting up database and storage secrets..." -ForegroundColor Cyan

# Get Cosmos DB connection details
Write-Host "Retrieving Cosmos DB details..."
try {
    $cosmosAccounts = az cosmosdb list --resource-group $ResourceGroup --query "[].name" -o tsv
    if ($cosmosAccounts) {
        $cosmosAccount = $cosmosAccounts.Split([Environment]::NewLine)[0]
        $cosmosEndpoint = "https://$cosmosAccount.documents.azure.com:443/"
        $cosmosKey = az cosmosdb keys list --name $cosmosAccount --resource-group $ResourceGroup --query "primaryMasterKey" -o tsv
        
        Set-Secret "cosmos-endpoint" $cosmosEndpoint "Cosmos DB endpoint"
        Set-Secret "cosmos-key" $cosmosKey "Cosmos DB primary key"
        
        $cosmosConnectionString = "AccountEndpoint=$cosmosEndpoint;AccountKey=$cosmosKey;"
        Set-Secret "cosmos-connection-string" $cosmosConnectionString "Cosmos DB connection string"
    } else {
        Write-Host "⚠️  No Cosmos DB found, using placeholder values" -ForegroundColor Yellow
        Set-Secret "cosmos-endpoint" "https://placeholder-cosmos.documents.azure.com:443/" "Cosmos DB endpoint (placeholder)"
        Set-Secret "cosmos-key" "placeholder-cosmos-key" "Cosmos DB key (placeholder)"
        Set-Secret "cosmos-connection-string" "AccountEndpoint=https://placeholder-cosmos.documents.azure.com:443/;AccountKey=placeholder-key;" "Cosmos DB connection string (placeholder)"
    }
} catch {
    Write-Host "⚠️  Error retrieving Cosmos DB, using placeholder values" -ForegroundColor Yellow
    Set-Secret "cosmos-endpoint" "https://placeholder-cosmos.documents.azure.com:443/" "Cosmos DB endpoint (placeholder)"
    Set-Secret "cosmos-key" "placeholder-cosmos-key" "Cosmos DB key (placeholder)"
    Set-Secret "cosmos-connection-string" "AccountEndpoint=https://placeholder-cosmos.documents.azure.com:443/;AccountKey=placeholder-key;" "Cosmos DB connection string (placeholder)"
}

# Get Redis connection string
Write-Host "Retrieving Redis details..."
try {
    $redisCaches = az redis list --resource-group $ResourceGroup --query "[].name" -o tsv
    if ($redisCaches) {
        $redisName = $redisCaches.Split([Environment]::NewLine)[0]
        $redisKey = az redis list-keys --name $redisName --resource-group $ResourceGroup --query "primaryKey" -o tsv
        $redisHostname = az redis show --name $redisName --resource-group $ResourceGroup --query "hostName" -o tsv
        $redisSslPort = az redis show --name $redisName --resource-group $ResourceGroup --query "sslPort" -o tsv
        
        $redisConnectionString = "$redisHostname`:$redisSslPort,password=$redisKey,ssl=True,abortConnect=False"
        Set-Secret "redis-connection-string" $redisConnectionString "Redis connection string"
    } else {
        Write-Host "⚠️  No Redis found, using placeholder value" -ForegroundColor Yellow
        Set-Secret "redis-connection-string" "localhost:6379,password=placeholder-redis-key" "Redis connection string (placeholder)"
    }
} catch {
    Write-Host "⚠️  Error retrieving Redis, using placeholder value" -ForegroundColor Yellow
    Set-Secret "redis-connection-string" "localhost:6379,password=placeholder-redis-key" "Redis connection string (placeholder)"
}

# Get Storage Account details
Write-Host "Retrieving Storage Account details..."
try {
    $storageAccounts = az storage account list --resource-group $ResourceGroup --query "[].name" -o tsv
    if ($storageAccounts) {
        $storageAccount = $storageAccounts.Split([Environment]::NewLine)[0]
        Set-Secret "storage-account-name" $storageAccount "Storage account name"
        
        $storageConnectionString = az storage account show-connection-string --name $storageAccount --resource-group $ResourceGroup --query "connectionString" -o tsv
        Set-Secret "storage-connection-string" $storageConnectionString "Storage account connection string"
    } else {
        Write-Host "⚠️  No Storage Account found, using placeholder value" -ForegroundColor Yellow
        Set-Secret "storage-account-name" "placeholder-storage" "Storage account name (placeholder)"
        Set-Secret "storage-connection-string" "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=placeholder;" "Storage connection string (placeholder)"
    }
} catch {
    Write-Host "⚠️  Error retrieving Storage Account, using placeholder value" -ForegroundColor Yellow
    Set-Secret "storage-account-name" "placeholder-storage" "Storage account name (placeholder)"
    Set-Secret "storage-connection-string" "DefaultEndpointsProtocol=https;AccountName=placeholder;AccountKey=placeholder;" "Storage connection string (placeholder)"
}

Write-Host "🧠 Setting up AI services secrets..." -ForegroundColor Cyan

# Get Cognitive Services details
try {
    $cognitiveAccounts = az cognitiveservices account list --resource-group $ResourceGroup --query "[].name" -o tsv
    if ($cognitiveAccounts) {
        $cognitiveAccount = $cognitiveAccounts.Split([Environment]::NewLine)[0]
        $cognitiveKey = az cognitiveservices account keys list --name $cognitiveAccount --resource-group $ResourceGroup --query "key1" -o tsv
        $cognitiveEndpoint = az cognitiveservices account show --name $cognitiveAccount --resource-group $ResourceGroup --query "properties.endpoint" -o tsv
        
        Set-Secret "cognitive-services-key" $cognitiveKey "Cognitive Services API key"
        Set-Secret "cognitive-services-endpoint" $cognitiveEndpoint "Cognitive Services endpoint"
    } else {
        Write-Host "⚠️  No Cognitive Services found, using placeholder values" -ForegroundColor Yellow
        Set-Secret "cognitive-services-key" "placeholder-cognitive-key" "Cognitive Services key (placeholder)"
        Set-Secret "cognitive-services-endpoint" "https://placeholder-cognitive.cognitiveservices.azure.com/" "Cognitive Services endpoint (placeholder)"
    }
} catch {
    Write-Host "⚠️  Error retrieving Cognitive Services, using placeholder values" -ForegroundColor Yellow
    Set-Secret "cognitive-services-key" "placeholder-cognitive-key" "Cognitive Services key (placeholder)"
    Set-Secret "cognitive-services-endpoint" "https://placeholder-cognitive.cognitiveservices.azure.com/" "Cognitive Services endpoint (placeholder)"
}

Write-Host "📊 Setting up monitoring secrets..." -ForegroundColor Cyan

# Get Application Insights connection string
try {
    $appInsights = az monitor app-insights component list --resource-group $ResourceGroup --query "[].name" -o tsv
    if ($appInsights) {
        $appInsightsName = $appInsights.Split([Environment]::NewLine)[0]
        $appInsightsConnectionString = az monitor app-insights component show --app $appInsightsName --resource-group $ResourceGroup --query "connectionString" -o tsv
        
        Set-Secret "application-insights-connection-string" $appInsightsConnectionString "Application Insights connection string"
    } else {
        Write-Host "⚠️  No Application Insights found, using placeholder value" -ForegroundColor Yellow
        Set-Secret "application-insights-connection-string" "InstrumentationKey=placeholder-key" "Application Insights connection string (placeholder)"
    }
} catch {
    Write-Host "⚠️  Error retrieving Application Insights, using placeholder value" -ForegroundColor Yellow
    Set-Secret "application-insights-connection-string" "InstrumentationKey=placeholder-key" "Application Insights connection string (placeholder)"
}

Write-Host "✅ Key Vault secrets setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "🔍 Summary of secrets created/updated:" -ForegroundColor Cyan
az keyvault secret list --vault-name $KeyVaultName --query "[].{Name:name, Created:attributes.created}" --output table

Write-Host ""
Write-Host "🔧 Next steps:" -ForegroundColor Yellow
Write-Host "1. Update Container Apps configuration to use secretRef instead of value"
Write-Host "2. Redeploy the Container Apps to pick up the new secrets"
Write-Host "3. Replace placeholder values with real Azure AD app registration details"
Write-Host ""
Write-Host "💡 To update Azure AD secrets with real values:" -ForegroundColor Cyan
Write-Host "   `$env:AZURE_CLIENT_ID = 'your-real-client-id'"
Write-Host "   `$env:AZURE_TENANT_ID = 'your-real-tenant-id'"
Write-Host "   .\setup-keyvault-secrets.ps1 -Environment $Environment"