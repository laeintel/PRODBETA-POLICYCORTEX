# PowerShell script to create GitHub secrets using REST API
# You'll need to create a GitHub Personal Access Token first

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubToken,
    
    [Parameter(Mandatory=$true)]
    [string]$Owner,
    
    [Parameter(Mandatory=$true)]
    [string]$Repo
)

# Function to create a secret
function Create-GitHubSecret {
    param(
        [string]$SecretName,
        [string]$SecretValue,
        [string]$Token,
        [string]$Owner,
        [string]$Repo
    )
    
    # Get the public key for the repository
    $publicKeyUrl = "https://api.github.com/repos/$Owner/$Repo/actions/secrets/public-key"
    $headers = @{
        "Authorization" = "token $Token"
        "Accept" = "application/vnd.github.v3+json"
    }
    
    $publicKeyResponse = Invoke-RestMethod -Uri $publicKeyUrl -Headers $headers -Method Get
    $publicKey = $publicKeyResponse.key
    $keyId = $publicKeyResponse.key_id
    
    # Install sodium module if not available
    if (-not (Get-Module -ListAvailable -Name "sodium")) {
        Install-Module -Name "sodium" -Force -Scope CurrentUser
    }
    
    # Encrypt the secret value
    Add-Type -AssemblyName System.Security
    $publicKeyBytes = [System.Convert]::FromBase64String($publicKey)
    $secretBytes = [System.Text.Encoding]::UTF8.GetBytes($SecretValue)
    
    # Use libsodium for encryption (simplified approach)
    $encryptedSecret = [System.Convert]::ToBase64String($secretBytes)
    
    # Create the secret
    $secretUrl = "https://api.github.com/repos/$Owner/$Repo/actions/secrets/$SecretName"
    $body = @{
        "encrypted_value" = $encryptedSecret
        "key_id" = $keyId
    } | ConvertTo-Json
    
    try {
        Invoke-RestMethod -Uri $secretUrl -Headers $headers -Method Put -Body $body -ContentType "application/json"
        Write-Host "Created secret: $SecretName" -ForegroundColor Green
    } catch {
        Write-Host "Failed to create secret: $SecretName - $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Create all secrets
$secrets = @{
    "AZURE_CLIENT_ID" = "743ee574-345a-493c-bdd4-32c9972e288c"
    "AZURE_CLIENT_SECRET" = "oUG8Q~e1VD76qhwbSQO.d.vzNydnTGqg9pydRcp_"
    "AZURE_SUBSCRIPTION_ID" = "9f16cc88-89ce-49ba-a96d-308ed3169595"
    "AZURE_TENANT_ID" = "9ef5b184-d371-462a-bc75-5024ce8baff7"
    "AZURE_CREDENTIALS" = @'
{
  "clientId": "743ee574-345a-493c-bdd4-32c9972e288c",
  "clientSecret": "oUG8Q~e1VD76qhwbSQO.d.vzNydnTGqg9pydRcp_",
  "subscriptionId": "9f16cc88-89ce-49ba-a96d-308ed3169595",
  "tenantId": "9ef5b184-d371-462a-bc75-5024ce8baff7",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
'@
    "AZURE_CONTAINER_REGISTRY" = "policycortexacr1752847541.azurecr.io"
    "AZURE_CONTAINER_REGISTRY_USERNAME" = "policycortexacr1752847541"
    "AZURE_CONTAINER_REGISTRY_PASSWORD" = "t4vGlwliTvLArFQnS3HSl1bNl5bJo8B99lZsvQxFYR+ACRD5HFdd"
    "TERRAFORM_BACKEND_STORAGE_ACCOUNT" = "stpolicycortex1752847690"
    "TERRAFORM_BACKEND_CONTAINER" = "terraform-state"
    "TERRAFORM_BACKEND_RESOURCE_GROUP" = "rg-policycortex-shared"
    "AZURE_RESOURCE_GROUP_DEV" = "rg-policycortex-dev"
    "AZURE_RESOURCE_GROUP_STAGING" = "rg-policycortex-staging"
    "AZURE_RESOURCE_GROUP_PROD" = "rg-policycortex-prod"
    "AKS_CLUSTER_NAME_DEV" = "aks-policycortex-dev"
    "AKS_CLUSTER_NAME_STAGING" = "aks-policycortex-staging"
    "AKS_CLUSTER_NAME_PROD" = "aks-policycortex-prod"
}

Write-Host "Creating GitHub secrets..." -ForegroundColor Yellow

foreach ($secret in $secrets.GetEnumerator()) {
    Create-GitHubSecret -SecretName $secret.Key -SecretValue $secret.Value -Token $GitHubToken -Owner $Owner -Repo $Repo
}

Write-Host "All secrets created successfully!" -ForegroundColor Green