# PowerShell script to fix container app deployments with proper images and logging
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'dev',
    
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "rg-pcx-app-$Environment",
    
    [Parameter(Mandatory=$false)]
    [string]$BuildId = 'latest',
    
    [Parameter(Mandatory=$false)]
    [switch]$DeployBicep = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$UpdateImages = $true
)

Write-Host "PolicyCortex Container Apps Deployment Fix" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Resource Group: $ResourceGroup" -ForegroundColor Yellow
Write-Host "Build ID: $BuildId" -ForegroundColor Yellow

# Check Azure CLI login
Write-Host "`nChecking Azure CLI authentication..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged in to Azure. Please run 'az login' first." -ForegroundColor Red
    exit 1
}
Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green

# Get container registry details
Write-Host "`nGetting container registry details..." -ForegroundColor Yellow
$registryName = "crpcx$Environment"
$registry = az acr show --name $registryName --resource-group $ResourceGroup 2>$null | ConvertFrom-Json
if (-not $registry) {
    Write-Host "Container registry $registryName not found" -ForegroundColor Red
    exit 1
}
$registryLoginServer = $registry.loginServer
Write-Host "Container Registry: $registryLoginServer" -ForegroundColor Green

# Deploy Bicep templates if requested
if ($DeployBicep) {
    Write-Host "`nDeploying Bicep infrastructure..." -ForegroundColor Yellow
    
    $bicepParams = @{
        environment = $Environment
        location = 'East US'
        deployContainerApps = $true
        jwtSecretKey = 'development-secret-key-change-in-production'
        deploySqlServer = $false
        deployMLWorkspace = $false
        deployOpenAI = $false
    }
    
    $paramString = ($bicepParams.GetEnumerator() | ForEach-Object { "$($_.Key)='$($_.Value)'" }) -join ' '
    
    Write-Host "Deploying main.bicep with parameters: $paramString" -ForegroundColor Yellow
    $deploymentResult = az deployment sub create `
        --location 'eastus' `
        --template-file 'infrastructure/bicep/main.bicep' `
        --parameters $paramString `
        --name "pcx-deployment-$(Get-Date -Format 'yyyyMMddHHmmss')" `
        --output json | ConvertFrom-Json
    
    if ($deploymentResult.properties.provisioningState -eq 'Succeeded') {
        Write-Host "Bicep deployment successful!" -ForegroundColor Green
    } else {
        Write-Host "Bicep deployment failed!" -ForegroundColor Red
        Write-Host $deploymentResult | ConvertTo-Json -Depth 10
        exit 1
    }
}

# Define service mappings
$services = @(
    @{Name='api_gateway'; AppName="ca-pcx-gateway-$Environment"; Image="policortex001-api-gateway"},
    @{Name='azure_integration'; AppName="ca-pcx-azureint-$Environment"; Image="policortex001-azure-integration"},
    @{Name='ai_engine'; AppName="ca-pcx-ai-$Environment"; Image="policortex001-ai-engine"},
    @{Name='data_processing'; AppName="ca-pcx-dataproc-$Environment"; Image="policortex001-data-processing"},
    @{Name='conversation'; AppName="ca-pcx-chat-$Environment"; Image="policortex001-conversation"},
    @{Name='notification'; AppName="ca-pcx-notify-$Environment"; Image="policortex001-notification"},
    @{Name='frontend'; AppName="ca-pcx-web-$Environment"; Image="policortex001-frontend"}
)

# Build and push Docker images if needed
Write-Host "`nChecking Docker images in registry..." -ForegroundColor Yellow
foreach ($service in $services) {
    $imageName = $service.Image
    $imageTag = $BuildId
    
    # Check if image exists in registry
    $imageExists = az acr repository show-tags `
        --name $registryName `
        --repository $imageName `
        --query "contains(@, '$imageTag')" `
        --output tsv 2>$null
    
    if ($imageExists -ne 'true') {
        Write-Host "Image $imageName:$imageTag not found in registry" -ForegroundColor Yellow
        
        # Build and push the image
        if ($service.Name -eq 'frontend') {
            Write-Host "Building and pushing frontend image..." -ForegroundColor Yellow
            docker build -t "$registryLoginServer/$imageName:$imageTag" -f frontend/Dockerfile frontend
            
            # Login to ACR
            az acr login --name $registryName
            
            # Push image
            docker push "$registryLoginServer/$imageName:$imageTag"
            
            # Also tag as latest
            docker tag "$registryLoginServer/$imageName:$imageTag" "$registryLoginServer/$imageName:latest"
            docker push "$registryLoginServer/$imageName:latest"
        } else {
            $servicePath = $service.Name.Replace('_', '-')
            if ($servicePath -eq 'azure-integration') { $servicePath = 'azure_integration' }
            if ($servicePath -eq 'ai-engine') { $servicePath = 'ai_engine' }
            if ($servicePath -eq 'data-processing') { $servicePath = 'data_processing' }
            
            Write-Host "Building and pushing $($service.Name) image..." -ForegroundColor Yellow
            docker build -t "$registryLoginServer/$imageName:$imageTag" `
                -f "backend/services/$servicePath/Dockerfile" backend
            
            # Login to ACR
            az acr login --name $registryName
            
            # Push image
            docker push "$registryLoginServer/$imageName:$imageTag"
            
            # Also tag as latest
            docker tag "$registryLoginServer/$imageName:$imageTag" "$registryLoginServer/$imageName:latest"
            docker push "$registryLoginServer/$imageName:latest"
        }
        
        Write-Host "Successfully pushed $imageName:$imageTag" -ForegroundColor Green
    } else {
        Write-Host "Image $imageName:$imageTag already exists in registry" -ForegroundColor Green
    }
}

# Update container apps with proper images
if ($UpdateImages) {
    Write-Host "`nUpdating container apps with images..." -ForegroundColor Yellow
    
    foreach ($service in $services) {
        $appName = $service.AppName
        $imageName = $service.Image
        $imageTag = $BuildId
        $fullImage = "$registryLoginServer/$imageName`:$imageTag"
        
        Write-Host "Updating $appName with image $fullImage..." -ForegroundColor Yellow
        
        # Check if container app exists
        $appExists = az containerapp show --name $appName --resource-group $ResourceGroup 2>$null
        
        if ($appExists) {
            # Update the container app
            $updateResult = az containerapp update `
                --name $appName `
                --resource-group $ResourceGroup `
                --image $fullImage `
                --revision-suffix "v$(Get-Date -Format 'yyyyMMddHHmm')" `
                --output json 2>$null | ConvertFrom-Json
            
            if ($updateResult) {
                Write-Host "Successfully updated $appName" -ForegroundColor Green
                
                # Get the latest revision
                $latestRevision = az containerapp revision list `
                    --name $appName `
                    --resource-group $ResourceGroup `
                    --query "[0].name" `
                    --output tsv
                
                Write-Host "Latest revision: $latestRevision" -ForegroundColor Cyan
                
                # Check revision status
                $revisionStatus = az containerapp revision show `
                    --name $appName `
                    --resource-group $ResourceGroup `
                    --revision $latestRevision `
                    --query "properties.runningState" `
                    --output tsv
                
                if ($revisionStatus -eq 'Running') {
                    Write-Host "Revision is running successfully" -ForegroundColor Green
                } else {
                    Write-Host "Revision status: $revisionStatus" -ForegroundColor Yellow
                    
                    # Get logs to debug issues
                    Write-Host "Fetching container logs..." -ForegroundColor Yellow
                    $logs = az containerapp logs show `
                        --name $appName `
                        --resource-group $ResourceGroup `
                        --type console `
                        --follow $false `
                        --tail 50 2>$null
                    
                    if ($logs) {
                        Write-Host "Recent logs:" -ForegroundColor Yellow
                        Write-Host $logs
                    }
                }
            } else {
                Write-Host "Failed to update $appName" -ForegroundColor Red
            }
        } else {
            Write-Host "Container app $appName not found. Skipping..." -ForegroundColor Yellow
        }
    }
}

# Verify Log Analytics configuration
Write-Host "`nVerifying Log Analytics configuration..." -ForegroundColor Yellow
$environment = az containerapp env show `
    --name "cae-pcx-$Environment" `
    --resource-group $ResourceGroup `
    --query "properties.appLogsConfiguration" `
    --output json 2>$null | ConvertFrom-Json

if ($environment.destination -eq 'log-analytics') {
    Write-Host "Log Analytics is configured correctly" -ForegroundColor Green
    Write-Host "Destination: $($environment.destination)" -ForegroundColor Cyan
} else {
    Write-Host "Log Analytics is not configured properly" -ForegroundColor Red
    Write-Host "Current configuration:" -ForegroundColor Yellow
    Write-Host ($environment | ConvertTo-Json -Depth 10)
}

# Verify Application Insights configuration  
Write-Host "`nVerifying Application Insights configuration..." -ForegroundColor Yellow
$appInsights = az monitor app-insights component show `
    --app "ai-pcx-$Environment" `
    --resource-group $ResourceGroup `
    --output json 2>$null | ConvertFrom-Json

if ($appInsights) {
    Write-Host "Application Insights is configured" -ForegroundColor Green
    Write-Host "Connection String: $($appInsights.connectionString)" -ForegroundColor Cyan
    Write-Host "Sampling Percentage: $($appInsights.samplingPercentage)%" -ForegroundColor Cyan
    
    # Update Key Vault secret with connection string if needed
    Write-Host "Updating Key Vault secret for Application Insights..." -ForegroundColor Yellow
    az keyvault secret set `
        --vault-name "kv-pcx-$Environment" `
        --name "application-insights-connection-string" `
        --value $appInsights.connectionString `
        --output none
    
    Write-Host "Key Vault secret updated" -ForegroundColor Green
} else {
    Write-Host "Application Insights not found" -ForegroundColor Red
}

Write-Host "`nDeployment fix completed!" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Cyan

# Display container app URLs
Write-Host "`nContainer App URLs:" -ForegroundColor Yellow
foreach ($service in $services) {
    if ($service.Name -in @('api_gateway', 'frontend')) {
        $fqdn = az containerapp show `
            --name $service.AppName `
            --resource-group $ResourceGroup `
            --query "properties.configuration.ingress.fqdn" `
            --output tsv 2>$null
        
        if ($fqdn) {
            Write-Host "$($service.Name): https://$fqdn" -ForegroundColor Cyan
        }
    }
}