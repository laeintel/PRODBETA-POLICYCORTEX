# Test Bicep deployment locally
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('dev', 'staging', 'prod')]
    [string]$Environment = 'dev',
    
    [Parameter(Mandatory=$false)]
    [switch]$ValidateOnly = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf = $false
)

Write-Host "Testing Bicep Deployment" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host "Environment: $Environment" -ForegroundColor Yellow
Write-Host "Validate Only: $ValidateOnly" -ForegroundColor Yellow
Write-Host "What-If Mode: $WhatIf" -ForegroundColor Yellow

# Check Azure CLI login
Write-Host "`nChecking Azure CLI authentication..." -ForegroundColor Yellow
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Not logged in to Azure. Please run 'az login' first." -ForegroundColor Red
    exit 1
}
Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green
Write-Host "Subscription: $($account.name) [$($account.id)]" -ForegroundColor Green

# Build Bicep to ARM template for validation
Write-Host "`nBuilding Bicep template..." -ForegroundColor Yellow
$buildResult = az bicep build --file infrastructure/bicep/main.bicep --stdout 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Bicep build failed:" -ForegroundColor Red
    Write-Host $buildResult -ForegroundColor Red
    exit 1
}

Write-Host "Bicep build successful!" -ForegroundColor Green

# Prepare parameters
$deploymentName = "pcx-$Environment-$(Get-Date -Format 'yyyyMMddHHmmss')"
$location = 'eastus'

$parameters = @{
    environment = $Environment
    location = 'East US'
    owner = 'AeoliTech'
    deployContainerApps = $true
    jwtSecretKey = 'development-secret-key-change-in-production'
    deploySqlServer = $false
    deployMLWorkspace = $false
    deployOpenAI = $false
    createTerraformAccessPolicy = $false
}

# Convert parameters to JSON string
$parametersJson = @{parameters = $parameters} | ConvertTo-Json -Depth 10 -Compress

# Validate deployment
if ($ValidateOnly -or $WhatIf) {
    Write-Host "`nValidating deployment..." -ForegroundColor Yellow
    
    if ($WhatIf) {
        # What-if deployment
        $result = az deployment sub what-if `
            --location $location `
            --template-file infrastructure/bicep/main.bicep `
            --parameters $parametersJson `
            --name $deploymentName `
            --output json 2>&1 | ConvertFrom-Json
            
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nWhat-If Results:" -ForegroundColor Green
            Write-Host ($result | ConvertTo-Json -Depth 10) -ForegroundColor Cyan
        } else {
            Write-Host "What-If failed:" -ForegroundColor Red
            Write-Host $result -ForegroundColor Red
            exit 1
        }
    } else {
        # Validate only
        $result = az deployment sub validate `
            --location $location `
            --template-file infrastructure/bicep/main.bicep `
            --parameters $parametersJson `
            --name $deploymentName `
            --output json 2>&1 | ConvertFrom-Json
            
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Validation successful!" -ForegroundColor Green
            if ($result.error) {
                Write-Host "Validation warnings/info:" -ForegroundColor Yellow
                Write-Host ($result.error | ConvertTo-Json -Depth 10) -ForegroundColor Yellow
            }
        } else {
            Write-Host "Validation failed:" -ForegroundColor Red
            Write-Host ($result | ConvertTo-Json -Depth 10) -ForegroundColor Red
            
            # Try to get more details
            if ($result.error) {
                Write-Host "`nError Details:" -ForegroundColor Red
                Write-Host "Code: $($result.error.code)" -ForegroundColor Red
                Write-Host "Message: $($result.error.message)" -ForegroundColor Red
                
                if ($result.error.details) {
                    Write-Host "`nDetailed Errors:" -ForegroundColor Red
                    foreach ($detail in $result.error.details) {
                        Write-Host "  - $($detail.message)" -ForegroundColor Red
                    }
                }
            }
            exit 1
        }
    }
} else {
    Write-Host "`nStarting actual deployment..." -ForegroundColor Yellow
    Write-Host "Deployment Name: $deploymentName" -ForegroundColor Cyan
    
    $confirmDeploy = Read-Host "Do you want to proceed with the deployment? (y/n)"
    if ($confirmDeploy -ne 'y') {
        Write-Host "Deployment cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    # Deploy
    $result = az deployment sub create `
        --location $location `
        --template-file infrastructure/bicep/main.bicep `
        --parameters $parametersJson `
        --name $deploymentName `
        --output json 2>&1 | ConvertFrom-Json
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nDeployment successful!" -ForegroundColor Green
        
        # Show outputs
        if ($result.properties.outputs) {
            Write-Host "`nDeployment Outputs:" -ForegroundColor Cyan
            foreach ($output in $result.properties.outputs.PSObject.Properties) {
                Write-Host "  $($output.Name): $($output.Value.value)" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "Deployment failed:" -ForegroundColor Red
        Write-Host ($result | ConvertTo-Json -Depth 10) -ForegroundColor Red
        
        # Get deployment operations for more details
        Write-Host "`nGetting deployment operation details..." -ForegroundColor Yellow
        $operations = az deployment operation sub list `
            --name $deploymentName `
            --query "[?properties.provisioningState=='Failed']" `
            --output json | ConvertFrom-Json
        
        if ($operations) {
            Write-Host "`nFailed Operations:" -ForegroundColor Red
            foreach ($op in $operations) {
                Write-Host "  Resource: $($op.properties.targetResource.id)" -ForegroundColor Red
                Write-Host "  Message: $($op.properties.statusMessage.error.message)" -ForegroundColor Red
            }
        }
        
        exit 1
    }
}

Write-Host "`nTest completed!" -ForegroundColor Green