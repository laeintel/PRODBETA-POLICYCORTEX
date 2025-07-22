#!/usr/bin/env pwsh
# Fix AI Services deployment issue with custom domain

param(
    [Parameter(Mandatory=$false)]
    [string]$Environment = "dev",
    
    [Parameter(Mandatory=$false)]
    [switch]$DeleteExisting,
    
    [Parameter(Mandatory=$false)]
    [switch]$ListOnly
)

Write-Host "PolicyCortex AI Services Deployment Fix" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check if logged in to Azure
$context = Get-AzContext
if (-not $context) {
    Write-Host "Please login to Azure first using Connect-AzAccount" -ForegroundColor Red
    exit 1
}

Write-Host "Current Azure Context:" -ForegroundColor Yellow
Write-Host "  Subscription: $($context.Subscription.Name)" -ForegroundColor Gray
Write-Host "  Tenant: $($context.Tenant.Id)" -ForegroundColor Gray

# Define resource names
$cognitiveServiceName = "policycortex-cognitive-$Environment"
$openAIServiceName = "policycortex-openai-$Environment"
$resourceGroupName = "rg-policycortex-app-$Environment"

Write-Host "`nChecking for existing AI Services resources..." -ForegroundColor Yellow

# Check Cognitive Services
$cognitiveService = Get-AzCognitiveServicesAccount -Name $cognitiveServiceName -ResourceGroupName $resourceGroupName -ErrorAction SilentlyContinue

if ($cognitiveService) {
    Write-Host "`nFound existing Cognitive Services account:" -ForegroundColor Green
    Write-Host "  Name: $($cognitiveService.AccountName)" -ForegroundColor Gray
    Write-Host "  Type: $($cognitiveService.Kind)" -ForegroundColor Gray
    Write-Host "  SKU: $($cognitiveService.Sku.Name)" -ForegroundColor Gray
    Write-Host "  Custom Domain: $($cognitiveService.CustomDomainName)" -ForegroundColor Gray
    Write-Host "  Endpoint: $($cognitiveService.Endpoint)" -ForegroundColor Gray
    
    if ($cognitiveService.CustomDomainName) {
        Write-Host "`n⚠️  This resource has a custom domain configured!" -ForegroundColor Yellow
        Write-Host "   Custom domains cannot be updated via ARM/Bicep deployments." -ForegroundColor Yellow
    }
}
else {
    Write-Host "  Cognitive Services account not found." -ForegroundColor Gray
}

# Check OpenAI Service
$openAIService = Get-AzCognitiveServicesAccount -Name $openAIServiceName -ResourceGroupName $resourceGroupName -ErrorAction SilentlyContinue

if ($openAIService) {
    Write-Host "`nFound existing OpenAI Service account:" -ForegroundColor Green
    Write-Host "  Name: $($openAIService.AccountName)" -ForegroundColor Gray
    Write-Host "  Type: $($openAIService.Kind)" -ForegroundColor Gray
    Write-Host "  SKU: $($openAIService.Sku.Name)" -ForegroundColor Gray
    Write-Host "  Custom Domain: $($openAIService.CustomDomainName)" -ForegroundColor Gray
    Write-Host "  Endpoint: $($openAIService.Endpoint)" -ForegroundColor Gray
    
    if ($openAIService.CustomDomainName) {
        Write-Host "`n⚠️  This resource has a custom domain configured!" -ForegroundColor Yellow
        Write-Host "   Custom domains cannot be updated via ARM/Bicep deployments." -ForegroundColor Yellow
    }
}
else {
    Write-Host "  OpenAI Service account not found." -ForegroundColor Gray
}

# If only listing, exit here
if ($ListOnly) {
    Write-Host "`nUse -DeleteExisting to remove these resources and allow fresh deployment." -ForegroundColor Cyan
    exit 0
}

# Provide options if resources exist
if ($cognitiveService -or $openAIService) {
    Write-Host "`n=======================================" -ForegroundColor Cyan
    Write-Host "Options to resolve the deployment issue:" -ForegroundColor Yellow
    Write-Host "=======================================" -ForegroundColor Cyan
    
    Write-Host "`n1. Delete existing resources (data loss warning!):" -ForegroundColor White
    Write-Host "   Run: .\fix-ai-services-deployment.ps1 -Environment $Environment -DeleteExisting" -ForegroundColor Gray
    
    Write-Host "`n2. Use different resource names:" -ForegroundColor White
    Write-Host "   Update the ai-services.bicep module to use unique names" -ForegroundColor Gray
    
    Write-Host "`n3. Import existing resources into Bicep state:" -ForegroundColor White
    Write-Host "   Use 'existing' keyword in Bicep to reference without modifying" -ForegroundColor Gray
    
    if ($DeleteExisting) {
        Write-Host "`n⚠️  WARNING: You are about to delete AI Services resources!" -ForegroundColor Red
        Write-Host "   This action cannot be undone and may result in data loss." -ForegroundColor Red
        
        $confirmation = Read-Host "`nType 'DELETE' to confirm deletion"
        
        if ($confirmation -eq 'DELETE') {
            # Delete Cognitive Services
            if ($cognitiveService) {
                Write-Host "`nDeleting Cognitive Services account: $cognitiveServiceName" -ForegroundColor Yellow
                Remove-AzCognitiveServicesAccount -Name $cognitiveServiceName -ResourceGroupName $resourceGroupName -Force
                Write-Host "✅ Cognitive Services account deleted" -ForegroundColor Green
            }
            
            # Delete OpenAI Service
            if ($openAIService) {
                Write-Host "`nDeleting OpenAI Service account: $openAIServiceName" -ForegroundColor Yellow
                Remove-AzCognitiveServicesAccount -Name $openAIServiceName -ResourceGroupName $resourceGroupName -Force
                Write-Host "✅ OpenAI Service account deleted" -ForegroundColor Green
            }
            
            Write-Host "`n✅ Resources deleted. You can now redeploy the infrastructure." -ForegroundColor Green
            Write-Host "   Run the GitHub Actions workflow or use:" -ForegroundColor Gray
            Write-Host "   az deployment sub create --location 'East US' --template-file infrastructure/bicep/main.bicep --parameters infrastructure/bicep/environments/$Environment.bicepparam" -ForegroundColor Gray
        }
        else {
            Write-Host "`n❌ Deletion cancelled. No resources were deleted." -ForegroundColor Red
        }
    }
}
else {
    Write-Host "`n✅ No conflicting AI Services resources found." -ForegroundColor Green
    Write-Host "   The deployment error might be caused by a different issue." -ForegroundColor Gray
    Write-Host "   Check the deployment details in Azure Portal for more information." -ForegroundColor Gray
}

Write-Host "`n=======================================" -ForegroundColor Cyan
Write-Host "Additional Troubleshooting Steps:" -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "1. Check Azure Portal > Deployments for detailed error messages" -ForegroundColor Gray
Write-Host "2. Review the ai-services.bicep module for any hardcoded values" -ForegroundColor Gray
Write-Host "3. Ensure you have proper permissions to create/modify AI Services" -ForegroundColor Gray
Write-Host "4. Check if there are any Azure Policy restrictions in your subscription" -ForegroundColor Gray 