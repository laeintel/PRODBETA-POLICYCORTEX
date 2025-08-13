# Create gallery image version from the snapshot

$resourceGroup = "shared-compute-image-repo"
$location = "centralus"
$galleryName = "sharedcomputeimagerepo"
$imageDefinitionName = "ubuntu-2404-selfhosted-runner"
$imageVersion = "1.0.0"
$snapshotName = "runner-snapshot-202508131135"

Write-Host "Creating gallery image version from snapshot..." -ForegroundColor Green
Write-Host "This may take 10-15 minutes..." -ForegroundColor Yellow

# Get snapshot ID
$snapshotId = az snapshot show --resource-group $resourceGroup --name $snapshotName --query id -o tsv

# Create image version with correct storage account type format
az sig image-version create `
    --resource-group $resourceGroup `
    --gallery-name $galleryName `
    --gallery-image-definition $imageDefinitionName `
    --gallery-image-version $imageVersion `
    --os-snapshot $snapshotId `
    --target-regions "centralus=1=standard_lrs"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "IMAGE CREATED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    
    Write-Host "`nImage Details:" -ForegroundColor Cyan
    Write-Host "Gallery: $galleryName"
    Write-Host "Image Definition: $imageDefinitionName"
    Write-Host "Version: $imageVersion"
    Write-Host "Location: $location"
    
    Write-Host "`n=== SAVE THIS IMAGE ID ===" -ForegroundColor Yellow
    Write-Host "/subscriptions/babdc901-44aa-41d9-a8d5-955ac92dbfe0/resourceGroups/$resourceGroup/providers/Microsoft.Compute/galleries/$galleryName/images/$imageDefinitionName/versions/$imageVersion" -ForegroundColor Cyan
    
    Write-Host "`n=== CREATE VM FROM IMAGE ===" -ForegroundColor Green
    Write-Host @"
az vm create \
    --resource-group <your-rg> \
    --name <vm-name> \
    --image "/subscriptions/babdc901-44aa-41d9-a8d5-955ac92dbfe0/resourceGroups/$resourceGroup/providers/Microsoft.Compute/galleries/$galleryName/images/$imageDefinitionName/versions/$imageVersion" \
    --size Standard_D4s_v3 \
    --admin-username azureuser \
    --generate-ssh-keys
"@ -ForegroundColor Yellow
    
    Write-Host "`n=== CREATE VMSS FROM IMAGE ===" -ForegroundColor Green
    Write-Host @"
az vmss create \
    --resource-group <your-rg> \
    --name <vmss-name> \
    --image "/subscriptions/babdc901-44aa-41d9-a8d5-955ac92dbfe0/resourceGroups/$resourceGroup/providers/Microsoft.Compute/galleries/$galleryName/images/$imageDefinitionName/versions/$imageVersion" \
    --instance-count 2 \
    --admin-username azureuser \
    --generate-ssh-keys
"@ -ForegroundColor Yellow
    
    Write-Host "`nCleanup Options:" -ForegroundColor Yellow
    $cleanupVM = Read-Host "Delete the source VM? (y/n)"
    if ($cleanupVM -eq 'y') {
        az vm delete --resource-group $resourceGroup --name runner-image-builder --yes
        Write-Host "VM deleted" -ForegroundColor Green
    }
    
    $cleanupSnapshot = Read-Host "Delete the snapshot? (y/n)"
    if ($cleanupSnapshot -eq 'y') {
        az snapshot delete --resource-group $resourceGroup --name $snapshotName --yes
        Write-Host "Snapshot deleted" -ForegroundColor Green
    }
    
    Write-Host "`nYour custom Ubuntu 24.04 image with all tools is ready!" -ForegroundColor Green
    Write-Host "It includes: PowerShell Core, Docker, Node.js, npm, Python packages, and all development tools." -ForegroundColor Cyan
} else {
    Write-Host "Failed to create image version from snapshot" -ForegroundColor Red
}