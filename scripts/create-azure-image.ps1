# PowerShell script to create and configure Azure VM Image
# Run this from your local machine with Azure CLI installed

$resourceGroup = "shared-compute-image-repo"
$location = "centralus"
$vmName = "runner-image-builder"
$imageName = "ubuntu-2404-runner"
$galleryName = "sharedcomputeimagerepo"
$imageDefinitionName = "ubuntu-2404-selfhosted-runner"
$imageVersion = "1.0.0"

Write-Host "Step 1: Creating VM with Ubuntu 24.04" -ForegroundColor Green
az vm create `
    --resource-group $resourceGroup `
    --name $vmName `
    --image "Canonical:ubuntu-24_04-lts:server:latest" `
    --size "Standard_D4s_v3" `
    --admin-username "azureuser" `
    --generate-ssh-keys `
    --public-ip-sku Standard

Write-Host "Step 2: Get VM Public IP" -ForegroundColor Green
$publicIp = az vm show -d -g $resourceGroup -n $vmName --query publicIps -o tsv
Write-Host "VM Public IP: $publicIp" -ForegroundColor Yellow

Write-Host "Step 3: Copy setup script to VM" -ForegroundColor Green
Write-Host "Run: scp setup-runner-image.sh azureuser@${publicIp}:~/" -ForegroundColor Yellow

Write-Host "Step 4: SSH into VM and run setup" -ForegroundColor Green
Write-Host "Run: ssh azureuser@$publicIp" -ForegroundColor Yellow
Write-Host "Then run: chmod +x setup-runner-image.sh && sudo ./setup-runner-image.sh" -ForegroundColor Yellow

Write-Host "Step 5: After setup completes, generalize the VM (run this ON the VM)" -ForegroundColor Green
Write-Host @"
sudo waagent -deprovision+user -force
exit
"@ -ForegroundColor Yellow

Write-Host "Step 6: Deallocate the VM" -ForegroundColor Green
Read-Host "Press Enter after you've generalized the VM..."
az vm deallocate --resource-group $resourceGroup --name $vmName

Write-Host "Step 7: Generalize the VM" -ForegroundColor Green
az vm generalize --resource-group $resourceGroup --name $vmName

Write-Host "Step 8: Create image definition in gallery (if not exists)" -ForegroundColor Green
az sig image-definition create `
    --resource-group $resourceGroup `
    --gallery-name $galleryName `
    --gallery-image-definition $imageDefinitionName `
    --publisher "MyOrganization" `
    --offer "SelfHostedRunner" `
    --sku "Ubuntu2404" `
    --os-type Linux `
    --os-state Generalized `
    --hyper-v-generation V2 `
    --features SecurityType=TrustedLaunchSupported `
    --architecture x64

Write-Host "Step 9: Create image version from VM" -ForegroundColor Green
$vmId = az vm show --resource-group $resourceGroup --name $vmName --query id -o tsv
az sig image-version create `
    --resource-group $resourceGroup `
    --gallery-name $galleryName `
    --gallery-image-definition $imageDefinitionName `
    --gallery-image-version $imageVersion `
    --target-regions $location `
    --replica-count 1 `
    --managed-image $vmId

Write-Host "Step 10: Clean up builder VM (optional)" -ForegroundColor Green
$cleanup = Read-Host "Delete the builder VM? (y/n)"
if ($cleanup -eq 'y') {
    az vm delete --resource-group $resourceGroup --name $vmName --yes
}

Write-Host "Image creation complete!" -ForegroundColor Green
Write-Host "Image location: /subscriptions/babdc901-44aa-41d9-a8d5-955ac92dbfe0/resourceGroups/$resourceGroup/providers/Microsoft.Compute/galleries/$galleryName/images/$imageDefinitionName/versions/$imageVersion" -ForegroundColor Cyan