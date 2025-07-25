# Fix Container Dependencies and Redeploy
Write-Host "Fixing container dependencies and redeploying..." -ForegroundColor Green

# Set variables
$RESOURCE_GROUP = "rg-policortex001-app-dev"
$REGISTRY_NAME = "crpolicortex001dev"
$ENVIRONMENT_NAME = "cae-policortex001-app-dev"

# Services to fix (use underscore for directory names)
$SERVICES = @("ai_engine", "azure_integration", "conversation")

foreach ($SERVICE in $SERVICES) {
    Write-Host "Processing $SERVICE..." -ForegroundColor Yellow
    
    # Build and push Docker image
    $SERVICE_DASH = $SERVICE.Replace("_", "-")
    $IMAGE_NAME = "policortex001-$SERVICE_DASH"
    $IMAGE_TAG = "$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $FULL_IMAGE_NAME = "$REGISTRY_NAME.azurecr.io/$IMAGE_NAME`:$IMAGE_TAG"
    $LATEST_IMAGE_NAME = "$REGISTRY_NAME.azurecr.io/$IMAGE_NAME`:latest"
    
    Write-Host "Building Docker image: $FULL_IMAGE_NAME"
    docker build -t $FULL_IMAGE_NAME -f "backend/services/$SERVICE/Dockerfile" backend/
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully built $FULL_IMAGE_NAME" -ForegroundColor Green
        
        # Tag as latest
        docker tag $FULL_IMAGE_NAME $LATEST_IMAGE_NAME
        
        # Push to registry
        Write-Host "Pushing to registry..."
        docker push $FULL_IMAGE_NAME
        docker push $LATEST_IMAGE_NAME
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully pushed $FULL_IMAGE_NAME" -ForegroundColor Green
            
            # Update container app
            $APP_NAME = "ca-$SERVICE_DASH-dev"
            Write-Host "Updating container app: $APP_NAME"
            
            az containerapp update `
                --name $APP_NAME `
                --resource-group $RESOURCE_GROUP `
                --image $LATEST_IMAGE_NAME `
                --revision-suffix $IMAGE_TAG.Replace("-", "").Replace(":", "")
                
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Successfully updated $APP_NAME" -ForegroundColor Green
            } else {
                Write-Host "Failed to update $APP_NAME" -ForegroundColor Red
            }
        } else {
            Write-Host "Failed to push $FULL_IMAGE_NAME" -ForegroundColor Red
        }
    } else {
        Write-Host "Failed to build $FULL_IMAGE_NAME" -ForegroundColor Red
    }
    
    Write-Host "---" -ForegroundColor Gray
}

Write-Host "Container fix process completed!" -ForegroundColor Green
Write-Host "Check container status with:"
Write-Host "az containerapp list --resource-group $RESOURCE_GROUP --query `"[].{Name:name, State:properties.runningStatus}`" --output table"