#!/bin/bash

# PolicyCortex Deployment Script
# Comprehensive deployment with health checks and rollback capability

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
ROLLBACK=${3:-false}

# Azure configuration
RESOURCE_GROUP="rg-policycortex-${ENVIRONMENT}"
CONTAINER_REGISTRY="policycortexacr.azurecr.io"
KEY_VAULT="kv-policycortex-${ENVIRONMENT}"

# Services to deploy
SERVICES=(
    "api-gateway:8000"
    "azure-integration:8001"
    "ai-engine:8002"
    "data-processing:8003"
    "conversation:8004"
    "notification:8005"
)

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check Azure CLI is installed
    if ! command -v az &> /dev/null; then
        error "Azure CLI is not installed"
        exit 1
    fi
    
    # Check logged in to Azure
    if ! az account show &> /dev/null; then
        error "Not logged in to Azure. Run 'az login' first"
        exit 1
    fi
    
    # Verify resource group exists
    if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        error "Resource group $RESOURCE_GROUP does not exist"
        exit 1
    fi
    
    # Verify container registry access
    if ! az acr show --name "${CONTAINER_REGISTRY%%.*}" &> /dev/null; then
        error "Cannot access container registry $CONTAINER_REGISTRY"
        exit 1
    fi
    
    log "Pre-deployment checks passed"
}

# Create deployment backup
create_backup() {
    log "Creating backup of current deployment..."
    
    BACKUP_TAG="backup-$(date +%Y%m%d-%H%M%S)"
    
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        # Get current revision
        current_revision=$(az containerapp show \
            --name "ca-${service}-${ENVIRONMENT}" \
            --resource-group "$RESOURCE_GROUP" \
            --query "properties.latestRevisionName" -o tsv 2>/dev/null || echo "")
        
        if [ -n "$current_revision" ]; then
            # Tag current revision for backup
            az containerapp revision label add \
                --name "ca-${service}-${ENVIRONMENT}" \
                --resource-group "$RESOURCE_GROUP" \
                --revision "$current_revision" \
                --label "$BACKUP_TAG" \
                --yes || warning "Failed to backup $service"
        fi
    done
    
    log "Backup created with tag: $BACKUP_TAG"
}

# Deploy service
deploy_service() {
    local service=$1
    local port=$2
    
    log "Deploying $service..."
    
    # Get secrets from Key Vault
    local db_url=$(az keyvault secret show \
        --vault-name "$KEY_VAULT" \
        --name "database-url" \
        --query "value" -o tsv)
    
    local redis_url=$(az keyvault secret show \
        --vault-name "$KEY_VAULT" \
        --name "redis-url" \
        --query "value" -o tsv)
    
    # Update Container App
    az containerapp update \
        --name "ca-${service}-${ENVIRONMENT}" \
        --resource-group "$RESOURCE_GROUP" \
        --image "${CONTAINER_REGISTRY}/${service}:${VERSION}" \
        --set-env-vars \
            "SERVICE_NAME=${service}" \
            "PORT=${port}" \
            "DATABASE_URL=secretref:database-url" \
            "REDIS_URL=secretref:redis-url" \
            "ENVIRONMENT=${ENVIRONMENT}" \
        --min-replicas 2 \
        --max-replicas 10 \
        --cpu 0.5 \
        --memory 1.0Gi \
        --revision-suffix "v${VERSION//\./-}" \
        --query "properties.latestRevisionFqdn" -o tsv
}

# Health check
health_check() {
    local service=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    log "Running health check for $service..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf "${url}/health" > /dev/null 2>&1; then
            log "Health check passed for $service"
            return 0
        fi
        
        warning "Health check attempt $attempt/$max_attempts failed for $service"
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed for $service after $max_attempts attempts"
    return 1
}

# Deploy all services
deploy_all_services() {
    log "Starting deployment of all services..."
    
    local failed_services=()
    
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        # Deploy service
        service_url=$(deploy_service "$service" "$port")
        
        if [ -z "$service_url" ]; then
            error "Failed to deploy $service"
            failed_services+=("$service")
            continue
        fi
        
        # Health check
        if ! health_check "$service" "https://$service_url"; then
            failed_services+=("$service")
        fi
        
        # Traffic routing (gradual rollout)
        log "Routing traffic to new revision for $service..."
        az containerapp ingress traffic set \
            --name "ca-${service}-${ENVIRONMENT}" \
            --resource-group "$RESOURCE_GROUP" \
            --label-weight "v${VERSION//\./-}=100"
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        error "Deployment failed for services: ${failed_services[*]}"
        return 1
    fi
    
    log "All services deployed successfully"
}

# Deploy frontend
deploy_frontend() {
    log "Deploying frontend..."
    
    # Build frontend with environment variables
    cd frontend
    
    # Create .env file with runtime config
    cat > .env.production << EOF
VITE_API_BASE_URL=https://ca-api-gateway-${ENVIRONMENT}.azurecontainerapps.io/api
VITE_WS_URL=wss://ca-api-gateway-${ENVIRONMENT}.azurecontainerapps.io/ws
VITE_AZURE_CLIENT_ID=$(az keyvault secret show --vault-name "$KEY_VAULT" --name "azure-client-id" --query "value" -o tsv)
VITE_AZURE_TENANT_ID=$(az keyvault secret show --vault-name "$KEY_VAULT" --name "azure-tenant-id" --query "value" -o tsv)
EOF
    
    # Build
    npm ci
    npm run build
    
    # Deploy to Azure Storage
    storage_account="stpolicycortex${ENVIRONMENT}"
    
    az storage blob upload-batch \
        --account-name "$storage_account" \
        --destination '$web' \
        --source dist \
        --overwrite
    
    # Enable CDN
    cdn_endpoint="cdn-policycortex-${ENVIRONMENT}"
    az cdn endpoint purge \
        --resource-group "$RESOURCE_GROUP" \
        --profile-name "cdn-policycortex" \
        --name "$cdn_endpoint" \
        --content-paths "/*"
    
    cd ..
    log "Frontend deployed successfully"
}

# Run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    local base_url="https://ca-api-gateway-${ENVIRONMENT}.azurecontainerapps.io"
    
    # Test health endpoints
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        if ! curl -sf "${base_url}/health/${service}" > /dev/null 2>&1; then
            error "Smoke test failed for $service"
            return 1
        fi
    done
    
    # Test authentication
    if ! curl -sf "${base_url}/api/v1/auth/health" > /dev/null 2>&1; then
        error "Authentication endpoint not responding"
        return 1
    fi
    
    log "All smoke tests passed"
}

# Rollback deployment
rollback_deployment() {
    log "Rolling back deployment..."
    
    for service_port in "${SERVICES[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        # Get latest backup revision
        backup_revision=$(az containerapp revision list \
            --name "ca-${service}-${ENVIRONMENT}" \
            --resource-group "$RESOURCE_GROUP" \
            --query "[?properties.labels.backup-* != null].name | [0]" -o tsv)
        
        if [ -n "$backup_revision" ]; then
            log "Rolling back $service to revision $backup_revision"
            
            az containerapp revision activate \
                --name "ca-${service}-${ENVIRONMENT}" \
                --resource-group "$RESOURCE_GROUP" \
                --revision "$backup_revision"
            
            # Route all traffic to backup
            az containerapp ingress traffic set \
                --name "ca-${service}-${ENVIRONMENT}" \
                --resource-group "$RESOURCE_GROUP" \
                --revision-weight "${backup_revision}=100"
        else
            warning "No backup found for $service"
        fi
    done
    
    log "Rollback completed"
}

# Send deployment notification
send_notification() {
    local status=$1
    local message=$2
    
    # Send to Teams/Slack webhook
    webhook_url=$(az keyvault secret show \
        --vault-name "$KEY_VAULT" \
        --name "deployment-webhook" \
        --query "value" -o tsv 2>/dev/null || echo "")
    
    if [ -n "$webhook_url" ]; then
        curl -H "Content-Type: application/json" \
            -d "{\"text\": \"PolicyCortex Deployment to ${ENVIRONMENT}: ${status}\\n${message}\"}" \
            "$webhook_url" 2>/dev/null || true
    fi
}

# Main deployment flow
main() {
    log "Starting PolicyCortex deployment to $ENVIRONMENT with version $VERSION"
    
    # Pre-deployment checks
    pre_deployment_checks
    
    # Create backup
    create_backup
    
    # Deploy services
    if deploy_all_services; then
        # Deploy frontend
        deploy_frontend
        
        # Run smoke tests
        if run_smoke_tests; then
            log "Deployment completed successfully!"
            send_notification "SUCCESS" "All services deployed and healthy"
            exit 0
        else
            error "Smoke tests failed"
            send_notification "FAILED" "Smoke tests failed"
        fi
    else
        error "Service deployment failed"
        send_notification "FAILED" "Service deployment failed"
    fi
    
    # Rollback on failure
    if [ "$ROLLBACK" = "true" ]; then
        rollback_deployment
        send_notification "ROLLBACK" "Deployment rolled back due to failures"
    fi
    
    exit 1
}

# Run main function
main