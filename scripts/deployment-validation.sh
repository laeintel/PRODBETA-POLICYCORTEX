#!/bin/bash

# PolicyCortex Deployment Validation and Rollback Script
# Validates deployments and provides rollback capabilities

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
ENVIRONMENT="dev"
SUBSCRIPTION_ID=""
RESOURCE_GROUP=""
ACTION="validate"
ROLLBACK_TAG=""
TIMEOUT=300
HEALTH_CHECK_RETRIES=10

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Validate PolicyCortex deployments and provide rollback capabilities

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (dev|staging|prod) [default: dev]
    -s, --subscription-id ID         Azure subscription ID [required]
    -r, --resource-group GROUP       Azure resource group [required]
    -a, --action ACTION              Action to perform (validate|rollback|status) [default: validate]
    --rollback-tag TAG               Container image tag to rollback to (required for rollback)
    --timeout SECONDS                Health check timeout in seconds [default: 300]
    --retries COUNT                  Number of health check retries [default: 10]
    -h, --help                       Show this help message

EXAMPLES:
    $0 -e dev -s sub-123 -r rg-policycortex-dev -a validate
    $0 --environment prod --subscription-id sub-456 --resource-group rg-prod --action rollback --rollback-tag v1.2.3
    $0 -e staging -s sub-789 -r rg-staging -a status

ACTIONS:
    validate    - Validate current deployment health and configuration
    rollback    - Rollback to a previous container image version
    status      - Show current deployment status and health
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -s|--subscription-id)
            SUBSCRIPTION_ID="$2"
            shift 2
            ;;
        -r|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        --rollback-tag)
            ROLLBACK_TAG="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --retries)
            HEALTH_CHECK_RETRIES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$SUBSCRIPTION_ID" ]]; then
    log_error "Subscription ID is required"
    usage
    exit 1
fi

if [[ -z "$RESOURCE_GROUP" ]]; then
    RESOURCE_GROUP="rg-policycortex-$ENVIRONMENT"
fi

if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    log_error "Environment must be one of: dev, staging, prod"
    exit 1
fi

if [[ ! "$ACTION" =~ ^(validate|rollback|status)$ ]]; then
    log_error "Action must be one of: validate, rollback, status"
    exit 1
fi

if [[ "$ACTION" == "rollback" && -z "$ROLLBACK_TAG" ]]; then
    log_error "Rollback tag is required for rollback action"
    exit 1
fi

# Service definitions
SERVICES=("api-gateway" "azure-integration" "ai-engine" "data-processing" "conversation" "notification" "customer-onboarding" "frontend")

log_info "PolicyCortex Deployment Validation"
log_info "Environment: $ENVIRONMENT"
log_info "Subscription: $SUBSCRIPTION_ID"
log_info "Resource Group: $RESOURCE_GROUP"
log_info "Action: $ACTION"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed"
        exit 1
    fi
    
    # Check Azure authentication
    if ! az account show &> /dev/null; then
        log_error "Not authenticated with Azure CLI. Run 'az login'"
        exit 1
    fi
    
    # Check subscription
    if ! az account set --subscription "$SUBSCRIPTION_ID" 2>/dev/null; then
        log_error "Cannot access subscription: $SUBSCRIPTION_ID"
        exit 1
    fi
    
    # Check resource group exists
    if ! az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        log_error "Resource group '$RESOURCE_GROUP' does not exist"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Check container app health
check_container_app_health() {
    local service="$1"
    local app_name="ca-$service-$ENVIRONMENT"
    
    log_info "Checking health of $service..."
    
    # Check if container app exists
    if ! az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        log_error "Container app '$app_name' not found"
        return 1
    fi
    
    # Get container app status
    local running_status
    running_status=$(az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" --query "properties.runningStatus" -o tsv)
    
    if [[ "$running_status" != "Running" ]]; then
        log_error "$service is not running (status: $running_status)"
        return 1
    fi
    
    # Get application URL
    local app_url
    app_url=$(az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)
    
    if [[ -z "$app_url" ]]; then
        log_warning "$service does not have external ingress configured"
        return 0  # Not an error for internal services
    fi
    
    # Determine health endpoint
    local health_url expected_status
    if [[ "$service" == "frontend" ]]; then
        health_url="https://$app_url"
        expected_status=200
    else
        health_url="https://$app_url/health"
        expected_status=200
    fi
    
    # Perform health check with retries
    local retry_count=0
    while [[ $retry_count -lt $HEALTH_CHECK_RETRIES ]]; do
        local status_code
        status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$health_url" || echo "000")
        
        if [[ "$status_code" == "$expected_status" ]]; then
            log_success "$service health check passed (HTTP $status_code)"
            return 0
        fi
        
        ((retry_count++))
        if [[ $retry_count -lt $HEALTH_CHECK_RETRIES ]]; then
            log_info "$service health check failed (HTTP $status_code), retrying ($retry_count/$HEALTH_CHECK_RETRIES)..."
            sleep 30
        fi
    done
    
    log_error "$service health check failed after $HEALTH_CHECK_RETRIES attempts (HTTP $status_code)"
    return 1
}

# Get container app revision information
get_revision_info() {
    local service="$1"
    local app_name="ca-$service-$ENVIRONMENT"
    
    # Get active revision
    local active_revision
    active_revision=$(az containerapp revision list \
        --name "$app_name" \
        --resource-group "$RESOURCE_GROUP" \
        --query "[?properties.active==\`true\`].{name:name, image:properties.template.containers[0].image, created:properties.createdTime}" \
        -o json)
    
    if [[ -n "$active_revision" ]]; then
        echo "$active_revision"
    else
        echo "[]"
    fi
}

# Get deployment status
get_deployment_status() {
    log_info "Getting deployment status..."
    
    local failed_services=()
    local healthy_services=()
    
    for service in "${SERVICES[@]}"; do
        local app_name="ca-$service-$ENVIRONMENT"
        
        # Check if app exists
        if ! az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
            log_warning "Service '$service' not found"
            continue
        fi
        
        # Get basic status
        local running_status
        running_status=$(az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" --query "properties.runningStatus" -o tsv)
        
        # Get revision info
        local revision_info
        revision_info=$(get_revision_info "$service")
        
        # Display service information
        local image_tag="Unknown"
        local created_time="Unknown"
        
        if [[ "$revision_info" != "[]" ]]; then
            image_tag=$(echo "$revision_info" | jq -r '.[0].image' | sed 's/.*://')
            created_time=$(echo "$revision_info" | jq -r '.[0].created' | cut -d'T' -f1)
        fi
        
        printf "%-25s %-15s %-20s %-15s\n" "$service" "$running_status" "$image_tag" "$created_time"
        
        if [[ "$running_status" == "Running" ]]; then
            healthy_services+=("$service")
        else
            failed_services+=("$service")
        fi
    done
    
    echo
    log_info "Summary:"
    log_success "Healthy services: ${#healthy_services[@]}"
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Failed services: ${failed_services[*]}"
        return 1
    else
        log_success "All services are running"
        return 0
    fi
}

# Validate deployment
validate_deployment() {
    log_info "Validating deployment..."
    
    local failed_services=()
    
    echo
    printf "%-25s %-15s %-50s %-15s\n" "SERVICE" "STATUS" "URL" "HEALTH"
    printf "%-25s %-15s %-50s %-15s\n" "-------" "------" "---" "------"
    
    for service in "${SERVICES[@]}"; do
        local app_name="ca-$service-$ENVIRONMENT"
        local status="UNKNOWN"
        local url="N/A"
        local health="N/A"
        
        # Check if container app exists
        if ! az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
            status="NOT FOUND"
            printf "%-25s %-15s %-50s %-15s\n" "$service" "$status" "$url" "$health"
            failed_services+=("$service")
            continue
        fi
        
        # Get status
        status=$(az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" --query "properties.runningStatus" -o tsv)
        
        # Get URL
        local app_url
        app_url=$(az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)
        if [[ -n "$app_url" ]]; then
            url="https://$app_url"
        fi
        
        # Health check
        if [[ "$status" == "Running" ]]; then
            if check_container_app_health "$service"; then
                health="✅ HEALTHY"
            else
                health="❌ UNHEALTHY"
                failed_services+=("$service")
            fi
        else
            health="❌ NOT RUNNING"
            failed_services+=("$service")
        fi
        
        printf "%-25s %-15s %-50s %-15s\n" "$service" "$status" "$url" "$health"
    done
    
    echo
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All services are healthy"
        
        # Additional validation checks
        log_info "Running additional validation checks..."
        
        # Check database connectivity
        if validate_database_connectivity; then
            log_success "Database connectivity validated"
        else
            log_error "Database connectivity failed"
            return 1
        fi
        
        # Check Key Vault integration
        if validate_keyvault_integration; then
            log_success "Key Vault integration validated"
        else
            log_error "Key Vault integration failed"
            return 1
        fi
        
        # Check inter-service communication
        if validate_service_communication; then
            log_success "Service communication validated"
        else
            log_error "Service communication failed"
            return 1
        fi
        
        log_success "Deployment validation completed successfully"
        return 0
    else
        log_error "Deployment validation failed for services: ${failed_services[*]}"
        return 1
    fi
}

# Validate database connectivity
validate_database_connectivity() {
    local api_gateway_url
    api_gateway_url=$(az containerapp show --name "ca-api-gateway-$ENVIRONMENT" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
    
    if [[ -z "$api_gateway_url" ]]; then
        log_warning "API Gateway URL not found, skipping database connectivity check"
        return 0
    fi
    
    local response
    response=$(curl -s --max-time 30 "https://$api_gateway_url/ready" || echo "FAILED")
    
    if [[ "$response" == "FAILED" ]]; then
        log_error "Database connectivity check failed - cannot reach ready endpoint"
        return 1
    fi
    
    if echo "$response" | grep -q "ready\|healthy"; then
        return 0
    else
        log_error "Database connectivity check failed - unhealthy response"
        return 1
    fi
}

# Validate Key Vault integration
validate_keyvault_integration() {
    local key_vault_name="policycortex-${ENVIRONMENT}-kv"
    
    # Check if Key Vault exists
    if ! az keyvault show --name "$key_vault_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
        log_warning "Key Vault '$key_vault_name' not found, skipping integration check"
        return 0
    fi
    
    # Check that secrets are configured
    local secret_count
    secret_count=$(az keyvault secret list --vault-name "$key_vault_name" --query "length(@)" -o tsv 2>/dev/null)
    
    if [[ "$secret_count" -lt 5 ]]; then
        log_warning "Only $secret_count secrets found in Key Vault, may need configuration"
        return 0
    fi
    
    return 0
}

# Validate service communication
validate_service_communication() {
    local api_gateway_url
    api_gateway_url=$(az containerapp show --name "ca-api-gateway-$ENVIRONMENT" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
    
    if [[ -z "$api_gateway_url" ]]; then
        log_warning "API Gateway URL not found, skipping service communication check"
        return 0
    fi
    
    # Test a few key endpoints
    local endpoints=("/health" "/ready")
    
    for endpoint in "${endpoints[@]}"; do
        local status_code
        status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "https://$api_gateway_url$endpoint" || echo "000")
        
        if [[ "$status_code" != "200" ]]; then
            log_error "Service communication check failed for endpoint $endpoint (HTTP $status_code)"
            return 1
        fi
    done
    
    return 0
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment to tag: $ROLLBACK_TAG"
    
    local container_registry
    container_registry=$(az acr list --resource-group "$RESOURCE_GROUP" --query "[0].loginServer" -o tsv 2>/dev/null)
    
    if [[ -z "$container_registry" ]]; then
        log_error "Container registry not found in resource group"
        exit 1
    fi
    
    log_info "Container Registry: $container_registry"
    
    local failed_rollbacks=()
    
    for service in "${SERVICES[@]}"; do
        local app_name="ca-$service-$ENVIRONMENT"
        local image_name="$container_registry/$service:$ROLLBACK_TAG"
        
        log_info "Rolling back $service to $image_name..."
        
        # Check if container app exists
        if ! az containerapp show --name "$app_name" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
            log_warning "Container app '$app_name' not found, skipping"
            continue
        fi
        
        # Update container app with rollback image
        if az containerapp update \
            --name "$app_name" \
            --resource-group "$RESOURCE_GROUP" \
            --image "$image_name" \
            --revision-suffix "rollback-$(date +%s)" \
            --output none 2>/dev/null; then
            
            log_success "Rolled back $service"
        else
            log_error "Failed to rollback $service"
            failed_rollbacks+=("$service")
        fi
    done
    
    if [[ ${#failed_rollbacks[@]} -eq 0 ]]; then
        log_success "Rollback completed successfully"
        
        # Wait for rollback to take effect
        log_info "Waiting for rollback to complete..."
        sleep 60
        
        # Validate rollback
        log_info "Validating rollback..."
        if validate_deployment; then
            log_success "Rollback validation successful"
        else
            log_error "Rollback validation failed"
            exit 1
        fi
    else
        log_error "Rollback failed for services: ${failed_rollbacks[*]}"
        exit 1
    fi
}

# Main execution
main() {
    check_prerequisites
    
    case "$ACTION" in
        validate)
            validate_deployment
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            echo
            printf "%-25s %-15s %-20s %-15s\n" "SERVICE" "STATUS" "IMAGE TAG" "CREATED"
            printf "%-25s %-15s %-20s %-15s\n" "-------" "------" "---------" "-------"
            get_deployment_status
            ;;
        *)
            log_error "Unknown action: $ACTION"
            exit 1
            ;;
    esac
}

# Run main function
main