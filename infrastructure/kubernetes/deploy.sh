#!/bin/bash

# Kubernetes deployment script for PolicyCortex
# This script deploys the PolicyCortex microservices to AKS

set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-dev}
ACR_NAME=${ACR_NAME:-""}
NAMESPACE="policycortex"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
    fi
    
    if ! command -v envsubst &> /dev/null; then
        error "envsubst is not installed"
    fi
    
    if [ -z "$ACR_NAME" ]; then
        error "ACR_NAME environment variable is required"
    fi
    
    # Check if connected to correct AKS cluster
    CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
    if [ -z "$CURRENT_CONTEXT" ]; then
        error "Not connected to any Kubernetes cluster. Run: az aks get-credentials --resource-group <rg> --name <cluster-name>"
    fi
    
    log "Connected to cluster: $CURRENT_CONTEXT"
}

# Deploy namespace and basic resources
deploy_namespace() {
    log "Deploying namespace and basic resources..."
    
    # Create namespace
    envsubst < manifests/namespace.yaml | kubectl apply -f -
    
    # Create ConfigMap
    kubectl apply -f manifests/configmap.yaml
    
    log "Namespace and ConfigMap deployed"
}

# Deploy secrets (template - requires manual secret creation)
deploy_secrets() {
    log "Checking secrets..."
    
    if kubectl get secret policycortex-secrets -n $NAMESPACE &> /dev/null; then
        log "Secrets already exist"
    else
        warn "Secrets do not exist. You need to create them manually or using Azure Key Vault CSI driver"
        warn "Template file: manifests/secrets.yaml"
        warn "Consider using: kubectl create secret generic policycortex-secrets --from-env-file=.env -n $NAMESPACE"
    fi
}

# Deploy application services
deploy_services() {
    log "Deploying microservices..."
    
    # Replace ACR_NAME in manifest files and deploy
    for manifest in manifests/*.yaml; do
        case $(basename "$manifest") in
            "namespace.yaml"|"configmap.yaml"|"secrets.yaml"|"ingress.yaml")
                # Skip these as they're handled separately
                continue
                ;;
            *)
                log "Deploying $(basename "$manifest" .yaml)..."
                envsubst < "$manifest" | kubectl apply -f -
                ;;
        esac
    done
    
    log "All services deployed"
}

# Deploy ingress
deploy_ingress() {
    log "Deploying ingress..."
    
    # Check if ingress controller is available
    if kubectl get ingressclass nginx &> /dev/null; then
        log "NGINX ingress controller found"
    elif kubectl get ingressclass azure-application-gateway &> /dev/null; then
        log "Azure Application Gateway ingress controller found"
    else
        warn "No ingress controller found. Install NGINX or AGIC first."
    fi
    
    envsubst < manifests/ingress.yaml | kubectl apply -f -
    log "Ingress deployed"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log "Waiting for deployments to be ready..."
    
    DEPLOYMENTS=("api-gateway" "azure-integration" "ai-engine" "data-processing" "conversation" "notification")
    
    for deployment in "${DEPLOYMENTS[@]}"; do
        log "Waiting for $deployment..."
        kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $NAMESPACE
    done
    
    log "All deployments are ready"
}

# Get service information
get_service_info() {
    log "Service information:"
    kubectl get services -n $NAMESPACE
    
    log "\nIngress information:"
    kubectl get ingress -n $NAMESPACE
    
    log "\nPod status:"
    kubectl get pods -n $NAMESPACE
}

# Main deployment function
main() {
    log "Starting PolicyCortex Kubernetes deployment..."
    log "Environment: $ENVIRONMENT"
    log "ACR Name: $ACR_NAME"
    log "Namespace: $NAMESPACE"
    
    check_prerequisites
    deploy_namespace
    deploy_secrets
    deploy_services
    deploy_ingress
    wait_for_deployments
    get_service_info
    
    log "Deployment completed successfully!"
    log "Access your application at: https://api.policycortex.com"
}

# Run main function
main "$@"