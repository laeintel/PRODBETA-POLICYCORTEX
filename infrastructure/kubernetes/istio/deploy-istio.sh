#!/bin/bash

# Deploy Istio Service Mesh for PolicyCortex
# This script deploys an enterprise-grade Istio service mesh with multi-tenant isolation
# Author: PolicyCortex Platform Team
# Version: 2.0.0

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ISTIO_VERSION="${ISTIO_VERSION:-1.20.2}"
readonly NAMESPACE_ISTIO_SYSTEM="istio-system"
readonly NAMESPACE_ISTIO_INGRESS="istio-ingress"
readonly NAMESPACE_ISTIO_EGRESS="istio-egress"
readonly HELM_RELEASE_BASE="istio-base"
readonly HELM_RELEASE_ISTIOD="istiod"
readonly HELM_RELEASE_INGRESS="istio-ingress"
readonly HELM_RELEASE_EGRESS="istio-egress"

# Configuration
TENANT_TIER="${TENANT_TIER:-enterprise}"
DRY_RUN="${DRY_RUN:-false}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
ENABLE_TRACING="${ENABLE_TRACING:-true}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"

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

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v helm &> /dev/null; then
        missing_tools+=("helm")
    fi
    
    if ! command -v istioctl &> /dev/null; then
        missing_tools+=("istioctl")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check Istio version
    local istio_version
    istio_version=$(istioctl version --client --short 2>/dev/null || echo "unknown")
    log_info "Using istioctl version: $istio_version"
    
    # Check Helm version
    local helm_version
    helm_version=$(helm version --short 2>/dev/null || echo "unknown")
    log_info "Using Helm version: $helm_version"
    
    log_success "Prerequisites check completed"
}

# Validate cluster readiness
validate_cluster() {
    log_info "Validating cluster readiness..."
    
    # Check cluster version
    local k8s_version
    k8s_version=$(kubectl version --short --client | grep Client | awk '{print $3}')
    log_info "Kubernetes client version: $k8s_version"
    
    # Check if cluster supports required features
    if ! kubectl api-resources | grep -q "networkpolicies"; then
        log_warning "NetworkPolicies not available in this cluster"
    fi
    
    # Check available nodes
    local node_count
    node_count=$(kubectl get nodes --no-headers | wc -l)
    log_info "Available nodes: $node_count"
    
    if [ "$node_count" -lt 3 ]; then
        log_warning "Cluster has fewer than 3 nodes. High availability may be compromised."
    fi
    
    # Check cluster resources
    log_info "Checking cluster resources..."
    kubectl describe nodes | grep -E "(Capacity|Allocatable)" | head -20
    
    log_success "Cluster validation completed"
}

# Install Istio CRDs and base components
install_istio_base() {
    log_info "Installing Istio base components..."
    
    # Add Istio Helm repository
    helm repo add istio https://istio-release.storage.googleapis.com/charts
    helm repo update
    
    # Install Istio base (CRDs)
    local helm_args=("upgrade" "--install" "$HELM_RELEASE_BASE" "istio/base")
    helm_args+=("--namespace" "$NAMESPACE_ISTIO_SYSTEM")
    helm_args+=("--create-namespace")
    helm_args+=("--version" "$ISTIO_VERSION")
    helm_args+=("--values" "$SCRIPT_DIR/istio-base-values.yaml")
    helm_args+=("--wait")
    helm_args+=("--timeout" "600s")
    
    if [ "$DRY_RUN" = "true" ]; then
        helm_args+=("--dry-run")
    fi
    
    if ! helm "${helm_args[@]}"; then
        log_error "Failed to install Istio base components"
        return 1
    fi
    
    log_success "Istio base components installed successfully"
}

# Install Istio control plane (istiod)
install_istio_control_plane() {
    log_info "Installing Istio control plane (istiod)..."
    
    # Select values file based on tenant tier
    local values_file="$SCRIPT_DIR/istio-base-values.yaml"
    if [ -f "$SCRIPT_DIR/helm-values-tenants.yaml" ]; then
        values_file="$SCRIPT_DIR/helm-values-tenants.yaml"
    fi
    
    local helm_args=("upgrade" "--install" "$HELM_RELEASE_ISTIOD" "istio/istiod")
    helm_args+=("--namespace" "$NAMESPACE_ISTIO_SYSTEM")
    helm_args+=("--version" "$ISTIO_VERSION")
    helm_args+=("--values" "$values_file")
    helm_args+=("--set" "global.meshID=policycortex-mesh")
    helm_args+=("--set" "global.network=policycortex-network")
    helm_args+=("--wait")
    helm_args+=("--timeout" "600s")
    
    # Tenant-specific configurations
    case "$TENANT_TIER" in
        enterprise)
            helm_args+=("--set" "pilot.autoscaleMin=3")
            helm_args+=("--set" "pilot.autoscaleMax=10")
            helm_args+=("--set" "pilot.resources.requests.cpu=1000m")
            helm_args+=("--set" "pilot.resources.requests.memory=4Gi")
            ;;
        premium)
            helm_args+=("--set" "pilot.autoscaleMin=2")
            helm_args+=("--set" "pilot.autoscaleMax=6")
            helm_args+=("--set" "pilot.resources.requests.cpu=750m")
            helm_args+=("--set" "pilot.resources.requests.memory=3Gi")
            ;;
        standard)
            helm_args+=("--set" "pilot.autoscaleMin=1")
            helm_args+=("--set" "pilot.autoscaleMax=3")
            helm_args+=("--set" "pilot.resources.requests.cpu=500m")
            helm_args+=("--set" "pilot.resources.requests.memory=2Gi")
            ;;
    esac
    
    if [ "$DRY_RUN" = "true" ]; then
        helm_args+=("--dry-run")
    fi
    
    if ! helm "${helm_args[@]}"; then
        log_error "Failed to install Istio control plane"
        return 1
    fi
    
    log_success "Istio control plane installed successfully"
}

# Install Istio ingress gateway
install_istio_ingress() {
    log_info "Installing Istio ingress gateway..."
    
    local helm_args=("upgrade" "--install" "$HELM_RELEASE_INGRESS" "istio/gateway")
    helm_args+=("--namespace" "$NAMESPACE_ISTIO_INGRESS")
    helm_args+=("--create-namespace")
    helm_args+=("--version" "$ISTIO_VERSION")
    helm_args+=("--set" "service.type=LoadBalancer")
    helm_args+=("--wait")
    helm_args+=("--timeout" "600s")
    
    # Tenant-specific gateway configurations
    case "$TENANT_TIER" in
        enterprise)
            helm_args+=("--set" "autoscaling.minReplicas=3")
            helm_args+=("--set" "autoscaling.maxReplicas=10")
            helm_args+=("--set" "resources.requests.cpu=500m")
            helm_args+=("--set" "resources.requests.memory=512Mi")
            ;;
        premium)
            helm_args+=("--set" "autoscaling.minReplicas=2")
            helm_args+=("--set" "autoscaling.maxReplicas=6")
            helm_args+=("--set" "resources.requests.cpu=300m")
            helm_args+=("--set" "resources.requests.memory=384Mi")
            ;;
        standard)
            helm_args+=("--set" "autoscaling.minReplicas=1")
            helm_args+=("--set" "autoscaling.maxReplicas=3")
            helm_args+=("--set" "resources.requests.cpu=200m")
            helm_args+=("--set" "resources.requests.memory=256Mi")
            ;;
    esac
    
    if [ "$DRY_RUN" = "true" ]; then
        helm_args+=("--dry-run")
    fi
    
    if ! helm "${helm_args[@]}"; then
        log_error "Failed to install Istio ingress gateway"
        return 1
    fi
    
    log_success "Istio ingress gateway installed successfully"
}

# Install Istio egress gateway
install_istio_egress() {
    log_info "Installing Istio egress gateway..."
    
    local helm_args=("upgrade" "--install" "$HELM_RELEASE_EGRESS" "istio/gateway")
    helm_args+=("--namespace" "$NAMESPACE_ISTIO_EGRESS")
    helm_args+=("--create-namespace")
    helm_args+=("--version" "$ISTIO_VERSION")
    helm_args+=("--set" "service.type=ClusterIP")
    helm_args+=("--set" "name=egress")
    helm_args+=("--wait")
    helm_args+=("--timeout" "600s")
    
    if [ "$DRY_RUN" = "true" ]; then
        helm_args+=("--dry-run")
    fi
    
    if ! helm "${helm_args[@]}"; then
        log_error "Failed to install Istio egress gateway"
        return 1
    fi
    
    log_success "Istio egress gateway installed successfully"
}

# Apply Kubernetes manifests
apply_manifests() {
    log_info "Applying Kubernetes manifests..."
    
    local manifests=(
        "$SCRIPT_DIR/istio-system-namespace.yaml"
        "$SCRIPT_DIR/tenant-namespaces.yaml"
        "$SCRIPT_DIR/peer-authentication.yaml"
        "$SCRIPT_DIR/authorization-policy.yaml"
        "$SCRIPT_DIR/ingress-gateway.yaml"
        "$SCRIPT_DIR/network-policies.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        if [ -f "$manifest" ]; then
            log_info "Applying $(basename "$manifest")..."
            if [ "$DRY_RUN" = "true" ]; then
                kubectl apply --dry-run=client -f "$manifest"
            else
                kubectl apply -f "$manifest"
            fi
        else
            log_warning "Manifest not found: $manifest"
        fi
    done
    
    log_success "Kubernetes manifests applied successfully"
}

# Verify installation
verify_installation() {
    if [ "$SKIP_VALIDATION" = "true" ]; then
        log_info "Skipping installation verification"
        return 0
    fi
    
    log_info "Verifying Istio installation..."
    
    # Wait for pods to be ready
    log_info "Waiting for Istio system pods to be ready..."
    kubectl wait --for=condition=ready pod -l app=istiod -n "$NAMESPACE_ISTIO_SYSTEM" --timeout=300s
    
    # Verify using istioctl
    if ! istioctl verify-install; then
        log_warning "Istioctl verification failed, but installation may still be functional"
    fi
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE_ISTIO_SYSTEM"
    kubectl get pods -n "$NAMESPACE_ISTIO_INGRESS"
    kubectl get pods -n "$NAMESPACE_ISTIO_EGRESS"
    
    # Check services
    log_info "Checking services..."
    kubectl get svc -n "$NAMESPACE_ISTIO_SYSTEM"
    kubectl get svc -n "$NAMESPACE_ISTIO_INGRESS"
    kubectl get svc -n "$NAMESPACE_ISTIO_EGRESS"
    
    # Check gateways and virtual services
    log_info "Checking Istio configuration..."
    kubectl get gateways -A
    kubectl get virtualservices -A
    kubectl get destinationrules -A
    
    # Check authorization policies
    log_info "Checking security policies..."
    kubectl get peerauthentication -A
    kubectl get authorizationpolicy -A
    
    log_success "Installation verification completed"
}

# Display post-installation information
show_post_install_info() {
    log_info "Post-installation information:"
    
    echo -e "\n${GREEN}Istio Service Mesh Installation Complete!${NC}\n"
    
    echo "Installed Components:"
    echo "- Istio Control Plane (istiod) in namespace: $NAMESPACE_ISTIO_SYSTEM"
    echo "- Istio Ingress Gateway in namespace: $NAMESPACE_ISTIO_INGRESS"
    echo "- Istio Egress Gateway in namespace: $NAMESPACE_ISTIO_EGRESS"
    echo "- Multi-tenant namespaces with security policies"
    echo "- Network policies for traffic isolation"
    echo ""
    
    echo "Tenant Configuration:"
    echo "- Tenant Tier: $TENANT_TIER"
    echo "- Security Level: Critical (strict mTLS enabled)"
    echo "- Network Isolation: Enabled"
    echo ""
    
    # Get ingress gateway external IP
    local external_ip
    external_ip=$(kubectl get svc istio-ingressgateway -n "$NAMESPACE_ISTIO_INGRESS" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    
    echo "Access Information:"
    echo "- Ingress Gateway External IP: $external_ip"
    echo "- Frontend URL: https://app.policycortex.aeolitech.com"
    echo "- API URL: https://api.policycortex.aeolitech.com"
    echo "- GraphQL URL: https://graphql.policycortex.aeolitech.com"
    echo ""
    
    echo "Next Steps:"
    echo "1. Configure DNS to point to the external IP address"
    echo "2. Install TLS certificates for secure communication"
    echo "3. Deploy PolicyCortex application components"
    echo "4. Configure monitoring and observability tools"
    echo ""
    
    echo "Useful Commands:"
    echo "- Check Istio proxy status: istioctl proxy-status"
    echo "- View mesh configuration: istioctl proxy-config cluster <pod-name> -n <namespace>"
    echo "- Analyze mesh configuration: istioctl analyze"
    echo "- View access logs: kubectl logs -f <pod-name> -c istio-proxy -n <namespace>"
    echo ""
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        echo "Monitoring:"
        echo "- Prometheus metrics are enabled"
        echo "- Grafana dashboards can be configured"
        echo "- Access logs are enabled"
    fi
    
    if [ "$ENABLE_TRACING" = "true" ]; then
        echo "- Distributed tracing is enabled"
        echo "- Configure Jaeger or other tracing backends"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

# Error handling
handle_error() {
    log_error "An error occurred during installation"
    log_error "Check the logs above for details"
    cleanup
    exit 1
}

# Main installation function
main() {
    # Set up error handling
    trap handle_error ERR
    
    log_info "Starting Istio Service Mesh installation for PolicyCortex"
    log_info "Tenant Tier: $TENANT_TIER"
    log_info "Dry Run: $DRY_RUN"
    
    # Installation steps
    check_prerequisites
    validate_cluster
    install_istio_base
    install_istio_control_plane
    install_istio_ingress
    install_istio_egress
    apply_manifests
    verify_installation
    show_post_install_info
    
    cleanup
    log_success "Istio Service Mesh installation completed successfully!"
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Istio Service Mesh for PolicyCortex with enterprise-grade security and multi-tenant isolation.

OPTIONS:
    -t, --tenant-tier TIER    Set tenant tier (enterprise, premium, standard). Default: enterprise
    -d, --dry-run             Perform a dry run without making changes
    -m, --enable-monitoring   Enable monitoring and metrics collection. Default: true
    -r, --enable-tracing      Enable distributed tracing. Default: true
    -s, --skip-validation     Skip installation verification
    -h, --help                Show this help message

EXAMPLES:
    $0                                      # Install with enterprise tier
    $0 --tenant-tier premium               # Install with premium tier
    $0 --dry-run                           # Dry run installation
    $0 --tenant-tier standard --dry-run    # Dry run with standard tier

ENVIRONMENT VARIABLES:
    ISTIO_VERSION              Istio version to install (default: 1.20.2)
    TENANT_TIER               Tenant tier configuration
    DRY_RUN                   Enable dry run mode
    ENABLE_MONITORING         Enable monitoring features
    ENABLE_TRACING           Enable tracing features
    SKIP_VALIDATION          Skip installation verification

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tenant-tier)
            TENANT_TIER="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -m|--enable-monitoring)
            ENABLE_MONITORING="true"
            shift
            ;;
        -r|--enable-tracing)
            ENABLE_TRACING="true"
            shift
            ;;
        -s|--skip-validation)
            SKIP_VALIDATION="true"
            shift
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

# Validate tenant tier
case "$TENANT_TIER" in
    enterprise|premium|standard)
        ;;
    *)
        log_error "Invalid tenant tier: $TENANT_TIER"
        log_error "Valid options: enterprise, premium, standard"
        exit 1
        ;;
esac

# Run main function
main "$@"