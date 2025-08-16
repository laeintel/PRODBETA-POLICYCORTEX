#!/bin/bash
set -e

echo "🚀 Deploying Istio Service Mesh for PolicyCortex..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "❌ Helm not found. Please install Helm first."
    exit 1
fi

# Add Istio Helm repository
echo "📦 Adding Istio Helm repository..."
helm repo add istio https://istio-release.storage.googleapis.com/charts
helm repo update

# Create istio-system namespace
echo "🏗️ Creating istio-system namespace..."
kubectl apply -f istio-system-namespace.yaml

# Install Istio base components
echo "⚙️ Installing Istio base components..."
helm install istio-base istio/base -n istio-system --create-namespace

# Install Istio control plane (istiod)
echo "🎮 Installing Istio control plane..."
helm install istiod istio/istiod -n istio-system --wait -f istio-base-values.yaml

# Install Istio ingress gateway
echo "🚪 Installing Istio ingress gateway..."
kubectl apply -f ingress-gateway.yaml

# Wait for Istio components to be ready
echo "⏳ Waiting for Istio components to be ready..."
kubectl wait --for=condition=ready pod -l app=istiod -n istio-system --timeout=300s
kubectl wait --for=condition=ready pod -l istio=ingressgateway -n istio-system --timeout=300s

# Create tenant namespaces
echo "🏢 Creating tenant namespaces..."
kubectl apply -f tenant-namespaces.yaml

# Apply PeerAuthentication for mTLS
echo "🔐 Configuring mTLS..."
kubectl apply -f peer-authentication.yaml

# Apply AuthorizationPolicies
echo "🛡️ Configuring authorization policies..."
kubectl apply -f authorization-policy.yaml

# Apply NetworkPolicies for tenant isolation
echo "🔒 Applying network policies for tenant isolation..."
kubectl apply -f network-policies.yaml

# Create resource quotas for each tenant
echo "📊 Creating resource quotas..."
for tenant in default alpha beta enterprise; do
    kubectl create quota tenant-quota-$tenant \
        --namespace=tenant-$tenant \
        --hard=requests.cpu=1,requests.memory=1Gi,limits.cpu=2,limits.memory=2Gi \
        --dry-run=client -o yaml | kubectl apply -f -
done

# Label namespaces for Istio injection
echo "💉 Enabling Istio injection for application namespaces..."
kubectl label namespace tenant-default istio-injection=enabled --overwrite
kubectl label namespace tenant-alpha istio-injection=enabled --overwrite
kubectl label namespace tenant-beta istio-injection=enabled --overwrite
kubectl label namespace tenant-enterprise istio-injection=enabled --overwrite

# Verify installation
echo "✅ Verifying Istio installation..."
kubectl get pods -n istio-system
kubectl get svc -n istio-system

echo "🎉 Istio Service Mesh deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Deploy PolicyCortex applications to tenant namespaces"
echo "2. Configure TLS certificates for the ingress gateway"
echo "3. Set up monitoring with Kiali, Prometheus, and Grafana"
echo "4. Configure external DNS for the ingress gateway LoadBalancer IP"
echo ""
echo "🔍 To check Istio status: istioctl proxy-status"
echo "📊 To view Istio dashboard: istioctl dashboard kiali"