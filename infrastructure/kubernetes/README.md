# PolicyCortex Kubernetes Deployment

This directory contains Kubernetes manifests and deployment scripts for running PolicyCortex on Azure Kubernetes Service (AKS).

## Architecture

The Kubernetes deployment includes:
- **AKS Cluster**: Managed Kubernetes cluster with auto-scaling
- **6 Microservices**: All PolicyCortex services as Kubernetes deployments
- **Load Balancing**: Service-level load balancing and ingress
- **Secrets Management**: Integration with Azure Key Vault
- **Monitoring**: Built-in monitoring with Azure Monitor
- **AI Workloads**: Optional dedicated GPU node pool

## Prerequisites

1. **Azure CLI** with AKS permissions
2. **kubectl** configured for your AKS cluster
3. **Helm** (for ingress controller installation)
4. **Container Registry** with pushed images

## Quick Start

### 1. Deploy Infrastructure

First, deploy the AKS cluster using Terraform:

```bash
cd infrastructure/terraform

# Enable Kubernetes deployment
terraform plan -var="deploy_kubernetes=true" -var-file="environments/dev/terraform.tfvars"
terraform apply -var="deploy_kubernetes=true" -var-file="environments/dev/terraform.tfvars"
```

### 2. Configure kubectl

```bash
# Get AKS credentials
az aks get-credentials --resource-group policycortex-rg-dev --name policycortex-aks-dev

# Verify connection
kubectl cluster-info
```

### 3. Deploy Applications

```bash
cd infrastructure/kubernetes

# Set required environment variables
export ACR_NAME="your-acr-name"
export ENVIRONMENT="dev"

# Run deployment script
chmod +x deploy.sh
./deploy.sh
```

## Manual Deployment Steps

If you prefer manual deployment:

### 1. Create Namespace
```bash
kubectl apply -f manifests/namespace.yaml
```

### 2. Create Secrets
```bash
# Create from environment file
kubectl create secret generic policycortex-secrets \
  --from-env-file=.env \
  --namespace=policycortex

# Or manually create secrets
kubectl create secret generic policycortex-secrets \
  --from-literal=database-url="your-db-url" \
  --from-literal=redis-url="your-redis-url" \
  --namespace=policycortex
```

### 3. Deploy Services
```bash
# Replace ACR_NAME with your registry
export ACR_NAME="youracr"

# Deploy all services
for manifest in manifests/*.yaml; do
  envsubst < "$manifest" | kubectl apply -f -
done
```

## Configuration

### Environment Variables

Set these environment variables before deployment:

```bash
export ACR_NAME="your-acr-name"
export ENVIRONMENT="dev"  # or staging, prod
```

### Terraform Variables

Configure Kubernetes deployment in your `terraform.tfvars`:

```hcl
# Enable Kubernetes deployment
deploy_kubernetes = true

# Kubernetes configuration
kubernetes_version = "1.28.3"
kubernetes_node_count = 3
kubernetes_node_vm_size = "Standard_D4s_v3"
kubernetes_enable_auto_scaling = true
kubernetes_min_node_count = 2
kubernetes_max_node_count = 10

# AI workloads (optional)
kubernetes_enable_ai_node_pool = true
kubernetes_ai_node_vm_size = "Standard_NC6s_v3"
kubernetes_ai_node_count = 1
```

## Ingress Configuration

### NGINX Ingress Controller

The deployment uses NGINX for internal routing:

```bash
# Install NGINX ingress controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace
```

### Azure Application Gateway (AGIC)

For production, use Azure Application Gateway:

```bash
# Enable AGIC in Terraform
enable_application_gateway = true
```

## Services

The deployment includes these microservices:

| Service | Port | Path | Description |
|---------|------|------|-------------|
| API Gateway | 8000 | `/api/v1/gateway` | Main entry point |
| Azure Integration | 8001 | `/api/v1/azure` | Azure API integration |
| AI Engine | 8002 | `/api/v1/ai` | ML model inference |
| Data Processing | 8003 | `/api/v1/data` | ETL pipelines |
| Conversation | 8004 | `/api/v1/conversation` | Natural language interface |
| Notification | 8005 | `/api/v1/notification` | Alerts and notifications |

## Scaling

### Manual Scaling
```bash
# Scale a specific service
kubectl scale deployment api-gateway --replicas=5 -n policycortex
```

### Auto-scaling
```bash
# Enable HPA (Horizontal Pod Autoscaler)
kubectl autoscale deployment api-gateway --cpu-percent=70 --min=2 --max=10 -n policycortex
```

## Monitoring

### Check Pod Status
```bash
kubectl get pods -n policycortex
kubectl logs -f deployment/api-gateway -n policycortex
```

### Check Services
```bash
kubectl get services -n policycortex
kubectl get ingress -n policycortex
```

### Port Forwarding (Development)
```bash
# Forward API Gateway to local port
kubectl port-forward service/api-gateway-service 8080:80 -n policycortex
```

## Troubleshooting

### Common Issues

1. **Images not pulling**: Ensure ACR authentication is configured
2. **Secrets not found**: Create secrets before deploying services
3. **Ingress not working**: Check ingress controller installation
4. **Pods not starting**: Check resource requests and node capacity

### Debug Commands

```bash
# Describe failing pod
kubectl describe pod <pod-name> -n policycortex

# Check events
kubectl get events -n policycortex --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl top pods -n policycortex
kubectl top nodes
```

## Security

- All secrets are stored in Kubernetes secrets (integrate with Azure Key Vault for production)
- Network policies can be added for micro-segmentation
- RBAC is configured at the cluster level
- Images should be scanned before deployment

## Migration from Container Apps

To migrate from Azure Container Apps to Kubernetes:

1. Deploy AKS infrastructure with `deploy_kubernetes=true`
2. Build and push images to ACR
3. Deploy Kubernetes manifests
4. Update DNS to point to AKS ingress
5. Disable Container Apps with `deploy_container_apps=false`