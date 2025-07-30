# Kubernetes deployment option for PolicyCortex
# This file provides an alternative to Azure Container Apps deployment

# Create the AKS cluster (optional - controlled by variable)
module "kubernetes" {
  count  = var.deploy_kubernetes ? 1 : 0
  source = "./modules/kubernetes"

  name_prefix         = var.name_prefix
  environment         = var.environment
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  
  # Networking
  subnet_id         = module.networking.private_subnet_id
  dns_service_ip    = "10.2.0.10"
  service_cidr      = "10.2.0.0/24"
  
  # Node configuration
  kubernetes_version = var.kubernetes_version
  node_count         = var.kubernetes_node_count
  node_vm_size       = var.kubernetes_node_vm_size
  enable_auto_scaling = var.kubernetes_enable_auto_scaling
  min_node_count     = var.kubernetes_min_node_count
  max_node_count     = var.kubernetes_max_node_count
  
  # AI node pool (optional)
  enable_ai_node_pool = var.kubernetes_enable_ai_node_pool
  ai_node_vm_size     = var.kubernetes_ai_node_vm_size
  ai_node_count       = var.kubernetes_ai_node_count
  
  # Dependencies
  container_registry_id        = azurerm_container_registry.acr.id
  key_vault_id                = azurerm_key_vault.main.id
  log_analytics_workspace_id  = module.monitoring[0].log_analytics_workspace_id
  
  tags = local.common_tags
}

# Kubernetes provider configuration (only when Kubernetes is deployed)
provider "kubernetes" {
  count = var.deploy_kubernetes ? 1 : 0
  
  host                   = var.deploy_kubernetes ? module.kubernetes[0].host : ""
  client_certificate     = var.deploy_kubernetes ? base64decode(module.kubernetes[0].client_certificate) : null
  client_key             = var.deploy_kubernetes ? base64decode(module.kubernetes[0].client_key) : null
  cluster_ca_certificate = var.deploy_kubernetes ? base64decode(module.kubernetes[0].cluster_ca_certificate) : null
}

# Helm provider configuration (only when Kubernetes is deployed)
provider "helm" {
  count = var.deploy_kubernetes ? 1 : 0
  
  kubernetes {
    host                   = var.deploy_kubernetes ? module.kubernetes[0].host : ""
    client_certificate     = var.deploy_kubernetes ? base64decode(module.kubernetes[0].client_certificate) : null
    client_key             = var.deploy_kubernetes ? base64decode(module.kubernetes[0].client_key) : null
    cluster_ca_certificate = var.deploy_kubernetes ? base64decode(module.kubernetes[0].cluster_ca_certificate) : null
  }
}

# Install NGINX Ingress Controller
resource "helm_release" "nginx_ingress" {
  count      = var.deploy_kubernetes ? 1 : 0
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  namespace  = "ingress-nginx"
  
  create_namespace = true
  
  set {
    name  = "controller.service.type"
    value = "LoadBalancer"
  }
  
  set {
    name  = "controller.service.annotations.service\\.beta\\.kubernetes\\.io/azure-load-balancer-health-probe-request-path"
    value = "/healthz"
  }
  
  depends_on = [module.kubernetes]
}

# Install cert-manager for TLS certificates
resource "helm_release" "cert_manager" {
  count      = var.deploy_kubernetes ? 1 : 0
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  namespace  = "cert-manager"
  version    = "v1.13.0"
  
  create_namespace = true
  
  set {
    name  = "installCRDs"
    value = "true"
  }
  
  depends_on = [module.kubernetes]
}

# Azure Application Gateway Ingress Controller (AGIC)
resource "helm_release" "agic" {
  count      = var.deploy_kubernetes && var.enable_application_gateway ? 1 : 0
  name       = "ingress-azure"
  repository = "https://appgwingress.blob.core.windows.net/ingress-azure-helm-package/"
  chart      = "ingress-azure"
  namespace  = "default"
  
  set {
    name  = "appgw.name"
    value = "${var.name_prefix}-appgw-${var.environment}"
  }
  
  set {
    name  = "appgw.resourceGroup"
    value = azurerm_resource_group.main.name
  }
  
  set {
    name  = "appgw.subscriptionId"
    value = data.azurerm_client_config.current.subscription_id
  }
  
  set {
    name  = "kubernetes.watchNamespace"
    value = "policycortex"
  }
  
  depends_on = [module.kubernetes]
}

# Output Kubernetes information when deployed
output "kubernetes_cluster_name" {
  description = "Name of the AKS cluster"
  value       = var.deploy_kubernetes ? module.kubernetes[0].cluster_name : null
}

output "kubernetes_cluster_fqdn" {
  description = "FQDN of the AKS cluster"
  value       = var.deploy_kubernetes ? module.kubernetes[0].cluster_fqdn : null
}

output "kubernetes_kube_config" {
  description = "Kubernetes config for kubectl"
  value       = var.deploy_kubernetes ? module.kubernetes[0].kube_config : null
  sensitive   = true
}