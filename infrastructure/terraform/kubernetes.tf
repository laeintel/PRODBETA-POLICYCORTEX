# Kubernetes deployment option for PolicyCortex
# This file provides an alternative to Azure Container Apps deployment

# Create the AKS cluster (optional - controlled by variable)
module "kubernetes" {
  count  = var.deploy_kubernetes ? 1 : 0
  source = "./modules/kubernetes"

  name_prefix         = var.name_prefix
  environment         = var.environment
  location            = var.location
  resource_group_name = azurerm_resource_group.app.name
  
  # Networking - use data_services subnet for AKS
  subnet_id         = module.networking.subnet_ids["data_services"]
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
  container_registry_id        = azurerm_container_registry.main.id
  key_vault_id                = azurerm_key_vault.main.id
  log_analytics_workspace_id  = module.monitoring[0].log_analytics_workspace_id
  
  tags = local.common_tags
}

#  module name "kubernetes" {
#   count  = var.deploy_kubernetes ? 1 : 0
#   source = "./modules/kubernetes"

#   name_prefix         = var.name_prefix
#   environment         = var.environment
#   location            = var.location
#   resource_group_name = azurerm_resource_group.app.name
  
#   # Networking - use data_services subnet for AKS
#   subnet_id         = module.networking.subnet_ids["data_services"]
#   dns_service_ip    = "
   
#  }

# Note: Providers cannot use count. They will be configured when Kubernetes is available.
# Due to provider configuration limitations with conditional deployment

# Helm charts will be installed manually or via separate deployment
# Due to provider configuration limitations with conditional deployment

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