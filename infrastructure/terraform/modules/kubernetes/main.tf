# AKS Cluster Terraform Module
# This module creates an Azure Kubernetes Service (AKS) cluster for PolicyCortex

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.name_prefix}-aks-${var.environment}"
  location            = var.location
  resource_group_name = var.resource_group_name
  dns_prefix          = "${var.name_prefix}-aks-${var.environment}"
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name                = "default"
    node_count          = var.node_count
    vm_size             = var.node_vm_size
    vnet_subnet_id      = var.subnet_id
    enable_auto_scaling = var.enable_auto_scaling
    min_count           = var.enable_auto_scaling ? var.min_node_count : null
    max_count           = var.enable_auto_scaling ? var.max_node_count : null
    
    upgrade_settings {
      max_surge = "10%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    dns_service_ip    = var.dns_service_ip
    service_cidr      = var.service_cidr
  }

  azure_policy_enabled = true
  
  oms_agent {
    log_analytics_workspace_id = var.log_analytics_workspace_id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  workload_identity_enabled = true
  oidc_issuer_enabled      = true

  tags = var.tags
}

# Role assignment for AKS to pull images from ACR
resource "azurerm_role_assignment" "aks_acr_pull" {
  principal_id                     = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                           = var.container_registry_id
  skip_service_principal_aad_check = true
}

# Role assignment for AKS to access Key Vault
resource "azurerm_role_assignment" "aks_key_vault_secrets" {
  principal_id                     = azurerm_kubernetes_cluster.aks.identity[0].principal_id
  role_definition_name             = "Key Vault Secrets User"
  scope                           = var.key_vault_id
  skip_service_principal_aad_check = true
}

# Additional node pool for AI workloads (optional)
resource "azurerm_kubernetes_cluster_node_pool" "ai_workload" {
  count                 = var.enable_ai_node_pool ? 1 : 0
  name                  = "aiworkload"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.ai_node_vm_size
  node_count            = var.ai_node_count
  vnet_subnet_id        = var.subnet_id
  
  node_taints = ["workload=ai:NoSchedule"]
  node_labels = {
    "workload" = "ai"
  }

  upgrade_settings {
    max_surge = "10%"
  }

  tags = var.tags
}