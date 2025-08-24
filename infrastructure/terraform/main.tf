# PolicyCortex Complete Infrastructure
# This creates EVERYTHING from scratch including state storage and private endpoints

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.115.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6.0"
    }
  }

  # Backend configuration - will be initialized with backend config file
  backend "azurerm" {}
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    cognitive_account {
      purge_soft_delete_on_destroy = true
    }
  }
  skip_provider_registration = false
}

# Data sources
data "azurerm_client_config" "current" {}
data "azurerm_subscription" "current" {}

# Variables
variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "ai_location" {
  description = "Azure region for AI services"
  type        = string
  default     = "eastus2"
}

# Locals for consistent naming
locals {
  project = "cortex"
  # Fixed suffix for globally unique resources - NO RANDOM NUMBERS
  # This ensures resources can be recreated with exact same names
  hash_suffix = "3p0bata" # Fixed suffix, never changes

  # Resource names
  resource_names = {
    # Core Infrastructure
    resource_group = "rg-${local.project}-${var.environment}"
    tfstate_rg     = "rg-tfstate-${local.project}-${var.environment}"

    # Networking
    vnet = "vnet-${local.project}-${var.environment}"

    # AKS
    aks_cluster = "aks-${local.project}-${var.environment}"
    app_gateway = "agw-${local.project}-${var.environment}"

    # Storage & Registry
    container_registry = "cr${local.project}${var.environment}${local.hash_suffix}"
    storage_account    = "st${local.project}${var.environment}${local.hash_suffix}"
    tfstate_storage    = "sttf${local.project}${var.environment}${local.hash_suffix}"

    # Data
    key_vault  = "kv-${local.project}-${var.environment}-${local.hash_suffix}"
    postgresql = "psql-${local.project}-${var.environment}"
    cosmos_db  = "cosmos-${local.project}-${var.environment}-${local.hash_suffix}"

    # Monitoring
    log_workspace = "log-${local.project}-${var.environment}"
    app_insights  = "appi-${local.project}-${var.environment}"

    # AI Services
    openai     = "cogao-${local.project}-${var.environment}"
    ai_hub     = "policycortex-gpt4o-resource"
    ai_project = "policycortex_gpt4o"
  }

  common_tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
    ManagedBy   = "Terraform"
    Repository  = "github.com/laeintel/policycortex"
  }
}

# ===========================
# TERRAFORM STATE INFRASTRUCTURE
# ===========================

resource "azurerm_resource_group" "tfstate" {
  name     = local.resource_names.tfstate_rg
  location = var.location
  tags     = local.common_tags

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [tags]
  }
}

resource "azurerm_storage_account" "tfstate" {
  name                     = local.resource_names.tfstate_storage
  resource_group_name      = azurerm_resource_group.tfstate.name
  location                 = azurerm_resource_group.tfstate.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  blob_properties {
    versioning_enabled = true
  }

  tags = local.common_tags

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [tags]
  }
}

resource "azurerm_storage_container" "tfstate" {
  name                  = "tfstate"
  storage_account_name  = azurerm_storage_account.tfstate.name
  container_access_type = "private"
}

# ===========================
# MAIN RESOURCE GROUP
# ===========================

resource "azurerm_resource_group" "main" {
  name     = local.resource_names.resource_group
  location = var.location
  tags     = local.common_tags
}

# ===========================
# NETWORKING WITH PRIVATE ENDPOINTS
# ===========================

resource "azurerm_virtual_network" "main" {
  name                = local.resource_names.vnet
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.0.0.0/16"]
  tags                = local.common_tags
}

# Subnet for AKS Nodes
resource "azurerm_subnet" "aks" {
  name                 = "snet-aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.0.0/22"] # Large subnet for AKS nodes

  # Service endpoints for resources that need them
  service_endpoints = [
    "Microsoft.Storage",
    "Microsoft.KeyVault",
    "Microsoft.ContainerRegistry",
    "Microsoft.AzureCosmosDB",
    "Microsoft.Web",
    "Microsoft.Sql"
  ]
}

# Subnet for Application Gateway (AGIC)
resource "azurerm_subnet" "appgateway" {
  name                 = "snet-appgateway"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.12.0/24"]
}

# Subnet for Private Endpoints
resource "azurerm_subnet" "private_endpoints" {
  name                 = "snet-private-endpoints"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.4.0/24"]

  private_endpoint_network_policies = "Disabled"
}

# Subnet for Database
resource "azurerm_subnet" "database" {
  name                 = "snet-database"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.5.0/24"]

  delegation {
    name = "postgresql"
    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

# Private DNS Zones
resource "azurerm_private_dns_zone" "keyvault" {
  name                = "privatelink.vaultcore.azure.net"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone" "storage" {
  name                = "privatelink.blob.core.windows.net"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone" "acr" {
  name                = "privatelink.azurecr.io"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone" "cosmosdb" {
  name                = "privatelink.documents.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone" "postgresql" {
  name                = "privatelink.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

resource "azurerm_private_dns_zone" "openai" {
  name                = "privatelink.openai.azure.com"
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

# Link DNS zones to VNet
resource "azurerm_private_dns_zone_virtual_network_link" "keyvault" {
  name                  = "keyvault-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.keyvault.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "storage" {
  name                  = "storage-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.storage.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "acr" {
  name                  = "acr-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.acr.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "cosmosdb" {
  name                  = "cosmosdb-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.cosmosdb.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgresql" {
  name                  = "postgresql-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgresql.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "openai" {
  name                  = "openai-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.openai.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

# ===========================
# MONITORING
# ===========================

resource "azurerm_log_analytics_workspace" "main" {
  name                = local.resource_names.log_workspace
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = local.common_tags
}

resource "azurerm_application_insights" "main" {
  name                = local.resource_names.app_insights
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  tags                = local.common_tags
}

# Smart Detection Alert Rule
resource "azurerm_monitor_smart_detector_alert_rule" "failure_anomalies" {
  name                = "Failure Anomalies - ${local.resource_names.app_insights}"
  resource_group_name = azurerm_resource_group.main.name
  severity            = "Sev3"
  scope_resource_ids  = [azurerm_application_insights.main.id]
  frequency           = "PT1M"
  detector_type       = "FailureAnomaliesDetector"

  action_group {
    ids = []
  }

  tags = local.common_tags
}

# ===========================
# CONTAINER REGISTRY WITH PRIVATE ENDPOINT
# ===========================

resource "azurerm_container_registry" "main" {
  name                = local.resource_names.container_registry
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Premium" # Premium required for private endpoints
  admin_enabled       = true

  # Remove network rules - we'll use private endpoints instead
  public_network_access_enabled = true # Temporarily enable for initial setup

  tags = local.common_tags
}

resource "azurerm_private_endpoint" "acr" {
  name                = "pe-${local.resource_names.container_registry}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-acr"
    private_connection_resource_id = azurerm_container_registry.main.id
    subresource_names              = ["registry"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "pdz-acr"
    private_dns_zone_ids = [azurerm_private_dns_zone.acr.id]
  }

  tags = local.common_tags
}

# ===========================
# STORAGE ACCOUNT WITH PRIVATE ENDPOINT
# ===========================

resource "azurerm_storage_account" "main" {
  name                     = local.resource_names.storage_account
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  # Temporarily allow public access for initial setup
  public_network_access_enabled = true

  tags = local.common_tags
}

resource "azurerm_private_endpoint" "storage" {
  name                = "pe-${local.resource_names.storage_account}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-storage"
    private_connection_resource_id = azurerm_storage_account.main.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "pdz-storage"
    private_dns_zone_ids = [azurerm_private_dns_zone.storage.id]
  }

  tags = local.common_tags
}

# ===========================
# KEY VAULT WITH PRIVATE ENDPOINT
# ===========================

resource "azurerm_key_vault" "main" {
  name                       = local.resource_names.key_vault
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  soft_delete_retention_days = 7
  purge_protection_enabled   = false

  # Temporarily allow public access for initial setup
  public_network_access_enabled = true
  network_acls {
    default_action = "Allow"
    bypass         = "AzureServices"
  }

  access_policy {
    tenant_id = data.azurerm_client_config.current.tenant_id
    object_id = data.azurerm_client_config.current.object_id

    secret_permissions = [
      "Get", "List", "Set", "Delete", "Purge", "Recover"
    ]

    key_permissions = [
      "Get", "List", "Create", "Delete", "Purge", "Recover"
    ]
  }

  tags = local.common_tags
}

resource "azurerm_private_endpoint" "keyvault" {
  name                = "pe-${local.resource_names.key_vault}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-keyvault"
    private_connection_resource_id = azurerm_key_vault.main.id
    subresource_names              = ["vault"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "pdz-keyvault"
    private_dns_zone_ids = [azurerm_private_dns_zone.keyvault.id]
  }

  tags = local.common_tags
}

# ===========================
# POSTGRESQL WITH PRIVATE ENDPOINT
# ===========================

resource "random_password" "db_password" {
  length  = 32
  special = true
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                = local.resource_names.postgresql
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  version             = "15"
  delegated_subnet_id = azurerm_subnet.database.id
  private_dns_zone_id = azurerm_private_dns_zone.postgresql.id

  administrator_login    = "psqladmin"
  administrator_password = random_password.db_password.result

  storage_mb = 32768
  sku_name   = "B_Standard_B1ms"
  zone       = "1"

  # Disable public network access when using private endpoints
  public_network_access_enabled = false

  tags = local.common_tags

  depends_on = [
    azurerm_private_dns_zone_virtual_network_link.postgresql
  ]
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "policycortex"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# Store DB password in Key Vault
resource "azurerm_key_vault_secret" "db_password" {
  name         = "db-admin-password"
  value        = random_password.db_password.result
  key_vault_id = azurerm_key_vault.main.id
}

# ===========================
# COSMOS DB WITH PRIVATE ENDPOINT
# ===========================

resource "azurerm_cosmosdb_account" "main" {
  name                = local.resource_names.cosmos_db
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  capabilities {
    name = "EnableServerless"
  }

  # Temporarily disable network restrictions for initial setup
  is_virtual_network_filter_enabled = false
  public_network_access_enabled     = true

  tags = local.common_tags
}

resource "azurerm_private_endpoint" "cosmosdb" {
  name                = "pe-${local.resource_names.cosmos_db}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-cosmosdb"
    private_connection_resource_id = azurerm_cosmosdb_account.main.id
    subresource_names              = ["Sql"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "pdz-cosmosdb"
    private_dns_zone_ids = [azurerm_private_dns_zone.cosmosdb.id]
  }

  tags = local.common_tags
}

# ===========================
# AZURE OPENAI WITH PRIVATE ENDPOINT
# ===========================

resource "azurerm_cognitive_account" "openai" {
  name                = local.resource_names.openai
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = "S0"

  custom_subdomain_name = local.resource_names.openai

  # Temporarily allow public access for initial setup
  public_network_access_enabled = true
  network_acls {
    default_action = "Allow"
  }

  tags = local.common_tags
}

resource "azurerm_cognitive_deployment" "gpt4" {
  name                 = "gpt-4o-mini"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4o-mini"
    version = "2024-07-18"
  }

  scale {
    type = "Standard"
  }
}

resource "azurerm_private_endpoint" "openai" {
  name                = "pe-${local.resource_names.openai}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "psc-openai"
    private_connection_resource_id = azurerm_cognitive_account.openai.id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "pdz-openai"
    private_dns_zone_ids = [azurerm_private_dns_zone.openai.id]
  }

  tags = local.common_tags
}

# ===========================
# AI FOUNDRY HUB AND PROJECT
# ===========================

resource "azurerm_resource_group" "ai_hub" {
  name     = local.resource_names.ai_hub
  location = var.ai_location
  tags     = local.common_tags
}

resource "azurerm_machine_learning_workspace" "ai_hub" {
  name                    = local.resource_names.ai_hub
  location                = azurerm_resource_group.ai_hub.location
  resource_group_name     = azurerm_resource_group.ai_hub.name
  application_insights_id = azurerm_application_insights.main.id
  key_vault_id            = azurerm_key_vault.main.id
  storage_account_id      = azurerm_storage_account.main.id

  identity {
    type = "SystemAssigned"
  }

  tags = local.common_tags
}

# ===========================
# AZURE KUBERNETES SERVICE (AKS)
# ===========================

resource "azurerm_kubernetes_cluster" "main" {
  name                = local.resource_names.aks_cluster
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = "aks-${local.project}-${var.environment}"
  kubernetes_version  = "1.29"

  default_node_pool {
    name                = "default"
    node_count          = 3
    vm_size             = "Standard_D4s_v3"
    os_disk_size_gb     = 100
    vnet_subnet_id      = azurerm_subnet.aks.id
    type                = "VirtualMachineScaleSets"
    enable_auto_scaling = true
    min_count           = 2
    max_count           = 5

    node_labels = {
      "environment" = var.environment
      "nodepool"    = "default"
    }

    tags = local.common_tags
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    service_cidr      = "10.1.0.0/16"
    dns_service_ip    = "10.1.0.10"
  }

  ingress_application_gateway {
    gateway_name = local.resource_names.app_gateway
    subnet_id    = azurerm_subnet.appgateway.id
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  azure_policy_enabled = true

  tags = local.common_tags
}

# ===========================
# AKS NODE POOL FOR ML WORKLOADS
# ===========================

resource "azurerm_kubernetes_cluster_node_pool" "ml" {
  name                  = "ml"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_NC6s_v3" # GPU-enabled for ML
  node_count            = 1
  enable_auto_scaling   = true
  min_count             = 0
  max_count             = 3
  vnet_subnet_id        = azurerm_subnet.aks.id

  node_labels = {
    "workload" = "ml"
    "gpu"      = "true"
  }

  node_taints = [
    "ml=true:NoSchedule"
  ]

  tags = local.common_tags
}

# Grant ACR pull permissions to AKS
resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

# Grant AKS access to Key Vault
resource "azurerm_role_assignment" "aks_keyvault_secrets" {
  scope                            = azurerm_key_vault.main.id
  role_definition_name             = "Key Vault Secrets User"
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

# ===========================
# KUBERNETES DEPLOYMENTS
# Note: The actual deployments are managed via kubectl apply
# using the manifests in k8s/dev or k8s/prod directories
# ===========================

# The AKS cluster will host:
# - Frontend (Next.js) on port 3000
# - Core API (Rust) on port 8080
# - GraphQL Gateway on port 4000
# - ML Services
# - Backend Python Services

# Deployments are handled by CI/CD pipeline using:
# kubectl apply -f k8s/${environment}/

# ===========================
# OUTPUTS
# ===========================

output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "tfstate_resource_group" {
  value = azurerm_resource_group.tfstate.name
}

output "tfstate_storage_account" {
  value = azurerm_storage_account.tfstate.name
}

output "container_registry_name" {
  value = azurerm_container_registry.main.name
}

output "container_registry_url" {
  value = azurerm_container_registry.main.login_server
}

output "aks_cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "aks_kube_config" {
  value     = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive = true
}

output "aks_host" {
  value = azurerm_kubernetes_cluster.main.kube_config[0].host
}

output "aks_ingress_ip" {
  value       = "Application Gateway IP will be available after deployment"
  description = "The public IP of the Application Gateway Ingress Controller"
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "postgresql_server_name" {
  value = azurerm_postgresql_flexible_server.main.name
}

output "cosmos_db_endpoint" {
  value = azurerm_cosmosdb_account.main.endpoint
}

output "openai_endpoint" {
  value = azurerm_cognitive_account.openai.endpoint
}

output "app_insights_connection_string" {
  value     = azurerm_application_insights.main.connection_string
  sensitive = true
}

