# Main Terraform configuration for PolicyCortex infrastructure
# Service principal now has Owner permissions on subscription
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
  }
  
  backend "azurerm" {
    # Backend configuration will be provided via init command
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    cognitive_account {
      purge_soft_delete_on_destroy = true
    }
  }
}

# Local variables
locals {
  common_tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
    Owner       = "AeoliTech"
    ManagedBy   = "Terraform"
  }
}

# Data source for current client configuration
data "azurerm_client_config" "current" {}

# Resource provider registrations for Container Apps
# Note: These providers are typically pre-registered in Azure subscriptions
# If you get import errors, manually register them in Azure portal or CLI:
# az provider register --namespace Microsoft.App
# az provider register --namespace Microsoft.OperationalInsights

# Resource group for the environment
resource "azurerm_resource_group" "main" {
  name     = "rg-policycortex-${var.environment}"
  location = var.location
  tags     = local.common_tags
}

# Storage account for application data (with security compliance)
resource "azurerm_storage_account" "app_storage" {
  name                     = "stpolicycortex${var.environment}stg"
  resource_group_name      = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  
  # Security configurations
  min_tls_version                 = "TLS1_2"
  https_traffic_only_enabled      = true
  public_network_access_enabled   = true
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = true
  
  # Network rules (Allow during initial setup)
  network_rules {
    default_action = "Allow"
    bypass         = ["AzureServices"]
  }
  
  # Blob properties for security
  blob_properties {
    delete_retention_policy {
      days = 30
    }
    container_delete_retention_policy {
      days = 30
    }
    versioning_enabled = true
  }
  
  # Queue properties for logging
  queue_properties {
    logging {
      delete                = true
      read                  = true
      write                 = true
      version               = "1.0"
      retention_policy_days = 30
    }
  }
  
  tags = local.common_tags
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                = "kvpolicycortex${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  # Security settings
  purge_protection_enabled   = true
  soft_delete_retention_days = 30
  
  tags = local.common_tags
}

# Key Vault access policy for Terraform (conditional creation)
resource "azurerm_key_vault_access_policy" "terraform" {
  count = var.create_terraform_access_policy ? 1 : 0
  
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id
  
  key_permissions = [
    "Get", "List", "Create", "Delete", "Update", "Recover", "Purge"
  ]
  
  secret_permissions = [
    "Get", "List", "Set", "Delete", "Recover", "Purge"
  ]
}

# Container Registry for Docker images
resource "azurerm_container_registry" "main" {
  name                = "crpolicycortex${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  
  tags = local.common_tags
}

# Virtual Network using networking module
module "networking" {
  source = "./modules/networking"
  
  project_name        = "policycortex"
  environment         = var.environment
  location            = var.location
  resource_group_name = azurerm_resource_group.main.name
  
  vnet_address_space = ["10.0.0.0/16"]
  
  # DDoS protection disabled to avoid high costs
  enable_ddos_protection = false
  
  # Flow logs disabled to avoid storage costs
  enable_flow_logs = false
  
  # Log Analytics integration
  log_analytics_workspace_id          = azurerm_log_analytics_workspace.main.workspace_id
  log_analytics_workspace_resource_id = azurerm_log_analytics_workspace.main.id
  
  subnet_configurations = {
    container_apps = {
      address_prefixes = ["10.0.0.0/23"]
      service_endpoints = ["Microsoft.Storage", "Microsoft.KeyVault"]
      delegation = null
    }
    app_gateway = {
      address_prefixes = ["10.0.2.0/24"]
      service_endpoints = ["Microsoft.Storage"]
    }
    data_services = {
      address_prefixes = ["10.0.4.0/24"]
      service_endpoints = ["Microsoft.Sql", "Microsoft.AzureCosmosDB", "Microsoft.Storage", "Microsoft.KeyVault"]
    }
    ai_services = {
      address_prefixes = ["10.0.5.0/24"]
      service_endpoints = ["Microsoft.MachineLearningServices", "Microsoft.CognitiveServices", "Microsoft.Storage", "Microsoft.KeyVault"]
    }
  }
  
  common_tags = local.common_tags
}

# Data Services Module
module "data_services" {
  source = "./modules/data-services"
  
  project_name                  = "policycortex"
  environment                   = var.environment
  resource_group_name           = azurerm_resource_group.main.name
  vnet_name                     = module.networking.vnet_name
  data_services_subnet_name     = "data-services-subnet"
  key_vault_name                = azurerm_key_vault.main.name
  
  # SQL Server configuration
  sql_admin_username            = var.sql_admin_username
  sql_azuread_admin_login       = var.sql_azuread_admin_login
  sql_azuread_admin_object_id   = var.sql_azuread_admin_object_id
  sql_database_sku              = var.sql_database_sku
  sql_database_max_size_gb      = var.sql_database_max_size_gb
  
  # Cosmos DB configuration
  cosmos_consistency_level      = var.cosmos_consistency_level
  cosmos_failover_location      = var.cosmos_failover_location
  cosmos_max_throughput         = var.cosmos_max_throughput
  
  # Redis configuration
  redis_capacity                = var.redis_capacity
  redis_sku_name                = var.redis_sku_name
  
  tags = local.common_tags
  
  depends_on = [
    module.networking,
    azurerm_key_vault.main
  ]
}

# AI Services Module
module "ai_services" {
  source = "./modules/ai-services"
  
  project_name                     = "policycortex"
  environment                      = var.environment
  resource_group_name              = azurerm_resource_group.main.name
  vnet_name                        = module.networking.vnet_name
  ai_services_subnet_name          = "ai-services-subnet"
  key_vault_name                   = azurerm_key_vault.main.name
  storage_account_name             = azurerm_storage_account.app_storage.name
  application_insights_name        = azurerm_application_insights.main.name
  
  # Container Registry configuration
  create_container_registry        = var.create_ml_container_registry
  existing_container_registry_id   = var.create_ml_container_registry ? null : azurerm_container_registry.main.id
  
  # Compute configuration
  training_cluster_vm_size         = var.training_cluster_vm_size
  training_cluster_max_nodes       = var.training_cluster_max_nodes
  compute_instance_vm_size         = var.compute_instance_vm_size
  
  # Cognitive Services configuration
  cognitive_services_sku           = var.cognitive_services_sku
  deploy_openai                    = var.deploy_openai
  openai_sku                       = var.openai_sku
  
  tags = local.common_tags
  
  depends_on = [
    module.networking,
    azurerm_key_vault.main,
    azurerm_storage_account.app_storage,
    azurerm_application_insights.main
  ]
}

# Log Analytics Workspace for monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-policycortex-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = local.common_tags
}

# Application Insights for monitoring
resource "azurerm_application_insights" "main" {
  name                = "ai-policycortex-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  
  tags = local.common_tags
}

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-policycortex-${var.environment}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  infrastructure_subnet_id   = module.networking.subnet_ids["container_apps"]
  
  # Resource providers should be manually registered if needed
  # depends_on = [
  #   azurerm_resource_provider_registration.container_apps,
  #   azurerm_resource_provider_registration.operational_insights
  # ]
  
  tags = local.common_tags
}

# User-assigned managed identity for Container Apps
resource "azurerm_user_assigned_identity" "container_apps" {
  name                = "id-policycortex-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  
  tags = local.common_tags
}

# Role assignment for Container Apps to pull from ACR
resource "azurerm_role_assignment" "container_apps_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Role assignment for Container Apps to access Key Vault
resource "azurerm_role_assignment" "container_apps_keyvault" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Role assignment for Container Apps to access Storage
resource "azurerm_role_assignment" "container_apps_storage" {
  scope                = azurerm_storage_account.app_storage.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Container Apps resources moved to container-apps.tf
# Set deploy_container_apps = true to deploy them

# Output values
output "resource_group_name" {
  value = azurerm_resource_group.main.name
}

output "storage_account_name" {
  value = azurerm_storage_account.app_storage.name
}

output "key_vault_name" {
  value = azurerm_key_vault.main.name
}

output "container_registry_name" {
  value = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  value = azurerm_container_registry.main.login_server
}

output "container_app_environment_name" {
  value = azurerm_container_app_environment.main.name
}

output "container_app_environment_fqdn" {
  value = azurerm_container_app_environment.main.default_domain
}

output "container_apps_identity_id" {
  value = azurerm_user_assigned_identity.container_apps.id
}

output "log_analytics_workspace_id" {
  value = azurerm_log_analytics_workspace.main.id
}

output "application_insights_instrumentation_key" {
  value = azurerm_application_insights.main.instrumentation_key
  sensitive = true
}

# Data Services Outputs
output "sql_server_fqdn" {
  value = module.data_services.sql_server_fqdn
}

output "cosmos_account_endpoint" {
  value = module.data_services.cosmos_account_endpoint
}

output "redis_cache_hostname" {
  value = module.data_services.redis_cache_hostname
}

# AI Services Outputs
output "ml_workspace_name" {
  value = module.ai_services.ml_workspace_name
}

output "cognitive_services_endpoint" {
  value = module.ai_services.cognitive_services_endpoint
}

output "openai_endpoint" {
  value = module.ai_services.openai_endpoint
}

# Container Apps URLs moved to container-apps.tf