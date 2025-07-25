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
    # Backend configuration will be provided via init command or tfvars
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

# Network Resource Group for networking infrastructure
resource "azurerm_resource_group" "network" {
  name     = "rg-policycortex-network-${var.environment}"
  location = var.location
  tags     = merge(local.common_tags, {
    ResourceType = "Networking"
  })
}

# Application Resource Group for application resources
resource "azurerm_resource_group" "app" {
  name     = "rg-policycortex-app-${var.environment}"
  location = var.location
  tags     = merge(local.common_tags, {
    ResourceType = "Application"
  })
}

# Storage account for application data (with security compliance)
resource "azurerm_storage_account" "app_storage" {
  name                     = "stpolicycortex${var.environment}stg"
  resource_group_name      = azurerm_resource_group.app.name
  location                = azurerm_resource_group.app.location
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
  
  # User-assigned managed identity
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  tags = local.common_tags
}

# Key Vault for secrets management
resource "azurerm_key_vault" "main" {
  name                = "kvpolicycortex${var.environment}v2"
  location            = azurerm_resource_group.app.location
  resource_group_name = azurerm_resource_group.app.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  # Security settings
  purge_protection_enabled   = true
  soft_delete_retention_days = 30
  
  # Enable RBAC authorization (temporarily disabled for migration)
  enable_rbac_authorization = false
  
  tags = local.common_tags
}

# Key Vault access policy for current client (service principal or user)
resource "azurerm_key_vault_access_policy" "current_client" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = data.azurerm_client_config.current.object_id
  
  key_permissions = [
    "Get", "List", "Create", "Delete", "Update", "Recover", "Purge", "GetRotationPolicy", "SetRotationPolicy"
  ]
  
  secret_permissions = [
    "Get", "List", "Set", "Delete", "Recover", "Purge"
  ]
  
  certificate_permissions = [
    "Get", "List", "Create", "Delete", "Update"
  ]
}

# Key Vault access policy for Container Apps managed identity
resource "azurerm_key_vault_access_policy" "container_apps" {
  key_vault_id = azurerm_key_vault.main.id
  tenant_id    = data.azurerm_client_config.current.tenant_id
  object_id    = azurerm_user_assigned_identity.container_apps.principal_id
  
  secret_permissions = [
    "Get", "List"
  ]
  
  depends_on = [azurerm_user_assigned_identity.container_apps]
}

# RBAC role assignments for Key Vault (for future migration to RBAC)
# Key Vault Administrator role for current client
resource "azurerm_role_assignment" "key_vault_admin_current_client" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Administrator"
  principal_id         = data.azurerm_client_config.current.object_id
}

# Key Vault Secrets User role for Container Apps managed identity
resource "azurerm_role_assignment" "key_vault_secrets_user_container_apps" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  
  depends_on = [azurerm_user_assigned_identity.container_apps]
}

# Container Registry for Docker images
resource "azurerm_container_registry" "main" {
  name                = "crpolicycortex${var.environment}"
  resource_group_name = azurerm_resource_group.app.name
  location            = azurerm_resource_group.app.location
  sku                 = "Basic"
  admin_enabled       = true
  
  # User-assigned managed identity
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  tags = local.common_tags
}

# Virtual Network using networking module
module "networking" {
  source = "./modules/networking"
  
  project_name        = "policycortex"
  environment         = var.environment
  location            = var.location
  resource_group_name = azurerm_resource_group.network.name
  
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
      delegation = {
        name = "Microsoft.App.environments"
        service_delegation = {
          name = "Microsoft.App/environments"
          actions = ["Microsoft.Network/virtualNetworks/subnets/action"]
        }
      }
    }
    app_gateway = {
      address_prefixes = ["10.0.2.0/24"]
      service_endpoints = ["Microsoft.Storage"]
    }
    private_endpoints = {
      address_prefixes = ["10.0.3.0/24"]
      service_endpoints = ["Microsoft.Storage", "Microsoft.KeyVault", "Microsoft.Sql", "Microsoft.AzureCosmosDB", "Microsoft.CognitiveServices"]
    }
    data_services = {
      address_prefixes = ["10.0.4.0/24"]
      service_endpoints = ["Microsoft.Sql", "Microsoft.AzureCosmosDB", "Microsoft.Storage", "Microsoft.KeyVault"]
    }
    ai_services = {
      address_prefixes = ["10.0.5.0/24"]
      service_endpoints = ["Microsoft.CognitiveServices", "Microsoft.Storage", "Microsoft.KeyVault"]
    }
  }
  
  common_tags = local.common_tags
}

# Data Services Module
module "data_services" {
  source = "./modules/data-services"
  
  project_name                  = "policycortex"
  environment                   = var.environment
  location                      = var.location
  resource_group_name           = azurerm_resource_group.app.name
  network_resource_group_name   = azurerm_resource_group.network.name
  vnet_name                     = module.networking.vnet_name
  data_services_subnet_name     = "policycortex-${var.environment}-subnet-data_services"
  private_endpoints_subnet_name = "policycortex-${var.environment}-subnet-private_endpoints"
  key_vault_name                = azurerm_key_vault.main.name
  managed_identity_id           = azurerm_user_assigned_identity.container_apps.id
  
  # SQL Server configuration
  deploy_sql_server             = var.deploy_sql_server
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
  location                         = var.location
  resource_group_name              = azurerm_resource_group.app.name
  network_resource_group_name      = azurerm_resource_group.network.name
  vnet_name                        = module.networking.vnet_name
  ai_services_subnet_name          = "policycortex-${var.environment}-subnet-ai_services"
  private_endpoints_subnet_name    = "policycortex-${var.environment}-subnet-private_endpoints"
  key_vault_name                   = azurerm_key_vault.main.name
  storage_account_name             = azurerm_storage_account.app_storage.name
  application_insights_name        = azurerm_application_insights.main.name
  managed_identity_id              = azurerm_user_assigned_identity.container_apps.id
  
  # ML Workspace configuration
  deploy_ml_workspace              = var.deploy_ml_workspace
  
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

# Monitoring Module
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name                     = "policycortex"
  environment                      = var.environment
  resource_group_name              = azurerm_resource_group.app.name
  log_analytics_workspace_name     = azurerm_log_analytics_workspace.main.name
  application_insights_name        = azurerm_application_insights.main.name
  subscription_id                  = data.azurerm_client_config.current.subscription_id
  
  # Alert configuration
  critical_alert_emails            = var.critical_alert_emails
  warning_alert_emails             = var.warning_alert_emails
  budget_alert_emails              = var.budget_alert_emails
  
  # Container Apps monitoring
  container_app_environment_id     = azurerm_container_app_environment.main.id
  
  # Database monitoring
  cosmos_db_account_id             = module.data_services.cosmos_account_id
  sql_database_id                  = module.data_services.sql_database_id
  storage_account_id               = azurerm_storage_account.app_storage.id
  
  # Monitoring deployment flags
  deploy_cosmos_monitoring         = true
  deploy_sql_monitoring            = var.deploy_sql_server
  deploy_storage_monitoring        = true
  
  # Budget configuration
  monthly_budget_amount            = var.monthly_budget_amount
  
  tags = local.common_tags
  
  depends_on = [
    module.data_services,
    module.ai_services,
    azurerm_container_app_environment.main
  ]
}

# Log Analytics Workspace for monitoring
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-policycortex-${var.environment}"
  location            = azurerm_resource_group.app.location
  resource_group_name = azurerm_resource_group.app.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  
  tags = local.common_tags
}

# Application Insights for monitoring
resource "azurerm_application_insights" "main" {
  name                = "ai-policycortex-${var.environment}"
  location            = azurerm_resource_group.app.location
  resource_group_name = azurerm_resource_group.app.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "web"
  
  tags = local.common_tags
}

# Container Apps Environment with Dedicated Workload Profiles
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-policycortex-${var.environment}"
  location                   = azurerm_resource_group.app.location
  resource_group_name        = azurerm_resource_group.app.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  infrastructure_subnet_id   = module.networking.subnet_ids["container_apps"]
  
  # Dedicated workload profiles for improved performance and isolation
  workload_profile {
    name                  = "Consumption"
    workload_profile_type = "Consumption"
  }
  
  workload_profile {
    name                  = "Dedicated-D4"
    workload_profile_type = "D4"
    minimum_count         = 1
    maximum_count         = 3
  }
  
  workload_profile {
    name                  = "Dedicated-D8"
    workload_profile_type = "D8"
    minimum_count         = 0
    maximum_count         = 2
  }
  
  workload_profile {
    name                  = "Dedicated-D16"
    workload_profile_type = "D16"
    minimum_count         = 0
    maximum_count         = 2
  }
  
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
  location            = azurerm_resource_group.app.location
  resource_group_name = azurerm_resource_group.app.name
  
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

# Role assignment for Container Apps to access Cosmos DB
resource "azurerm_role_assignment" "container_apps_cosmos" {
  scope                = module.data_services.cosmos_account_id
  role_definition_name = "Cosmos DB Built-in Data Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  
  depends_on = [
    module.data_services,
    azurerm_user_assigned_identity.container_apps
  ]
  
  # Add delay to ensure Cosmos DB is fully provisioned
  provisioner "local-exec" {
    command = "sleep 30"
  }
}

# Role assignment for Container Apps to access Redis
resource "azurerm_role_assignment" "container_apps_redis" {
  scope                = module.data_services.redis_cache_id
  role_definition_name = "Redis Cache Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  
  depends_on = [
    module.data_services,
    azurerm_user_assigned_identity.container_apps
  ]
  
  # Add delay to ensure Redis is fully provisioned
  provisioner "local-exec" {
    command = "sleep 30"
  }
}

# Role assignment for Container Apps to access Cognitive Services
resource "azurerm_role_assignment" "container_apps_cognitive" {
  scope                = module.ai_services.cognitive_services_id
  role_definition_name = "Cognitive Services User"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
  
  depends_on = [
    module.ai_services,
    azurerm_user_assigned_identity.container_apps
  ]
  
  # Add delay to ensure Cognitive Services is fully provisioned
  provisioner "local-exec" {
    command = "sleep 30"
  }
}

# Role assignment for Container Apps to access Application Insights
resource "azurerm_role_assignment" "container_apps_appinsights" {
  scope                = azurerm_application_insights.main.id
  role_definition_name = "Monitoring Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Role assignment for Container Apps to access Log Analytics Workspace
resource "azurerm_role_assignment" "container_apps_log_analytics" {
  scope                = azurerm_log_analytics_workspace.main.id
  role_definition_name = "Log Analytics Contributor"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Role assignment for Container Apps to read Resource Group
resource "azurerm_role_assignment" "container_apps_rg_reader" {
  scope                = azurerm_resource_group.app.id
  role_definition_name = "Reader"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# Additional Key Vault secrets for container apps
resource "azurerm_key_vault_secret" "jwt_secret_key" {
  name         = "jwt-secret-key"
  value        = var.jwt_secret_key
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_client]
}

resource "azurerm_key_vault_secret" "managed_identity_client_id" {
  name         = "managed-identity-client-id"
  value        = azurerm_user_assigned_identity.container_apps.client_id
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_client]
}

resource "azurerm_key_vault_secret" "storage_account_name" {
  name         = "storage-account-name"
  value        = azurerm_storage_account.app_storage.name
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_client]
}

resource "azurerm_key_vault_secret" "application_insights_connection_string" {
  name         = "application-insights-connection-string"
  value        = azurerm_application_insights.main.connection_string
  key_vault_id = azurerm_key_vault.main.id
  
  depends_on = [azurerm_key_vault_access_policy.current_client]
}

# Container Apps resources moved to container-apps.tf
# NOTE: For initial deployment, set deploy_container_apps = false to deploy infrastructure first
# Then set deploy_container_apps = true and run terraform apply again to deploy container apps
# This ensures all Azure resources are fully provisioned before container apps are created

# Output values
output "resource_group_name" {
  value = azurerm_resource_group.app.name
}

output "network_resource_group_name" {
  value = azurerm_resource_group.network.name
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