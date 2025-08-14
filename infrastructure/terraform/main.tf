# PolicyCortex Infrastructure - Optimized for Azure Free Tier
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

  backend "azurerm" {
    # Backend config will be provided during init
    use_azuread_auth = true
  }
}

provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = true
    }
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
  }
  use_oidc                   = true
  skip_provider_registration = true
}

# Data sources
data "azurerm_client_config" "current" {}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

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

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default = {
    Owner     = "AeoliTech"
    Project   = "PolicyCortex"
    ManagedBy = "Terraform"
  }
}


locals {
  common_tags = merge(var.tags, {
    Environment = var.environment
    CreatedDate = timestamp()
  })

  # Naming convention: <resource>-cortex-<env>
  env_suffix    = var.environment
  unique_suffix = random_string.suffix.result
}

# Resource Group - New group for Terraform-managed resources
resource "azurerm_resource_group" "main" {
  name     = "rg-cortex-${local.env_suffix}"
  location = var.location
  tags     = local.common_tags
}


# ===========================================================================
# FREE TIER RESOURCES
# ===========================================================================

# Storage Account - Using free tier limits
# Free: 5GB Hot storage, 100GB File storage
resource "azurerm_storage_account" "main" {
  name                     = "stcortex${local.env_suffix}${local.unique_suffix}"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS" # Locally redundant (cheapest)
  account_kind             = "StorageV2"

  blob_properties {
    delete_retention_policy {
      days = 7
    }
  }

  tags = local.common_tags
}

# ===================== Azure OpenAI (Cognitive Services) =====================

locals {
  openai_account_name         = "cogao-cortex-${local.env_suffix}"
  openai_custom_subdomain     = "cortexao-${local.env_suffix}"
  openai_realtime_model_name  = var.openai_realtime_model_name != null ? var.openai_realtime_model_name : "gpt-4o-realtime-preview"
  openai_realtime_model_ver   = var.openai_realtime_model_version != null ? var.openai_realtime_model_version : "2024-05-01-preview"
  openai_chat_model_name      = var.openai_chat_model_name != null ? var.openai_chat_model_name : "gpt-4o-mini"
  openai_chat_model_ver       = var.openai_chat_model_version != null ? var.openai_chat_model_version : "2024-05-01-preview"
  openai_realtime_deploy_name = "realtime-${local.env_suffix}"
  openai_chat_deploy_name     = "chat-${local.env_suffix}"
}

# Optional input override variables (declare with defaults when using a single file module)
variable "openai_realtime_model_name" {
  type    = string
  default = null
}
variable "openai_realtime_model_version" {
  type    = string
  default = null
}
variable "openai_chat_model_name" {
  type    = string
  default = null
}
variable "openai_chat_model_version" {
  type    = string
  default = null
}

# Toggle to create Azure OpenAI deployments. Default false to avoid
# failures when models are not available in the selected region.
variable "enable_openai_deployments" {
  description = "Create Azure OpenAI deployments (set true only if models are supported in region)"
  type        = bool
  default     = false
}

# Generic list of OpenAI deployments you want to create. If empty, none will
# be created unless you use the legacy per-model variables above.
variable "openai_deployments" {
  description = "List of OpenAI deployments to create"
  type = list(object({
    deploy_name   = string
    model_name    = string
    model_version = string
    format        = string
    scale_type    = string
  }))
  default = []
}

resource "azurerm_cognitive_account" "openai" {
  name                          = local.openai_account_name
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  kind                          = "OpenAI"
  sku_name                      = "S0"
  custom_subdomain_name         = local.openai_custom_subdomain
  public_network_access_enabled = true

  tags = local.common_tags
}

locals {
  # Build the deployment list from either explicit list, or legacy defaults
  _default_openai_deployments = []
  effective_openai_deployments = length(var.openai_deployments) > 0 ? var.openai_deployments : local._default_openai_deployments
}

resource "azurerm_cognitive_deployment" "openai" {
  for_each = var.enable_openai_deployments ? { for d in local.effective_openai_deployments : d.deploy_name => d } : {}

  name                 = each.value.deploy_name
  cognitive_account_id = azurerm_cognitive_account.openai.id
  rai_policy_name      = null

  model {
    format  = each.value.format
    name    = each.value.model_name
    version = each.value.model_version
  }

  scale {
    type = each.value.scale_type
  }
}

# Storage containers for different purposes
resource "azurerm_storage_container" "data" {
  name                  = "data"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "backups" {
  name                  = "backups"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# Azure Database for PostgreSQL - Flexible Server
# Free tier: B1MS (1 vCore, 2GB RAM) - 750 hours/month
resource "azurerm_postgresql_flexible_server" "main" {
  name                = "psql-cortex-${local.env_suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  # Free tier eligible
  sku_name = "B_Standard_B1ms" # Burstable, 1 vCore, 2GB RAM
  version  = "15"

  # Free storage: 32GB
  storage_mb = 32768

  administrator_login    = "pcxadmin"
  administrator_password = random_password.postgres.result

  backup_retention_days        = 7
  geo_redundant_backup_enabled = false # Keep costs down

  zone = "1"

  tags = local.common_tags
}

# PostgreSQL Database
resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "policycortex"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}

# PostgreSQL Firewall Rule - Allow Azure services
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure" {
  name             = "AllowAzureServices"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Cosmos DB - Free Tier Account
# Free: 25GB storage, 1000 RU/s
resource "azurerm_cosmosdb_account" "main" {
  name                = "cosmos-cortex-${local.env_suffix}-${local.unique_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  # Enable free tier (only one per subscription)
  free_tier_enabled = var.environment == "dev" ? true : false

  consistency_policy {
    consistency_level       = "Session"
    max_interval_in_seconds = 5
    max_staleness_prefix    = 100
  }

  geo_location {
    location          = azurerm_resource_group.main.location
    failover_priority = 0
  }

  tags = local.common_tags
}

# Cosmos DB SQL Database
resource "azurerm_cosmosdb_sql_database" "main" {
  name                = "policycortex"
  resource_group_name = azurerm_cosmosdb_account.main.resource_group_name
  account_name        = azurerm_cosmosdb_account.main.name

  # Free tier gets 1000 RU/s shared
  throughput = 400
}

# Note: VM and Public IP removed due to Azure free tier limits
# The free tier only allows limited Basic SKU public IPs
# Container Apps provide sufficient compute for the application

# Virtual Network - No cost
resource "azurerm_virtual_network" "main" {
  name                = "vnet-cortex-${local.env_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = ["10.0.0.0/16"]

  tags = local.common_tags
}

# Subnet - No cost
resource "azurerm_subnet" "main" {
  name                 = "subnet-app"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.1.0/24"]
}

# Container Registry - Basic tier (lower cost than Standard)
resource "azurerm_container_registry" "main" {
  name                = "crcortex${local.env_suffix}${local.unique_suffix}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic" # Cheapest tier
  admin_enabled       = true

  tags = local.common_tags
}

# Key Vault - Standard tier (no free tier, but minimal cost)
resource "azurerm_key_vault" "main" {
  name                = "kv-cortex-${local.env_suffix}-${local.unique_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  enable_rbac_authorization       = true
  enabled_for_deployment          = true
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = true
  purge_protection_enabled        = false # Can be deleted immediately (dev friendly)

  tags = local.common_tags
}

# Log Analytics Workspace for Container Apps
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-cortex-${local.env_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30 # Minimum retention to save costs

  tags = local.common_tags
}

# Application Insights - Free tier includes 5GB/month
resource "azurerm_application_insights" "main" {
  name                = "appi-cortex-${local.env_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.main.id

  daily_data_cap_in_gb                  = 0.5 # Keep under free limit
  daily_data_cap_notifications_disabled = false
  retention_in_days                     = 30 # Minimum retention
  sampling_percentage                   = 50 # Sample to reduce data
  disable_ip_masking                    = false

  tags = local.common_tags
}

# Container Apps Environment - Using Consumption profile for cost efficiency
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-cortex-${local.env_suffix}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  # No workload_profile block = Consumption profile (serverless, pay-per-use)
  # This is the most cost-effective option

  tags = local.common_tags
}

# Import guidance (handled automatically by deploy.ps1 before apply)
# If running Terraform manually, import pre-existing resources first:
#   terraform import azurerm_container_app_environment.main \
#     "/subscriptions/<SUB>/resourceGroups/rg-cortex-<env>/providers/Microsoft.App/managedEnvironments/cae-cortex-<env>"
#   terraform import azurerm_container_app.core \
#     "/subscriptions/<SUB>/resourceGroups/rg-cortex-<env>/providers/Microsoft.App/containerApps/ca-cortex-core-<env>"
#   terraform import azurerm_container_app.frontend \
#     "/subscriptions/<SUB>/resourceGroups/rg-cortex-<env>/providers/Microsoft.App/containerApps/ca-cortex-frontend-<env>"

# Container App - Core API
resource "azurerm_container_app" "core" {
  name                         = "ca-cortex-core-${local.env_suffix}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name = "core-api"
      # Use nginx as placeholder until actual image is pushed
      image  = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
      cpu    = 0.25    # Minimum CPU (0.25 vCPU)
      memory = "0.5Gi" # Minimum memory (0.5 GB)

      env {
        name  = "POSTGRES_HOST"
        value = azurerm_postgresql_flexible_server.main.fqdn
      }
      env {
        name  = "POSTGRES_DB"
        value = azurerm_postgresql_flexible_server_database.main.name
      }
      env {
        name        = "POSTGRES_PASSWORD"
        secret_name = "postgres-password"
      }
      env {
        name  = "COSMOS_ENDPOINT"
        value = azurerm_cosmosdb_account.main.endpoint
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }
    }

    min_replicas = 0 # Scale to zero to save costs
    max_replicas = 2
  }

  ingress {
    external_enabled = true
    target_port      = 8080

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  secret {
    name  = "postgres-password"
    value = random_password.postgres.result
  }

  # Registry configuration removed - using public images initially
  # Uncomment and update when pushing custom images:
  # registry {
  #   server               = azurerm_container_registry.main.login_server
  #   username             = azurerm_container_registry.main.admin_username
  #   password_secret_name = "registry-password"
  # }
  # secret {
  #   name  = "registry-password"
  #   value = azurerm_container_registry.main.admin_password
  # }

  tags = local.common_tags
}

# Container App - Frontend
resource "azurerm_container_app" "frontend" {
  name                         = "ca-cortex-frontend-${local.env_suffix}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name = "frontend"
      # Use nginx as placeholder until actual image is pushed
      image  = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name = "NEXT_PUBLIC_API_URL"
        # Use stable app FQDN (not revision-specific) to avoid provider plan drift
        value = "https://${azurerm_container_app.core.ingress[0].fqdn}"
      }
    }

    min_replicas = 0 # Scale to zero
    max_replicas = 2
  }

  ingress {
    external_enabled = true
    target_port      = 3000

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  # Registry configuration removed - using public images initially
  # Uncomment and update when pushing custom images:
  # registry {
  #   server               = azurerm_container_registry.main.login_server
  #   username             = azurerm_container_registry.main.admin_username
  #   password_secret_name = "registry-password"
  # }
  # secret {
  #   name  = "registry-password"
  #   value = azurerm_container_registry.main.admin_password
  # }

  tags = local.common_tags
}

# Service Bus Namespace - Basic tier (lowest cost)
resource "azurerm_servicebus_namespace" "main" {
  count = var.environment == "prod" ? 1 : 0 # Only in prod to manage costs

  name                = "sb-cortex-${local.env_suffix}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "Basic" # Cheapest tier

  tags = local.common_tags
}

# Random password for PostgreSQL
resource "random_password" "postgres" {
  length  = 16
  special = true
}

# Note: Key Vault secret creation requires additional RBAC permissions
# The Service Principal needs "Key Vault Secrets Officer" role
# For now, we're passing the password directly to Container Apps

# ===========================================================================
# OUTPUTS
# ===========================================================================

output "resource_group_name" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "storage_account_name" {
  description = "Storage account name"
  value       = azurerm_storage_account.main.name
}

output "postgresql_server_name" {
  description = "PostgreSQL server name"
  value       = azurerm_postgresql_flexible_server.main.name
}

output "postgresql_fqdn" {
  description = "PostgreSQL server FQDN"
  value       = azurerm_postgresql_flexible_server.main.fqdn
}

output "cosmosdb_endpoint" {
  description = "Cosmos DB endpoint"
  value       = azurerm_cosmosdb_account.main.endpoint
}

output "openai_endpoint" {
  description = "Azure OpenAI endpoint"
  value       = azurerm_cognitive_account.openai.endpoint
}

output "container_registry_login_server" {
  description = "Container Registry login server"
  value       = azurerm_container_registry.main.login_server
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}

output "application_insights_instrumentation_key" {
  description = "Application Insights instrumentation key"
  value       = azurerm_application_insights.main.instrumentation_key
  sensitive   = true
}

output "container_app_environment_id" {
  description = "Container Apps Environment ID"
  value       = azurerm_container_app_environment.main.id
}

output "container_app_core_url" {
  description = "Core API Container App URL"
  # Use stable ingress FQDN to avoid plan churn on new revisions
  value = "https://${azurerm_container_app.core.ingress[0].fqdn}"
}

output "container_app_frontend_url" {
  description = "Frontend Container App URL"
  # Use stable ingress FQDN to avoid plan churn on new revisions
  value = "https://${azurerm_container_app.frontend.ingress[0].fqdn}"
}