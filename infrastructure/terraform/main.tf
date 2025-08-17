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

    # Container Apps
    container_env = "cae-${local.project}-${var.environment}"
    core_app      = "ca-${local.project}-core-${var.environment}"
    frontend_app  = "ca-${local.project}-frontend-${var.environment}"
    graphql_app   = "ca-${local.project}-graphql-${var.environment}"

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

# Subnet for Container Apps
resource "azurerm_subnet" "container_apps" {
  name                 = "snet-container-apps"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.0.0.0/23"]

  delegation {
    name = "container-apps"
    service_delegation {
      name    = "Microsoft.App/environments"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
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

  network_rule_set {
    default_action = "Deny"

    virtual_network {
      action    = "Allow"
      subnet_id = azurerm_subnet.container_apps.id
    }
  }

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

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [azurerm_subnet.container_apps.id]
  }

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

  network_acls {
    default_action             = "Deny"
    bypass                     = "AzureServices"
    virtual_network_subnet_ids = [azurerm_subnet.container_apps.id]
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

  is_virtual_network_filter_enabled = true

  virtual_network_rule {
    id = azurerm_subnet.container_apps.id
  }

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

  network_acls {
    default_action = "Deny"
    virtual_network_rules {
      subnet_id = azurerm_subnet.container_apps.id
    }
  }

  tags = local.common_tags
}

resource "azurerm_cognitive_deployment" "gpt4" {
  name                 = "gpt-4"
  cognitive_account_id = azurerm_cognitive_account.openai.id

  model {
    format  = "OpenAI"
    name    = "gpt-4"
    version = "0613"
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
# CONTAINER APPS ENVIRONMENT
# ===========================

resource "azurerm_container_app_environment" "main" {
  name                       = local.resource_names.container_env
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  infrastructure_subnet_id   = azurerm_subnet.container_apps.id

  tags = local.common_tags
}

# ===========================
# USER ASSIGNED IDENTITY FOR CONTAINER APPS
# ===========================

resource "azurerm_user_assigned_identity" "container_apps" {
  name                = "id-container-apps-${var.environment}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

# Role assignments
resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

resource "azurerm_role_assignment" "keyvault_secrets" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.container_apps.principal_id
}

# ===========================
# CONTAINER APPS
# ===========================

resource "azurerm_container_app" "core" {
  name                         = local.resource_names.core_app
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "core"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-core:latest"
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.main.connection_string
      }

      env {
        name  = "DATABASE_URL"
        value = "postgresql://psqladmin:${random_password.db_password.result}@${azurerm_postgresql_flexible_server.main.fqdn}/policycortex"
      }
    }
    min_replicas = 0
    max_replicas = 2
  }

  ingress {
    external_enabled = true
    target_port      = 8080
    transport        = "http"
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }

  tags = local.common_tags

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}

resource "azurerm_container_app" "frontend" {
  name                         = local.resource_names.frontend_app
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-frontend:latest"
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "NEXT_PUBLIC_API_URL"
        value = "https://${azurerm_container_app.core.latest_revision_fqdn}"
      }

      env {
        name  = "NEXT_PUBLIC_GRAPHQL_URL"
        value = "https://${azurerm_container_app.graphql.latest_revision_fqdn}"
      }
    }
    min_replicas = 0
    max_replicas = 2
  }

  ingress {
    external_enabled = true
    target_port      = 3000
    transport        = "http"
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }

  tags = local.common_tags

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}

resource "azurerm_container_app" "graphql" {
  name                         = local.resource_names.graphql_app
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = "graphql"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-graphql:latest"
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "CORE_API_URL"
        value = "https://${azurerm_container_app.core.latest_revision_fqdn}"
      }
    }
    min_replicas = 0
    max_replicas = 2
  }

  ingress {
    external_enabled = true
    target_port      = 4000
    transport        = "http"
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }

  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }

  tags = local.common_tags

  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}

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

output "container_apps_environment_id" {
  value = azurerm_container_app_environment.main.id
}

output "core_app_url" {
  value = "https://${azurerm_container_app.core.latest_revision_fqdn}"
}

output "frontend_app_url" {
  value = "https://${azurerm_container_app.frontend.latest_revision_fqdn}"
}

output "graphql_app_url" {
  value = "https://${azurerm_container_app.graphql.latest_revision_fqdn}"
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

