# Data Services Module for PolicyCortex
# Implements Azure SQL Database, Cosmos DB, and Redis Cache

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Data sources for existing resources
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

data "azurerm_virtual_network" "main" {
  name                = var.vnet_name
  resource_group_name = var.network_resource_group_name
}

data "azurerm_subnet" "data_services" {
  name                 = var.data_services_subnet_name
  virtual_network_name = var.vnet_name
  resource_group_name  = var.network_resource_group_name
}

data "azurerm_subnet" "private_endpoints" {
  name                 = var.private_endpoints_subnet_name
  virtual_network_name = var.vnet_name
  resource_group_name  = var.network_resource_group_name
}

data "azurerm_key_vault" "main" {
  name                = var.key_vault_name
  resource_group_name = var.resource_group_name
}

# Random password for SQL Server admin
resource "random_password" "sql_admin_password" {
  length  = 32
  special = true
}

# Store SQL admin password in Key Vault
resource "azurerm_key_vault_secret" "sql_admin_password" {
  name         = "sql-admin-password"
  value        = random_password.sql_admin_password.result
  key_vault_id = data.azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }

  tags = var.tags
}

# Azure SQL Server
resource "azurerm_mssql_server" "main" {
  count                        = var.deploy_sql_server ? 1 : 0
  name                         = "${var.project_name}-sql-${var.environment}"
  resource_group_name          = data.azurerm_resource_group.main.name
  location                     = data.azurerm_resource_group.main.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_username
  administrator_login_password = random_password.sql_admin_password.result

  # Security settings
  minimum_tls_version = "1.2"
  
  # Azure AD authentication
  dynamic "azuread_administrator" {
    for_each = var.sql_azuread_admin_object_id != "" && var.sql_azuread_admin_object_id != "00000000-0000-0000-0000-000000000000" ? [1] : []
    content {
      login_username = var.sql_azuread_admin_login
      object_id      = var.sql_azuread_admin_object_id
    }
  }

  # User-assigned managed identity
  identity {
    type         = "UserAssigned"
    identity_ids = [var.managed_identity_id]
  }

  tags = var.tags
}

# SQL Server firewall rule for Azure services
resource "azurerm_mssql_firewall_rule" "azure_services" {
  count            = var.deploy_sql_server ? 1 : 0
  name             = "AllowAzureServices"
  server_id        = azurerm_mssql_server.main[0].id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# Private endpoint for SQL Server
resource "azurerm_private_endpoint" "sql" {
  count               = var.deploy_sql_server ? 1 : 0
  name                = "${var.project_name}-sql-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "${var.project_name}-sql-psc-${var.environment}"
    private_connection_resource_id = azurerm_mssql_server.main[0].id
    subresource_names             = ["sqlServer"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "sql-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.sql.id]
  }

  tags = var.tags
}

# Private DNS zone for SQL Server
resource "azurerm_private_dns_zone" "sql" {
  name                = "privatelink.database.windows.net"
  resource_group_name = var.network_resource_group_name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "sql" {
  name                  = "sql-dns-vnet-link"
  resource_group_name   = var.network_resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.sql.name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# Azure SQL Database for PolicyCortex
resource "azurerm_mssql_database" "policycortex" {
  count     = var.deploy_sql_server ? 1 : 0
  name      = "${var.project_name}-db-${var.environment}"
  server_id = azurerm_mssql_server.main[0].id
  
  # Performance and sizing
  sku_name                    = var.sql_database_sku
  max_size_gb                 = var.sql_database_max_size_gb
  auto_pause_delay_in_minutes = var.environment == "dev" ? 60 : -1
  min_capacity                = var.environment == "dev" ? 0.5 : 2
  
  # Backup and retention
  short_term_retention_policy {
    retention_days           = var.sql_backup_retention_days
    backup_interval_in_hours = 12
  }

  long_term_retention_policy {
    weekly_retention  = "P4W"
    monthly_retention = "P12M"
    yearly_retention  = "P7Y"
    week_of_year      = 1
  }

  # Security
  transparent_data_encryption_enabled = true

  tags = var.tags
}

# Cosmos DB Account
resource "azurerm_cosmosdb_account" "main" {
  name                = "${var.project_name}-cosmos-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  # Consistency policy
  consistency_policy {
    consistency_level       = var.cosmos_consistency_level
    max_interval_in_seconds = 10
    max_staleness_prefix    = 200
  }

  # Geo-replication
  geo_location {
    location          = data.azurerm_resource_group.main.location
    failover_priority = 0
    zone_redundant    = var.environment == "prod"
  }

  dynamic "geo_location" {
    for_each = var.environment == "prod" ? [var.cosmos_failover_location] : []
    content {
      location          = geo_location.value
      failover_priority = 1
      zone_redundant    = true
    }
  }

  # Security and access
  public_network_access_enabled = false
  is_virtual_network_filter_enabled = true
  
  virtual_network_rule {
    id                                   = data.azurerm_subnet.data_services.id
    ignore_missing_vnet_service_endpoint = false
  }

  # Backup
  backup {
    type                = "Periodic"
    interval_in_minutes = 240
    retention_in_hours  = 720
    storage_redundancy  = "Geo"
  }

  # Capabilities
  capabilities {
    name = "EnableServerless"
  }

  capabilities {
    name = "EnableAggregationPipeline"
  }

  # User-assigned managed identity
  identity {
    type         = "UserAssigned"
    identity_ids = [var.managed_identity_id]
  }

  tags = var.tags
}

# Private endpoint for Cosmos DB
resource "azurerm_private_endpoint" "cosmos" {
  name                = "${var.project_name}-cosmos-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "${var.project_name}-cosmos-psc-${var.environment}"
    private_connection_resource_id = azurerm_cosmosdb_account.main.id
    subresource_names             = ["Sql"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "cosmos-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.cosmos.id]
  }

  tags = var.tags
}

# Private DNS zone for Cosmos DB
resource "azurerm_private_dns_zone" "cosmos" {
  name                = "privatelink.documents.azure.com"
  resource_group_name = var.network_resource_group_name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "cosmos" {
  name                  = "cosmos-dns-vnet-link"
  resource_group_name   = var.network_resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.cosmos.name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# Cosmos DB databases and containers
resource "azurerm_cosmosdb_sql_database" "governance" {
  name                = "governance"
  resource_group_name = data.azurerm_resource_group.main.name
  account_name        = azurerm_cosmosdb_account.main.name
}

resource "azurerm_cosmosdb_sql_container" "policies" {
  name                  = "policies"
  resource_group_name   = data.azurerm_resource_group.main.name
  account_name          = azurerm_cosmosdb_account.main.name
  database_name         = azurerm_cosmosdb_sql_database.governance.name
  partition_key_paths   = ["/tenantId"]
  partition_key_version = 1

  indexing_policy {
    indexing_mode = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/\"_etag\"/?"
    }
  }
}

resource "azurerm_cosmosdb_sql_container" "conversations" {
  name                  = "conversations"
  resource_group_name   = data.azurerm_resource_group.main.name
  account_name          = azurerm_cosmosdb_account.main.name
  database_name         = azurerm_cosmosdb_sql_database.governance.name
  partition_key_paths   = ["/userId"]
  partition_key_version = 1

  # TTL for conversation history
  default_ttl = 2592000  # 30 days
}

resource "azurerm_cosmosdb_sql_container" "audit_logs" {
  name                  = "audit-logs"
  resource_group_name   = data.azurerm_resource_group.main.name
  account_name          = azurerm_cosmosdb_account.main.name
  database_name         = azurerm_cosmosdb_sql_database.governance.name
  partition_key_paths   = ["/tenantId"]
  partition_key_version = 1

  # TTL for audit logs (7 years for compliance)
  default_ttl = 220898400  # 7 years
}

# Redis Cache
resource "azurerm_redis_cache" "main" {
  name                = "${var.project_name}-redis-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  capacity            = var.redis_capacity
  family              = var.redis_family
  sku_name            = var.redis_sku_name
  
  # Security
  minimum_tls_version = "1.2"
  public_network_access_enabled = false
  
  # Redis configuration
  redis_configuration {
    authentication_enabled = true
    maxmemory_reserved     = var.redis_maxmemory_reserved
    maxmemory_delta        = var.redis_maxmemory_delta
    maxmemory_policy       = "allkeys-lru"
  }

  # Backup for production
  dynamic "patch_schedule" {
    for_each = var.environment == "prod" ? ["enabled"] : []
    content {
      day_of_week        = "Sunday"
      start_hour_utc     = 2
      maintenance_window = "PT5H"
    }
  }

  # User-assigned managed identity
  identity {
    type         = "UserAssigned"
    identity_ids = [var.managed_identity_id]
  }

  tags = var.tags
}

# Private endpoint for Redis
resource "azurerm_private_endpoint" "redis" {
  name                = "${var.project_name}-redis-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.private_endpoints.id

  private_service_connection {
    name                           = "${var.project_name}-redis-psc-${var.environment}"
    private_connection_resource_id = azurerm_redis_cache.main.id
    subresource_names             = ["redisCache"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "redis-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.redis.id]
  }

  tags = var.tags
}

# Private DNS zone for Redis
resource "azurerm_private_dns_zone" "redis" {
  name                = "privatelink.redis.cache.windows.net"
  resource_group_name = var.network_resource_group_name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "redis" {
  name                  = "redis-dns-vnet-link"
  resource_group_name   = var.network_resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.redis.name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# Store Redis connection details in Key Vault
resource "azurerm_key_vault_secret" "redis_connection_string" {
  name         = "redis-connection-string"
  value        = azurerm_redis_cache.main.primary_connection_string
  key_vault_id = data.azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }

  depends_on = [
    azurerm_redis_cache.main,
    azurerm_private_endpoint.redis
  ]

  tags = var.tags
}

resource "azurerm_key_vault_secret" "cosmos_connection_string" {
  name         = "cosmos-connection-string"
  value        = azurerm_cosmosdb_account.main.primary_sql_connection_string
  key_vault_id = data.azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }

  depends_on = [
    azurerm_cosmosdb_account.main,
    azurerm_cosmosdb_sql_database.governance,
    azurerm_cosmosdb_sql_container.policies,
    azurerm_cosmosdb_sql_container.conversations,
    azurerm_cosmosdb_sql_container.audit_logs
  ]

  tags = var.tags
}

resource "azurerm_key_vault_secret" "sql_connection_string" {
  count = var.deploy_sql_server ? 1 : 0
  name  = "sql-connection-string"
  value = "Server=tcp:${azurerm_mssql_server.main[0].fully_qualified_domain_name},1433;Initial Catalog=${azurerm_mssql_database.policycortex[0].name};Persist Security Info=False;User ID=${var.sql_admin_username};Password=${random_password.sql_admin_password.result};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
  key_vault_id = data.azurerm_key_vault.main.id

  lifecycle {
    ignore_changes = [value]
  }

  tags = var.tags
}