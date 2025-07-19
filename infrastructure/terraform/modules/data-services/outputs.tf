# Data Services Module Outputs

# SQL Server outputs
output "sql_server_id" {
  description = "ID of the SQL Server"
  value       = azurerm_mssql_server.main.id
}

output "sql_server_fqdn" {
  description = "Fully qualified domain name of the SQL Server"
  value       = azurerm_mssql_server.main.fully_qualified_domain_name
}

output "sql_database_id" {
  description = "ID of the SQL Database"
  value       = azurerm_mssql_database.policycortex.id
}

output "sql_database_name" {
  description = "Name of the SQL Database"
  value       = azurerm_mssql_database.policycortex.name
}

output "sql_admin_username" {
  description = "SQL Server administrator username"
  value       = var.sql_admin_username
}

# Cosmos DB outputs
output "cosmos_account_id" {
  description = "ID of the Cosmos DB account"
  value       = azurerm_cosmosdb_account.main.id
}

output "cosmos_account_endpoint" {
  description = "Endpoint of the Cosmos DB account"
  value       = azurerm_cosmosdb_account.main.endpoint
}

output "cosmos_account_name" {
  description = "Name of the Cosmos DB account"
  value       = azurerm_cosmosdb_account.main.name
}

output "cosmos_database_name" {
  description = "Name of the Cosmos DB database"
  value       = azurerm_cosmosdb_sql_database.governance.name
}

output "cosmos_containers" {
  description = "Names of the Cosmos DB containers"
  value = {
    policies      = azurerm_cosmosdb_sql_container.policies.name
    conversations = azurerm_cosmosdb_sql_container.conversations.name
    audit_logs    = azurerm_cosmosdb_sql_container.audit_logs.name
  }
}

# Redis Cache outputs
output "redis_cache_id" {
  description = "ID of the Redis Cache"
  value       = azurerm_redis_cache.main.id
}

output "redis_cache_name" {
  description = "Name of the Redis Cache"
  value       = azurerm_redis_cache.main.name
}

output "redis_cache_hostname" {
  description = "Hostname of the Redis Cache"
  value       = azurerm_redis_cache.main.hostname
}

output "redis_cache_port" {
  description = "Port of the Redis Cache"
  value       = azurerm_redis_cache.main.port
}

output "redis_cache_ssl_port" {
  description = "SSL port of the Redis Cache"
  value       = azurerm_redis_cache.main.ssl_port
}

# Private DNS zones
output "private_dns_zones" {
  description = "Private DNS zones created for data services"
  value = {
    sql    = azurerm_private_dns_zone.sql.name
    cosmos = azurerm_private_dns_zone.cosmos.name
    redis  = azurerm_private_dns_zone.redis.name
  }
}

# Private endpoints
output "private_endpoints" {
  description = "Private endpoints created for data services"
  value = {
    sql    = azurerm_private_endpoint.sql.id
    cosmos = azurerm_private_endpoint.cosmos.id
    redis  = azurerm_private_endpoint.redis.id
  }
}

# Connection strings (stored in Key Vault)
output "connection_string_secrets" {
  description = "Names of Key Vault secrets containing connection strings"
  value = {
    sql    = azurerm_key_vault_secret.sql_connection_string.name
    cosmos = azurerm_key_vault_secret.cosmos_connection_string.name
    redis  = azurerm_key_vault_secret.redis_connection_string.name
  }
}