# Data Services Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "policycortex"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "vnet_name" {
  description = "Name of the virtual network"
  type        = string
}

variable "data_services_subnet_name" {
  description = "Name of the data services subnet"
  type        = string
  default     = "data-services-subnet"
}

variable "key_vault_name" {
  description = "Name of the Key Vault"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# SQL Server variables
variable "deploy_sql_server" {
  description = "Whether to deploy SQL Server"
  type        = bool
  default     = true
}

variable "sql_admin_username" {
  description = "SQL Server administrator username"
  type        = string
  default     = "sqladmin"
}

variable "sql_azuread_admin_login" {
  description = "Azure AD admin login for SQL Server"
  type        = string
}

variable "sql_azuread_admin_object_id" {
  description = "Azure AD admin object ID for SQL Server"
  type        = string
}

variable "sql_database_sku" {
  description = "SKU for the SQL database"
  type        = string
  default     = "GP_S_Gen5_2"
}

variable "sql_database_max_size_gb" {
  description = "Maximum size of the SQL database in GB"
  type        = number
  default     = 32
}

variable "sql_backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

# Cosmos DB variables
variable "cosmos_consistency_level" {
  description = "Cosmos DB consistency level"
  type        = string
  default     = "Session"
  validation {
    condition = contains(["Eventual", "ConsistentPrefix", "Session", "BoundedStaleness", "Strong"], var.cosmos_consistency_level)
    error_message = "Consistency level must be one of: Eventual, ConsistentPrefix, Session, BoundedStaleness, Strong."
  }
}

variable "cosmos_failover_location" {
  description = "Secondary region for Cosmos DB geo-replication"
  type        = string
  default     = "West US 2"
}

variable "cosmos_max_throughput" {
  description = "Maximum throughput for Cosmos DB database"
  type        = number
  default     = 4000
}

variable "cosmos_container_max_throughput" {
  description = "Maximum throughput for Cosmos DB containers"
  type        = number
  default     = 1000
}

# Redis Cache variables
variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 2
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "C"
}

variable "redis_sku_name" {
  description = "Redis cache SKU name"
  type        = string
  default     = "Standard"
  validation {
    condition = contains(["Basic", "Standard", "Premium"], var.redis_sku_name)
    error_message = "Redis SKU must be Basic, Standard, or Premium."
  }
}

variable "redis_maxmemory_reserved" {
  description = "Redis max memory reserved"
  type        = number
  default     = 2
}

variable "redis_maxmemory_delta" {
  description = "Redis max memory delta"
  type        = number
  default     = 2
}