# Centralized Variables for Resource Naming
# This ensures consistent naming across all resources

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "Environment must be either 'dev' or 'prod'."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "eastus"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "cortex"
}

# Computed locals for consistent naming
locals {
  # Generate consistent hash from subscription ID for uniqueness
  hash_input  = "${data.azurerm_client_config.current.subscription_id}-${var.project_name}"
  hash_suffix = substr(md5(local.hash_input), 0, 6)

  # Environment suffix
  env_suffix = var.environment

  # Resource names - single source of truth
  resource_names = {
    # Resource Group
    resource_group = "rg-${var.project_name}-${local.env_suffix}"

    # Container Apps
    container_env = "cae-${var.project_name}-${local.env_suffix}"
    core_app      = "ca-${var.project_name}-core-${local.env_suffix}"
    frontend_app  = "ca-${var.project_name}-frontend-${local.env_suffix}"
    graphql_app   = "ca-${var.project_name}-graphql-${local.env_suffix}"

    # Container Registry (global, needs unique name)
    container_registry = "cr${var.project_name}${local.env_suffix}${local.hash_suffix}"

    # Storage Account (global, needs unique name, no hyphens)
    storage_account = "st${var.project_name}${local.env_suffix}${local.hash_suffix}"

    # Key Vault
    key_vault = "kv-${var.project_name}-${local.env_suffix}-${local.hash_suffix}"

    # Database
    postgresql = "psql-${var.project_name}-${local.env_suffix}"
    cosmos_db  = "cosmos-${var.project_name}-${local.env_suffix}-${local.hash_suffix}"

    # Monitoring
    log_workspace = "log-${var.project_name}-${local.env_suffix}"
    app_insights  = "appi-${var.project_name}-${local.env_suffix}"

    # Networking
    vnet = "vnet-${var.project_name}-${local.env_suffix}"

    # AI Services
    openai = "cogao-${var.project_name}-${local.env_suffix}"
  }

  # Common tags
  common_tags = {
    Environment = var.environment
    Project     = "PolicyCortex"
    ManagedBy   = "Terraform"
    Repository  = "github.com/laeintel/policycortex"
    CostCenter  = var.environment == "prod" ? "Production" : "Development"
  }
}

# Outputs for use by other systems
output "resource_names" {
  description = "All resource names for this environment"
  value       = local.resource_names
}

output "container_registry_url" {
  description = "Container Registry URL"
  value       = "${local.resource_names.container_registry}.azurecr.io"
}

output "hash_suffix" {
  description = "Hash suffix used for globally unique names"
  value       = local.hash_suffix
}