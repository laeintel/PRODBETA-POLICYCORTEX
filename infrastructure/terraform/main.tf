# PolicyCortex Infrastructure - Main Configuration
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
  
  backend "azurerm" {
    # Backend configuration will be provided via CLI or environment variables
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
}

# Variables
variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "East US"
}

variable "owner" {
  description = "Owner tag for resources"
  type        = string
  default     = "AeoliTech"
}

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "rg-policycortex-${var.environment}"
  location = var.location

  tags = {
    Environment = var.environment
    Owner       = var.owner
    Project     = "PolicyCortex"
    ManagedBy   = "Terraform"
  }
}

# Container Registry
resource "azurerm_container_registry" "main" {
  name                = "crpolicycortex${var.environment}${random_string.suffix.result}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Standard"
  admin_enabled       = true

  tags = {
    Environment = var.environment
    Owner       = var.owner
    Project     = "PolicyCortex"
  }
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "main" {
  name                = "law-policycortex-${var.environment}-${random_string.suffix.result}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = {
    Environment = var.environment
    Owner       = var.owner
    Project     = "PolicyCortex"
  }
}

# Container Apps Environment
resource "azurerm_container_app_environment" "main" {
  name                       = "cae-policycortex-${var.environment}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  tags = {
    Environment = var.environment
    Owner       = var.owner
    Project     = "PolicyCortex"
  }
}

# Key Vault
resource "azurerm_key_vault" "main" {
  name                = "kv-pcx-${var.environment}-${random_string.suffix.result}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"

  enable_rbac_authorization = true
  purge_protection_enabled  = false

  tags = {
    Environment = var.environment
    Owner       = var.owner
    Project     = "PolicyCortex"
  }
}

# Current Azure configuration
data "azurerm_client_config" "current" {}

# Outputs
output "resource_group" {
  description = "Resource group name"
  value       = azurerm_resource_group.main.name
}

output "container_registry" {
  description = "Container registry name"
  value       = azurerm_container_registry.main.name
}

output "container_registry_login_server" {
  description = "Container registry login server"
  value       = azurerm_container_registry.main.login_server
}

output "container_apps_environment" {
  description = "Container Apps environment name"
  value       = azurerm_container_app_environment.main.name
}

output "key_vault_name" {
  description = "Key Vault name"
  value       = azurerm_key_vault.main.name
}

output "key_vault_uri" {
  description = "Key Vault URI"
  value       = azurerm_key_vault.main.vault_uri
}