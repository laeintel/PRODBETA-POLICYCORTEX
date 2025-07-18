# Main Terraform configuration for PolicyCortex infrastructure
# Service principal now has Owner permissions on subscription
terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
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

# Random strings for unique naming
resource "random_string" "storage_suffix" {
  length  = 6
  special = false
  upper   = false
}

resource "random_string" "kv_suffix" {
  length  = 6
  special = false
  upper   = false
}

# Resource group for the environment
resource "azurerm_resource_group" "main" {
  name     = "rg-policycortex-${var.environment}"
  location = var.location
  tags     = local.common_tags
}

# Storage account for application data (with security compliance)
resource "azurerm_storage_account" "app_storage" {
  name                     = "stpolicycortex${var.environment}${random_string.storage_suffix.result}"
  resource_group_name      = azurerm_resource_group.main.name
  location                = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  
  # Security configurations
  min_tls_version                 = "TLS1_2"
  enable_https_traffic_only       = true
  public_network_access_enabled   = false
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  
  # Network rules
  network_rules {
    default_action = "Deny"
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
  name                = "kv-policycortex-${var.environment}-${random_string.kv_suffix.result}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
  
  # Security settings
  purge_protection_enabled   = true
  soft_delete_retention_days = 30
  
  tags = local.common_tags
}

# Key Vault access policy for Terraform
resource "azurerm_key_vault_access_policy" "terraform" {
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