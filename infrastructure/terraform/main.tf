# Main Terraform configuration for PolicyCortex infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  
  backend "azurerm" {
    # Backend configuration will be provided via init command
  }
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    storage {
      storage_account {
        enable_https_traffic_only = true
        min_tls_version           = "TLS1_2"
      }
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
  account_replication_type = "GRS"  # Geographic redundancy for critical data
  
  # Security configurations
  min_tls_version                 = "TLS1_2"
  enable_https_traffic_only       = true
  public_network_access_enabled   = false  # Disable public access
  allow_nested_items_to_be_public = false  # Prevent blob anonymous access
  shared_access_key_enabled       = false  # Disable shared key auth
  
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

# Random string for storage account suffix
resource "random_string" "storage_suffix" {
  length  = 6
  special = false
  upper   = false
}

# Customer managed key for storage encryption
resource "azurerm_key_vault_key" "storage_key" {
  name         = "storage-encryption-key"
  key_vault_id = azurerm_key_vault.main.id
  key_type     = "RSA"
  key_size     = 2048
  
  key_opts = [
    "decrypt",
    "encrypt",
    "sign",
    "unwrapKey",
    "verify",
    "wrapKey",
  ]
  
  depends_on = [azurerm_key_vault_access_policy.terraform]
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

# Random string for Key Vault suffix
resource "random_string" "kv_suffix" {
  length  = 6
  special = false
  upper   = false
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