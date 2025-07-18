terraform {
  required_version = ">= 1.5"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.80"
    }
    azuread = {
      source  = "hashicorp/azuread"
      version = "~> 2.45"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # Backend configuration for remote state
  backend "azurerm" {
    # Values will be provided via backend configuration file
    # storage_account_name = ""
    # container_name      = ""
    # key                = ""
    # resource_group_name = ""
  }
}

# Configure the Azure Provider
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

# Configure the Azure Active Directory Provider
provider "azuread" {
  # Configuration for Azure AD resources
}

# Data source for current Azure client configuration
data "azurerm_client_config" "current" {}

# Data source for current subscription
data "azurerm_subscription" "current" {}

# Data source for Azure regions
data "azurerm_locations" "available" {
  location_filter {
    include_unavailable = false
  }
} 