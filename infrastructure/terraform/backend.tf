terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "stterraformstate${var.environment}"
    container_name       = "tfstate"
    key                  = "policycortex.${var.environment}.tfstate"
    
    # Enable state locking with Azure Blob Storage
    use_azuread_auth = true
    # State locking is automatic with Azure Storage backend
    # Each state file gets a lease that prevents concurrent modifications
  }
  
  required_version = ">= 1.6.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Create storage account for Terraform state with locking enabled
resource "azurerm_storage_account" "terraform_state" {
  name                     = "stterraformstate${var.environment}"
  resource_group_name      = azurerm_resource_group.terraform_state.name
  location                 = azurerm_resource_group.terraform_state.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
  
  # Enable blob versioning for state history
  blob_properties {
    versioning_enabled = true
    
    delete_retention_policy {
      days = 30
    }
  }
  
  # Enable soft delete for accidental deletion protection
  blob_properties {
    container_delete_retention_policy {
      days = 7
    }
  }
  
  tags = {
    purpose     = "terraform-state"
    environment = var.environment
    managed_by  = "terraform"
  }
}

resource "azurerm_storage_container" "terraform_state" {
  name                  = "tfstate"
  storage_account_name  = azurerm_storage_account.terraform_state.name
  container_access_type = "private"
}

# Lock table for additional safety (optional with Azure)
resource "azurerm_storage_table" "terraform_locks" {
  name                 = "terraformlocks"
  storage_account_name = azurerm_storage_account.terraform_state.name
}