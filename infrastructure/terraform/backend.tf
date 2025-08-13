# Backend configuration moved to environments/dev and environments/prod
# This file now only contains the storage account resources for state management

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