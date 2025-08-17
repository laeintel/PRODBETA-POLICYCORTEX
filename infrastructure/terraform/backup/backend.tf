# Backend configuration moved to environments/dev and environments/prod
# This file now only contains the storage account resources for state management

# Resource group to hold Terraform state resources
variable "manage_state_storage" {
  description = "Whether to create/own the Terraform state storage resources"
  type        = bool
  default     = false
}

resource "azurerm_resource_group" "terraform_state" {
  count    = var.manage_state_storage ? 1 : 0
  name     = "rg-terraform-state-${var.environment}"
  location = var.location
}

# Create storage account for Terraform state with locking enabled
resource "azurerm_storage_account" "terraform_state" {
  count                    = var.manage_state_storage ? 1 : 0
  name                     = "stterraformstate${var.environment}"
  resource_group_name      = azurerm_resource_group.terraform_state[0].name
  location                 = azurerm_resource_group.terraform_state[0].location
  account_tier             = "Standard"
  account_replication_type = "GRS"

  # Enable blob versioning for state history
  blob_properties {
    versioning_enabled = true

    delete_retention_policy {
      days = 30
    }

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
  count                 = var.manage_state_storage ? 1 : 0
  name                  = "tfstate"
  storage_account_name  = azurerm_storage_account.terraform_state[0].name
  container_access_type = "private"
}

# Lock table for additional safety (optional with Azure)
resource "azurerm_storage_table" "terraform_locks" {
  count                = var.manage_state_storage ? 1 : 0
  name                 = "terraformlocks"
  storage_account_name = azurerm_storage_account.terraform_state[0].name
}