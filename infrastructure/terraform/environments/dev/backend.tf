terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state-dev"
    storage_account_name = "stterraformstatedev"
    container_name       = "tfstate"
    key                  = "policycortex.dev.tfstate"

    # Enable state locking - automatic with Azure Storage
    use_azuread_auth = true
  }
}