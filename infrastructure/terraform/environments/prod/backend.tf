terraform {
  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "stterraformstateprod"
    container_name       = "tfstate"
    key                  = "policycortex.prod.tfstate"
    
    # Enable state locking - automatic with Azure Storage
    use_azuread_auth = true
  }
}