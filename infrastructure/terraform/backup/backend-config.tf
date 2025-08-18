# Backend Configuration for Terraform State
# These resources should NEVER be deleted as they maintain Terraform state

locals {
  # Backend configuration based on environment
  backend_config = {
    dev = {
      resource_group_name  = "rg-tfstate-cortex-dev"
      storage_account_name = "sttfcortexdev${local.hash_suffix}"
      container_name       = "tfstate"
      key                  = "dev.tfstate"
    }
    prod = {
      resource_group_name  = "rg-tfstate-cortex-prod"
      storage_account_name = "sttfcortexprod${local.hash_suffix}"
      container_name       = "tfstate"
      key                  = "prod.tfstate"
    }
  }
}

# Output backend configuration for reference
output "backend_config" {
  description = "Terraform backend configuration for this environment"
  value = {
    resource_group  = local.backend_config[var.environment].resource_group_name
    storage_account = local.backend_config[var.environment].storage_account_name
    container       = local.backend_config[var.environment].container_name
    key             = local.backend_config[var.environment].key
  }
  sensitive = false
}