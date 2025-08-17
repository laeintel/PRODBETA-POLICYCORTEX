terraform {
  required_version = ">= 1.6.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

variable "tenant_id" {
  description = "Unique tenant identifier"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "tier" {
  description = "Tenant tier (free, pro, enterprise)"
  type        = string
  default     = "free"
}

variable "azure_tenant_id" {
  description = "Azure AD tenant ID"
  type        = string
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

locals {
  kv_name = "kv-${substr(var.tenant_id, 0, 16)}"
  
  sku_mapping = {
    free       = "standard"
    pro        = "standard"
    enterprise = "premium"
  }
  
  retention_days = {
    free       = 7
    pro        = 30
    enterprise = 90
  }
}

resource "azurerm_key_vault" "tenant" {
  name                       = local.kv_name
  location                   = var.location
  resource_group_name        = var.resource_group_name
  tenant_id                  = var.azure_tenant_id
  sku_name                   = local.sku_mapping[var.tier]
  soft_delete_retention_days = local.retention_days[var.tier]
  purge_protection_enabled   = var.tier == "enterprise"
  
  enabled_for_deployment          = false
  enabled_for_disk_encryption     = true
  enabled_for_template_deployment = false
  
  public_network_access_enabled = false
  
  network_acls {
    default_action             = "Deny"
    bypass                     = "AzureServices"
    ip_rules                   = []
    virtual_network_subnet_ids = []
  }
  
  tags = merge(var.tags, {
    tenant_id = var.tenant_id
    tier      = var.tier
    managed   = "terraform"
  })
}

resource "azurerm_key_vault_access_policy" "tenant_app" {
  key_vault_id = azurerm_key_vault.tenant.id
  tenant_id    = var.azure_tenant_id
  object_id    = var.tenant_id # This would be the service principal ID
  
  secret_permissions = [
    "Get",
    "List",
    "Set",
    "Delete",
    "Recover",
    "Backup",
    "Restore"
  ]
  
  key_permissions = var.tier == "enterprise" ? [
    "Get",
    "List",
    "Create",
    "Delete",
    "Update",
    "Import",
    "Backup",
    "Restore",
    "Recover",
    "Encrypt",
    "Decrypt",
    "Sign",
    "Verify"
  ] : [
    "Get",
    "List",
    "Encrypt",
    "Decrypt"
  ]
  
  certificate_permissions = var.tier != "free" ? [
    "Get",
    "List",
    "Create",
    "Delete",
    "Update"
  ] : []
}

# Create initial secrets for the tenant
resource "azurerm_key_vault_secret" "db_connection" {
  name         = "db-connection-string"
  value        = "Server=tcp:${var.tenant_id}.database.windows.net;Database=policycortex;Authentication=Active Directory Managed Identity;"
  key_vault_id = azurerm_key_vault.tenant.id
  
  depends_on = [azurerm_key_vault_access_policy.tenant_app]
}

resource "azurerm_key_vault_secret" "api_key" {
  name         = "api-key"
  value        = random_password.api_key.result
  key_vault_id = azurerm_key_vault.tenant.id
  
  depends_on = [azurerm_key_vault_access_policy.tenant_app]
}

resource "random_password" "api_key" {
  length  = 32
  special = true
}

# Private endpoint for Key Vault
resource "azurerm_private_endpoint" "keyvault" {
  count = var.tier != "free" ? 1 : 0
  
  name                = "${local.kv_name}-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.subnet_id
  
  private_service_connection {
    name                           = "${local.kv_name}-psc"
    private_connection_resource_id = azurerm_key_vault.tenant.id
    subresource_names              = ["vault"]
    is_manual_connection           = false
  }
  
  tags = var.tags
}

# Monitoring and diagnostics
resource "azurerm_monitor_diagnostic_setting" "keyvault" {
  name               = "${local.kv_name}-diag"
  target_resource_id = azurerm_key_vault.tenant.id
  
  log_analytics_workspace_id = var.log_analytics_workspace_id
  
  enabled_log {
    category = "AuditEvent"
  }
  
  enabled_log {
    category = "AzurePolicyEvaluationDetails"
  }
  
  metric {
    category = "AllMetrics"
    enabled  = true
  }
}

# Backup configuration for enterprise tier
resource "azurerm_backup_protected_vault" "tenant" {
  count = var.tier == "enterprise" ? 1 : 0
  
  resource_group_name = var.resource_group_name
  recovery_vault_name = var.recovery_vault_name
  source_vault_id     = azurerm_key_vault.tenant.id
  backup_policy_id    = var.backup_policy_id
}

output "key_vault_id" {
  value       = azurerm_key_vault.tenant.id
  description = "Key Vault resource ID"
}

output "key_vault_uri" {
  value       = azurerm_key_vault.tenant.vault_uri
  description = "Key Vault URI"
}

output "key_vault_name" {
  value       = azurerm_key_vault.tenant.name
  description = "Key Vault name"
}