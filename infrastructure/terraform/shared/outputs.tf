# Resource Group Outputs
output "resource_group_name" {
  description = "Name of the resource group"
  value       = var.resource_group_name
}

output "resource_group_location" {
  description = "Location of the resource group"
  value       = var.location
}

# Network Outputs
output "vnet_id" {
  description = "ID of the virtual network"
  value       = ""
}

output "vnet_name" {
  description = "Name of the virtual network"
  value       = ""
}

output "subnet_ids" {
  description = "Map of subnet names to subnet IDs"
  value       = {}
}

# AKS Outputs
output "aks_cluster_name" {
  description = "Name of the AKS cluster"
  value       = ""
}

output "aks_cluster_id" {
  description = "ID of the AKS cluster"
  value       = ""
}

output "aks_kube_config" {
  description = "Kubernetes configuration for the AKS cluster"
  value       = ""
  sensitive   = true
}

# Data Services Outputs
output "sql_server_name" {
  description = "Name of the SQL server"
  value       = ""
}

output "sql_server_fqdn" {
  description = "Fully qualified domain name of the SQL server"
  value       = ""
}

output "cosmos_db_account_name" {
  description = "Name of the Cosmos DB account"
  value       = ""
}

output "cosmos_db_endpoint" {
  description = "Endpoint URL of the Cosmos DB account"
  value       = ""
}

output "storage_account_name" {
  description = "Name of the storage account"
  value       = ""
}

output "storage_account_primary_access_key" {
  description = "Primary access key for the storage account"
  value       = ""
  sensitive   = true
}

# AI/ML Outputs
output "ml_workspace_name" {
  description = "Name of the Machine Learning workspace"
  value       = ""
}

output "ml_workspace_id" {
  description = "ID of the Machine Learning workspace"
  value       = ""
}

output "container_registry_name" {
  description = "Name of the container registry"
  value       = ""
}

output "container_registry_login_server" {
  description = "Login server URL for the container registry"
  value       = ""
}

# Security Outputs
output "key_vault_name" {
  description = "Name of the Key Vault"
  value       = ""
}

output "key_vault_uri" {
  description = "URI of the Key Vault"
  value       = ""
}

# Monitoring Outputs
output "log_analytics_workspace_id" {
  description = "ID of the Log Analytics workspace"
  value       = ""
}

output "application_insights_instrumentation_key" {
  description = "Instrumentation key for Application Insights"
  value       = ""
  sensitive   = true
}

output "application_insights_connection_string" {
  description = "Connection string for Application Insights"
  value       = ""
  sensitive   = true
}

# Common Information
output "common_tags" {
  description = "Common tags applied to resources"
  value       = merge(var.common_tags, var.additional_tags, { Environment = var.environment })
}

output "naming_convention" {
  description = "Naming convention used for resources"
  value = {
    prefix    = var.project_name
    suffix    = var.naming_suffix
    separator = "-"
    environment = var.environment
  }
} 