# Monitoring Module Outputs

# Action Groups
output "critical_action_group_id" {
  description = "ID of the critical alerts action group"
  value       = azurerm_monitor_action_group.critical.id
}

output "warning_action_group_id" {
  description = "ID of the warning alerts action group"
  value       = azurerm_monitor_action_group.warning.id
}

# Metric Alerts
output "container_apps_cpu_alert_ids" {
  description = "IDs of Container Apps CPU metric alerts"
  value       = azurerm_monitor_metric_alert.container_apps_cpu[*].id
}

output "container_apps_memory_alert_ids" {
  description = "IDs of Container Apps memory metric alerts"
  value       = azurerm_monitor_metric_alert.container_apps_memory[*].id
}

output "container_apps_restart_alert_ids" {
  description = "IDs of Container Apps restart metric alerts"
  value       = azurerm_monitor_metric_alert.container_apps_restarts[*].id
}

# Log Analytics Alerts
output "application_errors_alert_id" {
  description = "ID of the application errors alert"
  value       = azurerm_monitor_scheduled_query_rules_alert_v2.application_errors.id
}

output "auth_failures_alert_id" {
  description = "ID of the authentication failures alert"
  value       = azurerm_monitor_scheduled_query_rules_alert_v2.auth_failures.id
}

# Database Alerts
output "cosmos_ru_alert_id" {
  description = "ID of the Cosmos DB RU consumption alert"
  value       = var.cosmos_db_account_id != null ? azurerm_monitor_metric_alert.cosmos_ru_consumption[0].id : null
}

output "sql_dtu_alert_id" {
  description = "ID of the SQL Database DTU alert"
  value       = var.sql_database_id != null ? azurerm_monitor_metric_alert.sql_dtu_consumption[0].id : null
}

# Storage Alerts
output "storage_transactions_alert_id" {
  description = "ID of the Storage Account transactions alert"
  value       = var.storage_account_id != null ? azurerm_monitor_metric_alert.storage_transactions[0].id : null
}

# Dashboard and Analytics
output "dashboard_id" {
  description = "ID of the Azure Dashboard"
  value       = azurerm_portal_dashboard.main.id
}

output "dashboard_url" {
  description = "URL of the Azure Dashboard"
  value       = "https://portal.azure.com/#@/dashboard/arm${azurerm_portal_dashboard.main.id}"
}

output "workbook_id" {
  description = "ID of the Application Insights Workbook"
  value       = azurerm_application_insights_workbook.main.id
}

# Budget
output "budget_id" {
  description = "ID of the consumption budget"
  value       = azurerm_consumption_budget_resource_group.main.id
}

# Activity Log Alerts
output "resource_group_changes_alert_id" {
  description = "ID of the resource group changes alert"
  value       = azurerm_monitor_activity_log_alert.resource_group_changes.id
}