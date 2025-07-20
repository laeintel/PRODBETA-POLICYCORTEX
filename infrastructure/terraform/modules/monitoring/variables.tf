# Monitoring Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "policycortex"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

variable "log_analytics_workspace_name" {
  description = "Name of the Log Analytics workspace"
  type        = string
}

variable "application_insights_name" {
  description = "Name of the Application Insights"
  type        = string
}

variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Alert configuration
variable "critical_alert_emails" {
  description = "Email addresses for critical alerts"
  type        = list(string)
  default     = []
}

variable "warning_alert_emails" {
  description = "Email addresses for warning alerts"
  type        = list(string)
  default     = []
}

variable "critical_alert_sms" {
  description = "SMS numbers for critical alerts"
  type = list(object({
    country_code = string
    phone_number = string
  }))
  default = []
}

variable "webhook_urls" {
  description = "Webhook URLs for alert integration"
  type        = list(string)
  default     = []
}

variable "budget_alert_emails" {
  description = "Email addresses for budget alerts"
  type        = list(string)
  default     = []
}

# Container Apps monitoring
variable "container_app_names" {
  description = "Names of Container Apps to monitor"
  type        = list(string)
  default     = []
}

variable "container_app_resource_ids" {
  description = "Resource IDs of Container Apps to monitor"
  type        = list(string)
  default     = []
}

variable "container_app_environment_id" {
  description = "Resource ID of the Container App Environment"
  type        = string
  default     = null
}

# Threshold configuration
variable "cpu_threshold_percentage" {
  description = "CPU usage threshold percentage for alerts"
  type        = number
  default     = 80
}

variable "memory_threshold_percentage" {
  description = "Memory usage threshold percentage for alerts"
  type        = number
  default     = 85
}

variable "restart_threshold_count" {
  description = "Restart count threshold for alerts"
  type        = number
  default     = 5
}

variable "error_threshold_count" {
  description = "Error count threshold for alerts"
  type        = number
  default     = 10
}

variable "auth_failure_threshold" {
  description = "Authentication failure threshold for alerts"
  type        = number
  default     = 20
}

# Database monitoring
variable "cosmos_db_account_id" {
  description = "Resource ID of the Cosmos DB account"
  type        = string
  default     = null
}

variable "deploy_cosmos_monitoring" {
  description = "Whether to deploy Cosmos DB monitoring alerts"
  type        = bool
  default     = true
}

variable "cosmos_ru_threshold" {
  description = "Cosmos DB RU consumption threshold"
  type        = number
  default     = 10000
}

variable "sql_database_id" {
  description = "Resource ID of the SQL Database"
  type        = string
  default     = null
}

variable "deploy_sql_monitoring" {
  description = "Whether to deploy SQL Database monitoring alerts"
  type        = bool
  default     = true
}

variable "sql_dtu_threshold_percentage" {
  description = "SQL Database DTU threshold percentage"
  type        = number
  default     = 80
}

# Storage monitoring
variable "storage_account_id" {
  description = "Resource ID of the Storage Account"
  type        = string
  default     = null
}

variable "deploy_storage_monitoring" {
  description = "Whether to deploy Storage Account monitoring alerts"
  type        = bool
  default     = true
}

variable "storage_transaction_threshold" {
  description = "Storage Account transaction threshold"
  type        = number
  default     = 100000
}

# Budget configuration
variable "monthly_budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 1000
}