# Terraform variables for PolicyCortex infrastructure

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "Azure region for resources"
  type        = string
  default     = "East US"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "policycortex"
}

variable "name_prefix" {
  description = "Prefix for all resource names"
  type        = string
  default     = "policycortex"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "AeoliTech"
}

variable "allowed_ips" {
  description = "List of allowed IP addresses for storage account access"
  type        = list(string)
  default     = []
}

variable "create_terraform_access_policy" {
  description = "Whether to create Terraform access policy for Key Vault (set to false if already exists)"
  type        = bool
  default     = true
}

variable "deploy_container_apps" {
  description = "Whether to deploy Container Apps (set to false for initial infrastructure deployment)"
  type        = bool
  default     = false
}

# Kubernetes deployment variables
variable "deploy_kubernetes" {
  description = "Whether to deploy AKS cluster instead of Container Apps"
  type        = bool
  default     = false
}

variable "kubernetes_version" {
  description = "Kubernetes version for AKS cluster"
  type        = string
  default     = "1.28.3"
}

variable "kubernetes_node_count" {
  description = "Number of nodes in the default node pool"
  type        = number
  default     = 3
}

variable "kubernetes_node_vm_size" {
  description = "VM size for Kubernetes nodes"
  type        = string
  default     = "Standard_D4s_v3"
}

variable "kubernetes_enable_auto_scaling" {
  description = "Enable auto-scaling for Kubernetes node pool"
  type        = bool
  default     = true
}

variable "kubernetes_min_node_count" {
  description = "Minimum number of nodes when auto-scaling is enabled"
  type        = number
  default     = 2
}

variable "kubernetes_max_node_count" {
  description = "Maximum number of nodes when auto-scaling is enabled" 
  type        = number
  default     = 10
}

variable "kubernetes_enable_ai_node_pool" {
  description = "Enable dedicated node pool for AI workloads"
  type        = bool
  default     = false
}

variable "kubernetes_ai_node_vm_size" {
  description = "VM size for AI node pool"
  type        = string
  default     = "Standard_NC6s_v3"
}

variable "kubernetes_ai_node_count" {
  description = "Number of nodes in AI node pool"
  type        = number
  default     = 1
}

variable "enable_application_gateway" {
  description = "Enable Application Gateway for Kubernetes ingress"
  type        = bool
  default     = false
}

variable "jwt_secret_key" {
  description = "Secret key for JWT token signing"
  type        = string
  default     = "development-secret-key-change-in-production"
  sensitive   = true
}

# Data Services Variables
variable "deploy_sql_server" {
  description = "Whether to deploy SQL Server"
  type        = bool
  default     = true
}

variable "sql_admin_username" {
  description = "SQL Server administrator username"
  type        = string
  default     = "sqladmin"
}

variable "sql_azuread_admin_login" {
  description = "Azure AD admin login for SQL Server"
  type        = string
  default     = "admin@yourdomain.com"
}

variable "sql_azuread_admin_object_id" {
  description = "Azure AD admin object ID for SQL Server"
  type        = string
  default     = "00000000-0000-0000-0000-000000000000"
}

variable "sql_database_sku" {
  description = "SKU for the SQL database"
  type        = string
  default     = "GP_S_Gen5_2"
}

variable "sql_database_max_size_gb" {
  description = "Maximum size of the SQL database in GB"
  type        = number
  default     = 32
}

variable "cosmos_consistency_level" {
  description = "Cosmos DB consistency level"
  type        = string
  default     = "Session"
}

variable "cosmos_failover_location" {
  description = "Secondary region for Cosmos DB geo-replication"
  type        = string
  default     = "West US 2"
}

variable "cosmos_max_throughput" {
  description = "Maximum throughput for Cosmos DB database"
  type        = number
  default     = 4000
}

variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 2
}

variable "redis_sku_name" {
  description = "Redis cache SKU name"
  type        = string
  default     = "Standard"
}

# AI Services Variables
variable "deploy_ml_workspace" {
  description = "Whether to deploy ML workspace"
  type        = bool
  default     = true
}

variable "create_ml_container_registry" {
  description = "Whether to create a separate Container Registry for ML"
  type        = bool
  default     = false
}

variable "training_cluster_vm_size" {
  description = "VM size for the ML training cluster"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "training_cluster_max_nodes" {
  description = "Maximum number of nodes in the training cluster"
  type        = number
  default     = 4
}

variable "compute_instance_vm_size" {
  description = "VM size for the compute instance"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "cognitive_services_sku" {
  description = "SKU for Cognitive Services"
  type        = string
  default     = "S0"
}

variable "deploy_openai" {
  description = "Whether to deploy Azure OpenAI service"
  type        = bool
  default     = true
}

variable "openai_sku" {
  description = "SKU for Azure OpenAI"
  type        = string
  default     = "S0"
}

# Monitoring Variables
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

variable "budget_alert_emails" {
  description = "Email addresses for budget alerts"
  type        = list(string)
  default     = []
}

variable "monthly_budget_amount" {
  description = "Monthly budget amount in USD"
  type        = number
  default     = 1000
}