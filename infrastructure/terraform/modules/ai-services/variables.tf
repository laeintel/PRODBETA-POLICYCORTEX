# AI Services Module Variables

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
  description = "Name of the application resource group"
  type        = string
}

variable "network_resource_group_name" {
  description = "Name of the network resource group"
  type        = string
}

variable "location" {
  description = "Azure region location"
  type        = string
}

variable "vnet_name" {
  description = "Name of the virtual network"
  type        = string
}

variable "ai_services_subnet_name" {
  description = "Name of the AI services subnet"
  type        = string
  default     = "ai-services-subnet"
}

variable "private_endpoints_subnet_name" {
  description = "Name of the private endpoints subnet"
  type        = string
}

variable "managed_identity_id" {
  description = "ID of the user-assigned managed identity"
  type        = string
}

variable "key_vault_name" {
  description = "Name of the Key Vault"
  type        = string
}

variable "storage_account_name" {
  description = "Name of the Storage Account"
  type        = string
}

variable "application_insights_name" {
  description = "Name of the Application Insights"
  type        = string
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}

# Container Registry variables
variable "create_container_registry" {
  description = "Whether to create a new Container Registry for ML"
  type        = bool
  default     = true
}

variable "existing_container_registry_id" {
  description = "ID of existing Container Registry (if create_container_registry is false)"
  type        = string
  default     = null
}

variable "acr_sku" {
  description = "SKU for the Container Registry"
  type        = string
  default     = "Premium"
  validation {
    condition = contains(["Basic", "Standard", "Premium"], var.acr_sku)
    error_message = "ACR SKU must be Basic, Standard, or Premium."
  }
}

# Machine Learning Workspace variables
variable "deploy_ml_workspace" {
  description = "Whether to deploy ML workspace"
  type        = bool
  default     = true
}

variable "image_build_compute_name" {
  description = "Name of the compute instance for image builds"
  type        = string
  default     = null
}

variable "encryption_status" {
  description = "Encryption status for the ML workspace"
  type        = string
  default     = "Enabled"
  validation {
    condition = contains(["Enabled", "Disabled"], var.encryption_status)
    error_message = "Encryption status must be Enabled or Disabled."
  }
}

variable "encryption_key_vault_key_id" {
  description = "Key Vault key ID for encryption"
  type        = string
  default     = null
}

# Compute Instance variables
variable "deploy_ml_compute" {
  description = "Whether to deploy ML compute resources"
  type        = bool
  default     = false
}

variable "compute_instance_vm_size" {
  description = "VM size for the compute instance"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "compute_instance_ssh_public_key" {
  description = "SSH public key for the compute instance"
  type        = string
  default     = ""
}

# Compute Cluster variables
variable "training_cluster_vm_priority" {
  description = "VM priority for the training cluster"
  type        = string
  default     = "LowPriority"
  validation {
    condition = contains(["Dedicated", "LowPriority"], var.training_cluster_vm_priority)
    error_message = "VM priority must be Dedicated or LowPriority."
  }
}

variable "training_cluster_vm_size" {
  description = "VM size for the training cluster"
  type        = string
  default     = "Standard_DS3_v2"
}

variable "training_cluster_min_nodes" {
  description = "Minimum number of nodes in the training cluster"
  type        = number
  default     = 0
}

variable "training_cluster_max_nodes" {
  description = "Maximum number of nodes in the training cluster"
  type        = number
  default     = 4
}

# Cognitive Services variables
variable "cognitive_services_sku" {
  description = "SKU for Cognitive Services"
  type        = string
  default     = "S0"
}

# OpenAI variables
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

variable "openai_available_regions" {
  description = "List of regions where Azure OpenAI is available"
  type        = list(string)
  default = [
    "East US",
    "East US 2", 
    "West US 2",
    "West Europe",
    "North Central US",
    "South Central US",
    "UK South",
    "France Central",
    "Switzerland North",
    "Australia East",
    "Japan East",
    "Canada East"
  ]
}