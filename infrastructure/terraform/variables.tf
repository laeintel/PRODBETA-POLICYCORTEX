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