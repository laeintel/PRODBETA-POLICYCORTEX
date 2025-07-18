# Required Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
}

variable "location" {
  description = "Azure region for resources"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
}

# Networking Configuration
variable "vnet_address_space" {
  description = "Address space for the virtual network"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "subnet_configurations" {
  description = "Configuration for subnets"
  type = map(object({
    address_prefixes = list(string)
    service_endpoints = list(string)
    delegation = optional(object({
      name = string
      service_delegation = object({
        name    = string
        actions = list(string)
      })
    }))
  }))
}

variable "custom_dns_servers" {
  description = "Custom DNS servers for the VNet"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_ddos_protection" {
  description = "Enable DDoS protection for the VNet"
  type        = bool
  default     = false
}

variable "enable_network_watcher" {
  description = "Enable Network Watcher for monitoring"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable NSG flow logs"
  type        = bool
  default     = true
}

variable "enable_firewall" {
  description = "Enable Azure Firewall for centralized security"
  type        = bool
  default     = false
}

variable "firewall_private_ip" {
  description = "Private IP address of the Azure Firewall"
  type        = string
  default     = ""
}

# Monitoring Integration
variable "log_analytics_workspace_id" {
  description = "Log Analytics workspace ID for flow logs"
  type        = string
  default     = ""
}

variable "log_analytics_workspace_resource_id" {
  description = "Log Analytics workspace resource ID for flow logs"
  type        = string
  default     = ""
}

# Naming and Tagging
variable "naming_suffix" {
  description = "Suffix for resource naming"
  type        = string
  default     = ""
}

variable "common_tags" {
  description = "Common tags to apply to resources"
  type        = map(string)
  default     = {}
} 