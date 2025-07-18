# Core Configuration Variables
variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "policycortex"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*[a-z0-9]$", var.project_name))
    error_message = "Project name must be lowercase, start with letter, and contain only letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "The deployment environment (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "location" {
  description = "The Azure region for resource deployment"
  type        = string
  default     = "East US"
}

variable "location_short" {
  description = "Short abbreviation for the Azure region"
  type        = string
  default     = "eus"
}

# Resource Naming Variables
variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
  default     = ""
}

variable "naming_suffix" {
  description = "Suffix to append to resource names for uniqueness"
  type        = string
  default     = ""
}

# Tagging Variables
variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "PolicyCortex"
    Owner       = "Platform Team"
    CostCenter  = "Engineering"
    Backup      = "Required"
    Monitoring  = "Required"
  }
}

variable "additional_tags" {
  description = "Additional tags specific to the environment"
  type        = map(string)
  default     = {}
}

# Network Configuration Variables
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
  default = {
    aks = {
      address_prefixes  = ["10.0.1.0/24"]
      service_endpoints = ["Microsoft.ContainerRegistry", "Microsoft.KeyVault", "Microsoft.Storage"]
    }
    aks_system = {
      address_prefixes  = ["10.0.2.0/24"]
      service_endpoints = ["Microsoft.ContainerRegistry", "Microsoft.KeyVault"]
    }
    data = {
      address_prefixes  = ["10.0.10.0/24"]
      service_endpoints = ["Microsoft.Sql", "Microsoft.Storage", "Microsoft.KeyVault"]
    }
    ai = {
      address_prefixes  = ["10.0.20.0/24"]
      service_endpoints = ["Microsoft.CognitiveServices", "Microsoft.MachineLearningServices"]
    }
    gateway = {
      address_prefixes  = ["10.0.100.0/24"]
      service_endpoints = []
    }
    private_endpoints = {
      address_prefixes  = ["10.0.200.0/24"]
      service_endpoints = []
    }
  }
}

# AKS Configuration Variables
variable "aks_config" {
  description = "Configuration for Azure Kubernetes Service"
  type = object({
    kubernetes_version = string
    node_pools = map(object({
      vm_size             = string
      node_count          = number
      min_count           = number
      max_count           = number
      availability_zones  = list(string)
      node_taints         = list(string)
      node_labels         = map(string)
    }))
    enable_auto_scaling = bool
    enable_rbac        = bool
    network_plugin     = string
    network_policy     = string
  })
  default = {
    kubernetes_version = "1.28"
    node_pools = {
      system = {
        vm_size            = "Standard_D4s_v3"
        node_count         = 3
        min_count          = 3
        max_count          = 10
        availability_zones = ["1", "2", "3"]
        node_taints        = ["CriticalAddonsOnly=true:NoSchedule"]
        node_labels        = { "node-type" = "system" }
      }
      workload = {
        vm_size            = "Standard_D8s_v3"
        node_count         = 3
        min_count          = 3
        max_count          = 20
        availability_zones = ["1", "2", "3"]
        node_taints        = []
        node_labels        = { "node-type" = "workload" }
      }
      ai = {
        vm_size            = "Standard_NC6s_v3"
        node_count         = 1
        min_count          = 0
        max_count          = 5
        availability_zones = ["1", "2", "3"]
        node_taints        = ["ai-workload=true:NoSchedule"]
        node_labels        = { "node-type" = "ai", "accelerator" = "nvidia-tesla-v100" }
      }
    }
    enable_auto_scaling = true
    enable_rbac        = true
    network_plugin     = "azure"
    network_policy     = "calico"
  }
}

# Data Services Configuration
variable "sql_config" {
  description = "Configuration for Azure SQL Database"
  type = object({
    sku_name                     = string
    max_size_gb                  = number
    zone_redundant              = bool
    backup_retention_days       = number
    geo_redundant_backup_enabled = bool
    elastic_pool = object({
      max_size_gb = number
      sku = object({
        name     = string
        tier     = string
        capacity = number
      })
    })
  })
  default = {
    sku_name                     = "S2"
    max_size_gb                  = 250
    zone_redundant              = false
    backup_retention_days       = 35
    geo_redundant_backup_enabled = true
    elastic_pool = {
      max_size_gb = 500
      sku = {
        name     = "StandardPool"
        tier     = "Standard"
        capacity = 100
      }
    }
  }
}

variable "cosmos_config" {
  description = "Configuration for Azure Cosmos DB"
  type = object({
    consistency_policy = object({
      consistency_level       = string
      max_interval_in_seconds = number
      max_staleness_prefix    = number
    })
    geo_locations = list(object({
      location          = string
      failover_priority = number
      zone_redundant    = bool
    }))
    enable_automatic_failover = bool
    enable_multiple_write_locations = bool
  })
  default = {
    consistency_policy = {
      consistency_level       = "Session"
      max_interval_in_seconds = 5
      max_staleness_prefix    = 100
    }
    geo_locations = [
      {
        location          = "East US"
        failover_priority = 0
        zone_redundant    = true
      }
    ]
    enable_automatic_failover = true
    enable_multiple_write_locations = false
  }
}

# AI/ML Configuration
variable "ml_config" {
  description = "Configuration for Azure Machine Learning"
  type = object({
    compute_instances = map(object({
      vm_size = string
      min_node_count = number
      max_node_count = number
    }))
    enable_public_ip = bool
    enable_node_public_ip = bool
  })
  default = {
    compute_instances = {
      training = {
        vm_size = "Standard_DS3_v2"
        min_node_count = 0
        max_node_count = 10
      }
      inference = {
        vm_size = "Standard_F4s_v2"
        min_node_count = 1
        max_node_count = 5
      }
    }
    enable_public_ip = false
    enable_node_public_ip = false
  }
}

# Security Configuration
variable "security_config" {
  description = "Security configuration settings"
  type = object({
    enable_private_endpoints = bool
    enable_network_watcher   = bool
    allowed_ip_ranges       = list(string)
    enable_ddos_protection  = bool
  })
  default = {
    enable_private_endpoints = true
    enable_network_watcher   = true
    allowed_ip_ranges       = []
    enable_ddos_protection  = false
  }
}

# Monitoring Configuration
variable "monitoring_config" {
  description = "Monitoring and observability configuration"
  type = object({
    log_retention_days = number
    enable_application_insights = bool
    enable_container_insights = bool
    alerting = object({
      email_receivers = list(string)
      sms_receivers = list(object({
        name = string
        country_code = string
        phone_number = string
      }))
    })
  })
  default = {
    log_retention_days = 90
    enable_application_insights = true
    enable_container_insights = true
    alerting = {
      email_receivers = []
      sms_receivers = []
    }
  }
}

# Cost Optimization Configuration
variable "cost_config" {
  description = "Cost optimization configuration"
  type = object({
    enable_spot_instances = bool
    auto_shutdown_enabled = bool
    budget_amount = number
    budget_alerts = list(object({
      threshold = number
      contact_emails = list(string)
    }))
  })
  default = {
    enable_spot_instances = false
    auto_shutdown_enabled = true
    budget_amount = 1000
    budget_alerts = [
      {
        threshold = 80
        contact_emails = []
      },
      {
        threshold = 100
        contact_emails = []
      }
    ]
  }
} 