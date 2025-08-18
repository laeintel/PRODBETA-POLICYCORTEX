variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "policycortex"
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
}

variable "deployment_type" {
  description = "Deployment type: kubernetes (Neo4j) or cosmosdb (Gremlin)"
  type        = string
  default     = "kubernetes"
  validation {
    condition     = contains(["kubernetes", "cosmosdb"], var.deployment_type)
    error_message = "Deployment type must be either 'kubernetes' or 'cosmosdb'."
  }
}

# Neo4j on Kubernetes variables
variable "kubernetes_namespace" {
  description = "Kubernetes namespace for Neo4j"
  type        = string
  default     = "graph-db"
}

variable "neo4j_chart_version" {
  description = "Neo4j Helm chart version"
  type        = string
  default     = "5.14.0"
}

variable "neo4j_version" {
  description = "Neo4j version"
  type        = string
  default     = "5.14.0"
}

variable "cluster_enabled" {
  description = "Enable Neo4j clustering"
  type        = bool
  default     = true
}

variable "core_replicas" {
  description = "Number of Neo4j core replicas"
  type        = number
  default     = 3
}

variable "replica_replicas" {
  description = "Number of Neo4j read replicas"
  type        = number
  default     = 2
}

variable "memory_heap" {
  description = "Neo4j heap memory"
  type        = string
  default     = "4Gi"
}

variable "memory_pagecache" {
  description = "Neo4j page cache memory"
  type        = string
  default     = "4Gi"
}

variable "cpu_request" {
  description = "CPU request for Neo4j"
  type        = string
  default     = "2"
}

variable "cpu_limit" {
  description = "CPU limit for Neo4j"
  type        = string
  default     = "4"
}

variable "storage_size" {
  description = "Storage size for Neo4j"
  type        = string
  default     = "100Gi"
}

variable "storage_class" {
  description = "Storage class for Neo4j"
  type        = string
  default     = "managed-premium"
}

variable "auth_enabled" {
  description = "Enable authentication"
  type        = bool
  default     = true
}

variable "ssl_enabled" {
  description = "Enable SSL"
  type        = bool
  default     = true
}

variable "neo4j_password" {
  description = "Neo4j admin password"
  type        = string
  sensitive   = true
}

# Cosmos DB variables
variable "consistency_level" {
  description = "Cosmos DB consistency level"
  type        = string
  default     = "Session"
}

variable "enable_multi_region" {
  description = "Enable multi-region replication"
  type        = bool
  default     = false
}

variable "secondary_locations" {
  description = "Secondary locations for replication"
  type        = list(string)
  default     = []
}

variable "enable_automatic_failover" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "max_throughput" {
  description = "Maximum total throughput"
  type        = number
  default     = 50000
}

variable "database_max_throughput" {
  description = "Database max throughput"
  type        = number
  default     = 10000
}

variable "graph_max_throughput" {
  description = "Graph max throughput"
  type        = number
  default     = 5000
}

variable "enable_private_endpoint" {
  description = "Enable private endpoint"
  type        = bool
  default     = true
}

variable "subnet_id" {
  description = "Subnet ID for private endpoint"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}