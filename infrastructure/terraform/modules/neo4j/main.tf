terraform {
  required_version = ">= 1.6.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

locals {
  name_prefix = "${var.environment}-${var.project_name}"
  common_tags = merge(
    var.tags,
    {
      Environment = var.environment
      ManagedBy   = "Terraform"
      Project     = var.project_name
      Service     = "neo4j-graph"
    }
  )
}

# Option 1: Neo4j on AKS using Helm
resource "helm_release" "neo4j" {
  count = var.deployment_type == "kubernetes" ? 1 : 0

  name       = "neo4j-${var.environment}"
  repository = "https://helm.neo4j.com/neo4j"
  chart      = "neo4j"
  version    = var.neo4j_chart_version
  namespace  = var.kubernetes_namespace

  values = [
    templatefile("${path.module}/values/neo4j-values.yaml", {
      neo4j_version     = var.neo4j_version
      cluster_enabled   = var.cluster_enabled
      core_replicas     = var.core_replicas
      replica_replicas  = var.replica_replicas
      memory_heap       = var.memory_heap
      memory_pagecache  = var.memory_pagecache
      cpu_request       = var.cpu_request
      cpu_limit         = var.cpu_limit
      storage_size      = var.storage_size
      storage_class     = var.storage_class
      auth_enabled      = var.auth_enabled
      ssl_enabled       = var.ssl_enabled
    })
  ]

  set_sensitive {
    name  = "auth.neo4j.password"
    value = var.neo4j_password
  }
}

# Option 2: Azure Cosmos DB with Gremlin API
resource "azurerm_cosmosdb_account" "gremlin" {
  count = var.deployment_type == "cosmosdb" ? 1 : 0

  name                = "${local.name_prefix}-cosmos-graph"
  location            = var.location
  resource_group_name = var.resource_group_name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  capabilities {
    name = "EnableGremlin"
  }

  consistency_policy {
    consistency_level       = var.consistency_level
    max_interval_in_seconds = 10
    max_staleness_prefix    = 200
  }

  geo_location {
    location          = var.location
    failover_priority = 0
  }

  dynamic "geo_location" {
    for_each = var.enable_multi_region ? var.secondary_locations : []
    content {
      location          = geo_location.value
      failover_priority = index(var.secondary_locations, geo_location.value) + 1
    }
  }

  enable_automatic_failover = var.enable_automatic_failover
  enable_multiple_write_locations = var.enable_multi_region

  backup {
    type                = "Continuous"
    storage_redundancy  = "Geo"
    tier                = "Continuous30Days"
  }

  analytical_storage {
    schema_type = "WellDefined"
  }

  capacity {
    total_throughput_limit = var.max_throughput
  }

  tags = local.common_tags
}

resource "azurerm_cosmosdb_gremlin_database" "main" {
  count = var.deployment_type == "cosmosdb" ? 1 : 0

  name                = "policycortex-graph"
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.gremlin[0].name

  autoscale_settings {
    max_throughput = var.database_max_throughput
  }
}

resource "azurerm_cosmosdb_gremlin_graph" "governance" {
  count = var.deployment_type == "cosmosdb" ? 1 : 0

  name                = "governance-graph"
  resource_group_name = var.resource_group_name
  account_name        = azurerm_cosmosdb_account.gremlin[0].name
  database_name       = azurerm_cosmosdb_gremlin_database.main[0].name

  partition_key_path = "/tenantId"
  partition_key_version = 2

  autoscale_settings {
    max_throughput = var.graph_max_throughput
  }

  index_policy {
    automatic      = true
    indexing_mode  = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/\"_etag\"/?"
    }
  }

  conflict_resolution_policy {
    mode                     = "LastWriterWins"
    conflict_resolution_path = "/_ts"
  }
}

# Private endpoint for secure access
resource "azurerm_private_endpoint" "graph" {
  count = var.deployment_type == "cosmosdb" && var.enable_private_endpoint ? 1 : 0

  name                = "${local.name_prefix}-graph-pe"
  location            = var.location
  resource_group_name = var.resource_group_name
  subnet_id           = var.subnet_id

  private_service_connection {
    name                           = "${local.name_prefix}-graph-psc"
    private_connection_resource_id = azurerm_cosmosdb_account.gremlin[0].id
    subresource_names              = ["Gremlin"]
    is_manual_connection           = false
  }

  tags = local.common_tags
}

# Outputs for application configuration
output "graph_endpoint" {
  value = var.deployment_type == "kubernetes" ? (
    helm_release.neo4j[0].status == "deployed" ? 
    "bolt://neo4j-${var.environment}.${var.kubernetes_namespace}.svc.cluster.local:7687" : ""
  ) : (
    var.deployment_type == "cosmosdb" ? 
    azurerm_cosmosdb_account.gremlin[0].endpoint : ""
  )
  sensitive = false
}

output "graph_connection_string" {
  value = var.deployment_type == "cosmosdb" ? 
    azurerm_cosmosdb_account.gremlin[0].primary_key : ""
  sensitive = true
}

output "graph_type" {
  value = var.deployment_type == "kubernetes" ? "neo4j" : "cosmosdb-gremlin"
}