# Container Apps Configuration - Managed by Terraform
# This file ensures Container Apps are properly managed without deletion

locals {
  # Container App names based on environment
  container_apps = {
    core = {
      name = "ca-cortex-core-${var.environment}"
      image = "${var.registry_name}.azurecr.io/policycortex-core:latest"
      cpu = 0.5
      memory = "1Gi"
      min_replicas = 0
      max_replicas = 1
      port = 8080
      ingress_enabled = true
      external_enabled = true
    }
    frontend = {
      name = "ca-cortex-frontend-${var.environment}"
      image = "${var.registry_name}.azurecr.io/policycortex-frontend:latest"
      cpu = 0.5
      memory = "1Gi"
      min_replicas = 0
      max_replicas = 1
      port = 3000
      ingress_enabled = true
      external_enabled = true
    }
    graphql = {
      name = "ca-cortex-graphql-${var.environment}"
      image = "${var.registry_name}.azurecr.io/policycortex-graphql:latest"
      cpu = 0.5
      memory = "1Gi"
      min_replicas = 0
      max_replicas = 1
      port = 4000
      ingress_enabled = true
      external_enabled = true
    }
  }
}

# Import existing Container App Environment if it exists
import {
  id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${azurerm_resource_group.main.name}/providers/Microsoft.App/managedEnvironments/cae-cortex-${var.environment}"
  to = azurerm_container_app_environment.main
}

# Import existing Container Apps if they exist
import {
  id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${azurerm_resource_group.main.name}/providers/Microsoft.App/containerApps/ca-cortex-core-${var.environment}"
  to = azurerm_container_app.apps["core"]
}

import {
  id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${azurerm_resource_group.main.name}/providers/Microsoft.App/containerApps/ca-cortex-frontend-${var.environment}"
  to = azurerm_container_app.apps["frontend"]
}

import {
  id = "/subscriptions/${data.azurerm_client_config.current.subscription_id}/resourceGroups/${azurerm_resource_group.main.name}/providers/Microsoft.App/containerApps/ca-cortex-graphql-${var.environment}"
  to = azurerm_container_app.apps["graphql"]
}

# Dynamic Container Apps using for_each
resource "azurerm_container_app" "apps" {
  for_each = local.container_apps

  name                         = each.value.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"

  template {
    container {
      name   = each.key
      image  = each.value.image
      cpu    = each.value.cpu
      memory = each.value.memory

      dynamic "env" {
        for_each = var.container_app_env_vars[each.key]
        content {
          name  = env.value.name
          value = env.value.value
        }
      }
    }

    min_replicas = each.value.min_replicas
    max_replicas = each.value.max_replicas
  }

  ingress {
    external_enabled = each.value.external_enabled
    target_port      = each.value.port
    transport        = "http"

    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }

  registry {
    server               = "${var.registry_name}.azurecr.io"
    identity             = azurerm_user_assigned_identity.aca_identity.id
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.aca_identity.id]
  }

  tags = local.common_tags

  lifecycle {
    ignore_changes = [
      template[0].container[0].image, # Ignore image changes as they're updated by CI/CD
      secret,                         # Ignore secrets managed by deployment
    ]
  }
}

# User-assigned identity for Container Apps
resource "azurerm_user_assigned_identity" "aca_identity" {
  name                = "id-aca-${var.environment}-${random_string.suffix.result}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.common_tags
}

# Grant AcrPull permission to the identity
resource "azurerm_role_assignment" "aca_acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.aca_identity.principal_id
}

# Variables for container app configuration
variable "registry_name" {
  description = "Container Registry name"
  type        = string
  default     = "crcortexdev5sug2t"
}

variable "container_app_env_vars" {
  description = "Environment variables for each container app"
  type = map(list(object({
    name  = string
    value = string
  })))
  default = {
    core = [
      {
        name  = "ENVIRONMENT"
        value = "dev"
      },
      {
        name  = "PORT"
        value = "8080"
      }
    ]
    frontend = [
      {
        name  = "NEXT_PUBLIC_API_URL"
        value = "https://ca-cortex-core-dev.azurecontainerapps.io"
      }
    ]
    graphql = [
      {
        name  = "APOLLO_PORT"
        value = "4000"
      }
    ]
  }
}