# Container Apps - Deployed after infrastructure and images are ready

# Key Vault secrets configuration
locals {
  keyvault_secrets = [
    {
      name      = "client-id"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-id"
    },
    {
      name      = "client-secret"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-secret"
    },
    {
      name      = "tenant-id"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/tenant-id"
    },
    {
      name      = "subscription-id"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/subscription-id"
    },
    {
      name      = "resource-group"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/resource-group"
    },
    {
      name      = "key-vault-name"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/key-vault-name"
    },
    {
      name      = "managed-identity-client-id"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/managed-identity-client-id"
    },
    {
      name      = "sql-server"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/sql-server"
    },
    {
      name      = "sql-username"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/sql-username"
    },
    {
      name      = "sql-password"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/sql-password"
    },
    {
      name      = "cosmos-endpoint"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cosmos-endpoint"
    },
    {
      name      = "cosmos-connection-string"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cosmos-connection-string"
    },
    {
      name      = "cosmos-key"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cosmos-key"
    },
    {
      name      = "redis-connection-string"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/redis-connection-string"
    },
    {
      name      = "storage-account-name"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/storage-account-name"
    },
    {
      name      = "application-insights-connection-string"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/application-insights-connection-string"
    },
    {
      name      = "cognitive-services-endpoint"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cognitive-services-endpoint"
    },
    {
      name      = "cognitive-services-key"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cognitive-services-key"
    },
    {
      name      = "ml-workspace-name"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/ml-workspace-name"
    },
    {
      name      = "jwt-secret-key"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/jwt-secret-key"
    },
    {
      name      = "service-bus-namespace"
      key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/service-bus-namespace"
    }
  ]
}

# Common environment variables for all container apps using Key Vault secrets
locals {
  common_env_vars = [
    {
      name  = "ENVIRONMENT"
      value = var.environment
    },
    {
      name  = "DEBUG"
      value = var.environment == "dev" ? "true" : "false"
    },
    {
      name  = "LOG_LEVEL"
      value = var.environment == "dev" ? "debug" : "info"
    },
    {
      name  = "PORT"
      value = "80"
    }
  ]

  # Secret-based environment variables
  common_secret_env_vars = [
    {
      name      = "AZURE_CLIENT_ID"
      secret_name = "client-id"
    },
    {
      name      = "CLIENT_ID"
      secret_name = "client-id"
    },
    {
      name      = "CLIENT_SECRET"
      secret_name = "client-secret"
    },
    {
      name      = "TENANT_ID"
      secret_name = "tenant-id"
    },
    {
      name      = "SUBSCRIPTION_ID"
      secret_name = "subscription-id"
    },
    {
      name      = "RESOURCE_GROUP"
      secret_name = "resource-group"
    },
    {
      name      = "KEY_VAULT_NAME"
      secret_name = "key-vault-name"
    },
    {
      name      = "SQL_SERVER"
      secret_name = "sql-server"
    },
    {
      name      = "SQL_USERNAME"
      secret_name = "sql-username"
    },
    {
      name      = "SQL_PASSWORD"
      secret_name = "sql-password"
    },
    {
      name      = "COSMOS_ENDPOINT"
      secret_name = "cosmos-endpoint"
    },
    {
      name      = "COSMOS_CONNECTION_STRING"
      secret_name = "cosmos-connection-string"
    },
    {
      name      = "COSMOS_KEY"
      secret_name = "cosmos-key"
    },
    {
      name      = "REDIS_CONNECTION_STRING"
      secret_name = "redis-connection-string"
    },
    {
      name      = "STORAGE_ACCOUNT_NAME"
      secret_name = "storage-account-name"
    },
    {
      name      = "APPLICATIONINSIGHTS_CONNECTION_STRING"
      secret_name = "application-insights-connection-string"
    },
    {
      name      = "COGNITIVE_SERVICES_ENDPOINT"
      secret_name = "cognitive-services-endpoint"
    },
    {
      name      = "COGNITIVE_SERVICES_KEY"
      secret_name = "cognitive-services-key"
    },
    {
      name      = "ML_WORKSPACE_NAME"
      secret_name = "ml-workspace-name"
    },
    {
      name      = "JWT_SECRET_KEY"
      secret_name = "jwt-secret-key"
    },
    {
      name      = "SERVICE_BUS_NAMESPACE"
      secret_name = "service-bus-namespace"
    }
  ]
}

# Container Apps - Only deploy if flag is set
# API Gateway
resource "azurerm_container_app" "api_gateway" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-api-gateway-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  depends_on = [
    azurerm_container_app_environment.main,
    azurerm_user_assigned_identity.container_apps,
    module.data_services,
    module.ai_services,
    azurerm_role_assignment.container_apps_keyvault,
    azurerm_role_assignment.container_apps_storage,
    azurerm_role_assignment.container_apps_cosmos,
    azurerm_role_assignment.container_apps_redis,
    azurerm_role_assignment.container_apps_cognitive,
    azurerm_role_assignment.container_apps_appinsights,
    azurerm_role_assignment.container_apps_rg_reader
  ]
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  dynamic "secret" {
    for_each = local.keyvault_secrets
    content {
      name                = secret.value.name
      key_vault_secret_id = secret.value.key_vault_secret_id
      identity            = azurerm_user_assigned_identity.container_apps.id
    }
  }

  template {
    min_replicas = 1
    max_replicas = 10
    
    container {
      name   = "api-gateway"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-api_gateway:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "api_gateway"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8000"
      }
      
      # Service URLs
      env {
        name  = "AZURE_INTEGRATION_URL"
        value = var.deploy_container_apps ? "https://ca-azure-integration-${var.environment}.${azurerm_container_app_environment.main.default_domain}" : ""
      }
      
      env {
        name  = "AI_ENGINE_URL"
        value = var.deploy_container_apps ? "https://ca-ai-engine-${var.environment}.${azurerm_container_app_environment.main.default_domain}" : ""
      }
      
      env {
        name  = "DATA_PROCESSING_URL"
        value = var.deploy_container_apps ? "https://ca-data-processing-${var.environment}.${azurerm_container_app_environment.main.default_domain}" : ""
      }
      
      env {
        name  = "CONVERSATION_URL"
        value = var.deploy_container_apps ? "https://ca-conversation-${var.environment}.${azurerm_container_app_environment.main.default_domain}" : ""
      }
      
      env {
        name  = "NOTIFICATION_URL"
        value = var.deploy_container_apps ? "https://ca-notification-${var.environment}.${azurerm_container_app_environment.main.default_domain}" : ""
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # Secret-based environment variables
      dynamic "env" {
        for_each = local.common_secret_env_vars
        content {
          name        = env.value.name
          secret_name = env.value.secret_name
        }
      }
    }
  }
  
  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Azure Integration Service
resource "azurerm_container_app" "azure_integration" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-azure-integration-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  dynamic "secret" {
    for_each = local.keyvault_secrets
    content {
      name                = secret.value.name
      key_vault_secret_id = secret.value.key_vault_secret_id
      identity            = azurerm_user_assigned_identity.container_apps.id
    }
  }

  template {
    min_replicas = 1
    max_replicas = 5
    
    container {
      name   = "azure-integration"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-azure_integration:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "azure_integration"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8001"
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # Secret-based environment variables
      dynamic "env" {
        for_each = local.common_secret_env_vars
        content {
          name        = env.value.name
          secret_name = env.value.secret_name
        }
      }
    }
  }
  
  ingress {
    external_enabled = false
    target_port      = 8001
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# AI Engine Service
resource "azurerm_container_app" "ai_engine" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-ai-engine-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 8
    
    # Note: workload_profile_name not yet supported in azurerm provider v3.80
    # Will use environment-level workload profiles for now
    
    container {
      name   = "ai-engine"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-ai_engine:latest"
      cpu    = 1.0
      memory = "2Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "ai_engine"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8002"
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }
    }
  }
  
  ingress {
    external_enabled = false
    target_port      = 8002
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Data Processing Service
resource "azurerm_container_app" "data_processing" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-data-processing-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 5
    
    # Note: workload_profile_name not yet supported in azurerm provider v3.80
    # Will use environment-level workload profiles for now
    
    container {
      name   = "data-processing"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-data_processing:latest"
      cpu    = 0.75
      memory = "1.5Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "data_processing"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8003"
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }
    }
  }
  
  ingress {
    external_enabled = false
    target_port      = 8003
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Conversation Service
resource "azurerm_container_app" "conversation" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-conversation-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 6
    
    # Use Consumption profile for Conversation (interactive, variable load)
    
    container {
      name   = "conversation"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-conversation:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "conversation"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8004"
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }
    }
  }
  
  ingress {
    external_enabled = false
    target_port      = 8004
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Notification Service
resource "azurerm_container_app" "notification" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-notification-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 4
    
    # Use Consumption profile for Notification (lightweight, event-driven)
    
    container {
      name   = "notification"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-notification:latest"
      cpu    = 0.25
      memory = "0.5Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "notification"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8005"
      }
      
      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }
    }
  }
  
  ingress {
    external_enabled = false
    target_port      = 8005
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Frontend Container App
resource "azurerm_container_app" "frontend" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-frontend-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  dynamic "secret" {
    for_each = local.keyvault_secrets
    content {
      name                = secret.value.name
      key_vault_secret_id = secret.value.key_vault_secret_id
      identity            = azurerm_user_assigned_identity.container_apps.id
    }
  }

  template {
    min_replicas = 1
    max_replicas = 10
    
    # Use Consumption profile for Frontend (static content, variable load)
    
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-frontend:latest"
      cpu    = 0.25
      memory = "0.5Gi"
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "frontend"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8080"
      }
      
      env {
        name  = "VITE_API_BASE_URL"
        value = "https://ca-api-gateway-${var.environment}.${azurerm_container_app_environment.main.default_domain}"
      }
      
      env {
        name        = "VITE_AZURE_CLIENT_ID"
        secret_name = "client-id"
      }
      
      env {
        name  = "VITE_AZURE_TENANT_ID"
        value = data.azurerm_client_config.current.tenant_id
      }

      # Common environment variables
      dynamic "env" {
        for_each = local.common_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
      }

      # Secret-based environment variables
      dynamic "env" {
        for_each = local.common_secret_env_vars
        content {
          name        = env.value.name
          secret_name = env.value.secret_name
        }
      }
    }
  }
  
  ingress {
    external_enabled = true
    target_port      = 8080
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
  
  depends_on = [
    azurerm_container_app.api_gateway
  ]
}

# Container Apps URLs (conditional outputs)
output "api_gateway_url" {
  value = var.deploy_container_apps && length(azurerm_container_app.api_gateway) > 0 ? "https://${azurerm_container_app.api_gateway[0].latest_revision_fqdn}" : "Not deployed yet"
}

output "frontend_url" {
  value = var.deploy_container_apps && length(azurerm_container_app.frontend) > 0 ? "https://${azurerm_container_app.frontend[0].latest_revision_fqdn}" : "Not deployed yet"
}

output "container_apps_fqdns" {
  value = var.deploy_container_apps ? {
    api_gateway      = length(azurerm_container_app.api_gateway) > 0 ? azurerm_container_app.api_gateway[0].latest_revision_fqdn : "Not deployed"
    azure_integration = length(azurerm_container_app.azure_integration) > 0 ? azurerm_container_app.azure_integration[0].latest_revision_fqdn : "Not deployed"
    ai_engine        = length(azurerm_container_app.ai_engine) > 0 ? azurerm_container_app.ai_engine[0].latest_revision_fqdn : "Not deployed"
    data_processing  = length(azurerm_container_app.data_processing) > 0 ? azurerm_container_app.data_processing[0].latest_revision_fqdn : "Not deployed"
    conversation     = length(azurerm_container_app.conversation) > 0 ? azurerm_container_app.conversation[0].latest_revision_fqdn : "Not deployed"
    notification     = length(azurerm_container_app.notification) > 0 ? azurerm_container_app.notification[0].latest_revision_fqdn : "Not deployed"
    frontend         = length(azurerm_container_app.frontend) > 0 ? azurerm_container_app.frontend[0].latest_revision_fqdn : "Not deployed"
  } : {
    api_gateway      = "Not deployed"
    azure_integration = "Not deployed"
    ai_engine        = "Not deployed"
    data_processing  = "Not deployed"
    conversation     = "Not deployed"
    notification     = "Not deployed"
    frontend         = "Not deployed"
  }
}