# Container Apps - Deployed after infrastructure and images are ready
# Enhanced for Enterprise-Grade PolicyCortex Platform

# Common environment variables for all container apps
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
      name  = "SERVICE_MESH_ENABLED"
      value = "true"
    },
    {
      name  = "METRICS_ENABLED"
      value = "true"
    },
    {
      name  = "TRACING_ENABLED"
      value = "true"
    },
    {
      name  = "CIRCUIT_BREAKER_ENABLED"
      value = "true"
    },
    {
      name  = "RATE_LIMITING_ENABLED"
      value = "true"
    }
  ]
  
  # Enhanced security configurations
  security_env_vars = [
    {
      name  = "JWT_SECRET"
      secret_name = "jwt-secret"
    },
    {
      name  = "ENCRYPTION_KEY"
      secret_name = "encryption-key"
    },
    {
      name  = "API_KEY_HEADER"
      value = "X-API-Key"
    },
    {
      name  = "CORS_MAX_AGE"
      value = "86400"
    }
  ]
  
  # AI/ML specific configurations
  ai_ml_env_vars = [
    {
      name  = "MODEL_CACHE_ENABLED"
      value = "true"
    },
    {
      name  = "MODEL_UPDATE_INTERVAL"
      value = "3600"
    },
    {
      name  = "PREDICTION_TIMEOUT"
      value = "30000"
    },
    {
      name  = "ENSEMBLE_VOTING_STRATEGY"
      value = "weighted"
    },
    {
      name  = "FUZZY_LOGIC_ENABLED"
      value = "true"
    },
    {
      name  = "TEMPORAL_ANALYSIS_ENABLED"
      value = "true"
    }
  ]
}

# Container Apps - Only deploy if flag is set
# API Gateway - Enhanced for Enterprise Security
resource "azurerm_container_app" "api_gateway" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-api-gateway-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"  # Enhanced for enterprise load
  
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
  
  # Enhanced secrets for enterprise security
  secret {
    name                = "client-id"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-id"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "managed-identity-client-id"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/managed-identity-client-id"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "jwt-secret"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/jwt-secret"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "encryption-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/encryption-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 20  # Enhanced for enterprise scale
    
    container {
      name   = "api-gateway"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-api_gateway:latest"
      cpu    = 1.0  # Enhanced CPU allocation
      memory = "2Gi"  # Enhanced memory allocation
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "api_gateway"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8000"
      }
      
      # Enhanced service URLs with load balancing
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
      
      env {
        name        = "AZURE_CLIENT_ID"
        secret_name = "client-id"
      }
      
      # Enhanced security environment variables
      dynamic "env" {
        for_each = local.security_env_vars
        content {
          name        = env.value.name
          secret_name = env.value.secret_name
        }
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
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
    
    # Enhanced security headers
    custom_domain {
      name = "api.${var.environment}.policycortex.com"
      certificate_id = azurerm_container_app_environment_certificate.api_gateway.id
    }
  }
  
  registry {
    server   = azurerm_container_registry.main.login_server
    identity = azurerm_user_assigned_identity.container_apps.id
  }
  
  tags = local.common_tags
}

# Azure Integration Service - Enhanced for Enterprise Scale
resource "azurerm_container_app" "azure_integration" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-azure-integration-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"  # Enhanced for enterprise scale
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  # Enhanced secrets for enterprise integration
  secret {
    name                = "client-id"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-id"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "managed-identity-client-id"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/managed-identity-client-id"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "client-secret"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-secret"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "cosmos-connection-string"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cosmos-connection-string"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "redis-connection-string"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/redis-connection-string"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "sql-connection-string"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/sql-connection-string"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "cognitive-services-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cognitive-services-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  secret {
    name                = "cognitive-services-endpoint"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/cognitive-services-endpoint"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 10  # Enhanced for enterprise scale
    
    container {
      name   = "azure-integration"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-azure_integration:latest"
      cpu    = 1.0  # Enhanced CPU allocation
      memory = "2Gi"  # Enhanced memory allocation
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8001
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8001
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "azure_integration"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8001"
      }
      
      # Enhanced Azure service connections
      env {
        name        = "AZURE_CLIENT_ID"
        secret_name = "client-id"
      }
      
      env {
        name        = "AZURE_CLIENT_SECRET"
        secret_name = "client-secret"
      }
      
      env {
        name        = "COSMOS_CONNECTION_STRING"
        secret_name = "cosmos-connection-string"
      }
      
      env {
        name        = "REDIS_CONNECTION_STRING"
        secret_name = "redis-connection-string"
      }
      
      env {
        name        = "SQL_CONNECTION_STRING"
        secret_name = "sql-connection-string"
      }
      
      env {
        name        = "COGNITIVE_SERVICES_KEY"
        secret_name = "cognitive-services-key"
      }
      
      env {
        name        = "COGNITIVE_SERVICES_ENDPOINT"
        secret_name = "cognitive-services-endpoint"
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

# AI Engine Service - Enhanced for Enterprise AI/ML Workloads
resource "azurerm_container_app" "ai_engine" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-ai-engine-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D16"  # Enhanced for AI/ML workloads
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  # Enhanced secrets for AI/ML services
  secret {
    name                = "model-storage-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/model-storage-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "mlflow-tracking-uri"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/mlflow-tracking-uri"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 16  # Enhanced for enterprise AI scale
    
    container {
      name   = "ai-engine"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-ai_engine:latest"
      cpu    = 2.0  # Enhanced CPU allocation for AI workloads
      memory = "4Gi"  # Enhanced memory allocation for AI workloads
      
      # Enhanced health checks for AI services
      liveness_probe {
        transport = "HTTP"
        port      = 8002
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8002
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "ai_engine"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8002"
      }
      
      # AI/ML specific environment variables
      dynamic "env" {
        for_each = local.ai_ml_env_vars
        content {
          name  = env.value.name
          value = env.value.value
        }
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

# Data Processing Service - Enhanced for Enterprise Data Workloads
resource "azurerm_container_app" "data_processing" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-data-processing-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"  # Enhanced for data processing
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  # Enhanced secrets for data processing
  secret {
    name                = "data-lake-connection-string"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/data-lake-connection-string"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "stream-analytics-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/stream-analytics-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 10  # Enhanced for enterprise data processing
    
    container {
      name   = "data-processing"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-data_processing:latest"
      cpu    = 1.5  # Enhanced CPU allocation for data processing
      memory = "3Gi"  # Enhanced memory allocation for data processing
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8003
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8003
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "data_processing"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8003"
      }
      
      # Data processing specific environment variables
      env {
        name  = "BATCH_SIZE"
        value = "1000"
      }
      
      env {
        name  = "STREAM_BUFFER_SIZE"
        value = "10000"
      }
      
      env {
        name  = "PROCESSING_TIMEOUT"
        value = "300"
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

# Conversation Service - Enhanced for Enterprise Conversational AI
resource "azurerm_container_app" "conversation" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-conversation-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"  # Enhanced for conversational AI
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  # Enhanced secrets for conversational AI
  secret {
    name                = "openai-api-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/openai-api-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "conversation-db-connection"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/conversation-db-connection"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 12  # Enhanced for enterprise conversational AI
    
    container {
      name   = "conversation"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-conversation:latest"
      cpu    = 1.0  # Enhanced CPU allocation for conversational AI
      memory = "2Gi"  # Enhanced memory allocation for conversational AI
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8004
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8004
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "conversation"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8004"
      }
      
      # Conversational AI specific environment variables
      env {
        name  = "MAX_CONVERSATION_LENGTH"
        value = "50"
      }
      
      env {
        name  = "RESPONSE_TIMEOUT"
        value = "30000"
      }
      
      env {
        name  = "CONTEXT_WINDOW_SIZE"
        value = "10"
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

# Notification Service - Enhanced for Enterprise Notifications
resource "azurerm_container_app" "notification" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-notification-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D4"  # Enhanced for notification processing
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  # Enhanced secrets for notifications
  secret {
    name                = "sendgrid-api-key"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/sendgrid-api-key"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }
  
  secret {
    name                = "teams-webhook-url"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/teams-webhook-url"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 8  # Enhanced for enterprise notification scale
    
    container {
      name   = "notification"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-notification:latest"
      cpu    = 0.5  # Optimized CPU allocation for notifications
      memory = "1Gi"  # Optimized memory allocation for notifications
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8005
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8005
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "notification"
      }
      
      env {
        name  = "SERVICE_PORT"
        value = "8005"
      }
      
      # Notification specific environment variables
      env {
        name  = "MAX_RETRY_ATTEMPTS"
        value = "3"
      }
      
      env {
        name  = "NOTIFICATION_TIMEOUT"
        value = "10000"
      }
      
      env {
        name  = "BATCH_SIZE"
        value = "100"
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

# Frontend Container App - Enhanced for Enterprise UI
resource "azurerm_container_app" "frontend" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-frontend-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.app.name
  revision_mode                = "Single"
  workload_profile_name        = "Dedicated-D8"  # Enhanced for enterprise UI
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  secret {
    name                = "client-id"
    key_vault_secret_id = "${azurerm_key_vault.main.vault_uri}secrets/client-id"
    identity            = azurerm_user_assigned_identity.container_apps.id
  }

  template {
    min_replicas = 2  # Enhanced for high availability
    max_replicas = 20  # Enhanced for enterprise UI scale
    
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-frontend:latest"
      cpu    = 0.5  # Optimized CPU allocation for frontend
      memory = "1Gi"  # Optimized memory allocation for frontend
      
      # Enhanced health checks
      liveness_probe {
        transport = "HTTP"
        port      = 8080
        path      = "/health"
      }
      
      readiness_probe {
        transport = "HTTP"
        port      = 8080
        path      = "/ready"
      }
      
      # Service-specific configuration
      env {
        name  = "SERVICE_NAME"
        value = "frontend"
      }
      
      env {
        name  = "PORT"
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
      
      # Enhanced frontend environment variables
      env {
        name  = "VITE_ENVIRONMENT"
        value = var.environment
      }
      
      env {
        name  = "VITE_ENABLE_ANALYTICS"
        value = "true"
      }
      
      env {
        name  = "VITE_ENABLE_ERROR_TRACKING"
        value = "true"
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
    external_enabled = true
    target_port      = 8080
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
    
    # Enhanced security headers
    custom_domain {
      name = "app.${var.environment}.policycortex.com"
      certificate_id = azurerm_container_app_environment_certificate.frontend.id
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