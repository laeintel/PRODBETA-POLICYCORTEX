# Container Apps - Deployed after infrastructure and images are ready

# Container Apps - Only deploy if flag is set
# API Gateway
resource "azurerm_container_app" "api_gateway" {
  count = var.deploy_container_apps ? 1 : 0
  
  name                         = "ca-api-gateway-${var.environment}"
  container_app_environment_id = azurerm_container_app_environment.main.id
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 10
    
    container {
      name   = "api-gateway"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-api_gateway:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      env {
        name  = "PORT"
        value = "8000"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 5
    
    container {
      name   = "azure-integration"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-azure_integration:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      env {
        name  = "PORT"
        value = "8001"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 8
    
    container {
      name   = "ai-engine"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-ai_engine:latest"
      cpu    = 1.0
      memory = "2Gi"
      
      env {
        name  = "PORT"
        value = "8002"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 5
    
    container {
      name   = "data-processing"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-data_processing:latest"
      cpu    = 0.75
      memory = "1.5Gi"
      
      env {
        name  = "PORT"
        value = "8003"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 6
    
    container {
      name   = "conversation"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-conversation:latest"
      cpu    = 0.5
      memory = "1Gi"
      
      env {
        name  = "PORT"
        value = "8004"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 4
    
    container {
      name   = "notification"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-notification:latest"
      cpu    = 0.25
      memory = "0.5Gi"
      
      env {
        name  = "PORT"
        value = "8005"
      }
      
      env {
        name  = "AZURE_CLIENT_ID"
        value = azurerm_user_assigned_identity.container_apps.client_id
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
  resource_group_name          = azurerm_resource_group.main.name
  revision_mode                = "Single"
  
  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.container_apps.id]
  }
  
  template {
    min_replicas = 1
    max_replicas = 10
    
    container {
      name   = "frontend"
      image  = "${azurerm_container_registry.main.login_server}/policycortex-frontend:latest"
      cpu    = 0.25
      memory = "0.5Gi"
      
      env {
        name  = "PORT"
        value = "80"
      }
      
      env {
        name  = "VITE_API_BASE_URL"
        value = "https://${azurerm_container_app.api_gateway[0].latest_revision_fqdn}"
      }
    }
  }
  
  ingress {
    external_enabled = true
    target_port      = 80
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