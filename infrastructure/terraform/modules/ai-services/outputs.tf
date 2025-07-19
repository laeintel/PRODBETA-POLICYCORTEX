# AI Services Module Outputs

# Machine Learning Workspace outputs
output "ml_workspace_id" {
  description = "ID of the Machine Learning workspace"
  value       = azurerm_machine_learning_workspace.main.id
}

output "ml_workspace_name" {
  description = "Name of the Machine Learning workspace"
  value       = azurerm_machine_learning_workspace.main.name
}

output "ml_workspace_discovery_url" {
  description = "Discovery URL of the Machine Learning workspace"
  value       = azurerm_machine_learning_workspace.main.discovery_url
}

# Container Registry outputs
output "ml_container_registry_id" {
  description = "ID of the ML Container Registry"
  value       = var.create_container_registry ? azurerm_container_registry.ml[0].id : var.existing_container_registry_id
}

output "ml_container_registry_name" {
  description = "Name of the ML Container Registry"
  value       = var.create_container_registry ? azurerm_container_registry.ml[0].name : null
}

output "ml_container_registry_login_server" {
  description = "Login server of the ML Container Registry"
  value       = var.create_container_registry ? azurerm_container_registry.ml[0].login_server : null
}

# Compute Instance outputs
output "compute_instance_id" {
  description = "ID of the compute instance"
  value       = var.environment == "dev" ? azurerm_machine_learning_compute_instance.dev[0].id : null
}

output "compute_instance_name" {
  description = "Name of the compute instance"
  value       = var.environment == "dev" ? azurerm_machine_learning_compute_instance.dev[0].name : null
}

# Compute Cluster outputs
output "training_cluster_id" {
  description = "ID of the training cluster"
  value       = var.deploy_ml_compute ? azurerm_machine_learning_compute_cluster.training[0].id : null
}

output "training_cluster_name" {
  description = "Name of the training cluster"
  value       = var.deploy_ml_compute ? azurerm_machine_learning_compute_cluster.training[0].name : null
}

# Cognitive Services outputs
output "cognitive_services_id" {
  description = "ID of the Cognitive Services account"
  value       = azurerm_cognitive_account.main.id
}

output "cognitive_services_name" {
  description = "Name of the Cognitive Services account"
  value       = azurerm_cognitive_account.main.name
}

output "cognitive_services_endpoint" {
  description = "Endpoint of the Cognitive Services account"
  value       = azurerm_cognitive_account.main.endpoint
}

# OpenAI outputs
output "openai_id" {
  description = "ID of the Azure OpenAI account"
  value       = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_cognitive_account.openai[0].id : null
}

output "openai_name" {
  description = "Name of the Azure OpenAI account"
  value       = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_cognitive_account.openai[0].name : null
}

output "openai_endpoint" {
  description = "Endpoint of the Azure OpenAI account"
  value       = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_cognitive_account.openai[0].endpoint : null
}

# Event Grid outputs
output "eventgrid_topic_id" {
  description = "ID of the Event Grid topic"
  value       = azurerm_eventgrid_topic.ml_operations.id
}

output "eventgrid_topic_name" {
  description = "Name of the Event Grid topic"
  value       = azurerm_eventgrid_topic.ml_operations.name
}

output "eventgrid_topic_endpoint" {
  description = "Endpoint of the Event Grid topic"
  value       = azurerm_eventgrid_topic.ml_operations.endpoint
}

# Private DNS zones
output "private_dns_zones" {
  description = "Private DNS zones created for AI services"
  value = {
    ml        = azurerm_private_dns_zone.ml.name
    cognitive = azurerm_private_dns_zone.cognitive.name
    openai    = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_private_dns_zone.openai[0].name : null
  }
}

# Private endpoints
output "private_endpoints" {
  description = "Private endpoints created for AI services"
  value = {
    ml_workspace = azurerm_private_endpoint.ml_workspace.id
    cognitive    = azurerm_private_endpoint.cognitive.id
    openai       = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_private_endpoint.openai[0].id : null
    eventgrid    = azurerm_private_endpoint.eventgrid.id
  }
}

# Key Vault secrets
output "key_vault_secrets" {
  description = "Names of Key Vault secrets containing AI service credentials"
  value = {
    cognitive_services_key      = azurerm_key_vault_secret.cognitive_services_key.name
    cognitive_services_endpoint = azurerm_key_vault_secret.cognitive_services_endpoint.name
    openai_key                 = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_key_vault_secret.openai_key[0].name : null
    openai_endpoint            = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? azurerm_key_vault_secret.openai_endpoint[0].name : null
  }
}