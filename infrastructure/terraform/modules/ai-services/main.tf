# AI Services Module for PolicyCortex
# Implements Azure Machine Learning, Cognitive Services, and OpenAI

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

# Data sources for existing resources
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

data "azurerm_virtual_network" "main" {
  name                = var.vnet_name
  resource_group_name = var.resource_group_name
}

data "azurerm_subnet" "ai_services" {
  name                 = var.ai_services_subnet_name
  virtual_network_name = var.vnet_name
  resource_group_name  = var.resource_group_name
}

data "azurerm_key_vault" "main" {
  name                = var.key_vault_name
  resource_group_name = var.resource_group_name
}

data "azurerm_storage_account" "main" {
  name                = var.storage_account_name
  resource_group_name = var.resource_group_name
}

data "azurerm_application_insights" "main" {
  name                = var.application_insights_name
  resource_group_name = var.resource_group_name
}

# Container Registry for ML models (if not already existing)
resource "azurerm_container_registry" "ml" {
  count               = var.create_container_registry ? 1 : 0
  name                = "${var.project_name}mlacr${var.environment}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  sku                 = var.acr_sku
  admin_enabled       = false
  
  # Network access
  public_network_access_enabled = false
  network_rule_bypass_option    = "AzureServices"

  # Identity for accessing Key Vault
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Private endpoint for Container Registry
resource "azurerm_private_endpoint" "acr" {
  count               = var.create_container_registry ? 1 : 0
  name                = "${var.project_name}-ml-acr-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.ai_services.id

  private_service_connection {
    name                           = "${var.project_name}-ml-acr-psc-${var.environment}"
    private_connection_resource_id = azurerm_container_registry.ml[0].id
    subresource_names             = ["registry"]
    is_manual_connection          = false
  }

  tags = var.tags
}

# Azure Machine Learning Workspace
resource "azurerm_machine_learning_workspace" "main" {
  name                          = "${var.project_name}-ml-${var.environment}"
  location                      = data.azurerm_resource_group.main.location
  resource_group_name           = data.azurerm_resource_group.main.name
  application_insights_id       = data.azurerm_application_insights.main.id
  key_vault_id                  = data.azurerm_key_vault.main.id
  storage_account_id            = data.azurerm_storage_account.main.id
  container_registry_id         = var.create_container_registry ? azurerm_container_registry.ml[0].id : var.existing_container_registry_id
  
  # Security and access
  public_network_access_enabled = false
  image_build_compute_name      = var.image_build_compute_name

  # Identity
  identity {
    type = "SystemAssigned"
  }

  # Encryption (optional)
  dynamic "encryption" {
    for_each = var.encryption_key_vault_key_id != null ? [1] : []
    content {
      key_id        = var.encryption_key_vault_key_id
      key_vault_id  = data.azurerm_key_vault.main.id
    }
  }

  tags = var.tags
}

# Private endpoint for ML Workspace
resource "azurerm_private_endpoint" "ml_workspace" {
  name                = "${var.project_name}-ml-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.ai_services.id

  private_service_connection {
    name                           = "${var.project_name}-ml-psc-${var.environment}"
    private_connection_resource_id = azurerm_machine_learning_workspace.main.id
    subresource_names             = ["amlworkspace"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "ml-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.ml.id]
  }

  tags = var.tags
}

# Private DNS zone for ML Workspace
resource "azurerm_private_dns_zone" "ml" {
  name                = "privatelink.api.azureml.ms"
  resource_group_name = data.azurerm_resource_group.main.name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "ml" {
  name                  = "ml-dns-vnet-link"
  resource_group_name   = data.azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.ml.name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# ML Compute Instance for development
resource "azurerm_machine_learning_compute_instance" "dev" {
  count                         = var.environment == "dev" ? 1 : 0
  name                          = "${var.project_name}-ci-${var.environment}"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  virtual_machine_size          = var.compute_instance_vm_size
  
  # Security
  authorization_type = "personal"
  ssh {
    public_key = var.compute_instance_ssh_public_key
  }

  # Subnet assignment
  subnet_resource_id = data.azurerm_subnet.ai_services.id

  tags = var.tags
}

# ML Compute Cluster for training
resource "azurerm_machine_learning_compute_cluster" "training" {
  name                          = "${var.project_name}-cluster-${var.environment}"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.main.id
  location                      = data.azurerm_resource_group.main.location
  vm_priority                   = var.training_cluster_vm_priority
  vm_size                       = var.training_cluster_vm_size
  
  # Scaling
  scale_settings {
    min_node_count                       = var.training_cluster_min_nodes
    max_node_count                       = var.training_cluster_max_nodes
    scale_down_nodes_after_idle_duration = "PT300S"  # 5 minutes
  }

  # Network
  subnet_resource_id = data.azurerm_subnet.ai_services.id

  # Identity
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Cognitive Services Multi-Service Account
resource "azurerm_cognitive_account" "main" {
  name                = "${var.project_name}-cognitive-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  kind                = "CognitiveServices"
  sku_name            = var.cognitive_services_sku

  # Security
  public_network_access_enabled = false
  custom_subdomain_name         = "${var.project_name}-cognitive-${var.environment}"

  # Network rules
  network_acls {
    default_action = "Deny"
    
    virtual_network_rules {
      subnet_id                            = data.azurerm_subnet.ai_services.id
      ignore_missing_vnet_service_endpoint = false
    }
  }

  # Identity
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Private endpoint for Cognitive Services
resource "azurerm_private_endpoint" "cognitive" {
  name                = "${var.project_name}-cognitive-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.ai_services.id

  private_service_connection {
    name                           = "${var.project_name}-cognitive-psc-${var.environment}"
    private_connection_resource_id = azurerm_cognitive_account.main.id
    subresource_names             = ["account"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "cognitive-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.cognitive.id]
  }

  tags = var.tags
}

# Private DNS zone for Cognitive Services
resource "azurerm_private_dns_zone" "cognitive" {
  name                = "privatelink.cognitiveservices.azure.com"
  resource_group_name = data.azurerm_resource_group.main.name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "cognitive" {
  name                  = "cognitive-dns-vnet-link"
  resource_group_name   = data.azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.cognitive.name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# Azure OpenAI Service (if available in region)
resource "azurerm_cognitive_account" "openai" {
  count               = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name                = "${var.project_name}-openai-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  kind                = "OpenAI"
  sku_name            = var.openai_sku

  # Security
  public_network_access_enabled = false
  custom_subdomain_name         = "${var.project_name}-openai-${var.environment}"

  # Network rules
  network_acls {
    default_action = "Deny"
    
    virtual_network_rules {
      subnet_id                            = data.azurerm_subnet.ai_services.id
      ignore_missing_vnet_service_endpoint = false
    }
  }

  # Identity
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Private endpoint for OpenAI
resource "azurerm_private_endpoint" "openai" {
  count               = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name                = "${var.project_name}-openai-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.ai_services.id

  private_service_connection {
    name                           = "${var.project_name}-openai-psc-${var.environment}"
    private_connection_resource_id = azurerm_cognitive_account.openai[0].id
    subresource_names             = ["account"]
    is_manual_connection          = false
  }

  private_dns_zone_group {
    name                 = "openai-dns-zone-group"
    private_dns_zone_ids = [azurerm_private_dns_zone.openai[0].id]
  }

  tags = var.tags
}

# Private DNS zone for OpenAI
resource "azurerm_private_dns_zone" "openai" {
  count               = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name                = "privatelink.openai.azure.com"
  resource_group_name = data.azurerm_resource_group.main.name

  tags = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "openai" {
  count                 = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name                  = "openai-dns-vnet-link"
  resource_group_name   = data.azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.openai[0].name
  virtual_network_id    = data.azurerm_virtual_network.main.id

  tags = var.tags
}

# Store AI service keys in Key Vault
resource "azurerm_key_vault_secret" "cognitive_services_key" {
  name         = "cognitive-services-key"
  value        = azurerm_cognitive_account.main.primary_access_key
  key_vault_id = data.azurerm_key_vault.main.id

  tags = var.tags
}

resource "azurerm_key_vault_secret" "cognitive_services_endpoint" {
  name         = "cognitive-services-endpoint"
  value        = azurerm_cognitive_account.main.endpoint
  key_vault_id = data.azurerm_key_vault.main.id

  tags = var.tags
}

resource "azurerm_key_vault_secret" "openai_key" {
  count        = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name         = "openai-key"
  value        = azurerm_cognitive_account.openai[0].primary_access_key
  key_vault_id = data.azurerm_key_vault.main.id

  tags = var.tags
}

resource "azurerm_key_vault_secret" "openai_endpoint" {
  count        = var.deploy_openai && contains(var.openai_available_regions, data.azurerm_resource_group.main.location) ? 1 : 0
  name         = "openai-endpoint"
  value        = azurerm_cognitive_account.openai[0].endpoint
  key_vault_id = data.azurerm_key_vault.main.id

  tags = var.tags
}

# Event Grid Topic for ML operations
resource "azurerm_eventgrid_topic" "ml_operations" {
  name                = "${var.project_name}-ml-events-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name

  # Security
  public_network_access_enabled = false
  local_auth_enabled            = false

  # Identity
  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# Private endpoint for Event Grid
resource "azurerm_private_endpoint" "eventgrid" {
  name                = "${var.project_name}-eventgrid-pe-${var.environment}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  subnet_id           = data.azurerm_subnet.ai_services.id

  private_service_connection {
    name                           = "${var.project_name}-eventgrid-psc-${var.environment}"
    private_connection_resource_id = azurerm_eventgrid_topic.ml_operations.id
    subresource_names             = ["topic"]
    is_manual_connection          = false
  }

  tags = var.tags
}