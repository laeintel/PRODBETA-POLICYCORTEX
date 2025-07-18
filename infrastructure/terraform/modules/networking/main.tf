# Generate random suffix for unique naming
resource "random_string" "network_suffix" {
  length  = 8
  special = false
  upper   = false
}

# Local values for consistent naming
locals {
  naming_prefix = "${var.project_name}-${var.environment}"
  naming_suffix = var.naming_suffix != "" ? var.naming_suffix : random_string.network_suffix.result
  
  common_tags = merge(var.common_tags, {
    Module      = "networking"
    Environment = var.environment
  })
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${local.naming_prefix}-vnet-${local.naming_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name
  address_space       = var.vnet_address_space
  
  # DNS servers for custom DNS resolution
  dns_servers = var.custom_dns_servers

  # DDoS protection configuration
  dynamic "ddos_protection_plan" {
    for_each = var.enable_ddos_protection ? [1] : []
    content {
      id     = azurerm_network_ddos_protection_plan.main[0].id
      enable = true
    }
  }

  tags = local.common_tags
}

# Network Security Groups for each subnet
resource "azurerm_network_security_group" "subnet_nsgs" {
  for_each = var.subnet_configurations

  name                = "${local.naming_prefix}-nsg-${each.key}-${local.naming_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = merge(local.common_tags, {
    Subnet = each.key
  })
}

# Basic security rules for Container Apps
resource "azurerm_network_security_rule" "container_apps_inbound_allow" {
  count = contains(keys(var.subnet_configurations), "container_apps") ? 1 : 0

  name                        = "AllowContainerAppsInbound"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_ranges     = ["443", "80"]
  source_address_prefix       = "Internet"
  destination_address_prefix  = "*"
  resource_group_name         = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.subnet_nsgs["container_apps"].name
}

# Removed hardcoded data subnet rules - not needed for Container Apps

# Removed hardcoded AI subnet rules - not needed for Container Apps

# Basic security rules for Application Gateway
resource "azurerm_network_security_rule" "app_gateway_inbound_allow" {
  count = contains(keys(var.subnet_configurations), "app_gateway") ? 1 : 0

  name                        = "AllowAppGatewayInbound"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_ranges     = ["80", "443", "65200-65535"]
  source_address_prefix       = "*"
  destination_address_prefix  = "*"
  resource_group_name         = var.resource_group_name
  network_security_group_name = azurerm_network_security_group.subnet_nsgs["app_gateway"].name
}

# Subnets
resource "azurerm_subnet" "subnets" {
  for_each = var.subnet_configurations

  name                 = "${local.naming_prefix}-subnet-${each.key}-${local.naming_suffix}"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = each.value.address_prefixes
  service_endpoints    = each.value.service_endpoints

  # Delegation for specific services
  dynamic "delegation" {
    for_each = each.value.delegation != null ? [each.value.delegation] : []
    content {
      name = delegation.value.name
      service_delegation {
        name    = delegation.value.service_delegation.name
        actions = delegation.value.service_delegation.actions
      }
    }
  }
}

# Associate NSGs with subnets
resource "azurerm_subnet_network_security_group_association" "subnet_nsg_associations" {
  for_each = var.subnet_configurations

  subnet_id                 = azurerm_subnet.subnets[each.key].id
  network_security_group_id = azurerm_network_security_group.subnet_nsgs[each.key].id
}

# Route Table for custom routing
resource "azurerm_route_table" "main" {
  name                = "${local.naming_prefix}-rt-${local.naming_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name

  bgp_route_propagation_enabled = true

  tags = local.common_tags
}

# Custom route for internet traffic through firewall (if enabled)
resource "azurerm_route" "internet_via_firewall" {
  count = var.enable_firewall ? 1 : 0

  name                = "InternetViaFirewall"
  resource_group_name = var.resource_group_name
  route_table_name    = azurerm_route_table.main.name
  address_prefix      = "0.0.0.0/0"
  next_hop_type       = "VirtualAppliance"
  next_hop_in_ip_address = var.firewall_private_ip
}

# Associate route table with data and AI subnets for enhanced security
# Route table association for container apps subnet only
resource "azurerm_subnet_route_table_association" "container_apps" {
  count = contains(keys(var.subnet_configurations), "container_apps") ? 1 : 0

  subnet_id      = azurerm_subnet.subnets["container_apps"].id
  route_table_id = azurerm_route_table.main.id
}

# Network Watcher for monitoring and diagnostics
resource "azurerm_network_watcher" "main" {
  count = var.enable_network_watcher ? 1 : 0

  name                = "${local.naming_prefix}-nw-${local.naming_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = local.common_tags
}

# DDoS Protection Plan (optional, for production workloads)
resource "azurerm_network_ddos_protection_plan" "main" {
  count = var.enable_ddos_protection ? 1 : 0

  name                = "${local.naming_prefix}-ddos-${local.naming_suffix}"
  location            = var.location
  resource_group_name = var.resource_group_name

  tags = local.common_tags
}

# DDoS protection is now configured directly in the virtual network resource above

# Private DNS Zone for internal name resolution
resource "azurerm_private_dns_zone" "internal" {
  name                = "${var.project_name}.internal"
  resource_group_name = var.resource_group_name

  tags = local.common_tags
}

# Link Private DNS Zone to VNet
resource "azurerm_private_dns_zone_virtual_network_link" "internal" {
  name                  = "${local.naming_prefix}-dns-link-${local.naming_suffix}"
  resource_group_name   = var.resource_group_name
  private_dns_zone_name = azurerm_private_dns_zone.internal.name
  virtual_network_id    = azurerm_virtual_network.main.id
  registration_enabled  = true

  tags = local.common_tags
}

# Flow Logs for Network Security Groups (for monitoring and security analysis)
resource "azurerm_storage_account" "flow_logs" {
  count = var.enable_flow_logs ? 1 : 0

  name                     = "pcx${var.environment}fl${local.naming_suffix}"
  resource_group_name      = var.resource_group_name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  
  # Enable versioning and soft delete for security
  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 30
    }
  }

  tags = local.common_tags
}

# Network Watcher Flow Logs
resource "azurerm_network_watcher_flow_log" "nsg_flow_logs" {
  for_each = var.enable_flow_logs ? var.subnet_configurations : {}

  network_watcher_name = azurerm_network_watcher.main[0].name
  resource_group_name  = var.resource_group_name
  name                 = "${local.naming_prefix}-flowlog-${each.key}-${local.naming_suffix}"

  network_security_group_id = azurerm_network_security_group.subnet_nsgs[each.key].id
  storage_account_id        = azurerm_storage_account.flow_logs[0].id
  enabled                   = true

  retention_policy {
    enabled = true
    days    = 90
  }

  traffic_analytics {
    enabled               = true
    workspace_id          = var.log_analytics_workspace_id
    workspace_region      = var.location
    workspace_resource_id = var.log_analytics_workspace_resource_id
    interval_in_minutes   = 10
  }

  tags = local.common_tags
} 