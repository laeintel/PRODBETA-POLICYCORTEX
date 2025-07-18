# Virtual Network Outputs
output "vnet_id" {
  description = "ID of the virtual network"
  value       = azurerm_virtual_network.main.id
}

output "vnet_name" {
  description = "Name of the virtual network"
  value       = azurerm_virtual_network.main.name
}

output "vnet_address_space" {
  description = "Address space of the virtual network"
  value       = azurerm_virtual_network.main.address_space
}

# Subnet Outputs
output "subnet_ids" {
  description = "Map of subnet names to their IDs"
  value = {
    for k, v in azurerm_subnet.subnets : k => v.id
  }
}

output "subnet_names" {
  description = "Map of subnet keys to their names"
  value = {
    for k, v in azurerm_subnet.subnets : k => v.name
  }
}

output "subnet_address_prefixes" {
  description = "Map of subnet names to their address prefixes"
  value = {
    for k, v in azurerm_subnet.subnets : k => v.address_prefixes
  }
}

# Network Security Group Outputs
output "nsg_ids" {
  description = "Map of NSG names to their IDs"
  value = {
    for k, v in azurerm_network_security_group.subnet_nsgs : k => v.id
  }
}

output "nsg_names" {
  description = "Map of NSG keys to their names"
  value = {
    for k, v in azurerm_network_security_group.subnet_nsgs : k => v.name
  }
}

# Route Table Outputs
output "route_table_id" {
  description = "ID of the main route table"
  value       = azurerm_route_table.main.id
}

output "route_table_name" {
  description = "Name of the main route table"
  value       = azurerm_route_table.main.name
}

# Private DNS Zone Outputs
output "private_dns_zone_id" {
  description = "ID of the private DNS zone"
  value       = azurerm_private_dns_zone.internal.id
}

output "private_dns_zone_name" {
  description = "Name of the private DNS zone"
  value       = azurerm_private_dns_zone.internal.name
}

# Network Watcher Outputs
output "network_watcher_id" {
  description = "ID of the Network Watcher"
  value       = var.enable_network_watcher ? azurerm_network_watcher.main[0].id : null
}

output "network_watcher_name" {
  description = "Name of the Network Watcher"
  value       = var.enable_network_watcher ? azurerm_network_watcher.main[0].name : null
}

# DDoS Protection Plan Outputs
output "ddos_protection_plan_id" {
  description = "ID of the DDoS protection plan"
  value       = var.enable_ddos_protection ? azurerm_network_ddos_protection_plan.main[0].id : null
}

# Flow Logs Storage Account
output "flow_logs_storage_account_id" {
  description = "ID of the storage account for flow logs"
  value       = var.enable_flow_logs ? azurerm_storage_account.flow_logs[0].id : null
}

output "flow_logs_storage_account_name" {
  description = "Name of the storage account for flow logs"
  value       = var.enable_flow_logs ? azurerm_storage_account.flow_logs[0].name : null
}

# Networking Information for Other Modules
output "network_info" {
  description = "Comprehensive network information for use by other modules"
  value = {
    vnet_id           = azurerm_virtual_network.main.id
    vnet_name         = azurerm_virtual_network.main.name
    address_space     = azurerm_virtual_network.main.address_space
    resource_group    = var.resource_group_name
    location          = var.location
    private_dns_zone  = azurerm_private_dns_zone.internal.name
    
    subnets = {
      for k, v in azurerm_subnet.subnets : k => {
        id               = v.id
        name             = v.name
        address_prefixes = v.address_prefixes
        nsg_id          = azurerm_network_security_group.subnet_nsgs[k].id
        nsg_name        = azurerm_network_security_group.subnet_nsgs[k].name
      }
    }
  }
} 