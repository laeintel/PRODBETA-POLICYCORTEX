// Networking module
param environment string
param location string
param tags object = {}

var vnetName = 'policycortex-${environment}-vnet'
var vnetAddressSpace = '10.0.0.0/16'

// Subnets configuration
var subnetConfigs = [
  {
    name: 'container_apps'
    addressPrefix: '10.0.0.0/23'  // Container Apps requires /23 or larger
    delegations: [
      {
        name: 'Microsoft.App.environments'
        properties: {
          serviceName: 'Microsoft.App/environments'
        }
      }
    ]
  }
  {
    name: 'app_gateway'
    addressPrefix: '10.0.2.0/24'
    delegations: []
  }
  {
    name: 'private_endpoints'
    addressPrefix: '10.0.3.0/24'
    delegations: []
  }
  {
    name: 'data_services'
    addressPrefix: '10.0.4.0/24'
    delegations: []
  }
  {
    name: 'ai_services'
    addressPrefix: '10.0.5.0/24'
    delegations: []
  }
]

// Private DNS Zones - Minimal set for initial deployment
var privateDnsZoneNames = [
  'policycortex-${environment}.internal'
  'privatelink.documents.azure.com'
  'privatelink.redis.cache.windows.net'
  'privatelink.cognitiveservices.azure.com'
]

// Virtual Network
resource virtualNetwork 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: [vnetAddressSpace]
    }
    subnets: [for config in subnetConfigs: {
      name: config.name
      properties: {
        addressPrefix: config.addressPrefix
        delegations: config.delegations
        privateEndpointNetworkPolicies: 'Disabled'
        privateLinkServiceNetworkPolicies: 'Enabled'
      }
    }]
    enableDdosProtection: false
  }
}

// Network Security Groups
resource networkSecurityGroups 'Microsoft.Network/networkSecurityGroups@2023-05-01' = [for config in subnetConfigs: {
  name: 'policycortex-${environment}-nsg-${config.name}'
  location: location
  tags: tags
  properties: {
    securityRules: config.name == 'container_apps' ? [
      {
        name: 'AllowContainerAppsInbound'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '*'
          sourceAddressPrefix: 'VirtualNetwork'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowHTTPSInbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 200
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowHTTPInbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '80'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 300
          direction: 'Inbound'
        }
      }
    ] : config.name == 'app_gateway' ? [
      {
        name: 'AllowGatewayManager'
        properties: {
          protocol: '*'
          sourcePortRange: '*'
          destinationPortRange: '65200-65535'
          sourceAddressPrefix: 'GatewayManager'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 100
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowHTTPSInbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '443'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 200
          direction: 'Inbound'
        }
      }
      {
        name: 'AllowHTTPInbound'
        properties: {
          protocol: 'Tcp'
          sourcePortRange: '*'
          destinationPortRange: '80'
          sourceAddressPrefix: 'Internet'
          destinationAddressPrefix: '*'
          access: 'Allow'
          priority: 300
          direction: 'Inbound'
        }
      }
    ] : []
  }
}]

// Route Table
resource routeTable 'Microsoft.Network/routeTables@2023-05-01' = {
  name: 'policycortex-${environment}-rt'
  location: location
  tags: tags
  properties: {
    routes: [
      {
        name: 'DefaultRoute'
        properties: {
          addressPrefix: '0.0.0.0/0'
          nextHopType: 'Internet'
        }
      }
    ]
    disableBgpRoutePropagation: false
  }
}

// Network Watcher - Commented out as it's a singleton resource per region
// Azure automatically creates this when first networking resource is deployed
// resource networkWatcher 'Microsoft.Network/networkWatchers@2023-05-01' = {
//   name: 'policycortex-${environment}-nw'
//   location: location
//   tags: tags
//   properties: {}
// }

// Private DNS Zones
resource privateDnsZoneResources 'Microsoft.Network/privateDnsZones@2020-06-01' = [for zone in privateDnsZoneNames: {
  name: zone
  location: 'global'
  tags: tags
  properties: {}
}]

// Link Private DNS Zones to VNet
resource privateDnsZoneLinks 'Microsoft.Network/privateDnsZones/virtualNetworkLinks@2020-06-01' = [for (zone, i) in privateDnsZoneNames: {
  parent: privateDnsZoneResources[i]
  name: '${zone}-link'
  location: 'global'
  properties: {
    registrationEnabled: false
    virtualNetwork: {
      id: virtualNetwork.id
    }
  }
}]

// Outputs
output vnetId string = virtualNetwork.id
output vnetName string = virtualNetwork.name
output containerAppsSubnetId string = '${virtualNetwork.id}/subnets/container_apps'
output appGatewaySubnetId string = '${virtualNetwork.id}/subnets/app_gateway'
output privateEndpointsSubnetId string = '${virtualNetwork.id}/subnets/private_endpoints'
output dataServicesSubnetId string = '${virtualNetwork.id}/subnets/data_services'
output aiServicesSubnetId string = '${virtualNetwork.id}/subnets/ai_services'
output privateDnsZones object = {
  internal: privateDnsZoneResources[0].id
  cosmos: privateDnsZoneResources[1].id
  redis: privateDnsZoneResources[2].id
  cognitive: privateDnsZoneResources[3].id
  sql: ''  // Not deployed in minimal setup
  ml: ''   // Not deployed in minimal setup
  openai: ''  // Not deployed in minimal setup
}