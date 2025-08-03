// Private Endpoints module - deployed to network resource group
param environment string
param location string
param tags object = {}
param privateEndpointsSubnetId string
param privateDnsZones object

// Data Services Resource IDs (from app resource group)
param cosmosAccountId string
param cosmosAccountName string
param redisId string
param redisName string
param sqlServerId string = ''
param sqlServerName string = ''
param deploySqlServer bool = true

// AI Services Resource IDs (from app resource group)
param cognitiveServicesId string
param openAIServiceId string = ''
param mlWorkspaceId string = ''
param eventGridTopicId string
param deployOpenAI bool = true
param deployMLWorkspace bool = true

// Data Services Private Endpoints
resource cosmosPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-pcx-cosmos-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'cosmos-connection'
        properties: {
          privateLinkServiceId: cosmosAccountId
          groupIds: ['Sql']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Cosmos DB private endpoint'
          }
        }
      }
    ]
  }
}

resource redisPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-pcx-redis-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'redis-connection'
        properties: {
          privateLinkServiceId: redisId
          groupIds: ['redisCache']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Redis Cache private endpoint'
          }
        }
      }
    ]
  }
}

resource sqlPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = if (deploySqlServer) {
  name: 'pe-pcx-sql-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'sql-connection'
        properties: {
          privateLinkServiceId: sqlServerId
          groupIds: ['sqlServer']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'SQL Server private endpoint'
          }
        }
      }
    ]
  }
}

// AI Services Private Endpoints
resource cognitivePrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-pcx-cog-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'cognitive-connection'
        properties: {
          privateLinkServiceId: cognitiveServicesId
          groupIds: ['account']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'Cognitive Services private endpoint'
          }
        }
      }
    ]
  }
}

resource openAIPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = if (deployOpenAI) {
  name: 'pe-pcx-oai-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'openai-connection'
        properties: {
          privateLinkServiceId: openAIServiceId
          groupIds: ['account']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'OpenAI private endpoint'
          }
        }
      }
    ]
  }
}

resource eventGridPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-pcx-eg-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'eventgrid-connection'
        properties: {
          privateLinkServiceId: eventGridTopicId
          groupIds: ['topic']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'EventGrid private endpoint'
          }
        }
      }
    ]
  }
}

resource mlPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = if (deployMLWorkspace) {
  name: 'pe-pcx-ml-${environment}'
  location: location
  tags: tags
  properties: {
    subnet: {
      id: privateEndpointsSubnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'ml-connection'
        properties: {
          privateLinkServiceId: mlWorkspaceId
          groupIds: ['amlworkspace']
          privateLinkServiceConnectionState: {
            status: 'Approved'
            description: 'ML Workspace private endpoint'
          }
        }
      }
    ]
  }
}

// Private DNS Zone Groups
resource cosmosDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = {
  parent: cosmosPrivateEndpoint
  name: 'cosmos-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'cosmos-config'
        properties: {
          privateDnsZoneId: privateDnsZones.cosmos
        }
      }
    ]
  }
}

resource redisDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = {
  parent: redisPrivateEndpoint
  name: 'redis-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'redis-config'
        properties: {
          privateDnsZoneId: privateDnsZones.redis
        }
      }
    ]
  }
}

resource sqlDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = if (deploySqlServer) {
  parent: sqlPrivateEndpoint
  name: 'sql-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'sql-config'
        properties: {
          privateDnsZoneId: privateDnsZones.sql
        }
      }
    ]
  }
}

resource cognitiveDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = {
  parent: cognitivePrivateEndpoint
  name: 'cognitive-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'cognitive-config'
        properties: {
          privateDnsZoneId: privateDnsZones.cognitive
        }
      }
    ]
  }
}

resource openAIDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = if (deployOpenAI) {
  parent: openAIPrivateEndpoint
  name: 'openai-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'openai-config'
        properties: {
          privateDnsZoneId: privateDnsZones.openai
        }
      }
    ]
  }
}

resource mlDnsZoneGroup 'Microsoft.Network/privateEndpoints/privateDnsZoneGroups@2023-05-01' = if (deployMLWorkspace) {
  parent: mlPrivateEndpoint
  name: 'ml-dns-zone-group'
  properties: {
    privateDnsZoneConfigs: [
      {
        name: 'ml-config'
        properties: {
          privateDnsZoneId: privateDnsZones.ml
        }
      }
    ]
  }
}

// Outputs
output privateEndpointIds array = [
  cosmosPrivateEndpoint.id
  redisPrivateEndpoint.id
  cognitivePrivateEndpoint.id
  eventGridPrivateEndpoint.id
]