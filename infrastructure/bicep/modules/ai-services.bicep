// AI Services module
param environment string
param location string
param tags object = {}
param keyVaultName string
param privateEndpointsSubnetId string
param privateDnsZones object
param storageAccountId string
param containerRegistryId string
param applicationInsightsId string
param deployMLWorkspace bool = true
param createMLContainerRegistry bool = false
param trainingClusterVMSize string = 'Standard_DS3_v2'
param trainingClusterMaxNodes int = 4
param computeInstanceVMSize string = 'Standard_DS3_v2'
param cognitiveServicesSku string = 'S0'
param deployOpenAI bool = true
param openAISku string = 'S0'

// Generate a unique suffix to avoid conflicts with existing resources
var uniqueSuffix = uniqueString(resourceGroup().id, 'v2')
var shortUniqueSuffix = substring(uniqueSuffix, 0, 6)

// Cognitive Services Account
resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'policycortex-cog-inc-${environment}'
  location: location
  tags: tags
  sku: {
    name: cognitiveServicesSku
  }
  kind: 'CognitiveServices'
  properties: {
    apiProperties: {}
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

// Azure OpenAI Service (conditional)
resource openAIService 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployOpenAI) {
  name: 'policycortex-oai-inc-${environment}'
  location: location
  tags: tags
  sku: {
    name: openAISku
  }
  kind: 'OpenAI'
  properties: {
    apiProperties: {}
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

// ML Workspace (conditional)
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = if (deployMLWorkspace) {
  name: 'policycortex-ml-inc-${environment}'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'PolicyCortex ML Workspace'
    description: 'Machine Learning workspace for PolicyCortex AI models'
    storageAccount: storageAccountId
    containerRegistry: createMLContainerRegistry ? mlContainerRegistry.id : containerRegistryId
    applicationInsights: applicationInsightsId
    keyVault: resourceId('Microsoft.KeyVault/vaults', keyVaultName)
    hbiWorkspace: false
    allowPublicAccessWhenBehindVnet: true
    discoveryUrl: 'https://${location}.api.azureml.ms/discovery'
    publicNetworkAccess: 'Enabled'
  }
}

// ML Container Registry (conditional)
resource mlContainerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = if (createMLContainerRegistry) {
  name: 'crpcmlinc${environment}'
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// ML Compute Cluster (conditional)
resource mlComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = if (deployMLWorkspace) {
  parent: mlWorkspace
  name: 'training-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: trainingClusterVMSize
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: trainingClusterMaxNodes
        nodeIdleTimeBeforeScaleDown: 'PT2M'
      }
      osType: 'Linux'
      enableNodePublicIp: true
      isolatedNetwork: false
    }
  }
}

// ML Compute Instance (conditional)
resource mlComputeInstance 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = if (deployMLWorkspace) {
  parent: mlWorkspace
  name: 'compute-instance'
  location: location
  properties: {
    computeType: 'ComputeInstance'
    properties: {
      vmSize: computeInstanceVMSize
      enableNodePublicIp: true
    }
  }
}

// EventGrid Topic for ML Operations
resource eventGridTopic 'Microsoft.EventGrid/topics@2023-12-15-preview' = {
  name: 'pc-ml-events-inc-${environment}'
  location: location
  tags: tags
  properties: {
    inputSchema: 'EventGridSchema'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}

// Private Endpoints
resource cognitivePrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pc-cog-pe-inc-${environment}'
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
          privateLinkServiceId: cognitiveServices.id
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
  name: 'pc-oai-pe-inc-${environment}'
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
          privateLinkServiceId: openAIService.id
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
  name: 'pc-eg-pe-inc-${environment}'
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
          privateLinkServiceId: eventGridTopic.id
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
  name: 'pc-ml-pe-inc-${environment}'
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
          privateLinkServiceId: mlWorkspace.id
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

// Key Vault Secrets for AI services
resource cognitiveServicesKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cognitive-services-key'
  properties: {
    value: cognitiveServices.listKeys().key1
    contentType: 'text/plain'
  }
}

resource cognitiveServicesEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cognitive-services-endpoint'
  properties: {
    value: cognitiveServices.properties.endpoint
    contentType: 'text/plain'
  }
}

// OpenAI secrets will be created separately if needed

// Outputs
output cognitiveServicesId string = cognitiveServices.id
output cognitiveServicesName string = cognitiveServices.name
output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output openAIServiceId string = deployOpenAI ? openAIService.id : ''
output openAIServiceName string = deployOpenAI ? openAIService.name : ''
output openAIEndpoint string = ''  // Will be set manually if OpenAI is deployed
output mlWorkspaceId string = deployMLWorkspace ? mlWorkspace.id : ''
output mlWorkspaceName string = deployMLWorkspace ? mlWorkspace.name : ''
output eventGridTopicId string = eventGridTopic.id
output eventGridTopicName string = eventGridTopic.name