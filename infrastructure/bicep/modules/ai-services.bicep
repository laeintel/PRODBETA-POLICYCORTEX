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

// Static naming for consistent resource management

// Cognitive Services Account
resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'cog-pcx-${environment}'
  location: location
  tags: tags
  sku: {
    name: cognitiveServicesSku
  }
  kind: 'CognitiveServices'
  properties: {
    apiProperties: {}
    customSubDomainName: 'cog-pcx-${environment}'
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
    // Remove restore property - handled automatically by Azure
  }
}

// Azure OpenAI Service (conditional)
resource openAIService 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployOpenAI) {
  name: 'oai-pcx-${environment}'
  location: location
  tags: tags
  sku: {
    name: openAISku
  }
  kind: 'OpenAI'
  properties: {
    apiProperties: {}
    customSubDomainName: 'oai-pcx-${environment}'
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
    // Remove restore property - handled automatically by Azure
  }
}

// ML Workspace (conditional)
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = if (deployMLWorkspace) {
  name: 'ml-pcx-${environment}'
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'PolicyCortex ML Workspace'
    description: 'Machine Learning workspace for policortex AI models'
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
  name: 'crpcxml${environment}'
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
  name: 'eg-pcx-${environment}'
  location: location
  tags: tags
  properties: {
    inputSchema: 'EventGridSchema'
    publicNetworkAccess: 'Enabled'
    disableLocalAuth: false
  }
}


// Outputs
output cognitiveServicesId string = cognitiveServices.id
output cognitiveServicesName string = cognitiveServices.name
output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
// Removed cognitiveServicesKey output to avoid secrets in outputs
// Key will be retrieved directly in Key Vault secrets module
output openAIServiceId string = deployOpenAI ? openAIService.id : ''
output openAIServiceName string = deployOpenAI ? openAIService.name : ''
output openAIEndpoint string = ''  // Will be set manually if OpenAI is deployed
output mlWorkspaceId string = deployMLWorkspace ? mlWorkspace.id : ''
output mlWorkspaceName string = deployMLWorkspace ? mlWorkspace.name : ''
output eventGridTopicId string = eventGridTopic.id
output eventGridTopicName string = eventGridTopic.name