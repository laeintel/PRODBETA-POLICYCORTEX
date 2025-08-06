@description('Complete PolicyCortex Infrastructure - All Phases')
param environment string = 'dev'
param location string = resourceGroup().location
param owner string = 'AeoliTech'
param projectName string = 'policycortex'

@description('SQL Server administrator login')
@secure()
param sqlAdminLogin string = 'sqladmin'

@description('SQL Server administrator password')
@secure()
param sqlAdminPassword string

@description('Azure AD tenant ID for authentication')
param aadTenantId string

@description('Azure AD application client ID')
param aadClientId string

@description('Azure AD application client secret')
@secure()
param aadClientSecret string

// Variables for resource naming
var resourcePrefix = '${projectName}-${environment}'
var storageAccountName = replace('${resourcePrefix}storage', '-', '')
var keyVaultName = '${resourcePrefix}-kv'
var cosmosDbName = '${resourcePrefix}-cosmos'
var sqlServerName = '${resourcePrefix}-sql'
var redisCacheName = '${resourcePrefix}-redis'
var serviceBusName = '${resourcePrefix}-servicebus'
var appInsightsName = '${resourcePrefix}-insights'
var logAnalyticsName = '${resourcePrefix}-logs'
var containerRegistryName = replace('${resourcePrefix}acr', '-', '')
var containerAppEnvName = '${resourcePrefix}-containerenv'
var mlWorkspaceName = '${resourcePrefix}-ml'
var openAIName = '${resourcePrefix}-openai'
var communicationServiceName = '${resourcePrefix}-communication'
var eventHubNamespaceName = '${resourcePrefix}-eventhub'
var functionAppName = '${resourcePrefix}-functions'
var appServicePlanName = '${resourcePrefix}-asp'

// Tags
var commonTags = {
  Environment: environment
  Owner: owner
  Project: projectName
  ManagedBy: 'Bicep'
}

// Resource Outputs
output resourceOutputs object = {
  // Storage
  storageAccount: {
    name: storageAccount.name
    id: storageAccount.id
    connectionString: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
    blobEndpoint: storageAccount.properties.primaryEndpoints.blob
  }
  
  // Key Vault
  keyVault: {
    name: keyVault.name
    id: keyVault.id
    uri: keyVault.properties.vaultUri
  }
  
  // Databases
  cosmosDb: {
    name: cosmosDbAccount.name
    id: cosmosDbAccount.id
    endpoint: cosmosDbAccount.properties.documentEndpoint
    connectionString: cosmosDbAccount.listConnectionStrings().connectionStrings[0].connectionString
  }
  
  sqlServer: {
    name: sqlServer.name
    id: sqlServer.id
    fullyQualifiedDomainName: sqlServer.properties.fullyQualifiedDomainName
    connectionString: 'Server=tcp:${sqlServer.properties.fullyQualifiedDomainName},1433;Initial Catalog=${sqlDatabase.name};Persist Security Info=False;User ID=${sqlAdminLogin};Password=${sqlAdminPassword};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;'
  }
  
  // Cache
  redisCache: {
    name: redisCache.name
    id: redisCache.id
    hostName: redisCache.properties.hostName
    sslPort: redisCache.properties.sslPort
    primaryKey: redisCache.listKeys().primaryKey
    connectionString: '${redisCache.properties.hostName}:${redisCache.properties.sslPort},password=${redisCache.listKeys().primaryKey},ssl=True,abortConnect=False'
  }
  
  // Service Bus
  serviceBus: {
    name: serviceBus.name
    id: serviceBus.id
    connectionString: serviceBus.listKeys('RootManageSharedAccessKey', serviceBus.apiVersion).primaryConnectionString
  }
  
  // Container Registry
  containerRegistry: {
    name: containerRegistry.name
    id: containerRegistry.id
    loginServer: containerRegistry.properties.loginServer
  }
  
  // Container Apps Environment
  containerAppsEnvironment: {
    name: containerAppsEnvironment.name
    id: containerAppsEnvironment.id
    defaultDomain: containerAppsEnvironment.properties.defaultDomain
  }
  
  // AI/ML Services
  openAI: {
    name: openAIAccount.name
    id: openAIAccount.id
    endpoint: openAIAccount.properties.endpoint
  }
  
  mlWorkspace: {
    name: mlWorkspace.name
    id: mlWorkspace.id
  }
  
  // Communication
  communicationService: {
    name: communicationService.name
    id: communicationService.id
    connectionString: communicationService.listKeys().primaryConnectionString
  }
  
  // Event Hub
  eventHubNamespace: {
    name: eventHubNamespace.name
    id: eventHubNamespace.id
    connectionString: eventHubNamespace.listKeys('RootManageSharedAccessKey', eventHubNamespace.apiVersion).primaryConnectionString
  }
  
  // Monitoring
  applicationInsights: {
    name: applicationInsights.name
    id: applicationInsights.id
    connectionString: applicationInsights.properties.ConnectionString
    instrumentationKey: applicationInsights.properties.InstrumentationKey
  }
  
  logAnalyticsWorkspace: {
    name: logAnalyticsWorkspace.name
    id: logAnalyticsWorkspace.id
    workspaceId: logAnalyticsWorkspace.properties.customerId
  }
}

// ==================== STORAGE ACCOUNT ====================
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: commonTags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    defaultToOAuthAuthentication: true
  }
}

// Storage Containers
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
}

resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'documents'
  properties: {
    publicAccess: 'None'
  }
}

resource reportsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'reports'
  properties: {
    publicAccess: 'None'
  }
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'ml-models'
  properties: {
    publicAccess: 'None'
  }
}

resource logsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobServices
  name: 'logs'
  properties: {
    publicAccess: 'None'
  }
}

// ==================== LOG ANALYTICS WORKSPACE ====================
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  tags: commonTags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// ==================== APPLICATION INSIGHTS ====================
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: commonTags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// ==================== KEY VAULT ====================
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  tags: commonTags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: aadTenantId
    enableRbacAuthorization: true
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// ==================== COSMOS DB ====================
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: cosmosDbName
  location: location
  tags: commonTags
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    databaseAccountOfferType: 'Standard'
    enableFreeTier: false
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    capabilities: [
      {
        name: 'EnableServerless'
      }
    ]
    publicNetworkAccess: 'Enabled'
  }
}

resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosDbAccount
  name: 'policycortex'
  properties: {
    resource: {
      id: 'policycortex'
    }
  }
}

// Cosmos DB Containers
var containers = [
  'alerts'
  'notifications'
  'sessions'
  'user-preferences'
  'analytics'
  'conversations'
  'policies'
  'compliance-reports'
  'audit-logs'
  'ml-training-data'
]

resource cosmosContainers 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = [for container in containers: {
  parent: cosmosDatabase
  name: container
  properties: {
    resource: {
      id: container
      partitionKey: {
        paths: ['/tenantId']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        automatic: true
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/"_etag"/?'
          }
        ]
      }
    }
  }
}]

// ==================== SQL SERVER & DATABASE ====================
resource sqlServer 'Microsoft.Sql/servers@2023-02-01-preview' = {
  name: sqlServerName
  location: location
  tags: commonTags
  properties: {
    administratorLogin: sqlAdminLogin
    administratorLoginPassword: sqlAdminPassword
    version: '12.0'
    publicNetworkAccess: 'Enabled'
  }
}

resource sqlServerFirewallRule 'Microsoft.Sql/servers/firewallRules@2023-02-01-preview' = {
  parent: sqlServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource sqlDatabase 'Microsoft.Sql/servers/databases@2023-02-01-preview' = {
  parent: sqlServer
  name: 'policycortex'
  location: location
  tags: commonTags
  sku: {
    name: 'S2'
    tier: 'Standard'
    capacity: 50
  }
  properties: {
    maxSizeBytes: 268435456000 // 250GB
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    catalogCollation: 'SQL_Latin1_General_CP1_CI_AS'
  }
}

// ==================== REDIS CACHE ====================
resource redisCache 'Microsoft.Cache/redis@2023-04-01' = {
  name: redisCacheName
  location: location
  tags: commonTags
  properties: {
    sku: {
      name: 'Standard'
      family: 'C'
      capacity: 1
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    redisConfiguration: {
      'maxmemory-policy': 'allkeys-lru'
    }
  }
}

// ==================== SERVICE BUS ====================
resource serviceBus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: serviceBusName
  location: location
  tags: commonTags
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

// Service Bus Queues
var queues = [
  'policy-processing'
  'compliance-analysis'
  'notifications'
  'data-processing'
  'ml-training'
  'audit-events'
]

resource serviceBusQueues 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = [for queue in queues: {
  parent: serviceBus
  name: queue
  properties: {
    maxSizeInMegabytes: 5120
    defaultMessageTimeToLive: 'P14D'
    deadLetteringOnMessageExpiration: true
    enablePartitioning: false
    requiresDuplicateDetection: false
    requiresSession: false
  }
}]

// ==================== CONTAINER REGISTRY ====================
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: containerRegistryName
  location: location
  tags: commonTags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// ==================== CONTAINER APPS ENVIRONMENT ====================
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerAppEnvName
  location: location
  tags: commonTags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// ==================== AZURE OPENAI ====================
resource openAIAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAIName
  location: location
  tags: commonTags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAIName
    publicNetworkAccess: 'Enabled'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// OpenAI Deployments
resource gpt4Deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAIAccount
  name: 'gpt-4'
  sku: {
    name: 'Standard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4'
      version: '0613'
    }
  }
}

resource gpt35TurboDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAIAccount
  name: 'gpt-35-turbo'
  sku: {
    name: 'Standard'
    capacity: 30
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-35-turbo'
      version: '0613'
    }
  }
  dependsOn: [gpt4Deployment]
}

resource textEmbeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAIAccount
  name: 'text-embedding-ada-002'
  sku: {
    name: 'Standard'
    capacity: 30
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
  }
  dependsOn: [gpt35TurboDeployment]
}

// ==================== MACHINE LEARNING WORKSPACE ====================
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: mlWorkspaceName
  location: location
  tags: commonTags
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    publicNetworkAccess: 'Enabled'
    hbiWorkspace: false
  }
}

// ==================== COMMUNICATION SERVICE ====================
resource communicationService 'Microsoft.Communication/communicationServices@2023-03-31' = {
  name: communicationServiceName
  location: 'global'
  tags: commonTags
  properties: {
    dataLocation: 'United States'
  }
}

// ==================== EVENT HUB NAMESPACE ====================
resource eventHubNamespace 'Microsoft.EventHub/namespaces@2022-10-01-preview' = {
  name: eventHubNamespaceName
  location: location
  tags: commonTags
  sku: {
    name: 'Standard'
    tier: 'Standard'
    capacity: 1
  }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

// Event Hubs
var eventHubs = [
  'policy-events'
  'compliance-events'
  'audit-events'
  'analytics-events'
  'ml-events'
]

resource eventHubs_resource 'Microsoft.EventHub/namespaces/eventhubs@2022-10-01-preview' = [for hub in eventHubs: {
  parent: eventHubNamespace
  name: hub
  properties: {
    messageRetentionInDays: 7
    partitionCount: 4
  }
}]

// ==================== APP SERVICE PLAN ====================
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: appServicePlanName
  location: location
  tags: commonTags
  kind: 'functionapp'
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  properties: {
    reserved: true
  }
}

// ==================== FUNCTION APP ====================
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: functionAppName
  location: location
  tags: commonTags
  kind: 'functionapp,linux'
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'Python|3.11'
      appSettings: [
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
        }
        {
          name: 'WEBSITE_CONTENTAZUREFILECONNECTIONSTRING'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
        }
        {
          name: 'WEBSITE_CONTENTSHARE'
          value: toLower(functionAppName)
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: applicationInsights.properties.ConnectionString
        }
      ]
    }
  }
}

// ==================== KEY VAULT SECRETS ====================
resource keyVaultSecrets 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = [
  for secret in [
    { name: 'AZURE-SQL-PASSWORD', value: sqlAdminPassword }
    { name: 'AZURE-SQL-USERNAME', value: sqlAdminLogin }
    { name: 'AZURE-SQL-SERVER', value: sqlServer.properties.fullyQualifiedDomainName }
    { name: 'AZURE-SQL-DATABASE', value: sqlDatabase.name }
    { name: 'AZURE-COSMOS-KEY', value: cosmosDbAccount.listKeys().primaryMasterKey }
    { name: 'AZURE-COSMOS-ENDPOINT', value: cosmosDbAccount.properties.documentEndpoint }
    { name: 'REDIS-PASSWORD', value: redisCache.listKeys().primaryKey }
    { name: 'REDIS-URL', value: '${redisCache.properties.hostName}:${redisCache.properties.sslPort}' }
    { name: 'AZURE-STORAGE-ACCOUNT-KEY', value: storageAccount.listKeys().keys[0].value }
    { name: 'AZURE-STORAGE-ACCOUNT-NAME', value: storageAccount.name }
    { name: 'AZURE-SERVICE-BUS-CONNECTION-STRING', value: serviceBus.listKeys('RootManageSharedAccessKey', serviceBus.apiVersion).primaryConnectionString }
    { name: 'APPLICATION-INSIGHTS-CONNECTION-STRING', value: applicationInsights.properties.ConnectionString }
    { name: 'APPLICATION-INSIGHTS-KEY', value: applicationInsights.properties.InstrumentationKey }
    { name: 'AZURE-OPENAI-KEY', value: openAIAccount.listKeys().key1 }
    { name: 'AZURE-OPENAI-ENDPOINT', value: openAIAccount.properties.endpoint }
    { name: 'AZURE-COMMUNICATION-CONNECTION-STRING', value: communicationService.listKeys().primaryConnectionString }
    { name: 'EVENT-HUB-CONNECTION-STRING', value: eventHubNamespace.listKeys('RootManageSharedAccessKey', eventHubNamespace.apiVersion).primaryConnectionString }
    { name: 'AZURE-AD-CLIENT-SECRET', value: aadClientSecret }
    { name: 'CONTAINER-REGISTRY-PASSWORD', value: containerRegistry.listCredentials().passwords[0].value }
    { name: 'JWT-SECRET-KEY', value: 'change-this-in-production-${uniqueString(resourceGroup().id)}' }
  ]
]: {
  parent: keyVault
  name: secret.name
  properties: {
    value: secret.value
    contentType: 'text/plain'
  }
}

// ==================== MANAGED IDENTITY ====================
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${resourcePrefix}-identity'
  location: location
  tags: commonTags
}

// ==================== RBAC ASSIGNMENTS ====================
// Key Vault Secrets User role for Managed Identity
resource keyVaultSecretsUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, managedIdentity.id, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Storage Blob Data Contributor role for Managed Identity
resource storageBlobDataContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, managedIdentity.id, 'Storage Blob Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Cosmos DB Built-in Data Contributor role for Managed Identity
resource cosmosDbDataContributorRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: cosmosDbAccount
  name: guid(cosmosDbAccount.id, managedIdentity.id, 'Cosmos DB Built-in Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00000000-0000-0000-0000-000000000002')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Container Registry Push role for Managed Identity
resource acrPushRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentity.id, 'AcrPush')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8311e382-0749-4cb8-b61a-304f252e45ec')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}