// Complete PolicyCortex Infrastructure - Fixed Version
// Deploys all resources required for the PolicyCortex platform

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Resource owner tag')
param owner string = 'PolicyCortex'

@description('SQL Server admin login')
@secure()
param sqlAdminLogin string = 'sqladmin'

@description('SQL Server admin password')
@secure()
param sqlAdminPassword string

@description('Azure AD client secret')
@secure()
param aadClientSecret string = ''

@description('JWT secret key')
@secure()
param jwtSecretKey string = 'change-this-in-production-${uniqueString(resourceGroup().id)}'

@description('Deploy SQL Server and database')
param deploySqlServer bool = true

@description('Deploy Machine Learning Workspace')
param deployMLWorkspace bool = true

@description('Deploy OpenAI resources')
param deployOpenAI bool = true

@description('Deploy Container Apps')
param deployContainerApps bool = true

@description('Allowed IP addresses for SQL firewall')
param allowedIps array = []

@description('Create Terraform access policy for Key Vault')
param createTerraformAccessPolicy bool = false

// Variables
var resourcePrefix = 'pcx-${environment}'
var tags = {
  Environment: environment
  Owner: owner
  ManagedBy: 'Bicep'
  Project: 'PolicyCortex'
  DeploymentId: uniqueString(resourceGroup().id)
}

// Resource names
var vnetName = 'vnet-${resourcePrefix}'
var keyVaultName = 'kv-${resourcePrefix}'
var logWorkspaceName = 'log-${resourcePrefix}'
var appInsightsName = 'appi-${resourcePrefix}'
var containerRegistryName = 'cr${replace(resourcePrefix, '-', '')}${uniqueString(resourceGroup().id)}'
var storageAccountName = 'st${replace(resourcePrefix, '-', '')}${uniqueString(resourceGroup().id)}'
var sqlServerName = 'sql-${resourcePrefix}'
var cosmosAccountName = 'cosmos-${resourcePrefix}'
var redisCacheName = 'redis-${resourcePrefix}'
var serviceBusName = 'sb-${resourcePrefix}'
var eventHubNamespaceName = 'eh-${resourcePrefix}'
var mlWorkspaceName = 'ml-${resourcePrefix}'
var openAIAccountName = 'openai-${resourcePrefix}'
var commServiceName = 'comm-${resourcePrefix}'
var containerAppsEnvName = 'cae-${resourcePrefix}'

// Networking
resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vnetName
  location: location
  tags: tags
  properties: {
    addressSpace: {
      addressPrefixes: ['10.0.0.0/16']
    }
    subnets: [
      {
        name: 'container-apps'
        properties: {
          addressPrefix: '10.0.1.0/24'
        }
      }
      {
        name: 'data-services'
        properties: {
          addressPrefix: '10.0.2.0/24'
          serviceEndpoints: [
            { service: 'Microsoft.Sql' }
            { service: 'Microsoft.Storage' }
            { service: 'Microsoft.KeyVault' }
          ]
        }
      }
      {
        name: 'private-endpoints'
        properties: {
          addressPrefix: '10.0.3.0/24'
        }
      }
    ]
  }
}

// User-assigned managed identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${resourcePrefix}'
  location: location
  tags: tags
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enabledForDeployment: true
    enabledForTemplateDeployment: true
    enabledForDiskEncryption: true
    enableRbacAuthorization: false
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: managedIdentity.properties.principalId
        permissions: {
          secrets: ['get', 'list']
          keys: ['get', 'list']
          certificates: ['get', 'list']
        }
      }
    ]
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Log Analytics Workspace
resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logWorkspaceName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logWorkspace.id
  }
}

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: containerRegistryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  tags: tags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// SQL Server
resource sqlServer 'Microsoft.Sql/servers@2022-05-01-preview' = if (deploySqlServer) {
  name: sqlServerName
  location: location
  tags: tags
  properties: {
    administratorLogin: sqlAdminLogin
    administratorLoginPassword: sqlAdminPassword
    version: '12.0'
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
  }
}

// SQL Database
resource sqlDatabase 'Microsoft.Sql/servers/databases@2022-05-01-preview' = if (deploySqlServer) {
  parent: sqlServer
  name: 'policycortex-${environment}'
  location: location
  tags: tags
  sku: {
    name: 'S0'
    tier: 'Standard'
  }
  properties: {
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    maxSizeBytes: 2147483648
  }
}

// SQL Firewall Rules
resource sqlFirewallRuleAzure 'Microsoft.Sql/servers/firewallRules@2022-05-01-preview' = if (deploySqlServer) {
  parent: sqlServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Cosmos DB Account
resource cosmosDbAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: cosmosAccountName
  location: location
  tags: tags
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
        failoverPriority: 0
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    capabilities: []
  }
}

// Cosmos DB Database
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2023-04-15' = {
  parent: cosmosDbAccount
  name: 'policycortex'
  properties: {
    resource: {
      id: 'policycortex'
    }
  }
}

// Redis Cache
resource redisCache 'Microsoft.Cache/redis@2023-04-01' = {
  name: redisCacheName
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'Basic'
      family: 'C'
      capacity: 0
    }
    minimumTlsVersion: '1.2'
  }
}

// Service Bus Namespace
resource serviceBus 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: serviceBusName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
}

// Service Bus Queue
resource serviceBusQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBus
  name: 'notifications'
  properties: {
    lockDuration: 'PT1M'
    maxSizeInMegabytes: 1024
    requiresDuplicateDetection: false
    requiresSession: false
    defaultMessageTimeToLive: 'P7D'
    duplicateDetectionHistoryTimeWindow: 'PT10M'
    maxDeliveryCount: 10
    enablePartitioning: false
    enableExpress: false
  }
}

// Event Hub Namespace
resource eventHubNamespace 'Microsoft.EventHub/namespaces@2022-10-01-preview' = {
  name: eventHubNamespaceName
  location: location
  tags: tags
  sku: {
    name: 'Standard'
    tier: 'Standard'
    capacity: 1
  }
}

// Event Hub
resource eventHub 'Microsoft.EventHub/namespaces/eventhubs@2022-10-01-preview' = {
  parent: eventHubNamespace
  name: 'telemetry'
  properties: {
    messageRetentionInDays: 7
    partitionCount: 2
  }
}

// Machine Learning Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = if (deployMLWorkspace) {
  name: mlWorkspaceName
  location: location
  tags: tags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    containerRegistry: containerRegistry.id
  }
}

// OpenAI Account
resource openAIAccount 'Microsoft.CognitiveServices/accounts@2023-05-01' = if (deployOpenAI) {
  name: openAIAccountName
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAIAccountName
    publicNetworkAccess: 'Enabled'
  }
}

// Communication Service
resource communicationService 'Microsoft.Communication/communicationServices@2023-03-31' = {
  name: commServiceName
  location: 'global'
  tags: tags
  properties: {
    dataLocation: 'United States'
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = if (deployContainerApps) {
  name: containerAppsEnvName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logWorkspace.properties.customerId
        sharedKey: listKeys(logWorkspace.id, '2022-10-01').primarySharedKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: vnet.properties.subnets[0].id
    }
  }
}

// Key Vault Secrets - Individual resources to avoid runtime evaluation issues
resource sqlPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deploySqlServer) {
  parent: keyVault
  name: 'AZURE-SQL-PASSWORD'
  properties: {
    value: sqlAdminPassword
  }
}

resource sqlUsernameSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deploySqlServer) {
  parent: keyVault
  name: 'AZURE-SQL-USERNAME'
  properties: {
    value: sqlAdminLogin
  }
}

resource sqlServerSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deploySqlServer) {
  parent: keyVault
  name: 'AZURE-SQL-SERVER'
  properties: {
    value: deploySqlServer ? sqlServer.properties.fullyQualifiedDomainName : 'not-deployed'
  }
}

resource sqlDatabaseSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deploySqlServer) {
  parent: keyVault
  name: 'AZURE-SQL-DATABASE'
  properties: {
    value: deploySqlServer ? sqlDatabase.name : 'not-deployed'
  }
}

resource cosmosKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-COSMOS-KEY'
  properties: {
    value: listKeys(cosmosDbAccount.id, '2023-04-15').primaryMasterKey
  }
}

resource cosmosEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-COSMOS-ENDPOINT'
  properties: {
    value: cosmosDbAccount.properties.documentEndpoint
  }
}

resource redisPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'REDIS-PASSWORD'
  properties: {
    value: listKeys(redisCache.id, '2023-04-01').primaryKey
  }
}

resource redisUrlSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'REDIS-URL'
  properties: {
    value: '${redisCache.properties.hostName}:${redisCache.properties.sslPort}'
  }
}

resource storageKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-STORAGE-ACCOUNT-KEY'
  properties: {
    value: listKeys(storageAccount.id, '2023-01-01').keys[0].value
  }
}

resource storageNameSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-STORAGE-ACCOUNT-NAME'
  properties: {
    value: storageAccount.name
  }
}

resource serviceBusConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-SERVICE-BUS-CONNECTION-STRING'
  properties: {
    value: listKeys('${serviceBus.id}/AuthorizationRules/RootManageSharedAccessKey', '2022-10-01-preview').primaryConnectionString
  }
}

resource appInsightsConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'APPLICATION-INSIGHTS-CONNECTION-STRING'
  properties: {
    value: applicationInsights.properties.ConnectionString
  }
}

resource appInsightsKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'APPLICATION-INSIGHTS-KEY'
  properties: {
    value: applicationInsights.properties.InstrumentationKey
  }
}

resource openAIKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deployOpenAI) {
  parent: keyVault
  name: 'AZURE-OPENAI-KEY'
  properties: {
    value: deployOpenAI ? listKeys(openAIAccount.id, '2023-05-01').key1 : 'not-deployed'
  }
}

resource openAIEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = if (deployOpenAI) {
  parent: keyVault
  name: 'AZURE-OPENAI-ENDPOINT'
  properties: {
    value: deployOpenAI ? openAIAccount.properties.endpoint : 'not-deployed'
  }
}

resource communicationConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-COMMUNICATION-CONNECTION-STRING'
  properties: {
    value: listKeys(communicationService.id, '2023-03-31').primaryConnectionString
  }
}

resource eventHubConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'EVENT-HUB-CONNECTION-STRING'
  properties: {
    value: listKeys('${eventHubNamespace.id}/AuthorizationRules/RootManageSharedAccessKey', '2022-10-01-preview').primaryConnectionString
  }
}

resource aadClientSecretSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-AD-CLIENT-SECRET'
  properties: {
    value: aadClientSecret
  }
}

resource containerRegistryPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'CONTAINER-REGISTRY-PASSWORD'
  properties: {
    value: listCredentials(containerRegistry.id, '2023-01-01-preview').passwords[0].value
  }
}

resource jwtSecretKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'JWT-SECRET-KEY'
  properties: {
    value: jwtSecretKey
  }
}

// Outputs
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output logWorkspaceId string = logWorkspace.id
output applicationInsightsConnectionString string = applicationInsights.properties.ConnectionString
output applicationInsightsInstrumentationKey string = applicationInsights.properties.InstrumentationKey
output managedIdentityId string = managedIdentity.id
output managedIdentityClientId string = managedIdentity.properties.clientId
output vnetId string = vnet.id
output containerAppsEnvironmentId string = deployContainerApps ? containerAppsEnvironment.id : ''
output sqlServerFqdn string = deploySqlServer ? sqlServer.properties.fullyQualifiedDomainName : ''
output cosmosEndpoint string = cosmosDbAccount.properties.documentEndpoint
output redisHostName string = redisCache.properties.hostName
output serviceBusNamespace string = serviceBus.name
output eventHubNamespace string = eventHubNamespace.name
output storageAccountName string = storageAccount.name
output mlWorkspaceId string = deployMLWorkspace ? mlWorkspace.id : ''
output openAIEndpoint string = deployOpenAI ? openAIAccount.properties.endpoint : ''
output communicationServiceEndpoint string = communicationService.properties.hostName