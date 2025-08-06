// Simple PolicyCortex Infrastructure for Development
// Deploys only essential resources for container apps

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Location for all resources')
param location string = resourceGroup().location

@description('Resource owner tag')
param owner string = 'PolicyCortex'

// Variables
var resourcePrefix = 'pcx-${environment}'
var tags = {
  Environment: environment
  Owner: owner
  ManagedBy: 'Bicep'
  Project: 'PolicyCortex'
}

// Resource names with unique suffixes
var uniqueSuffix = uniqueString(resourceGroup().id, subscription().subscriptionId)
var vnetName = 'vnet-${resourcePrefix}-${uniqueSuffix}'
var keyVaultName = 'kv-${resourcePrefix}-${uniqueSuffix}'
var logWorkspaceName = 'log-${resourcePrefix}-${uniqueSuffix}'
var appInsightsName = 'appi-${resourcePrefix}-${uniqueSuffix}'
var containerRegistryName = 'cr${replace(resourcePrefix, '-', '')}${uniqueSuffix}'
var storageAccountName = 'st${replace(resourcePrefix, '-', '')}${uniqueSuffix}'
var redisCacheName = 'redis-${resourcePrefix}-${uniqueSuffix}'
var containerAppsEnvName = 'cae-${resourcePrefix}-${uniqueSuffix}'

// User-assigned managed identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${resourcePrefix}'
  location: location
  tags: tags
}

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
    ]
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

// Storage Account (minimal for development)
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

// Redis Cache (minimal for development)
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

// Key Vault (minimal)
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
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
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

// Essential secrets only
resource redisPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'REDIS-PASSWORD'
  properties: {
    value: listKeys(redisCache.id, '2023-04-01').primaryKey
  }
}

resource jwtSecretKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'JWT-SECRET-KEY'
  properties: {
    value: 'development-jwt-secret-${uniqueString(resourceGroup().id)}'
  }
}

resource appInsightsConnectionStringSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'APPLICATION-INSIGHTS-CONNECTION-STRING'
  properties: {
    value: applicationInsights.properties.ConnectionString
  }
}

resource containerRegistryPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'CONTAINER-REGISTRY-PASSWORD'
  properties: {
    value: listCredentials(containerRegistry.id, '2023-01-01-preview').passwords[0].value
  }
}

// Development defaults for missing services
resource sqlPasswordDefaultSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-SQL-PASSWORD'
  properties: {
    value: 'development-password'
  }
}

resource cosmosKeyDefaultSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-COSMOS-KEY'
  properties: {
    value: 'development-cosmos-key'
  }
}

resource openAIKeyDefaultSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'AZURE-OPENAI-KEY'
  properties: {
    value: 'development-openai-key'
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
output containerAppsEnvironmentId string = containerAppsEnvironment.id
output redisHostName string = redisCache.properties.hostName
output storageAccountName string = storageAccount.name