// Key Vault Secrets module
param keyVaultName string
@secure()
param jwtSecretKey string
param managedIdentityClientId string
param storageAccountName string
param applicationInsightsConnectionString string
param cognitiveServicesKey string
param cognitiveServicesEndpoint string
param redisConnectionString string
param cosmosConnectionString string
param cosmosEndpoint string
param cosmosKey string
param tenantId string = tenant().tenantId
param resourceGroupName string

// JWT Secret Key
resource jwtSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/jwt-secret-key'
  properties: {
    value: jwtSecretKey
    contentType: 'text/plain'
  }
}

// Managed Identity Client ID
resource managedIdentitySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/managed-identity-client-id'
  properties: {
    value: managedIdentityClientId
    contentType: 'text/plain'
  }
}

// Storage Account Name
resource storageAccountSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/storage-account-name'
  properties: {
    value: storageAccountName
    contentType: 'text/plain'
  }
}

// Application Insights Connection String
resource appInsightsSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/application-insights-connection-string'
  properties: {
    value: applicationInsightsConnectionString
    contentType: 'text/plain'
  }
}

// Cognitive Services Key
resource cognitiveServicesKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cognitive-services-key'
  properties: {
    value: cognitiveServicesKey
    contentType: 'text/plain'
  }
}

// Cognitive Services Endpoint
resource cognitiveServicesEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cognitive-services-endpoint'
  properties: {
    value: cognitiveServicesEndpoint
    contentType: 'text/plain'
  }
}

// Redis Connection String
resource redisSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/redis-connection-string'
  properties: {
    value: redisConnectionString
    contentType: 'text/plain'
  }
}

// Cosmos DB Connection String
resource cosmosConnectionSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cosmos-connection-string'
  properties: {
    value: cosmosConnectionString
    contentType: 'text/plain'
  }
}

// Cosmos DB Endpoint
resource cosmosEndpointSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cosmos-endpoint'
  properties: {
    value: cosmosEndpoint
    contentType: 'text/plain'
  }
}

// Cosmos DB Key
resource cosmosKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/cosmos-key'
  properties: {
    value: cosmosKey
    contentType: 'text/plain'
  }
}

// SQL Server (placeholder for future use)
resource sqlServerSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/sql-server'
  properties: {
    value: 'not-configured'
    contentType: 'text/plain'
  }
}

// SQL Username (placeholder for future use)
resource sqlUsernameSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/sql-username'
  properties: {
    value: 'not-configured'
    contentType: 'text/plain'
  }
}

// SQL Password (placeholder for future use)
resource sqlPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/sql-password'
  properties: {
    value: 'not-configured'
    contentType: 'text/plain'
  }
}

// Key Vault Name (for application configuration)
resource keyVaultNameSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/key-vault-name'
  properties: {
    value: keyVaultName
    contentType: 'text/plain'
  }
}

// ML Workspace Name (placeholder for future use)
resource mlWorkspaceNameSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/ml-workspace-name'
  properties: {
    value: 'not-configured'
    contentType: 'text/plain'
  }
}

// Azure Tenant ID
resource tenantIdSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/tenant-id'
  properties: {
    value: tenantId
    contentType: 'text/plain'
  }
}

// Azure Client ID (same as managed identity for now)
resource clientIdSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/client-id'
  properties: {
    value: managedIdentityClientId
    contentType: 'text/plain'
  }
}

// Azure Client ID for frontend (specific name expected by Container Apps)
resource azureClientIdSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/azure-client-id'
  properties: {
    value: managedIdentityClientId
    contentType: 'text/plain'
  }
}

// Azure Tenant ID for frontend (specific name expected by Container Apps)
resource azureTenantIdSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/azure-tenant-id'
  properties: {
    value: tenantId
    contentType: 'text/plain'
  }
}

// Azure Client Secret (placeholder since using managed identity)
resource clientSecretSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/client-secret'
  properties: {
    value: 'not-configured-using-managed-identity'
    contentType: 'text/plain'
  }
}

// Azure Subscription ID
resource subscriptionIdSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/subscription-id'
  properties: {
    value: subscription().subscriptionId
    contentType: 'text/plain'
  }
}

// Resource Group Name
resource resourceGroupSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/resource-group'
  properties: {
    value: resourceGroupName
    contentType: 'text/plain'
  }
}

// Service Bus Namespace (placeholder for future use)
resource serviceBusNamespaceSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  name: '${keyVaultName}/service-bus-namespace'
  properties: {
    value: 'not-configured'
    contentType: 'text/plain'
  }
}

// Outputs
output secretNames array = [
  jwtSecret.name
  managedIdentitySecret.name
  storageAccountSecret.name
  appInsightsSecret.name
  cognitiveServicesKeySecret.name
  cognitiveServicesEndpointSecret.name
  redisSecret.name
  cosmosConnectionSecret.name
  cosmosEndpointSecret.name
  cosmosKeySecret.name
  sqlServerSecret.name
  sqlUsernameSecret.name
  sqlPasswordSecret.name
  keyVaultNameSecret.name
  mlWorkspaceNameSecret.name
  tenantIdSecret.name
  clientIdSecret.name
  azureClientIdSecret.name
  azureTenantIdSecret.name
  clientSecretSecret.name
  subscriptionIdSecret.name
  resourceGroupSecret.name
  serviceBusNamespaceSecret.name
]