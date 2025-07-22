// Key Vault Secrets module
param keyVaultName string
@secure()
param jwtSecretKey string
param managedIdentityClientId string
param storageAccountName string
param applicationInsightsConnectionString string

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

// Outputs
output secretNames array = [
  jwtSecret.name
  managedIdentitySecret.name
  storageAccountSecret.name
  appInsightsSecret.name
]