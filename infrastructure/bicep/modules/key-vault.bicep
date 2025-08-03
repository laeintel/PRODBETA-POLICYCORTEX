// Key Vault module
param keyVaultName string
param location string
param tags object = {}
param createTerraformAccessPolicy bool = true
param environment string = 'dev'
param managedIdentityPrincipalId string = ''


resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  tags: tags
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenant().tenantId
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7  // Minimum retention for faster cleanup
    enableRbacAuthorization: false
    enablePurgeProtection: false  // Allow purging for dev environment
    createMode: 'recover'  // Recover soft-deleted Key Vault if it exists
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
    accessPolicies: managedIdentityPrincipalId != '' ? [
      {
        tenantId: tenant().tenantId
        objectId: managedIdentityPrincipalId
        permissions: {
          secrets: [
            'Get'
            'List'
            'Set'
          ]
          keys: [
            'Get'
            'List'
          ]
          certificates: [
            'Get'
            'List'
          ]
        }
      }
    ] : []
  }
}

output keyVaultId string = keyVault.id
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri