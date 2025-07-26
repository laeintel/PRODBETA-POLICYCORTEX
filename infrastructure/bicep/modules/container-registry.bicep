// Container Registry module
param registryName string
param location string
param tags object = {}
param managedIdentityPrincipalId string = ''

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  tags: tags
  sku: {
    name: 'Premium'
  }
  properties: {
    adminUserEnabled: true
    policies: {
      quarantinePolicy: {
        status: 'disabled'
      }
      trustPolicy: {
        type: 'Notary'
        status: 'disabled'
      }
      retentionPolicy: {
        days: 30
        status: 'enabled'
      }
      exportPolicy: {
        status: 'enabled'
      }
    }
    encryption: {
      status: 'disabled'
    }
    dataEndpointEnabled: false
    publicNetworkAccess: 'Enabled'
    networkRuleBypassOptions: 'AzureServices'
    zoneRedundancy: 'Disabled'
  }
}

// Role assignment for managed identity to pull images
resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (managedIdentityPrincipalId != '') {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentityPrincipalId, 'AcrPull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// Role assignment for managed identity to push images (for CI/CD scenarios)
resource acrPushRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (managedIdentityPrincipalId != '') {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentityPrincipalId, 'AcrPush')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8311e382-0749-4cb8-b61a-304f252e45ec') // AcrPush
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

output registryId string = containerRegistry.id
output registryName string = containerRegistry.name
output loginServer string = containerRegistry.properties.loginServer