// Container Apps Environment module
param environmentName string
param location string
param tags object = {}
param logAnalyticsWorkspaceId string
param subnetId string

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    // Simplified configuration for consumption tier - no VNet integration for now
    zoneRedundant: false
    // No workloadProfiles for consumption-based pricing
  }
}

output environmentId string = containerAppsEnvironment.id
output environmentName string = containerAppsEnvironment.name
output defaultDomain string = containerAppsEnvironment.properties.defaultDomain