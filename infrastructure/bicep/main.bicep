// Main Bicep template for policortex infrastructure
targetScope = 'subscription'

// Parameters
@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for resources')
param location string = 'East US'


@description('Owner of the resources')
param owner string = 'AeoliTech'

@description('List of allowed IP addresses for storage account access')
param allowedIps array = []

@description('Whether to create Terraform access policy for Key Vault')
param createTerraformAccessPolicy bool = true

@description('Whether to deploy Container Apps')
param deployContainerApps bool = false

@description('Secret key for JWT token signing')
@secure()
param jwtSecretKey string

// Data Services Parameters
@description('Whether to deploy SQL Server')
param deploySqlServer bool = true

@description('SQL Server administrator username')
param sqlAdminUsername string = 'sqladmin'

@description('Azure AD admin login for SQL Server')
param sqlAzureadAdminLogin string = 'admin@yourdomain.com'

@description('Azure AD admin object ID for SQL Server')
param sqlAzureadAdminObjectId string = '00000000-0000-0000-0000-000000000000'

@description('SKU for the SQL database')
param sqlDatabaseSku string = 'GP_S_Gen5_2'

@description('Maximum size of the SQL database in GB')
param sqlDatabaseMaxSizeGB int = 32

@description('Cosmos DB consistency level')
@allowed(['Eventual', 'Session', 'Strong', 'ConsistentPrefix', 'BoundedStaleness'])
param cosmosConsistencyLevel string = 'Session'

@description('Secondary region for Cosmos DB geo-replication')
param cosmosFailoverLocation string = 'West US 2'

@description('Maximum throughput for Cosmos DB database')
param cosmosMaxThroughput int = 4000

@description('Redis cache capacity')
param redisCapacity int = 2

@description('Redis cache SKU name')
@allowed(['Basic', 'Standard', 'Premium'])
param redisSKUName string = 'Standard'

// AI Services Parameters
@description('Whether to deploy ML workspace')
param deployMLWorkspace bool = true

@description('Whether to create a separate Container Registry for ML')
param createMLContainerRegistry bool = false

@description('VM size for the ML training cluster')
param trainingClusterVMSize string = 'Standard_DS3_v2'

@description('Maximum number of nodes in the training cluster')
param trainingClusterMaxNodes int = 4

@description('VM size for the compute instance')
param computeInstanceVMSize string = 'Standard_DS3_v2'

@description('SKU for Cognitive Services')
param cognitiveServicesSku string = 'S0'

@description('Whether to deploy Azure OpenAI service')
param deployOpenAI bool = true

@description('SKU for Azure OpenAI')
param openAISku string = 'S0'

// Monitoring Parameters
@description('Email addresses for critical alerts')
param criticalAlertEmails array = []

@description('Email addresses for warning alerts')
param warningAlertEmails array = []

@description('Email addresses for budget alerts')
param budgetAlertEmails array = []

@description('Monthly budget amount in USD')
param monthlyBudgetAmount int = 1000

// Variables
var commonTags = {
  Environment: environment
  Project: 'policortex'
  Owner: owner
  ManagedBy: 'Bicep'
}

var networkResourceGroupName = 'rg-policortex001-network-${environment}'
var appResourceGroupName = 'rg-policortex001-app-${environment}'

// Network Resource Group for networking infrastructure
resource networkResourceGroup 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: networkResourceGroupName
  location: location
  tags: union(commonTags, {
    ResourceType: 'Networking'
  })
}

// Application Resource Group for application resources
resource appResourceGroup 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: appResourceGroupName
  location: location
  tags: union(commonTags, {
    ResourceType: 'Application'
  })
}

// Storage account for application data
module storageAccount 'modules/storage.bicep' = {
  scope: appResourceGroup
  name: 'storageAccount'
  params: {
    storageAccountName: 'stpolicortex001${environment}'
    location: location
    tags: commonTags
    allowedIps: allowedIps
  }
}

// Container Registry
module containerRegistry 'modules/container-registry.bicep' = {
  scope: appResourceGroup
  name: 'containerRegistry'
  params: {
    registryName: 'crpolicortex001${environment}'
    location: location
    tags: commonTags
    managedIdentityPrincipalId: userIdentity.outputs.principalId
  }
  dependsOn: [
    userIdentity
  ]
}

// Key Vault
module keyVault 'modules/key-vault.bicep' = {
  scope: appResourceGroup
  name: 'keyVault-${uniqueString(deployment().name)}'
  params: {
    keyVaultName: 'kv-pcx001-${environment}02'
    location: location
    tags: commonTags
    createTerraformAccessPolicy: createTerraformAccessPolicy
    environment: environment
    managedIdentityPrincipalId: userIdentity.outputs.principalId
  }
  dependsOn: [
    userIdentity
  ]
}

// Log Analytics Workspace
module logAnalytics 'modules/log-analytics.bicep' = {
  scope: appResourceGroup
  name: 'logAnalytics'
  params: {
    workspaceName: 'law-policortex001-${environment}'
    location: location
    tags: commonTags
  }
}

// Application Insights
module applicationInsights 'modules/application-insights.bicep' = {
  scope: appResourceGroup
  name: 'applicationInsights'
  params: {
    appInsightsName: 'ai-policortex001-${environment}'
    location: location
    tags: commonTags
    workspaceResourceId: logAnalytics.outputs.workspaceId
  }
}

// User Assigned Identity for Container Apps
module userIdentity 'modules/user-identity.bicep' = {
  scope: appResourceGroup
  name: 'userIdentity'
  params: {
    identityName: 'id-policortex001-${environment}'
    location: location
    tags: commonTags
  }
}

// Networking module
module networking 'modules/networking.bicep' = {
  scope: networkResourceGroup
  name: 'networking'
  params: {
    environment: environment
    location: location
    tags: commonTags
  }
}

// Data Services module
module dataServices 'modules/data-services.bicep' = {
  scope: appResourceGroup
  name: 'dataServices-${uniqueString(deployment().name)}'
  params: {
    environment: environment
    location: location
    tags: commonTags
    keyVaultName: keyVault.outputs.keyVaultName
    subnetId: networking.outputs.dataServicesSubnetId
    privateEndpointsSubnetId: networking.outputs.privateEndpointsSubnetId
    privateDnsZones: networking.outputs.privateDnsZones
    deploySqlServer: deploySqlServer
    sqlAdminUsername: sqlAdminUsername
    sqlAzureadAdminLogin: sqlAzureadAdminLogin
    sqlAzureadAdminObjectId: sqlAzureadAdminObjectId
    sqlDatabaseSku: sqlDatabaseSku
    sqlDatabaseMaxSizeGB: sqlDatabaseMaxSizeGB
    cosmosConsistencyLevel: cosmosConsistencyLevel
    cosmosFailoverLocation: cosmosFailoverLocation
    cosmosMaxThroughput: cosmosMaxThroughput
    redisCapacity: redisCapacity
    redisSKUName: redisSKUName
  }
}

// AI Services module
module aiServices 'modules/ai-services.bicep' = {
  scope: appResourceGroup
  name: 'aiServices'
  params: {
    environment: environment
    location: location
    tags: commonTags
    keyVaultName: keyVault.outputs.keyVaultName
    privateEndpointsSubnetId: networking.outputs.privateEndpointsSubnetId
    privateDnsZones: networking.outputs.privateDnsZones
    storageAccountId: storageAccount.outputs.storageAccountId
    containerRegistryId: containerRegistry.outputs.registryId
    applicationInsightsId: applicationInsights.outputs.appInsightsId
    deployMLWorkspace: deployMLWorkspace
    createMLContainerRegistry: createMLContainerRegistry
    trainingClusterVMSize: trainingClusterVMSize
    trainingClusterMaxNodes: trainingClusterMaxNodes
    computeInstanceVMSize: computeInstanceVMSize
    cognitiveServicesSku: cognitiveServicesSku
    deployOpenAI: deployOpenAI
    openAISku: openAISku
  }
}

// Container Apps Environment
module containerAppsEnvironment 'modules/container-apps-environment.bicep' = {
  scope: appResourceGroup
  name: 'containerAppsEnvironment-${uniqueString(deployment().name)}'
  params: {
    environmentName: 'cae-policortex001-${environment}'
    location: location
    tags: commonTags
    logAnalyticsWorkspaceId: logAnalytics.outputs.workspaceId
    subnetId: networking.outputs.containerAppsSubnetId
  }
}

// Container Apps (conditional deployment - using simplified version)
module containerApps 'modules/container-apps-simple.bicep' = if (deployContainerApps) {
  scope: appResourceGroup
  name: 'containerApps'
  params: {
    environment: environment
    location: location
    tags: commonTags
    containerAppsEnvironmentId: containerAppsEnvironment.outputs.environmentId
    containerRegistryLoginServer: containerRegistry.outputs.loginServer
    userAssignedIdentityId: userIdentity.outputs.identityId
    keyVaultName: keyVault.outputs.keyVaultName
    jwtSecretKey: jwtSecretKey
    containerAppsEnvironmentDefaultDomain: containerAppsEnvironment.outputs.defaultDomain
  }
  dependsOn: [
    keyVaultSecrets
  ]
}

// Key Vault Secrets
module keyVaultSecrets 'modules/key-vault-secrets.bicep' = {
  scope: appResourceGroup
  name: 'keyVaultSecrets'
  params: {
    keyVaultName: keyVault.outputs.keyVaultName
    jwtSecretKey: jwtSecretKey
    managedIdentityClientId: userIdentity.outputs.clientId
    storageAccountName: storageAccount.outputs.storageAccountName
    applicationInsightsConnectionString: applicationInsights.outputs.connectionString
    cognitiveServicesKey: 'placeholder-key-retrieved-directly-from-cognitive-services'
    cognitiveServicesEndpoint: aiServices.outputs.cognitiveServicesEndpoint
    redisConnectionString: dataServices.outputs.redisConnectionString
    cosmosConnectionString: dataServices.outputs.cosmosConnectionString
    cosmosEndpoint: dataServices.outputs.cosmosEndpoint
    cosmosKey: dataServices.outputs.cosmosKey
    resourceGroupName: appResourceGroup.name
  }
  dependsOn: [
    dataServices
    aiServices
  ]
}

// Monitoring module
module monitoring 'modules/monitoring.bicep' = {
  scope: appResourceGroup
  name: 'monitoring'
  params: {
    environment: environment
    tags: commonTags
    criticalAlertEmails: criticalAlertEmails
    warningAlertEmails: warningAlertEmails
    budgetAlertEmails: budgetAlertEmails
    monthlyBudgetAmount: monthlyBudgetAmount
    resourceGroupName: appResourceGroupName
  }
}

// Outputs
output resourceGroupName string = appResourceGroup.name
output networkResourceGroupName string = networkResourceGroup.name
output storageAccountName string = storageAccount.outputs.storageAccountName
output containerRegistryName string = containerRegistry.outputs.registryName
output containerRegistryLoginServer string = containerRegistry.outputs.loginServer
output keyVaultName string = keyVault.outputs.keyVaultName
output logAnalyticsWorkspaceId string = logAnalytics.outputs.workspaceId
output containerAppEnvironmentName string = containerAppsEnvironment.outputs.environmentName
output userAssignedIdentityId string = userIdentity.outputs.identityId
output vnetId string = networking.outputs.vnetId