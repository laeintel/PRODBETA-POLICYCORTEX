// Container Apps module
param environment string
param location string
param tags object = {}
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param userAssignedIdentityId string
param keyVaultName string
param containerAppsEnvironmentDefaultDomain string

// Service configurations
var services = [
  {
    name: 'api-gateway'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    ingress: true
    external: true
  }
  {
    name: 'azure-integration'
    port: 8001
    cpu: '0.25'
    memory: '0.5Gi'
    minReplicas: 1
    maxReplicas: 3
    ingress: false
    external: false
  }
  {
    name: 'ai-engine'
    port: 8002
    cpu: '1'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 10
    ingress: false
    external: false
  }
  {
    name: 'data-processing'
    port: 8003
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    ingress: false
    external: false
  }
  {
    name: 'conversation'
    port: 8004
    cpu: '0.25'
    memory: '0.5Gi'
    minReplicas: 1
    maxReplicas: 3
    ingress: false
    external: false
  }
  {
    name: 'notification'
    port: 8005
    cpu: '0.25'
    memory: '0.5Gi'
    minReplicas: 1
    maxReplicas: 3
    ingress: false
    external: false
  }
  {
    name: 'frontend'
    port: 80
    cpu: '0.25'
    memory: '0.5Gi'
    minReplicas: 1
    maxReplicas: 5
    ingress: true
    external: true
  }
]

// Container Apps
resource containerApps 'Microsoft.App/containerApps@2023-05-01' = [for service in services: {
  name: 'ca-${service.name}-${environment}'
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentityId}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: service.ingress ? {
        external: service.external
        targetPort: service.port
        transport: 'auto'
        allowInsecure: false
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      } : null
      registries: [
        {
          server: containerRegistryLoginServer
          identity: userAssignedIdentityId
        }
      ]
      secrets: [
        {
          name: 'jwt-secret-key'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/jwt-secret-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'managed-identity-client-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/managed-identity-client-id'
          identity: userAssignedIdentityId
        }
        {
          name: 'storage-account-name'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/storage-account-name'
          identity: userAssignedIdentityId
        }
        {
          name: 'application-insights-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/application-insights-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'cognitive-services-key'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/cognitive-services-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'cognitive-services-endpoint'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/cognitive-services-endpoint'
          identity: userAssignedIdentityId
        }
        {
          name: 'redis-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/redis-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'cosmos-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/cosmos-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'cosmos-endpoint'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/cosmos-endpoint'
          identity: userAssignedIdentityId
        }
        {
          name: 'cosmos-key'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/cosmos-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'sql-server'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/sql-server'
          identity: userAssignedIdentityId
        }
        {
          name: 'sql-username'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/sql-username'
          identity: userAssignedIdentityId
        }
        {
          name: 'sql-password'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/sql-password'
          identity: userAssignedIdentityId
        }
        {
          name: 'key-vault-name'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/key-vault-name'
          identity: userAssignedIdentityId
        }
        {
          name: 'ml-workspace-name'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/ml-workspace-name'
          identity: userAssignedIdentityId
        }
        {
          name: 'tenant-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/tenant-id'
          identity: userAssignedIdentityId
        }
        {
          name: 'client-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/client-id'
          identity: userAssignedIdentityId
        }
        {
          name: 'client-secret'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/client-secret'
          identity: userAssignedIdentityId
        }
        {
          name: 'resource-group'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/resource-group'
          identity: userAssignedIdentityId
        }
        {
          name: 'service-bus-namespace'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/service-bus-namespace'
          identity: userAssignedIdentityId
        }
        {
          name: 'subscription-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/subscription-id'
          identity: userAssignedIdentityId
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistryLoginServer}/policortex001-${service.name}:latest'
          name: service.name
          resources: {
            cpu: json(service.cpu)
            memory: service.memory
          }
          env: [
            {
              name: 'ENVIRONMENT'
              value: environment == 'dev' ? 'development' : environment == 'prod' ? 'production' : environment
            }
            {
              name: 'JWT_SECRET_KEY'
              secretRef: 'jwt-secret-key'
            }
            {
              name: 'AZURE_CLIENT_ID'
              secretRef: 'managed-identity-client-id'
            }
            {
              name: 'STORAGE_ACCOUNT_NAME'
              secretRef: 'storage-account-name'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'application-insights-connection-string'
            }
            {
              name: 'COGNITIVE_SERVICES_KEY'
              secretRef: 'cognitive-services-key'
            }
            {
              name: 'COGNITIVE_SERVICES_ENDPOINT'
              secretRef: 'cognitive-services-endpoint'
            }
            {
              name: 'REDIS_CONNECTION_STRING'
              secretRef: 'redis-connection-string'
            }
            {
              name: 'COSMOS_CONNECTION_STRING'
              secretRef: 'cosmos-connection-string'
            }
            {
              name: 'COSMOS_ENDPOINT'
              secretRef: 'cosmos-endpoint'
            }
            {
              name: 'COSMOS_KEY'
              secretRef: 'cosmos-key'
            }
            {
              name: 'SQL_SERVER'
              secretRef: 'sql-server'
            }
            {
              name: 'SQL_USERNAME'
              secretRef: 'sql-username'
            }
            {
              name: 'SQL_PASSWORD'
              secretRef: 'sql-password'
            }
            {
              name: 'KEY_VAULT_NAME'
              secretRef: 'key-vault-name'
            }
            {
              name: 'ML_WORKSPACE_NAME'
              secretRef: 'ml-workspace-name'
            }
            {
              name: 'TENANT_ID'
              secretRef: 'tenant-id'
            }
            {
              name: 'CLIENT_ID'
              secretRef: 'client-id'
            }
            {
              name: 'CLIENT_SECRET'
              secretRef: 'client-secret'
            }
            {
              name: 'RESOURCE_GROUP'
              secretRef: 'resource-group'
            }
            {
              name: 'SERVICE_BUS_NAMESPACE'
              secretRef: 'service-bus-namespace'
            }
            {
              name: 'SUBSCRIPTION_ID'
              secretRef: 'subscription-id'
            }
            {
              name: 'SERVICE_NAME'
              value: service.name
            }
            {
              name: 'PORT'
              value: string(service.port)
            }
            {
              name: 'VITE_AZURE_CLIENT_ID'
              secretRef: 'managed-identity-client-id'
            }
            {
              name: 'VITE_AZURE_TENANT_ID'
              value: tenant().tenantId
            }
            {
              name: 'VITE_API_BASE_URL'
              value: service.name == 'frontend' ? 'https://ca-api-gateway-${environment}.${containerAppsEnvironmentDefaultDomain}' : ''
            }
          ]
          // Health probes temporarily disabled for troubleshooting
          probes: []
        }
      ]
      scale: {
        minReplicas: service.minReplicas
        maxReplicas: service.maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '30'
              }
            }
          }
        ]
      }
    }
  }
}]

// Outputs
output containerAppUrls array = [for (service, i) in services: service.ingress && service.external ? {
  name: service.name
  url: 'https://${containerApps[i].properties.configuration.ingress.fqdn}'
} : {
  name: service.name
  url: 'internal'
}]

output apiGatewayUrl string = 'https://${containerApps[0].properties.configuration.ingress.fqdn}'
output frontendUrl string = 'https://${containerApps[6].properties.configuration.ingress.fqdn}'