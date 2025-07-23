// Container Apps module
param environment string
param location string
param tags object = {}
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param userAssignedIdentityId string
param keyVaultName string

// Service configurations
var services = [
  {
    name: 'api-gateway'
    port: 8000
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
    port: 3000
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
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/jwt-secret-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'managed-identity-client-id'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/managed-identity-client-id'
          identity: userAssignedIdentityId
        }
        {
          name: 'storage-account-name'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/storage-account-name'
          identity: userAssignedIdentityId
        }
        {
          name: 'application-insights-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/application-insights-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'cognitive-services-key'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/cognitive-services-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'cognitive-services-endpoint'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/cognitive-services-endpoint'
          identity: userAssignedIdentityId
        }
        {
          name: 'redis-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/redis-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'cosmos-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.${az.environment().suffixes.keyvaultDns}/secrets/cosmos-connection-string'
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
              value: environment
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
              name: 'PORT'
              value: string(service.port)
            }
          ]
          probes: service.name != 'frontend' ? [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 3
              failureThreshold: 3
            }
          ] : []
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