// Container Apps module with real images
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
    port: 8000
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 2
    maxReplicas: 20
    ingress: true
    external: true
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-api-gateway'
  }
  {
    name: 'azure-integration'
    port: 8001
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 10
    ingress: false
    external: false
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-azure-integration'
  }
  {
    name: 'ai-engine'
    port: 8002
    cpu: '2.0'
    memory: '4Gi'
    minReplicas: 1
    maxReplicas: 8
    ingress: false
    external: false
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-ai-engine'
  }
  {
    name: 'data-processing'
    port: 8003
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 10
    ingress: false
    external: false
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-data-processing'
  }
  {
    name: 'conversation'
    port: 8004
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 10
    ingress: false
    external: false
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-conversation'
  }
  {
    name: 'notification'
    port: 8005
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    ingress: false
    external: false
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-notification'
  }
  {
    name: 'frontend'
    port: 8080
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 2
    maxReplicas: 10
    ingress: true
    external: true
    workloadProfile: 'GeneralPurpose'
    imageName: 'policortex001-frontend'
  }
]

// Create Container Apps
resource containerApps 'Microsoft.App/containerApps@2024-03-01' = [for service in services: {
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
    workloadProfileName: service.workloadProfile
    configuration: {
      ingress: service.ingress ? {
        external: service.external
        targetPort: service.port
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
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
          name: 'jwt-secret'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/jwt-secret'
          identity: userAssignedIdentityId
        }
        {
          name: 'encryption-key'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/encryption-key'
          identity: userAssignedIdentityId
        }
        {
          name: 'azure-client-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/azure-client-id'
          identity: userAssignedIdentityId
        }
        {
          name: 'azure-tenant-id'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/azure-tenant-id'
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
          name: 'redis-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/redis-connection-string'
          identity: userAssignedIdentityId
        }
        {
          name: 'storage-account-name'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/storage-account-name'
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
          name: 'application-insights-connection-string'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/application-insights-connection-string'
          identity: userAssignedIdentityId
        }
      ]
    }
    template: {
      containers: [
        {
          name: service.name
          image: '${containerRegistryLoginServer}/${service.imageName}:latest'
          resources: {
            cpu: json(service.cpu)
            memory: service.memory
          }
          probes: service.ingress ? [
            {
              type: 'Startup'
              httpGet: {
                path: service.name == 'frontend' ? '/' : '/health'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              successThreshold: 1
              failureThreshold: 30
            }
            {
              type: 'Liveness'
              httpGet: {
                path: service.name == 'frontend' ? '/' : '/health'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: service.name == 'frontend' ? '/' : '/ready'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 5
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ] : [
            {
              type: 'Startup'
              httpGet: {
                path: '/health'
                port: service.port
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              successThreshold: 1
              failureThreshold: 30
            }
          ]
          env: service.name == 'frontend' ? [
            {
              name: 'ENVIRONMENT'
              value: environment
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
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'VITE_API_BASE_URL'
              value: 'https://ca-api-gateway-${environment}.${containerAppsEnvironmentDefaultDomain}/api'
            }
            {
              name: 'VITE_WS_URL'
              value: 'wss://ca-api-gateway-${environment}.${containerAppsEnvironmentDefaultDomain}/ws'
            }
            {
              name: 'VITE_AZURE_CLIENT_ID'
              secretRef: 'azure-client-id'
            }
            {
              name: 'VITE_AZURE_TENANT_ID'
              secretRef: 'azure-tenant-id'
            }
            {
              name: 'VITE_AZURE_REDIRECT_URI'
              value: 'https://ca-frontend-${environment}.${containerAppsEnvironmentDefaultDomain}'
            }
            {
              name: 'VITE_APP_VERSION'
              value: '1.0.0'
            }
          ] : [
            {
              name: 'ENVIRONMENT'
              value: environment
            }
            {
              name: 'SERVICE_NAME'
              value: service.name
            }
            {
              name: 'SERVICE_PORT'
              value: string(service.port)
            }
            {
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'JWT_SECRET_KEY'
              secretRef: 'jwt-secret'
            }
            {
              name: 'ENCRYPTION_KEY'
              secretRef: 'encryption-key'
            }
            {
              name: 'AZURE_CLIENT_ID'
              secretRef: 'azure-client-id'
            }
            {
              name: 'AZURE_TENANT_ID'
              secretRef: 'azure-tenant-id'
            }
            {
              name: 'AZURE_COSMOS_ENDPOINT'
              secretRef: 'cosmos-endpoint'
            }
            {
              name: 'AZURE_COSMOS_KEY'
              secretRef: 'cosmos-key'
            }
            {
              name: 'REDIS_CONNECTION_STRING'
              secretRef: 'redis-connection-string'
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_NAME'
              secretRef: 'storage-account-name'
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
              name: 'APPLICATION_INSIGHTS_CONNECTION_STRING'
              secretRef: 'application-insights-connection-string'
            }
          ]
        }
      ]
      scale: {
        minReplicas: service.minReplicas
        maxReplicas: service.maxReplicas
        rules: []
      }
    }
  }
}]

// Outputs
output containerAppNames array = [for service in services: 'ca-${service.name}-${environment}']
output containerAppUrls array = [for (service, i) in services: service.ingress && service.external ? 'https://${containerApps[i].properties.configuration.ingress.fqdn}' : '']