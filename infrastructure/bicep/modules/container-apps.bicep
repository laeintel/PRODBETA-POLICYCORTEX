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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfile: 'Consumption'
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
    workloadProfileName: service.workloadProfile == 'Consumption' ? null : service.workloadProfile
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
      secrets: []
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
              value: 'placeholder-client-id'
            }
            {
              name: 'VITE_AZURE_TENANT_ID'
              value: 'placeholder-tenant-id'
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
              value: 'placeholder-jwt-secret'
            }
            {
              name: 'ENCRYPTION_KEY'
              value: 'placeholder-encryption-key'
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: 'placeholder-client-id'
            }
            {
              name: 'AZURE_TENANT_ID'
              value: 'placeholder-tenant-id'
            }
            {
              name: 'AZURE_COSMOS_ENDPOINT'
              value: 'placeholder-cosmos-endpoint'
            }
            {
              name: 'AZURE_COSMOS_KEY'
              value: 'placeholder-cosmos-key'
            }
            {
              name: 'REDIS_CONNECTION_STRING'
              value: 'placeholder-redis-connection'
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT_NAME'
              value: 'placeholder-storage-account'
            }
            {
              name: 'COGNITIVE_SERVICES_KEY'
              value: 'placeholder-cognitive-key'
            }
            {
              name: 'COGNITIVE_SERVICES_ENDPOINT'
              value: 'placeholder-cognitive-endpoint'
            }
            {
              name: 'APPLICATION_INSIGHTS_CONNECTION_STRING'
              value: 'placeholder-insights-connection'
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