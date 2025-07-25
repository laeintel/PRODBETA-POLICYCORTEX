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
    port: 8000
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 2
    maxReplicas: 20
    ingress: true
    external: true
    workloadProfile: 'GeneralPurpose'
  }
  {
    name: 'ai-engine'
    port: 8002
    cpu: '2.0'
    memory: '4Gi'
    minReplicas: 2
    maxReplicas: 16
    ingress: false
    external: false
    workloadProfile: 'HighPerformance'
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
    }
    template: {
      containers: [
        {
          name: service.name
          image: service.name == 'frontend' ? 'nginx:alpine' : 'nginx:alpine'
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
              name: 'PORT'
              value: string(service.port)
            }
            {
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            {
              name: 'JWT_SECRET'
              value: 'placeholder-jwt-secret'
            }
            {
              name: 'ENCRYPTION_KEY'
              value: 'placeholder-encryption-key'
            }
          ]
        }
      ]
      scale: {
        minReplicas: service.minReplicas
        maxReplicas: service.maxReplicas
      }
    }
  }
}]

// Outputs
output containerAppNames array = [for service in services: 'ca-${service.name}-${environment}'] 