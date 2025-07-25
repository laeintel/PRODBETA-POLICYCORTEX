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
]

// Create Container Apps
resource containerApps 'Microsoft.App/containerApps@2024-03-01' = [for service in services: {
  name: 'ca-${service.name}-${environment}'
  location: location
  tags: tags
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
      secrets: [
        {
          name: 'jwt-secret'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/jwt-secret'
        }
        {
          name: 'encryption-key'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/encryption-key'
        }
      ]
    }
    template: {
      containers: [
        {
          name: service.name
          image: '${containerRegistryLoginServer}/${service.name}:latest'
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
              secretRef: 'jwt-secret'
            }
            {
              name: 'ENCRYPTION_KEY'
              secretRef: 'encryption-key'
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