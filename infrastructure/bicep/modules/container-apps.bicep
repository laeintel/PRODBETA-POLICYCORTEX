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
      registries: [
        {
          server: containerRegistryLoginServer
          username: 'crpolicortex001dev'
          passwordSecretRef: 'acr-password'
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
          name: 'acr-password'
          keyVaultUrl: 'https://${keyVaultName}.vault.azure.net/secrets/acr-password'
          identity: userAssignedIdentityId
        }
      ]
    }
    template: {
      containers: [
        {
          name: service.name
          image: '${containerRegistryLoginServer}/policortex001-${service.name}:latest'
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