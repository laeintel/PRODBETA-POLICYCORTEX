// Container Apps module for initial deployment (uses hello-world placeholder images)
param environment string
param location string
param tags object = {}
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param userAssignedIdentityId string
param keyVaultName string
param containerAppsEnvironmentDefaultDomain string

// Service configurations with placeholder images
var services = [
  {
    name: 'api-gateway'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: true
    external: true
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'azure-integration'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: false
    external: false
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'ai-engine'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: false
    external: false
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'data-processing'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: false
    external: false
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'conversation'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: false
    external: false
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'notification'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: false
    external: false
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
  }
  {
    name: 'frontend'
    port: 80
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 2
    ingress: true
    external: true
    workloadProfile: 'Consumption'
    imageName: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
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
    }
    template: {
      containers: [
        {
          name: service.name
          image: service.imageName
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
output containerAppUrls array = [for (service, i) in services: service.ingress && service.external ? 'https://${containerApps[i].properties.configuration.ingress.fqdn}' : '']