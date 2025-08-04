// Comprehensive Container Apps module for all PolicyCortex services
param environment string
param location string
param tags object = {}
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param userAssignedIdentityId string
param keyVaultName string
param keyVaultUri string
@secure()
param jwtSecretKey string

// Naming convention variables
var resourcePrefix = 'pcx'
var containerAppPrefix = 'ca-${resourcePrefix}'

// Service definitions with consistent naming
var services = [
  {
    name: '${containerAppPrefix}-gateway-${environment}'
    displayName: 'API Gateway'
    imageName: 'api-gateway'
    port: 8000
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 10
    env: [
      { name: 'SERVICE_NAME', value: 'api_gateway' }
      { name: 'SERVICE_PORT', value: '8000' }
      { name: 'AZURE_INTEGRATION_URL', value: 'http://${containerAppPrefix}-azureint-${environment}:8001' }
      { name: 'AI_ENGINE_URL', value: 'http://${containerAppPrefix}-ai-${environment}:8002' }
      { name: 'DATA_PROCESSING_URL', value: 'http://${containerAppPrefix}-dataproc-${environment}:8003' }
      { name: 'CONVERSATION_URL', value: 'http://${containerAppPrefix}-chat-${environment}:8004' }
      { name: 'NOTIFICATION_URL', value: 'http://${containerAppPrefix}-notify-${environment}:8005' }
    ]
  }
  {
    name: '${containerAppPrefix}-azureint-${environment}'
    displayName: 'Azure Integration'
    imageName: 'azure-integration'
    port: 8001
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    env: [
      { name: 'SERVICE_NAME', value: 'azure_integration' }
      { name: 'SERVICE_PORT', value: '8001' }
    ]
  }
  {
    name: '${containerAppPrefix}-ai-${environment}'
    displayName: 'AI Engine'
    imageName: 'ai-engine'
    port: 8002
    cpu: '1.0'
    memory: '2Gi'
    minReplicas: 1
    maxReplicas: 5
    env: [
      { name: 'SERVICE_NAME', value: 'ai_engine' }
      { name: 'SERVICE_PORT', value: '8002' }
    ]
  }
  {
    name: '${containerAppPrefix}-dataproc-${environment}'
    displayName: 'Data Processing'
    imageName: 'data-processing'
    port: 8003
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    env: [
      { name: 'SERVICE_NAME', value: 'data_processing' }
      { name: 'SERVICE_PORT', value: '8003' }
    ]
  }
  {
    name: '${containerAppPrefix}-chat-${environment}'
    displayName: 'Conversation Service'
    imageName: 'conversation'
    port: 8004
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 1
    maxReplicas: 5
    env: [
      { name: 'SERVICE_NAME', value: 'conversation' }
      { name: 'SERVICE_PORT', value: '8004' }
    ]
  }
  {
    name: '${containerAppPrefix}-notify-${environment}'
    displayName: 'Notification Service'
    imageName: 'notification'
    port: 8005
    cpu: '0.5'
    memory: '1Gi'
    minReplicas: 0
    maxReplicas: 3
    env: [
      { name: 'SERVICE_NAME', value: 'notification' }
      { name: 'SERVICE_PORT', value: '8005' }
    ]
  }
]

// Common environment variables for all services
var commonEnv = [
  { name: 'ENVIRONMENT', value: environment }
  { name: 'LOG_LEVEL', value: environment == 'prod' ? 'INFO' : 'DEBUG' }
  { name: 'AZURE_KEY_VAULT_URL', value: keyVaultUri }
]

// Common secrets
var commonSecrets = [
  {
    name: 'jwt-secret'
    value: jwtSecretKey
  }
]

// Deploy all backend services
@batchSize(1)
resource containerApps 'Microsoft.App/containerApps@2023-05-01' = [for service in services: {
  name: service.name
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
      ingress: {
        external: service.name == '${containerAppPrefix}-gateway-${environment}' ? true : false
        targetPort: service.port
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
        corsPolicy: service.name == '${containerAppPrefix}-gateway-${environment}' ? {
          allowedOrigins: [
            'http://localhost:3000'
            'http://localhost:5173'
            'https://*.azurecontainerapps.io'
          ]
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID']
          allowCredentials: true
        } : null
      }
      registries: [
        {
          server: containerRegistryLoginServer
          identity: userAssignedIdentityId
        }
      ]
      secrets: commonSecrets
    }
    template: {
      containers: [
        {
          // Use placeholder image - real images will be deployed by application pipeline
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: toLower(replace(service.displayName, ' ', '-'))  // DNS compliant container name
          resources: {
            cpu: json(service.cpu)
            memory: service.memory
          }
          env: concat(commonEnv, service.env, [
            {
              name: 'JWT_SECRET_KEY'
              secretRef: 'jwt-secret'
            }
          ])
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
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}]

// Deploy frontend
resource frontend 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${containerAppPrefix}-web-${environment}'
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
      ingress: {
        external: true
        targetPort: 80
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
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
          // Use placeholder image - real image will be deployed by application pipeline
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: 'frontend'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'VITE_API_BASE_URL'
              value: '/api'
            }
            {
              name: 'VITE_APP_TITLE'
              value: 'PolicyCortex'
            }
            {
              name: 'VITE_ENVIRONMENT'
              value: environment
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '100'
              }
            }
          }
        ]
      }
    }
  }
}

// Outputs
output containerAppNames array = [for i in range(0, length(services)): services[i].name]
output apiGatewayFqdn string = containerApps[0].properties.configuration.ingress.fqdn
output apiGatewayUrl string = 'https://${containerApps[0].properties.configuration.ingress.fqdn}'
output frontendFqdn string = frontend.properties.configuration.ingress.fqdn
output frontendUrl string = 'https://${frontend.properties.configuration.ingress.fqdn}'