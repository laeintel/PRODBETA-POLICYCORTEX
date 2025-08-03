// Simplified Container Apps module without Key Vault dependencies
param environment string
param location string
param tags object = {}
param containerAppsEnvironmentId string
param containerRegistryLoginServer string
param userAssignedIdentityId string
param keyVaultName string
param containerAppsEnvironmentDefaultDomain string
param jwtSecretKey string = 'development-secret-key-change-in-production'

// Only deploy API Gateway for now to avoid Key Vault issues
resource apiGateway 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-api-gateway-${environment}'
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
        targetPort: 8000
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
        corsPolicy: {
          allowedOrigins: [
            'http://localhost:3000'
            'http://localhost:5173'
            'https://*.azurecontainerapps.io'
          ]
          allowedMethods: [
            'GET'
            'POST'
            'PUT'
            'DELETE'
            'OPTIONS'
          ]
          allowedHeaders: [
            'Content-Type'
            'Authorization'
          ]
          allowCredentials: true
        }
      }
      registries: [
        {
          server: containerRegistryLoginServer
          identity: userAssignedIdentityId
        }
      ]
      // No Key Vault secrets - use environment variables directly
      secrets: [
        {
          name: 'jwt-secret-inline'
          value: jwtSecretKey
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistryLoginServer}/policortex001-api-gateway:latest'
          name: 'api-gateway'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'ENVIRONMENT'
              value: environment
            }
            {
              name: 'SERVICE_NAME'
              value: 'api_gateway'
            }
            {
              name: 'SERVICE_PORT'
              value: '8000'
            }
            {
              name: 'JWT_SECRET_KEY'
              secretRef: 'jwt-secret-inline'
            }
            {
              name: 'LOG_LEVEL'
              value: 'INFO'
            }
            // Placeholder values for services that don't need Key Vault
            {
              name: 'AZURE_CLIENT_ID'
              value: 'development-client-id'
            }
            {
              name: 'AZURE_TENANT_ID'
              value: 'development-tenant-id'
            }
            {
              name: 'REDIS_URL'
              value: 'redis://localhost:6379'
            }
            {
              name: 'DATABASE_URL'
              value: 'sqlite:///app/data/local.db'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
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

output apiGatewayFqdn string = apiGateway.properties.configuration.ingress.fqdn
output apiGatewayUrl string = 'https://${apiGateway.properties.configuration.ingress.fqdn}'