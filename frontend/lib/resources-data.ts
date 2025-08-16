/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

// Comprehensive Azure Resources data with real-world resources

export interface ResourceDefinition {
  id: string
  name: string
  type: string
  resourceGroup: string
  subscription: string
  location: string
  status: 'Running' | 'Stopped' | 'Deallocated' | 'Failed' | 'Creating' | 'Deleting' | 'Updating'
  provisioningState: 'Succeeded' | 'Failed' | 'Canceled' | 'Accepted' | 'Creating' | 'Deleting' | 'Moving' | 'Running' | 'Updating'
  sku?: {
    name: string
    tier: string
    size?: string
    family?: string
    capacity?: number
  }
  tags: Record<string, string>
  createdTime: string
  changedTime: string
  costPerMonth: number
  compliance: {
    status: 'Compliant' | 'NonCompliant' | 'Unknown'
    violations: number
    policies: string[]
  }
  performance?: {
    cpu?: number
    memory?: number
    disk?: number
    network?: number
  }
  security?: {
    score: number
    issues: number
    recommendations: string[]
  }
  properties?: Record<string, any>
}

export const azureResources: ResourceDefinition[] = [
  // Virtual Machines
  {
    id: '/subscriptions/12345/resourceGroups/rg-production/providers/Microsoft.Compute/virtualMachines/vm-web-prod-001',
    name: 'vm-web-prod-001',
    type: 'Microsoft.Compute/virtualMachines',
    resourceGroup: 'rg-production',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard_D4s_v3',
      tier: 'Standard',
      size: 'D4s_v3'
    },
    tags: {
      Environment: 'Production',
      Application: 'WebApp',
      Owner: 'WebTeam',
      CostCenter: 'CC-001',
      Department: 'Engineering'
    },
    createdTime: '2024-06-15T10:30:00Z',
    changedTime: '2025-01-07T14:20:00Z',
    costPerMonth: 285.50,
    compliance: {
      status: 'NonCompliant',
      violations: 2,
      policies: ['VMDiskEncryption', 'AzureBackupEnabled']
    },
    performance: {
      cpu: 45,
      memory: 62,
      disk: 35,
      network: 28
    },
    security: {
      score: 75,
      issues: 3,
      recommendations: ['Enable disk encryption', 'Configure backup', 'Update OS patches']
    },
    properties: {
      osType: 'Linux',
      osVersion: 'Ubuntu 20.04',
      vmSize: 'Standard_D4s_v3',
      numberOfCores: 4,
      memoryInGB: 16,
      diskSizeGB: 128
    }
  },
  {
    id: '/subscriptions/12345/resourceGroups/rg-production/providers/Microsoft.Compute/virtualMachines/vm-app-prod-001',
    name: 'vm-app-prod-001',
    type: 'Microsoft.Compute/virtualMachines',
    resourceGroup: 'rg-production',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard_D8s_v3',
      tier: 'Standard',
      size: 'D8s_v3'
    },
    tags: {
      Environment: 'Production',
      Application: 'AppServer',
      Owner: 'AppTeam'
    },
    createdTime: '2024-06-20T11:15:00Z',
    changedTime: '2025-01-06T09:30:00Z',
    costPerMonth: 571.00,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    performance: {
      cpu: 72,
      memory: 85,
      disk: 45,
      network: 55
    },
    security: {
      score: 92,
      issues: 0,
      recommendations: []
    }
  },
  {
    id: '/subscriptions/12345/resourceGroups/rg-development/providers/Microsoft.Compute/virtualMachines/vm-dev-test-001',
    name: 'vm-dev-test-001',
    type: 'Microsoft.Compute/virtualMachines',
    resourceGroup: 'rg-development',
    subscription: 'Development',
    location: 'westus',
    status: 'Stopped',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard_B2s',
      tier: 'Standard',
      size: 'B2s'
    },
    tags: {
      Environment: 'Development',
      AutoShutdown: 'true'
    },
    createdTime: '2024-09-10T14:00:00Z',
    changedTime: '2025-01-08T08:00:00Z',
    costPerMonth: 30.40,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    performance: {
      cpu: 0,
      memory: 0,
      disk: 0,
      network: 0
    },
    security: {
      score: 88,
      issues: 1,
      recommendations: ['Remove public IP when not in use']
    }
  },

  // Storage Accounts
  {
    id: '/subscriptions/12345/resourceGroups/rg-production/providers/Microsoft.Storage/storageAccounts/stproddata001',
    name: 'stproddata001',
    type: 'Microsoft.Storage/storageAccounts',
    resourceGroup: 'rg-production',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard_GRS',
      tier: 'Standard'
    },
    tags: {
      Environment: 'Production',
      DataClassification: 'Confidential',
      Backup: 'Enabled'
    },
    createdTime: '2024-01-10T09:00:00Z',
    changedTime: '2024-12-15T11:30:00Z',
    costPerMonth: 125.80,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    security: {
      score: 95,
      issues: 0,
      recommendations: []
    },
    properties: {
      kind: 'StorageV2',
      accessTier: 'Hot',
      encryption: 'Microsoft.Storage',
      httpsOnly: true,
      blobPublicAccess: false,
      usedCapacityGB: 2450
    }
  },
  {
    id: '/subscriptions/12345/resourceGroups/rg-development/providers/Microsoft.Storage/storageAccounts/stdevtest001',
    name: 'stdevtest001',
    type: 'Microsoft.Storage/storageAccounts',
    resourceGroup: 'rg-development',
    subscription: 'Development',
    location: 'westus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard_LRS',
      tier: 'Standard'
    },
    tags: {
      Environment: 'Development'
    },
    createdTime: '2024-03-20T10:30:00Z',
    changedTime: '2024-11-20T14:15:00Z',
    costPerMonth: 24.50,
    compliance: {
      status: 'NonCompliant',
      violations: 2,
      policies: ['SecureTransferToStorageAccounts', 'StorageAccountPublicAccess']
    },
    security: {
      score: 68,
      issues: 2,
      recommendations: ['Enable HTTPS only', 'Disable public blob access']
    },
    properties: {
      kind: 'StorageV2',
      accessTier: 'Cool',
      httpsOnly: false,
      blobPublicAccess: true
    }
  },

  // SQL Databases
  {
    id: '/subscriptions/12345/resourceGroups/rg-database/providers/Microsoft.Sql/servers/sql-prod-001/databases/db-customers',
    name: 'db-customers',
    type: 'Microsoft.Sql/servers/databases',
    resourceGroup: 'rg-database',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'S3',
      tier: 'Standard',
      capacity: 100
    },
    tags: {
      Environment: 'Production',
      Application: 'CustomerPortal',
      DataClassification: 'PII'
    },
    createdTime: '2024-02-01T08:00:00Z',
    changedTime: '2024-12-20T16:45:00Z',
    costPerMonth: 146.95,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    security: {
      score: 94,
      issues: 0,
      recommendations: []
    },
    properties: {
      collation: 'SQL_Latin1_General_CP1_CI_AS',
      maxSizeBytes: 268435456000,
      status: 'Online',
      databaseId: 'db-001',
      currentServiceObjectiveName: 'S3',
      requestedServiceObjectiveName: 'S3',
      defaultSecondaryLocation: 'westus',
      catalogCollation: 'SQL_Latin1_General_CP1_CI_AS',
      zoneRedundant: false,
      earliestRestoreDate: '2025-01-01T00:00:00Z',
      readScale: 'Disabled',
      currentBackupStorageRedundancy: 'Geo',
      requestedBackupStorageRedundancy: 'Geo'
    }
  },
  {
    id: '/subscriptions/12345/resourceGroups/rg-database/providers/Microsoft.Sql/servers/sql-prod-001/databases/db-analytics',
    name: 'db-analytics',
    type: 'Microsoft.Sql/servers/databases',
    resourceGroup: 'rg-database',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'P2',
      tier: 'Premium',
      capacity: 250
    },
    tags: {
      Environment: 'Production',
      Application: 'Analytics'
    },
    createdTime: '2024-03-15T09:30:00Z',
    changedTime: '2024-12-18T13:20:00Z',
    costPerMonth: 930.00,
    compliance: {
      status: 'NonCompliant',
      violations: 1,
      policies: ['SQLServerAuditing']
    },
    security: {
      score: 82,
      issues: 1,
      recommendations: ['Enable auditing']
    }
  },

  // App Services
  {
    id: '/subscriptions/12345/resourceGroups/rg-webapp/providers/Microsoft.Web/sites/app-portal-prod',
    name: 'app-portal-prod',
    type: 'Microsoft.Web/sites',
    resourceGroup: 'rg-webapp',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'P2v3',
      tier: 'PremiumV3',
      size: 'P2v3',
      family: 'Pv3',
      capacity: 2
    },
    tags: {
      Environment: 'Production',
      Application: 'CustomerPortal'
    },
    createdTime: '2024-04-01T10:00:00Z',
    changedTime: '2025-01-07T15:30:00Z',
    costPerMonth: 420.00,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    performance: {
      cpu: 35,
      memory: 48,
      network: 65
    },
    security: {
      score: 91,
      issues: 0,
      recommendations: []
    },
    properties: {
      state: 'Running',
      hostNames: ['app-portal-prod.azurewebsites.net', 'portal.company.com'],
      repositorySiteName: 'app-portal-prod',
      usageState: 'Normal',
      enabled: true,
      enabledHostNames: ['app-portal-prod.azurewebsites.net', 'portal.company.com'],
      availabilityState: 'Normal',
      sslCertificates: ['portal.company.com'],
      serverFarmId: '/subscriptions/12345/resourceGroups/rg-webapp/providers/Microsoft.Web/serverfarms/asp-prod-001',
      reserved: false,
      isXenon: false,
      hyperV: false,
      siteConfig: {
        numberOfWorkers: 2,
        defaultDocuments: ['index.html'],
        netFrameworkVersion: 'v6.0',
        phpVersion: '',
        pythonVersion: '',
        nodeVersion: '18-lts',
        powerShellVersion: '',
        linuxFxVersion: '',
        windowsFxVersion: null,
        requestTracingEnabled: false,
        remoteDebuggingEnabled: false,
        remoteDebuggingVersion: null,
        httpLoggingEnabled: true,
        // azureMonitorLogCategories duplicated; keep a single occurrence
        acrUseManagedIdentityCreds: false,
        acrUserManagedIdentityID: null,
        logsDirectorySizeLimit: 100,
        detailedErrorLoggingEnabled: true,
        publishingUsername: '$app-portal-prod',
        publishingPassword: null,
        appSettings: null,
        azureStorageAccounts: null,
        metadata: null,
        connectionStrings: null,
        machineKey: null,
        handlerMappings: null,
        documentRoot: null,
        scmType: 'GitHub',
        use32BitWorkerProcess: false,
        webSocketsEnabled: true,
        alwaysOn: true,
        javaVersion: null,
        javaContainer: null,
        javaContainerVersion: null,
        appCommandLine: '',
        managedPipelineMode: 'Integrated',
        virtualApplications: null,
        winAuthAdminState: 0,
        winAuthTenantState: 0,
        customAppPoolIdentityAdminState: false,
        customAppPoolIdentityTenantState: false,
        runtimeADUser: null,
        runtimeADUserPassword: null,
        loadBalancing: 'LeastRequests',
        routingRules: [],
        experiments: null,
        limits: null,
        autoHealEnabled: true,
        autoHealRules: null,
        tracingOptions: null,
        vnetName: '',
        vnetRouteAllEnabled: false,
        vnetPrivatePortsCount: 0,
        publicNetworkAccess: null,
        cors: null,
        push: null,
        apiDefinition: null,
        apiManagementConfig: null,
        autoSwapSlotName: null,
        localMySqlEnabled: false,
        managedServiceIdentityId: null,
        xManagedServiceIdentityId: null,
        keyVaultReferenceIdentity: null,
        ipSecurityRestrictions: null,
        ipSecurityRestrictionsDefaultAction: null,
        scmIpSecurityRestrictions: null,
        scmIpSecurityRestrictionsDefaultAction: null,
        scmIpSecurityRestrictionsUseMain: false,
        http20Enabled: true,
        minTlsVersion: '1.2',
        minTlsCipherSuite: null,
        supportedTlsCipherSuites: null,
        scmMinTlsVersion: '1.2',
        ftpsState: 'FtpsOnly',
        preWarmedInstanceCount: 0,
        functionAppScaleLimit: null,
        elasticWebAppScaleLimit: null,
        healthCheckPath: '/health',
        fileChangeAuditEnabled: false,
        functionsRuntimeScaleMonitoringEnabled: false,
        websiteTimeZone: null,
        minimumElasticInstanceCount: 0,
        azureMonitorLogCategories: null,
        appServiceLogs: null,
        sitePort: null
      }
    }
  },

  // Key Vaults
  {
    id: '/subscriptions/12345/resourceGroups/rg-security/providers/Microsoft.KeyVault/vaults/kv-prod-secrets',
    name: 'kv-prod-secrets',
    type: 'Microsoft.KeyVault/vaults',
    resourceGroup: 'rg-security',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'standard',
      tier: 'Standard'
    },
    tags: {
      Environment: 'Production',
      Purpose: 'Secrets'
    },
    createdTime: '2024-01-05T08:30:00Z',
    changedTime: '2024-12-10T12:00:00Z',
    costPerMonth: 5.00,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    security: {
      score: 98,
      issues: 0,
      recommendations: []
    },
    properties: {
      tenantId: '12345678-1234-1234-1234-123456789012',
      enabledForDeployment: true,
      enabledForDiskEncryption: true,
      enabledForTemplateDeployment: true,
      enableSoftDelete: true,
      softDeleteRetentionInDays: 90,
      enableRbacAuthorization: true,
      vaultUri: 'https://kv-prod-secrets.vault.azure.net/',
      provisioningState: 'Succeeded',
      publicNetworkAccess: 'Disabled'
    }
  },

  // Network Security Groups
  {
    id: '/subscriptions/12345/resourceGroups/rg-network/providers/Microsoft.Network/networkSecurityGroups/nsg-prod-web',
    name: 'nsg-prod-web',
    type: 'Microsoft.Network/networkSecurityGroups',
    resourceGroup: 'rg-network',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    tags: {
      Environment: 'Production',
      Purpose: 'WebTier'
    },
    createdTime: '2024-01-20T11:00:00Z',
    changedTime: '2024-11-15T14:30:00Z',
    costPerMonth: 0,
    compliance: {
      status: 'NonCompliant',
      violations: 1,
      policies: ['NSGFlowLogsEnabled']
    },
    security: {
      score: 85,
      issues: 1,
      recommendations: ['Enable flow logs']
    },
    properties: {
      securityRules: [
        {
          name: 'AllowHTTPS',
          priority: 100,
          sourceAddressPrefix: '*',
          sourcePortRange: '*',
          destinationAddressPrefix: '*',
          destinationPortRange: '443',
          protocol: 'Tcp',
          access: 'Allow',
          direction: 'Inbound'
        },
        {
          name: 'AllowHTTP',
          priority: 101,
          sourceAddressPrefix: '*',
          sourcePortRange: '*',
          destinationAddressPrefix: '*',
          destinationPortRange: '80',
          protocol: 'Tcp',
          access: 'Allow',
          direction: 'Inbound'
        }
      ]
    }
  },

  // Load Balancers
  {
    id: '/subscriptions/12345/resourceGroups/rg-network/providers/Microsoft.Network/loadBalancers/lb-prod-web',
    name: 'lb-prod-web',
    type: 'Microsoft.Network/loadBalancers',
    resourceGroup: 'rg-network',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard',
      tier: 'Regional'
    },
    tags: {
      Environment: 'Production'
    },
    createdTime: '2024-02-10T09:15:00Z',
    changedTime: '2024-12-05T10:20:00Z',
    costPerMonth: 18.25,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    security: {
      score: 90,
      issues: 0,
      recommendations: []
    }
  },

  // Container Instances
  {
    id: '/subscriptions/12345/resourceGroups/rg-containers/providers/Microsoft.ContainerInstance/containerGroups/aci-batch-processor',
    name: 'aci-batch-processor',
    type: 'Microsoft.ContainerInstance/containerGroups',
    resourceGroup: 'rg-containers',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Standard',
      tier: 'Standard'
    },
    tags: {
      Environment: 'Production',
      Application: 'BatchProcessing'
    },
    createdTime: '2024-07-01T12:00:00Z',
    changedTime: '2025-01-08T08:30:00Z',
    costPerMonth: 45.60,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    performance: {
      cpu: 25,
      memory: 40
    },
    security: {
      score: 88,
      issues: 0,
      recommendations: []
    },
    properties: {
      containers: [
        {
          name: 'batch-processor',
          image: 'mycompany.azurecr.io/batch-processor:latest',
          cpu: 2,
          memoryInGB: 4
        }
      ],
      osType: 'Linux',
      restartPolicy: 'OnFailure'
    }
  },

  // Kubernetes Service
  {
    id: '/subscriptions/12345/resourceGroups/rg-containers/providers/Microsoft.ContainerService/managedClusters/aks-prod-cluster',
    name: 'aks-prod-cluster',
    type: 'Microsoft.ContainerService/managedClusters',
    resourceGroup: 'rg-containers',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Base',
      tier: 'Standard'
    },
    tags: {
      Environment: 'Production',
      Application: 'Microservices'
    },
    createdTime: '2024-05-01T10:00:00Z',
    changedTime: '2024-12-28T16:45:00Z',
    costPerMonth: 584.00,
    compliance: {
      status: 'NonCompliant',
      violations: 1,
      policies: ['KubernetesRBAC']
    },
    performance: {
      cpu: 68,
      memory: 72
    },
    security: {
      score: 86,
      issues: 1,
      recommendations: ['Configure RBAC properly']
    },
    properties: {
      kubernetesVersion: '1.28.3',
      dnsPrefix: 'aks-prod-cluster',
      fqdn: 'aks-prod-cluster-12345.eastus.azmk8s.io',
      agentPoolProfiles: [
        {
          name: 'nodepool1',
          count: 3,
          vmSize: 'Standard_D4s_v3',
          osDiskSizeGB: 128,
          osDiskType: 'Managed',
          vnetSubnetID: '/subscriptions/12345/resourceGroups/rg-network/providers/Microsoft.Network/virtualNetworks/vnet-prod/subnets/subnet-aks',
          maxPods: 110,
          type: 'VirtualMachineScaleSets',
          orchestratorVersion: '1.28.3',
          nodeLabels: {},
          mode: 'System',
          osType: 'Linux'
        }
      ],
      nodeResourceGroup: 'MC_rg-containers_aks-prod-cluster_eastus',
      enableRBAC: true,
      networkProfile: {
        networkPlugin: 'azure',
        networkPolicy: 'azure',
        serviceCidr: '10.0.0.0/16',
        dnsServiceIP: '10.0.0.10',
        dockerBridgeCidr: '172.17.0.1/16'
      }
    }
  },

  // Cosmos DB
  {
    id: '/subscriptions/12345/resourceGroups/rg-database/providers/Microsoft.DocumentDB/databaseAccounts/cosmos-prod-db',
    name: 'cosmos-prod-db',
    type: 'Microsoft.DocumentDB/databaseAccounts',
    resourceGroup: 'rg-database',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    tags: {
      Environment: 'Production',
      Application: 'NoSQL'
    },
    createdTime: '2024-06-10T11:30:00Z',
    changedTime: '2024-12-22T14:00:00Z',
    costPerMonth: 235.00,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    security: {
      score: 93,
      issues: 0,
      recommendations: []
    },
    properties: {
      databaseAccountOfferType: 'Standard',
      consistencyPolicy: {
        defaultConsistencyLevel: 'Session'
      },
      locations: [
        {
          locationName: 'East US',
          failoverPriority: 0,
          isZoneRedundant: true
        }
      ]
    }
  },

  // Redis Cache
  {
    id: '/subscriptions/12345/resourceGroups/rg-cache/providers/Microsoft.Cache/Redis/redis-prod-cache',
    name: 'redis-prod-cache',
    type: 'Microsoft.Cache/Redis',
    resourceGroup: 'rg-cache',
    subscription: 'Production',
    location: 'eastus',
    status: 'Running',
    provisioningState: 'Succeeded',
    sku: {
      name: 'Premium',
      tier: 'Premium',
      family: 'P',
      capacity: 1
    },
    tags: {
      Environment: 'Production',
      Purpose: 'SessionCache'
    },
    createdTime: '2024-04-20T13:00:00Z',
    changedTime: '2024-12-15T09:30:00Z',
    costPerMonth: 310.00,
    compliance: {
      status: 'Compliant',
      violations: 0,
      policies: []
    },
    performance: {
      cpu: 15,
      memory: 45
    },
    security: {
      score: 96,
      issues: 0,
      recommendations: []
    },
    properties: {
      redisVersion: '6.0',
      sku: {
        name: 'Premium',
        family: 'P',
        capacity: 1
      },
      enableNonSslPort: false,
      minimumTlsVersion: '1.2',
      publicNetworkAccess: 'Disabled'
    }
  }
]

// Helper functions
export function getResourcesByType(type: string): ResourceDefinition[] {
  if (type === 'all') return azureResources
  return azureResources.filter(r => r.type.includes(type))
}

export function getResourceStatistics() {
  const total = azureResources.length
  const running = azureResources.filter(r => r.status === 'Running').length
  const compliant = azureResources.filter(r => r.compliance.status === 'Compliant').length
  const nonCompliant = azureResources.filter(r => r.compliance.status === 'NonCompliant').length
  const totalCost = azureResources.reduce((sum, r) => sum + r.costPerMonth, 0)
  
  const byType = azureResources.reduce((acc, r) => {
    const type = r.type.split('/').pop() || 'Unknown'
    acc[type] = (acc[type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const bySubscription = azureResources.reduce((acc, r) => {
    acc[r.subscription] = (acc[r.subscription] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const byLocation = azureResources.reduce((acc, r) => {
    acc[r.location] = (acc[r.location] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const costByResourceGroup = azureResources.reduce((acc, r) => {
    acc[r.resourceGroup] = (acc[r.resourceGroup] || 0) + r.costPerMonth
    return acc
  }, {} as Record<string, number>)

  return {
    total,
    running,
    compliant,
    nonCompliant,
    totalCost,
    byType,
    bySubscription,
    byLocation,
    costByResourceGroup,
    avgSecurityScore: Math.round(
      azureResources
        .filter(r => r.security)
        .reduce((sum, r) => sum + (r.security?.score || 0), 0) / 
      azureResources.filter(r => r.security).length
    )
  }
}

export function getResourceGroups(): string[] {
  return [...new Set(azureResources.map(r => r.resourceGroup))].sort()
}

export function getSubscriptions(): string[] {
  return [...new Set(azureResources.map(r => r.subscription))].sort()
}

export function getLocations(): string[] {
  return [...new Set(azureResources.map(r => r.location))].sort()
}