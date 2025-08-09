// Comprehensive Azure Policy data with real-world policies

export interface PolicyDefinition {
  id: string
  name: string
  displayName: string
  description: string
  category: string
  type: 'BuiltIn' | 'Custom' | 'Static'
  effect: 'Deny' | 'Audit' | 'AuditIfNotExists' | 'DeployIfNotExists' | 'Append' | 'Modify'
  status: 'Active' | 'Disabled' | 'Draft'
  version: string
  parameters: Record<string, any>
  resourceTypes: string[]
  metadata: {
    createdBy: string
    createdOn: string
    updatedBy: string
    updatedOn: string
    source: string
  }
  compliance: {
    compliant: number
    nonCompliant: number
    exempt: number
    unknown: number
    percentage: number
  }
  assignments: number
  remediationTasks?: number
  policyRule?: any
  scope?: string[]
  excludedScopes?: string[]
  complianceFrameworks?: string[]
}

export const azurePolicies: PolicyDefinition[] = [
  // Security Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/404c3081-a854-4457-ae30-26a93ef643f9',
    name: 'SecureTransferToStorageAccounts',
    displayName: 'Secure transfer to storage accounts should be enabled',
    description: 'Audit requirement of Secure transfer in your storage account. Secure transfer is an option that forces your storage account to accept requests only from secure connections (HTTPS).',
    category: 'Security',
    type: 'BuiltIn',
    effect: 'Audit',
    status: 'Active',
    version: '2.0.0',
    parameters: {
      effect: {
        type: 'String',
        defaultValue: 'Audit',
        allowedValues: ['Audit', 'Deny', 'Disabled']
      }
    },
    resourceTypes: ['Microsoft.Storage/storageAccounts'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-01-15T10:30:00Z',
      updatedBy: 'Microsoft',
      updatedOn: '2024-12-01T14:20:00Z',
      source: 'Azure Security Center'
    },
    compliance: {
      compliant: 142,
      nonCompliant: 8,
      exempt: 2,
      unknown: 0,
      percentage: 94.7
    },
    assignments: 5,
    remediationTasks: 3,
    complianceFrameworks: ['ISO 27001', 'PCI DSS', 'HIPAA']
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/34c877ad-507e-4c82-993e-3452a6e0ad3c',
    name: 'StorageAccountPublicAccess',
    displayName: 'Storage accounts should restrict public access',
    description: 'Anonymous public read access to containers and blobs in Azure Storage is a convenient way to share data but might present security risks.',
    category: 'Security',
    type: 'BuiltIn',
    effect: 'Deny',
    status: 'Active',
    version: '3.1.0',
    parameters: {
      effect: {
        type: 'String',
        defaultValue: 'Deny',
        allowedValues: ['Audit', 'Deny']
      }
    },
    resourceTypes: ['Microsoft.Storage/storageAccounts'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-02-10T09:15:00Z',
      updatedBy: 'Microsoft',
      updatedOn: '2024-11-20T16:45:00Z',
      source: 'Azure Security Center'
    },
    compliance: {
      compliant: 138,
      nonCompliant: 12,
      exempt: 2,
      unknown: 0,
      percentage: 92.0
    },
    assignments: 8,
    remediationTasks: 5
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/1c6e3c8a-7e5a-4b8d-9c3f-2a4d5e6f7g8h',
    name: 'VMDiskEncryption',
    displayName: 'Virtual machines should encrypt temp disks, caches, and data flows',
    description: 'By default, a virtual machine OS and data disks are encrypted-at-rest using platform-managed keys.',
    category: 'Security',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '2.0.3',
    parameters: {},
    resourceTypes: ['Microsoft.Compute/virtualMachines'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-03-05T11:20:00Z',
      updatedBy: 'SecurityTeam',
      updatedOn: '2024-12-15T09:30:00Z',
      source: 'Azure Security Center'
    },
    compliance: {
      compliant: 75,
      nonCompliant: 15,
      exempt: 0,
      unknown: 10,
      percentage: 83.3
    },
    assignments: 12,
    remediationTasks: 8,
    complianceFrameworks: ['SOC 2', 'ISO 27001']
  },

  // Governance Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/96670d01-0a4d-4649-9c89-2d3abc0a5025',
    name: 'RequireTagOnResources',
    displayName: 'Require a tag on resources',
    description: 'Enforces existence of a tag. Does not apply to resource groups.',
    category: 'Governance',
    type: 'BuiltIn',
    effect: 'Deny',
    status: 'Active',
    version: '1.0.1',
    parameters: {
      tagName: {
        type: 'String',
        defaultValue: 'Environment'
      }
    },
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-01-20T14:00:00Z',
      updatedBy: 'GovernanceTeam',
      updatedOn: '2024-10-15T10:00:00Z',
      source: 'Azure Policy'
    },
    compliance: {
      compliant: 450,
      nonCompliant: 50,
      exempt: 10,
      unknown: 0,
      percentage: 90.0
    },
    assignments: 15,
    remediationTasks: 12
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/2a0e14a6-b0a6-4fab-991a-187a4f81c498',
    name: 'AppendTagAndValue',
    displayName: 'Append a tag and its value to resources',
    description: 'Appends the specified tag and value when any resource which is missing this tag is created or updated.',
    category: 'Governance',
    type: 'BuiltIn',
    effect: 'Append',
    status: 'Active',
    version: '1.0.1',
    parameters: {
      tagName: {
        type: 'String',
        defaultValue: 'CostCenter'
      },
      tagValue: {
        type: 'String',
        defaultValue: 'Default'
      }
    },
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-02-01T12:30:00Z',
      updatedBy: 'FinanceTeam',
      updatedOn: '2024-11-10T15:20:00Z',
      source: 'Azure Policy'
    },
    compliance: {
      compliant: 480,
      nonCompliant: 20,
      exempt: 5,
      unknown: 0,
      percentage: 96.0
    },
    assignments: 10
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/custom-naming-convention',
    name: 'NamingConvention',
    displayName: 'Enforce naming convention for resources',
    description: 'Ensures all resources follow the organization naming convention pattern.',
    category: 'Governance',
    type: 'Custom',
    effect: 'Deny',
    status: 'Active',
    version: '1.2.0',
    parameters: {
      namePattern: {
        type: 'String',
        defaultValue: '^(dev|test|prod)-(eastus|westus)-[a-z]+-\\d{3}$'
      }
    },
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'CloudArchitect',
      createdOn: '2024-05-15T09:00:00Z',
      updatedBy: 'CloudArchitect',
      updatedOn: '2024-12-01T11:00:00Z',
      source: 'Custom'
    },
    compliance: {
      compliant: 380,
      nonCompliant: 120,
      exempt: 0,
      unknown: 0,
      percentage: 76.0
    },
    assignments: 20,
    remediationTasks: 45
  },

  // Compliance Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/e56962a6-4747-49cd-b67b-bf8b01975c4c',
    name: 'AllowedLocations',
    displayName: 'Allowed locations',
    description: 'This policy enables you to restrict the locations your organization can specify when deploying resources.',
    category: 'Compliance',
    type: 'BuiltIn',
    effect: 'Deny',
    status: 'Active',
    version: '1.0.0',
    parameters: {
      listOfAllowedLocations: {
        type: 'Array',
        defaultValue: ['eastus', 'westus', 'northeurope']
      }
    },
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-01-05T08:00:00Z',
      updatedBy: 'ComplianceTeam',
      updatedOn: '2024-09-20T14:30:00Z',
      source: 'Azure Policy'
    },
    compliance: {
      compliant: 495,
      nonCompliant: 5,
      exempt: 0,
      unknown: 0,
      percentage: 99.0
    },
    assignments: 25,
    complianceFrameworks: ['GDPR', 'Data Residency']
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/0a914e76-4921-4c19-b460-a2d36003525a',
    name: 'AuditResourceLocation',
    displayName: 'Audit resource location matches resource group location',
    description: 'Audit that the resource location matches its resource group location.',
    category: 'Compliance',
    type: 'BuiltIn',
    effect: 'Audit',
    status: 'Active',
    version: '1.0.0',
    parameters: {},
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-02-20T10:15:00Z',
      updatedBy: 'Microsoft',
      updatedOn: '2024-08-15T13:45:00Z',
      source: 'Azure Policy'
    },
    compliance: {
      compliant: 420,
      nonCompliant: 80,
      exempt: 0,
      unknown: 0,
      percentage: 84.0
    },
    assignments: 18
  },

  // Network Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/88c0b9da-ce96-4b03-9635-f29a937e2900',
    name: 'NetworkWatcherEnabled',
    displayName: 'Network Watcher should be enabled',
    description: 'Network Watcher is a regional service that enables you to monitor and diagnose conditions at a network scenario level.',
    category: 'Network',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '3.0.0',
    parameters: {
      listOfLocations: {
        type: 'Array',
        defaultValue: ['*']
      }
    },
    resourceTypes: ['Microsoft.Network/networkWatchers'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-03-10T11:00:00Z',
      updatedBy: 'NetworkTeam',
      updatedOn: '2024-12-05T16:00:00Z',
      source: 'Azure Security Center'
    },
    compliance: {
      compliant: 8,
      nonCompliant: 2,
      exempt: 0,
      unknown: 0,
      percentage: 80.0
    },
    assignments: 5,
    remediationTasks: 2
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/35f9c03a-cc27-418e-9c0c-539ff999d010',
    name: 'NSGFlowLogsEnabled',
    displayName: 'Network Security Group Flow Logs should be enabled',
    description: 'Audit for network security groups to verify if flow logs are configured.',
    category: 'Network',
    type: 'BuiltIn',
    effect: 'Audit',
    status: 'Active',
    version: '1.1.0',
    parameters: {
      retentionDays: {
        type: 'Integer',
        defaultValue: 90
      }
    },
    resourceTypes: ['Microsoft.Network/networkSecurityGroups'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-04-01T09:30:00Z',
      updatedBy: 'SecurityTeam',
      updatedOn: '2024-11-25T12:15:00Z',
      source: 'Azure Security Center'
    },
    compliance: {
      compliant: 45,
      nonCompliant: 15,
      exempt: 0,
      unknown: 0,
      percentage: 75.0
    },
    assignments: 8,
    remediationTasks: 10,
    complianceFrameworks: ['PCI DSS', 'SOC 2']
  },

  // Backup Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/013e242c-8828-4970-87b3-ab247555486d',
    name: 'AzureBackupEnabled',
    displayName: 'Azure Backup should be enabled for Virtual Machines',
    description: 'Ensure protection of your Azure Virtual Machines by enabling Azure Backup.',
    category: 'Backup',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '3.0.0',
    parameters: {},
    resourceTypes: ['Microsoft.Compute/virtualMachines'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-02-15T10:00:00Z',
      updatedBy: 'BackupTeam',
      updatedOn: '2024-12-10T14:00:00Z',
      source: 'Azure Backup'
    },
    compliance: {
      compliant: 65,
      nonCompliant: 35,
      exempt: 0,
      unknown: 0,
      percentage: 65.0
    },
    assignments: 10,
    remediationTasks: 20
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/custom-backup-retention',
    name: 'BackupRetentionPolicy',
    displayName: 'Enforce minimum backup retention period',
    description: 'Ensures all backup policies have a minimum retention period of 30 days.',
    category: 'Backup',
    type: 'Custom',
    effect: 'Deny',
    status: 'Active',
    version: '1.0.0',
    parameters: {
      minimumRetentionDays: {
        type: 'Integer',
        defaultValue: 30
      }
    },
    resourceTypes: ['Microsoft.RecoveryServices/vaults'],
    metadata: {
      createdBy: 'BackupAdmin',
      createdOn: '2024-06-01T08:30:00Z',
      updatedBy: 'BackupAdmin',
      updatedOn: '2024-11-15T10:45:00Z',
      source: 'Custom'
    },
    compliance: {
      compliant: 18,
      nonCompliant: 2,
      exempt: 0,
      unknown: 0,
      percentage: 90.0
    },
    assignments: 3
  },

  // Monitoring Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/b7ddfbdc-1260-477d-91fd-98bd9be789a6',
    name: 'DiagnosticSettingsEnabled',
    displayName: 'Diagnostic settings should be enabled',
    description: 'Audit enabling of diagnostic settings to track resource activities.',
    category: 'Monitoring',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '2.0.0',
    parameters: {
      logAnalyticsWorkspaceId: {
        type: 'String',
        defaultValue: ''
      }
    },
    resourceTypes: ['*'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-03-20T12:00:00Z',
      updatedBy: 'MonitoringTeam',
      updatedOn: '2024-12-08T15:30:00Z',
      source: 'Azure Monitor'
    },
    compliance: {
      compliant: 380,
      nonCompliant: 120,
      exempt: 0,
      unknown: 0,
      percentage: 76.0
    },
    assignments: 22,
    remediationTasks: 35
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/1bc1795e-d44a-4d48-9b3b-6fff0fd5f9ba',
    name: 'ActivityLogRetention',
    displayName: 'Activity log should be retained for at least one year',
    description: 'This policy audits the activity log if retention is not set for 365 days or forever.',
    category: 'Monitoring',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '2.0.0',
    parameters: {},
    resourceTypes: ['Microsoft.Insights/logProfiles'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-04-10T13:20:00Z',
      updatedBy: 'Microsoft',
      updatedOn: '2024-10-20T11:10:00Z',
      source: 'Azure Monitor'
    },
    compliance: {
      compliant: 490,
      nonCompliant: 10,
      exempt: 0,
      unknown: 0,
      percentage: 98.0
    },
    assignments: 30,
    complianceFrameworks: ['ISO 27001', 'SOC 2', 'HIPAA']
  },

  // Cost Management Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/custom-cost-limit',
    name: 'MonthlyCostLimit',
    displayName: 'Enforce monthly cost limits per resource group',
    description: 'Prevents resource groups from exceeding their assigned monthly cost budget.',
    category: 'Cost Management',
    type: 'Custom',
    effect: 'Deny',
    status: 'Active',
    version: '2.1.0',
    parameters: {
      maxMonthlyCost: {
        type: 'Integer',
        defaultValue: 10000
      },
      currency: {
        type: 'String',
        defaultValue: 'USD'
      }
    },
    resourceTypes: ['Microsoft.Resources/resourceGroups'],
    metadata: {
      createdBy: 'FinanceTeam',
      createdOn: '2024-07-01T09:00:00Z',
      updatedBy: 'FinanceTeam',
      updatedOn: '2024-12-15T14:00:00Z',
      source: 'Custom'
    },
    compliance: {
      compliant: 48,
      nonCompliant: 2,
      exempt: 0,
      unknown: 0,
      percentage: 96.0
    },
    assignments: 5
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/custom-vm-sizing',
    name: 'VMSizingPolicy',
    displayName: 'Restrict VM sizes to cost-effective options',
    description: 'Limits VM deployments to pre-approved cost-effective sizes.',
    category: 'Cost Management',
    type: 'Custom',
    effect: 'Deny',
    status: 'Active',
    version: '1.3.0',
    parameters: {
      allowedSizes: {
        type: 'Array',
        defaultValue: ['Standard_B2s', 'Standard_B2ms', 'Standard_D2s_v3', 'Standard_D4s_v3']
      }
    },
    resourceTypes: ['Microsoft.Compute/virtualMachines'],
    metadata: {
      createdBy: 'CloudArchitect',
      createdOn: '2024-08-10T10:30:00Z',
      updatedBy: 'CostOptimizationTeam',
      updatedOn: '2024-12-01T09:15:00Z',
      source: 'Custom'
    },
    compliance: {
      compliant: 88,
      nonCompliant: 12,
      exempt: 0,
      unknown: 0,
      percentage: 88.0
    },
    assignments: 7,
    remediationTasks: 5
  },

  // Database Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/22bee202-a82f-4305-9a2a-6d7f44d4dedb',
    name: 'SQLServerAuditing',
    displayName: 'Auditing on SQL server should be enabled',
    description: 'Auditing on your SQL Server should be enabled to track database activities.',
    category: 'SQL',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '2.0.0',
    parameters: {},
    resourceTypes: ['Microsoft.Sql/servers'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-05-05T11:30:00Z',
      updatedBy: 'DatabaseTeam',
      updatedOn: '2024-11-30T16:20:00Z',
      source: 'Azure SQL'
    },
    compliance: {
      compliant: 28,
      nonCompliant: 2,
      exempt: 0,
      unknown: 0,
      percentage: 93.3
    },
    assignments: 4,
    complianceFrameworks: ['PCI DSS', 'HIPAA', 'SOC 2']
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/17k78e20-9358-41c9-923c-fb736d382a12',
    name: 'SQLTransparentDataEncryption',
    displayName: 'Transparent Data Encryption on SQL databases should be enabled',
    description: 'Transparent data encryption should be enabled to protect data-at-rest.',
    category: 'SQL',
    type: 'BuiltIn',
    effect: 'AuditIfNotExists',
    status: 'Active',
    version: '2.0.0',
    parameters: {},
    resourceTypes: ['Microsoft.Sql/servers/databases'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-05-10T09:45:00Z',
      updatedBy: 'SecurityTeam',
      updatedOn: '2024-12-12T13:00:00Z',
      source: 'Azure SQL'
    },
    compliance: {
      compliant: 45,
      nonCompliant: 5,
      exempt: 0,
      unknown: 0,
      percentage: 90.0
    },
    assignments: 6,
    remediationTasks: 3
  },

  // Container Policies
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/febd0533-8e55-448f-b837-bd0e06f16469',
    name: 'KubernetesRBAC',
    displayName: 'Role-Based Access Control should be used on Kubernetes Services',
    description: 'To provide granular filtering on the actions that users can perform, use RBAC.',
    category: 'Kubernetes',
    type: 'BuiltIn',
    effect: 'Audit',
    status: 'Active',
    version: '1.0.2',
    parameters: {},
    resourceTypes: ['Microsoft.ContainerService/managedClusters'],
    metadata: {
      createdBy: 'Microsoft',
      createdOn: '2024-06-15T10:00:00Z',
      updatedBy: 'ContainerTeam',
      updatedOn: '2024-12-03T11:30:00Z',
      source: 'Azure Kubernetes Service'
    },
    compliance: {
      compliant: 12,
      nonCompliant: 3,
      exempt: 0,
      unknown: 0,
      percentage: 80.0
    },
    assignments: 2
  },
  {
    id: '/providers/Microsoft.Authorization/policyDefinitions/custom-container-registry',
    name: 'ContainerRegistryPolicy',
    displayName: 'Use only approved container registries',
    description: 'Ensures containers are only pulled from approved registries.',
    category: 'Kubernetes',
    type: 'Custom',
    effect: 'Deny',
    status: 'Active',
    version: '1.1.0',
    parameters: {
      allowedRegistries: {
        type: 'Array',
        defaultValue: ['mycompany.azurecr.io', 'mcr.microsoft.com']
      }
    },
    resourceTypes: ['Microsoft.ContainerInstance/containerGroups'],
    metadata: {
      createdBy: 'ContainerAdmin',
      createdOn: '2024-09-01T08:00:00Z',
      updatedBy: 'SecurityTeam',
      updatedOn: '2024-12-05T10:00:00Z',
      source: 'Custom'
    },
    compliance: {
      compliant: 35,
      nonCompliant: 5,
      exempt: 0,
      unknown: 0,
      percentage: 87.5
    },
    assignments: 3
  }
]

// Helper function to get policies by category
export function getPoliciesByCategory(category: string): PolicyDefinition[] {
  if (category === 'all') return azurePolicies
  return azurePolicies.filter(p => p.category === category)
}

// Helper function to get policy statistics
export function getPolicyStatistics() {
  const total = azurePolicies.length
  const active = azurePolicies.filter(p => p.status === 'Active').length
  const compliant = azurePolicies.reduce((sum, p) => sum + p.compliance.compliant, 0)
  const nonCompliant = azurePolicies.reduce((sum, p) => sum + p.compliance.nonCompliant, 0)
  const overallCompliance = compliant + nonCompliant > 0 
    ? ((compliant / (compliant + nonCompliant)) * 100).toFixed(1)
    : '0'
  
  const byCategory = azurePolicies.reduce((acc, p) => {
    acc[p.category] = (acc[p.category] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const byEffect = azurePolicies.reduce((acc, p) => {
    acc[p.effect] = (acc[p.effect] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  return {
    total,
    active,
    compliant,
    nonCompliant,
    overallCompliance,
    byCategory,
    byEffect,
    totalAssignments: azurePolicies.reduce((sum, p) => sum + p.assignments, 0),
    totalRemediationTasks: azurePolicies.reduce((sum, p) => sum + (p.remediationTasks || 0), 0)
  }
}

// Get non-compliant resources across all policies
export function getNonCompliantResources() {
  return [
    {
      id: 'res-001',
      name: 'stpolicycortexdev',
      type: 'Microsoft.Storage/storageAccounts',
      resourceGroup: 'rg-policycortex-dev',
      subscription: 'Development',
      violations: [
        { policy: 'SecureTransferToStorageAccounts', reason: 'HTTPS not enforced' },
        { policy: 'StorageAccountPublicAccess', reason: 'Public access enabled' }
      ],
      lastEvaluated: '2025-01-08T10:30:00Z',
      riskLevel: 'High'
    },
    {
      id: 'res-002',
      name: 'vm-web-prod-001',
      type: 'Microsoft.Compute/virtualMachines',
      resourceGroup: 'rg-production',
      subscription: 'Production',
      violations: [
        { policy: 'VMDiskEncryption', reason: 'Temp disk not encrypted' },
        { policy: 'AzureBackupEnabled', reason: 'Backup not configured' }
      ],
      lastEvaluated: '2025-01-08T11:00:00Z',
      riskLevel: 'Critical'
    },
    {
      id: 'res-003',
      name: 'sql-app-prod',
      type: 'Microsoft.Sql/servers',
      resourceGroup: 'rg-database',
      subscription: 'Production',
      violations: [
        { policy: 'SQLServerAuditing', reason: 'Auditing disabled' }
      ],
      lastEvaluated: '2025-01-08T09:45:00Z',
      riskLevel: 'Medium'
    },
    {
      id: 'res-004',
      name: 'kv-secrets-dev',
      type: 'Microsoft.KeyVault/vaults',
      resourceGroup: 'rg-security',
      subscription: 'Development',
      violations: [
        { policy: 'DiagnosticSettingsEnabled', reason: 'Diagnostic logs not configured' }
      ],
      lastEvaluated: '2025-01-08T12:15:00Z',
      riskLevel: 'Medium'
    },
    {
      id: 'res-005',
      name: 'aks-cluster-prod',
      type: 'Microsoft.ContainerService/managedClusters',
      resourceGroup: 'rg-containers',
      subscription: 'Production',
      violations: [
        { policy: 'KubernetesRBAC', reason: 'RBAC not fully configured' }
      ],
      lastEvaluated: '2025-01-08T08:30:00Z',
      riskLevel: 'High'
    }
  ]
}