import { NextRequest, NextResponse } from 'next/server';

// Mock data generator for comprehensive RBAC analysis
function generateUserPermissionDetail(userId: string) {
  const permissions = Array.from({ length: 25 }, (_, i) => ({
    permissionId: `perm-${i + 1}`,
    permissionName: [
      'Virtual Machine Contributor',
      'Storage Blob Data Reader',
      'Key Vault Secrets User',
      'Network Contributor',
      'SQL DB Contributor',
      'App Service Contributor',
      'AKS Cluster Admin',
      'Resource Group Owner',
      'Subscription Reader',
      'Cost Management Contributor',
      'Security Admin',
      'Policy Contributor',
      'Log Analytics Contributor',
      'Backup Contributor',
      'DNS Zone Contributor',
      'Load Balancer Contributor',
      'Application Gateway Contributor',
      'Cosmos DB Account Reader',
      'Service Bus Data Owner',
      'Event Hub Data Receiver',
      'Container Registry Contributor',
      'Cognitive Services User',
      'Data Factory Contributor',
      'DevTest Labs User',
      'HDInsight Cluster Operator'
    ][i] || `Permission ${i + 1}`,
    resourceType: ['VirtualMachine', 'Storage', 'KeyVault', 'Network', 'Database'][i % 5],
    resourceId: `/subscriptions/xxx/resourceGroups/rg-${i}/resources/res-${i}`,
    resourceName: `resource-${i + 1}`,
    scope: i % 3 === 0 ? 'Subscription' : i % 2 === 0 ? 'ResourceGroup' : 'Resource',
    actions: [
      'Microsoft.Compute/virtualMachines/read',
      'Microsoft.Compute/virtualMachines/write',
      'Microsoft.Storage/storageAccounts/blobServices/containers/read'
    ].slice(0, Math.floor(Math.random() * 3) + 1),
    notActions: [],
    dataActions: i % 4 === 0 ? ['Microsoft.Storage/storageAccounts/blobServices/containers/blobs/read'] : [],
    notDataActions: [],
    assignedDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
    assignedBy: `admin-${Math.floor(Math.random() * 5) + 1}@company.com`,
    assignmentType: ['direct', 'role_based', 'group_inherited'][i % 3] as any,
    lastUsed: i % 3 === 0 ? null : new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
    usageCount30d: Math.floor(Math.random() * 100),
    usageCount90d: Math.floor(Math.random() * 300),
    isHighPrivilege: i % 4 === 0,
    isCustom: i % 5 === 0,
    riskLevel: ['critical', 'high', 'medium', 'low', 'none'][Math.floor(Math.random() * 5)] as any,
    usagePattern: ['daily', 'weekly', 'monthly', 'occasional', 'rare', 'never'][Math.floor(Math.random() * 6)] as any,
    similarUsersHaveThis: Math.floor(Math.random() * 100),
    removalImpact: {
      severity: ['critical', 'high', 'medium', 'low', 'none'][Math.floor(Math.random() * 5)] as any,
      affectedWorkflows: i % 3 === 0 ? ['Daily backup process', 'Monitoring alerts'] : [],
      dependentUsers: [`user-${i + 10}`, `user-${i + 20}`],
      businessImpact: 'May affect automated deployment processes'
    }
  }));

  const roles = Array.from({ length: 8 }, (_, i) => ({
    roleId: `role-${i + 1}`,
    roleName: [
      'Contributor',
      'Reader',
      'Owner',
      'User Access Administrator',
      'Security Reader',
      'Backup Operator',
      'DevOps Engineer',
      'Data Scientist'
    ][i],
    roleType: i < 4 ? 'built_in' : 'custom' as any,
    isBuiltin: i < 4,
    assignedDate: new Date(Date.now() - Math.random() * 180 * 24 * 60 * 60 * 1000).toISOString(),
    assignedBy: 'admin@company.com',
    scope: '/subscriptions/xxx',
    permissionsCount: Math.floor(Math.random() * 50) + 10,
    highRiskPermissions: ['Delete', 'Write', 'Admin'].slice(0, Math.floor(Math.random() * 3) + 1),
    lastActivity: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    usageFrequency: ['very_high', 'high', 'medium', 'low', 'very_low', 'none'][Math.floor(Math.random() * 6)] as any,
    justification: 'Required for daily operations',
    expiryDate: i % 3 === 0 ? new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() : null,
    isEligible: i % 2 === 0,
    isActive: true
  }));

  const groups = Array.from({ length: 5 }, (_, i) => ({
    groupId: `group-${i + 1}`,
    groupName: ['IT Admins', 'Developers', 'Data Team', 'Security Team', 'DevOps'][i],
    membershipType: i % 2 === 0 ? 'direct' : 'dynamic' as any,
    joinedDate: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
    addedBy: 'admin@company.com',
    permissionsInherited: Math.floor(Math.random() * 20) + 5,
    nestedGroups: i % 3 === 0 ? [`group-parent-${i}`] : [],
    isDynamic: i % 2 === 1,
    dynamicRule: i % 2 === 1 ? `user.department -eq "${['IT', 'Dev', 'Data'][i % 3]}"` : null
  }));

  return {
    userId,
    displayName: `User ${userId}`,
    email: `user.${userId}@company.com`,
    department: ['Engineering', 'IT', 'Finance', 'Operations'][Math.floor(Math.random() * 4)],
    jobTitle: ['Senior Developer', 'DevOps Engineer', 'Data Analyst', 'System Administrator'][Math.floor(Math.random() * 4)],
    permissions,
    roles,
    groups,
    riskScore: Math.random() * 0.8 + 0.2,
    overProvisioningScore: Math.random() * 0.7 + 0.3,
    lastSignIn: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
    accountEnabled: true,
    recommendations: [
      {
        recommendationId: 'rec-1',
        recommendationType: 'remove_permission',
        title: 'Remove unused Virtual Machine Contributor permission',
        description: 'This permission has not been used in the last 90 days and can be safely removed.',
        impact: 'low',
        confidence: 0.92,
        affectedPermissions: ['perm-1'],
        suggestedAction: {
          actionType: 'remove',
          description: 'Remove the Virtual Machine Contributor role assignment',
          automated: true,
          requiresApproval: true,
          implementationSteps: [
            'Review current usage',
            'Get approval from manager',
            'Remove role assignment',
            'Verify access'
          ]
        },
        estimatedRiskReduction: 0.15,
        similarUsersImplemented: 23,
        autoRemediationAvailable: true,
        requiresApproval: true,
        approvalWorkflowId: 'wf-123'
      },
      {
        recommendationId: 'rec-2',
        recommendationType: 'convert_to_just_in_time',
        title: 'Convert Owner role to Just-In-Time access',
        description: 'High-privilege Owner role should be converted to JIT access for better security.',
        impact: 'high',
        confidence: 0.88,
        affectedPermissions: ['role-3'],
        suggestedAction: {
          actionType: 'convert_jit',
          description: 'Enable Privileged Identity Management for this role',
          automated: false,
          requiresApproval: true,
          implementationSteps: [
            'Enable PIM for subscription',
            'Configure eligible assignment',
            'Set activation requirements',
            'Remove permanent assignment'
          ]
        },
        estimatedRiskReduction: 0.35,
        similarUsersImplemented: 45,
        autoRemediationAvailable: false,
        requiresApproval: true,
        approvalWorkflowId: 'wf-124'
      }
    ],
    accessPatterns: {
      typicalAccessTimes: [
        { start: '09:00', end: '12:00' },
        { start: '13:00', end: '17:00' }
      ],
      typicalLocations: [
        { country: 'United States', region: 'California', city: 'San Francisco', ipRange: '10.0.0.0/24' }
      ],
      typicalDevices: [
        { deviceId: 'device-1', deviceType: 'Windows PC', os: 'Windows 11', isCompliant: true, isManaged: true }
      ],
      unusualActivities: [
        {
          activityId: 'ua-1',
          activityType: 'Off-hours access',
          timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
          riskScore: 0.6,
          description: 'User accessed resources at 2 AM local time',
          affectedResources: ['res-1', 'res-2'],
          detectionMethod: 'Anomaly detection',
          isInvestigated: false,
          investigationNotes: null
        }
      ],
      accessVelocity: 0.7,
      failedAttempts30d: 3,
      mfaUsageRate: 0.95,
      conditionalAccessCompliance: 0.88,
      privilegedOperationsCount: 12,
      dataAccessVolume: {
        readsGb: 45.2,
        writesGb: 12.8,
        downloadsGb: 8.5
      },
      serviceUsage: {
        'Azure Storage': {
          accessCount: 234,
          uniqueOperations: 12,
          dataVolume: { readsGb: 20.5, writesGb: 5.2, downloadsGb: 3.1 },
          peakTimes: ['10:00-11:00', '14:00-15:00']
        },
        'Azure Compute': {
          accessCount: 156,
          uniqueOperations: 8,
          dataVolume: { readsGb: 10.2, writesGb: 3.5, downloadsGb: 1.2 },
          peakTimes: ['09:00-10:00', '16:00-17:00']
        }
      }
    },
    complianceStatus: {
      isCompliant: false,
      violations: [
        {
          violationId: 'viol-1',
          policyId: 'pol-1',
          policyName: 'Require MFA for admin roles',
          violationType: 'MFA not enabled',
          severity: 'high',
          detectedDate: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          remediationDeadline: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
          remediationSteps: ['Enable MFA in Azure AD', 'Configure authentication methods', 'Test MFA login']
        }
      ],
      certifications: [
        {
          certificationId: 'cert-1',
          name: 'Annual Access Review',
          issuedDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          expiryDate: new Date(Date.now() + 335 * 24 * 60 * 60 * 1000).toISOString(),
          certifier: 'manager@company.com'
        }
      ],
      lastReviewDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
      nextReviewDate: new Date(Date.now() + 60 * 24 * 60 * 60 * 1000).toISOString(),
      reviewer: 'manager@company.com',
      attestationStatus: 'pending'
    }
  };
}

export async function GET(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    const { userId } = params;
    
    // Generate comprehensive user permission details
    const userDetail = generateUserPermissionDetail(userId);
    
    return NextResponse.json(userDetail);
  } catch (error) {
    console.error('Error fetching user permission details:', error);
    return NextResponse.json(
      { error: 'Failed to fetch user permission details' },
      { status: 500 }
    );
  }
}