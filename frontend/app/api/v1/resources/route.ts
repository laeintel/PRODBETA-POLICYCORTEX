import { NextResponse } from 'next/server'

// Azure resources endpoint
export async function GET() {
  const resources = {
    total: 342,
    byType: {
      virtualMachines: 45,
      storageAccounts: 23,
      databases: 12,
      networkInterfaces: 89,
      loadBalancers: 8,
      containerInstances: 34,
      functions: 67,
      appServices: 28,
      keyVaults: 15,
      other: 21
    },
    byRegion: {
      eastUs: 154,
      westEurope: 102,
      southeastAsia: 51,
      centralUs: 35
    },
    byStatus: {
      running: 298,
      stopped: 23,
      deallocated: 12,
      failed: 9
    },
    resources: [
      {
        id: 'res-001',
        name: 'vm-prod-web-01',
        type: 'Virtual Machine',
        region: 'East US',
        resourceGroup: 'rg-production',
        status: 'Running',
        cost: 450.00,
        tags: {
          environment: 'Production',
          owner: 'WebTeam',
          costCenter: 'CC-100'
        },
        created: '2024-01-15T08:00:00Z',
        lastModified: '2024-03-20T14:30:00Z'
      },
      {
        id: 'res-002',
        name: 'storage-data-primary',
        type: 'Storage Account',
        region: 'East US',
        resourceGroup: 'rg-storage',
        status: 'Available',
        cost: 125.50,
        tags: {
          environment: 'Production',
          owner: 'DataTeam',
          costCenter: 'CC-200'
        },
        created: '2023-11-20T10:00:00Z',
        lastModified: '2024-03-19T09:15:00Z'
      },
      {
        id: 'res-003',
        name: 'sql-analytics-db',
        type: 'SQL Database',
        region: 'West Europe',
        resourceGroup: 'rg-analytics',
        status: 'Online',
        cost: 890.75,
        tags: {
          environment: 'Production',
          owner: 'AnalyticsTeam',
          costCenter: 'CC-300'
        },
        created: '2023-09-10T12:00:00Z',
        lastModified: '2024-03-21T16:45:00Z'
      },
      {
        id: 'res-004',
        name: 'func-api-processor',
        type: 'Function App',
        region: 'Southeast Asia',
        resourceGroup: 'rg-serverless',
        status: 'Running',
        cost: 67.25,
        tags: {
          environment: 'Production',
          owner: 'APITeam',
          costCenter: 'CC-400'
        },
        created: '2024-02-01T14:00:00Z',
        lastModified: '2024-03-21T11:20:00Z'
      },
      {
        id: 'res-005',
        name: 'aks-cluster-main',
        type: 'AKS Cluster',
        region: 'Central US',
        resourceGroup: 'rg-containers',
        status: 'Running',
        cost: 1250.00,
        tags: {
          environment: 'Production',
          owner: 'PlatformTeam',
          costCenter: 'CC-500'
        },
        created: '2023-12-05T09:00:00Z',
        lastModified: '2024-03-20T13:30:00Z'
      }
    ],
    compliance: {
      compliant: 289,
      nonCompliant: 38,
      warning: 15,
      complianceRate: 84.5
    },
    costs: {
      daily: 4250.75,
      monthly: 127522.50,
      projected: 135000.00,
      trend: '+5.2%'
    },
    health: {
      healthy: 312,
      degraded: 18,
      unhealthy: 12,
      healthScore: 91.2
    }
  }

  return NextResponse.json(resources)
}