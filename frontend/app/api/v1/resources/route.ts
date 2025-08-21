import { NextRequest, NextResponse } from 'next/server';
import { withAuth } from '@/lib/auth/apiAuth';
import { resourceQuerySchema, createResourceSchema } from '@/lib/validation/schemas';
import { validateRequest } from '@/lib/validation/schemas';
import { apiRateLimiter } from '@/lib/middleware/rateLimiter';
import { csrfProtection } from '@/lib/middleware/csrf';
import { auditResourceAccess, auditLogger, AuditEventType, AuditSeverity } from '@/lib/logging/auditLogger';

// Mock data - in production this would come from database
const mockResourcesData = {
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
};

export async function GET(request: NextRequest) {
  // Apply rate limiting
  return apiRateLimiter.middleware(request, async (req) => {
    // Apply authentication
    return withAuth(req, async (authenticatedReq, user) => {
      try {
        // Extract query parameters
        const searchParams = authenticatedReq.nextUrl.searchParams;
        const queryParams = {
          page: searchParams.get('page') || '1',
          limit: searchParams.get('limit') || '20',
          type: searchParams.get('type') || undefined,
          status: searchParams.get('status') || undefined,
          search: searchParams.get('search') || undefined,
          sort: searchParams.get('sort') || 'name',
          order: searchParams.get('order') || 'asc',
        };

        // Validate input with Zod
        const validation = validateRequest(resourceQuerySchema, queryParams);
        if (!validation.success) {
          auditLogger.log({
            eventType: AuditEventType.INVALID_REQUEST,
            severity: AuditSeverity.WARNING,
            userId: user.sub,
            userEmail: user.email,
            success: false,
            details: {
              errors: validation.errors.flatten(),
              endpoint: '/api/v1/resources',
            },
          });
          
          return NextResponse.json(
            { error: 'Invalid query parameters', details: validation.errors.flatten() },
            { status: 400 }
          );
        }

        // Log resource access
        auditResourceAccess(
          user.sub,
          'resources',
          'list',
          'GET',
          true
        );

        // Return resources data with pagination info
        const { page, limit } = validation.data;
        const paginatedResources = {
          ...mockResourcesData,
          pagination: {
            page,
            limit,
            total: mockResourcesData.resources.length,
            totalPages: Math.ceil(mockResourcesData.resources.length / limit),
          },
        };

        return NextResponse.json(paginatedResources);
      } catch (error) {
        auditLogger.log({
          eventType: AuditEventType.ERROR_OCCURRED,
          severity: AuditSeverity.ERROR,
          userId: user.sub,
          userEmail: user.email,
          success: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          details: {
            endpoint: '/api/v1/resources',
            method: 'GET',
          },
        });
        
        console.error('Error fetching resources:', error);
        return NextResponse.json(
          { error: 'Failed to fetch resources' },
          { status: 500 }
        );
      }
    });
  });
}

export async function POST(request: NextRequest) {
  // Apply rate limiting
  return apiRateLimiter.middleware(request, async (req) => {
    // Apply CSRF protection for state-changing request
    return csrfProtection.middleware(req, async (csrfProtectedReq) => {
      // Apply authentication
      return withAuth(csrfProtectedReq, async (authenticatedReq, user) => {
        try {
          // Parse and validate request body
          const body = await authenticatedReq.json();
          
          const validation = validateRequest(createResourceSchema, body);
          if (!validation.success) {
            auditLogger.log({
              eventType: AuditEventType.INVALID_REQUEST,
              severity: AuditSeverity.WARNING,
              userId: user.sub,
              userEmail: user.email,
              success: false,
              details: {
                errors: validation.errors.flatten(),
                endpoint: '/api/v1/resources',
                method: 'POST',
              },
            });
            
            return NextResponse.json(
              { error: 'Invalid request body', details: validation.errors.flatten() },
              { status: 400 }
            );
          }

          const resourceData = validation.data;

          // Log resource creation attempt
          auditLogger.log({
            eventType: AuditEventType.RESOURCE_CREATED,
            severity: AuditSeverity.INFO,
            userId: user.sub,
            userEmail: user.email,
            success: true,
            resource: {
              type: resourceData.type,
              id: 'new-resource-id',
              name: resourceData.name,
            },
            details: {
              resourceGroup: resourceData.resourceGroup,
              location: resourceData.location,
              tags: resourceData.tags,
            },
          });

          // Create new resource (mock implementation)
          const newResource = {
            id: `res-${Date.now()}`,
            ...resourceData,
            status: 'pending',
            created: new Date().toISOString(),
            lastModified: new Date().toISOString(),
            createdBy: user.sub,
          };

          // In production, this would save to database
          return NextResponse.json(
            { 
              message: 'Resource created successfully',
              data: newResource 
            },
            { status: 201 }
          );
        } catch (error) {
          auditLogger.log({
            eventType: AuditEventType.ERROR_OCCURRED,
            severity: AuditSeverity.ERROR,
            userId: user.sub,
            userEmail: user.email,
            success: false,
            errorMessage: error instanceof Error ? error.message : 'Unknown error',
            details: {
              endpoint: '/api/v1/resources',
              method: 'POST',
            },
          });
          
          console.error('Error creating resource:', error);
          return NextResponse.json(
            { error: 'Failed to create resource' },
            { status: 500 }
          );
        }
      });
    });
  });
}