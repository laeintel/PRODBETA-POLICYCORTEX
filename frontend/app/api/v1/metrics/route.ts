import { NextRequest, NextResponse } from 'next/server';
import { withAuth } from '@/lib/auth/apiAuth';
import { apiRateLimiter } from '@/lib/middleware/rateLimiter';
import { auditLogger, AuditEventType, AuditSeverity } from '@/lib/logging/auditLogger';

// Real-time metrics data
export async function GET(request: NextRequest) {
  // Apply rate limiting
  return apiRateLimiter.middleware(request, async (req) => {
    // Apply authentication
    return withAuth(req, async (authenticatedReq, user) => {
      try {
        // Log access
        auditLogger.log({
          eventType: AuditEventType.RESOURCE_ACCESSED,
          severity: AuditSeverity.INFO,
          userId: user.sub,
          userEmail: user.email,
          success: true,
          details: {
            endpoint: '/api/v1/metrics',
            method: 'GET',
          },
        });

  // Check if we're in real data mode
  if (process.env.NEXT_PUBLIC_DEMO_MODE === 'false' || process.env.NEXT_PUBLIC_USE_REAL_DATA === 'true') {
    // In real mode, try to fetch from backend
    try {
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
      const response = await fetch(`${backendUrl}/api/v1/metrics`, {
        headers: {
          'Authorization': authenticatedReq.headers.get('authorization') || '',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      } else {
        // Fail-fast in real mode
        return NextResponse.json(
          { error: 'Backend service unavailable', details: 'Real data mode requires backend connection' },
          { status: 503 }
        );
      }
    } catch (error) {
      return NextResponse.json(
        { error: 'Failed to fetch metrics from backend', details: error },
        { status: 503 }
      );
    }
  }

  // Only return mock data in demo mode
  const metrics = {
    compliance: {
      score: 94,
      trend: 2.3,
      details: {
        totalPolicies: 89,
        compliantPolicies: 84,
        nonCompliantPolicies: 5,
        lastChecked: new Date().toISOString()
      }
    },
    risks: {
      active: 3,
      critical: 1,
      high: 1,
      medium: 1,
      low: 0,
      trend: -1,
      details: [
        { id: 1, title: 'Expired SSL Certificate', severity: 'critical', resource: 'api.policycortex.com' },
        { id: 2, title: 'Excessive permissions on storage', severity: 'high', resource: 'pcx-storage-prod' },
        { id: 3, title: 'Untagged resources', severity: 'medium', resource: '12 resources' }
      ]
    },
    costs: {
      currentMonth: 125000,
      lastMonth: 170000,
      savings: 45000,
      trend: 12,
      breakdown: {
        compute: 45000,
        storage: 25000,
        network: 15000,
        database: 20000,
        other: 20000
      }
    },
    resources: {
      total: 342,
      byType: {
        virtualMachines: 45,
        storageAccounts: 23,
        databases: 12,
        networkInterfaces: 89,
        loadBalancers: 8,
        containerInstances: 34,
        functions: 67,
        other: 64
      },
      byRegion: {
        eastUs: 154,
        westEurope: 102,
        southeastAsia: 51,
        centralUs: 35
      }
    },
    identity: {
      totalUsers: 1247,
      activeUsers: 892,
      privilegedUsers: 45,
      guestUsers: 123,
      servicePrincipals: 67,
      mfaEnabled: 1089,
      conditionalAccessPolicies: 23
    },
    ai: {
      predictions: 7,
      correlations: 3,
      conversations: 145,
      accuracy: {
        predictiveCompliance: 99.2,
        crossDomain: 96.8,
        conversational: 98.7,
        unified: 97.5
      },
      lastModelUpdate: new Date(Date.now() - 3600000).toISOString()
    }
  }

        return NextResponse.json(metrics);
      } catch (error) {
        console.error('Metrics API error:', error);
        return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
      }
    });
  });
}