import { NextRequest, NextResponse } from 'next/server';
import { withAuth } from '@/lib/auth/apiAuth';
import { correlationQuerySchema } from '@/lib/validation/schemas';
import { validateRequest } from '@/lib/validation/schemas';
import { apiRateLimiter } from '@/lib/middleware/rateLimiter';
import { auditLogger, AuditEventType, AuditSeverity } from '@/lib/logging/auditLogger';

// Cross-Domain Correlations endpoint for Patent #1
export async function GET(request: NextRequest) {
  // Apply rate limiting
  return apiRateLimiter.middleware(request, async (req) => {
    // Apply authentication
    return withAuth(req, async (authenticatedReq, user) => {
      try {
        // Extract and validate query parameters
        const searchParams = authenticatedReq.nextUrl.searchParams;
        const queryParams = {
          domain: searchParams.get('domain') || undefined,
          resourceIds: searchParams.get('resourceIds')?.split(',') || undefined,
          minCorrelation: searchParams.get('minCorrelation') ? parseFloat(searchParams.get('minCorrelation')!) : undefined,
        };

        // Validate with Zod
        const validation = validateRequest(correlationQuerySchema, queryParams);
        if (!validation.success) {
          auditLogger.log({
            eventType: AuditEventType.INVALID_REQUEST,
            severity: AuditSeverity.WARNING,
            userId: user.sub,
            userEmail: user.email,
            success: false,
            details: {
              errors: validation.errors.flatten(),
              endpoint: '/api/v1/correlations',
            },
          });
          
          return NextResponse.json(
            { error: 'Invalid query parameters', details: validation.errors.flatten() },
            { status: 400 }
          );
        }

        // Log correlation analysis access
        auditLogger.log({
          eventType: AuditEventType.RESOURCE_ACCESSED,
          severity: AuditSeverity.INFO,
          userId: user.sub,
          userEmail: user.email,
          success: true,
          resource: {
            type: 'correlations',
            id: 'analysis',
          },
          details: {
            domain: validation.data.domain,
            endpoint: '/api/v1/correlations',
          },
        });

        // Check if we're in real data mode
        if (process.env.NEXT_PUBLIC_DEMO_MODE === 'false' || process.env.NEXT_PUBLIC_USE_REAL_DATA === 'true') {
          // In real mode, try to fetch from backend
          try {
            const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
            const queryString = authenticatedReq.nextUrl.searchParams.toString();
            const response = await fetch(`${backendUrl}/api/v1/correlations${queryString ? `?${queryString}` : ''}`, {
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
              { error: 'Failed to fetch correlations from backend', details: error },
              { status: 503 }
            );
          }
        }

        // Only return mock data in demo mode
        const correlations = {
          total: 23,
          critical: 3,
          high: 7,
          medium: 8,
          low: 5,
          lastAnalysis: '2024-03-21T12:00:00Z',
          domains: ['security', 'compliance', 'cost', 'performance', 'identity'],
          correlations: [
            {
              id: 'CORR-001',
              title: 'Security-Cost Correlation Detected',
              description: 'Increased security incidents correlating with reduced security spending',
              domains: ['security', 'cost'],
              confidence: 89.5,
              impact: 'Critical',
              pattern: 'inverse_correlation',
              affectedResources: 12,
              insight: 'Security budget cuts in Q4 2023 leading to 45% increase in incidents',
              recommendation: 'Restore security tooling budget to Q3 2023 levels',
              status: 'Active',
              detectedAt: '2024-03-21T11:30:00Z'
            },
            {
              id: 'CORR-002',
              title: 'Compliance-Performance Pattern',
              description: 'Compliance violations spike during high-load periods',
              domains: ['compliance', 'performance'],
              confidence: 92.3,
              impact: 'High',
              pattern: 'temporal_correlation',
              affectedResources: 8,
              insight: 'Auto-scaling policies override compliance controls during peak traffic',
              recommendation: 'Implement compliance-aware scaling policies',
              status: 'Active',
              detectedAt: '2024-03-21T10:15:00Z'
            },
            {
              id: 'CORR-003',
              title: 'Identity-Access Anomaly',
              description: 'Unusual access patterns from privileged accounts',
              domains: ['identity', 'security'],
              confidence: 78.2,
              impact: 'High',
              pattern: 'anomaly_detection',
              affectedResources: 5,
              insight: 'Multiple failed login attempts followed by successful access from new locations',
              recommendation: 'Enable MFA and review access logs for affected accounts',
              status: 'Investigating',
              detectedAt: '2024-03-21T09:45:00Z'
            },
            {
              id: 'CORR-004',
              title: 'Cost-Compliance Trade-off',
              description: 'Cost optimization conflicts with compliance requirements',
              domains: ['cost', 'compliance'],
              confidence: 85.7,
              impact: 'Medium',
              pattern: 'conflict_detection',
              affectedResources: 15,
              insight: 'Reserved instances in non-compliant regions offering 40% savings',
              recommendation: 'Evaluate compliance-friendly cost optimization strategies',
              status: 'Pending Review',
              detectedAt: '2024-03-20T16:20:00Z'
            },
            {
              id: 'CORR-005',
              title: 'Performance-Security Balance',
              description: 'Security controls impacting application performance',
              domains: ['performance', 'security'],
              confidence: 91.1,
              impact: 'Medium',
              pattern: 'causal_relationship',
              affectedResources: 7,
              insight: 'WAF rules adding 200ms latency to API responses',
              recommendation: 'Optimize WAF rules or implement caching strategy',
              status: 'Active',
              detectedAt: '2024-03-20T14:00:00Z'
            }
          ],
          algorithms: {
            pearson: { enabled: true, threshold: 0.7 },
            spearman: { enabled: true, threshold: 0.65 },
            kendall: { enabled: false, threshold: 0.6 },
            mutual_information: { enabled: true, threshold: 0.5 },
            granger_causality: { enabled: true, p_value: 0.05 }
          },
          performance: {
            analysisTime: 234,
            dataPoints: 1500000,
            timeWindow: '30d',
            lastUpdate: '2024-03-21T12:00:00Z'
          }
        };

        // Filter by domain if specified
        if (validation.data.domain) {
          correlations.correlations = correlations.correlations.filter(c => 
            c.domains.includes(validation.data.domain!)
          );
        }

        // Filter by minimum correlation confidence if specified
        if (validation.data.minCorrelation) {
          correlations.correlations = correlations.correlations.filter(c => 
            c.confidence >= validation.data.minCorrelation!
          );
        }

        return NextResponse.json(correlations);
      } catch (error) {
        auditLogger.log({
          eventType: AuditEventType.ERROR_OCCURRED,
          severity: AuditSeverity.ERROR,
          userId: user.sub,
          userEmail: user.email,
          success: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          details: {
            endpoint: '/api/v1/correlations',
            method: 'GET',
          },
        });
        
        console.error('Error fetching correlations:', error);
        return NextResponse.json(
          { error: 'Failed to fetch correlations' },
          { status: 500 }
        );
      }
    });
  });
}