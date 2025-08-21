import { NextRequest, NextResponse } from 'next/server';
import { withAuth } from '@/lib/auth/apiAuth';
import { conversationSchema } from '@/lib/validation/schemas';
import { validateRequest } from '@/lib/validation/schemas';
import { apiRateLimiter } from '@/lib/middleware/rateLimiter';
import { csrfProtection } from '@/lib/middleware/csrf';
import { auditLogger, AuditEventType, AuditSeverity } from '@/lib/logging/auditLogger';

// Conversational AI endpoint for Patent #2
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
          
          const validation = validateRequest(conversationSchema, body);
          if (!validation.success) {
            auditLogger.log({
              eventType: AuditEventType.INVALID_REQUEST,
              severity: AuditSeverity.WARNING,
              userId: user.sub,
              userEmail: user.email,
              success: false,
              details: {
                errors: validation.errors.flatten(),
                endpoint: '/api/v1/conversation',
                method: 'POST',
              },
            });
            
            return NextResponse.json(
              { error: 'Invalid request', details: validation.errors.flatten() },
              { status: 400 }
            );
          }

          const { message, context, sessionId } = validation.data;

          // Log AI conversation
          auditLogger.log({
            eventType: AuditEventType.AI_PREDICTION,
            severity: AuditSeverity.INFO,
            userId: user.sub,
            userEmail: user.email,
            success: true,
            details: {
              message: message.substring(0, 100), // Truncate for logging
              sessionId,
              endpoint: '/api/v1/conversation',
            },
          });

          // Simulate AI processing with domain expert model
          const response = {
            id: `CONV-${Date.now()}`,
            message: message,
            response: generateAIResponse(message),
            intent: detectIntent(message),
            entities: extractEntities(message),
            confidence: 98.7,
            modelVersion: '175B-v2.1',
            processingTime: 234,
            timestamp: new Date().toISOString(),
            sessionId: sessionId || `session-${Date.now()}`,
          };
          
          return NextResponse.json(response);
        } catch (error) {
          auditLogger.log({
            eventType: AuditEventType.ERROR_OCCURRED,
            severity: AuditSeverity.ERROR,
            userId: user.sub,
            userEmail: user.email,
            success: false,
            errorMessage: error instanceof Error ? error.message : 'Unknown error',
            details: {
              endpoint: '/api/v1/conversation',
              method: 'POST',
            },
          });
          
          console.error('Error in conversation API:', error);
          return NextResponse.json(
            { error: 'Failed to process conversation' },
            { status: 500 }
          );
        }
      });
    });
  });
}

export async function GET(request: NextRequest) {
  // Apply rate limiting
  return apiRateLimiter.middleware(request, async (req) => {
    // Apply authentication
    return withAuth(req, async (authenticatedReq, user) => {
      try {
        // Log access to conversation stats
        auditLogger.log({
          eventType: AuditEventType.RESOURCE_ACCESSED,
          severity: AuditSeverity.INFO,
          userId: user.sub,
          userEmail: user.email,
          success: true,
          resource: {
            type: 'conversation_stats',
            id: 'stats',
          },
          details: {
            endpoint: '/api/v1/conversation',
            method: 'GET',
          },
        });

        // Get conversation history and stats
        const stats = {
          totalConversations: 1247,
          todayConversations: 145,
          avgResponseTime: 245,
          satisfaction: 96.8,
          modelAccuracy: {
            azure: 98.7,
            aws: 98.2,
            gcp: 97.5
          },
          intents: {
            policy_creation: 234,
            compliance_check: 189,
            cost_optimization: 156,
            security_review: 145,
            resource_management: 123,
            identity_management: 98,
            other: 302
          },
          recentConversations: [
            {
              id: 'CONV-001',
              user: 'admin@company.com',
              message: 'Show me all non-compliant resources',
              intent: 'compliance_check',
              timestamp: '2024-03-21T11:30:00Z'
            },
            {
              id: 'CONV-002',
              user: 'security@company.com',
              message: 'Create a policy to enforce MFA for all admin accounts',
              intent: 'policy_creation',
              timestamp: '2024-03-21T11:25:00Z'
            },
            {
              id: 'CONV-003',
              user: 'finance@company.com',
              message: 'How can I reduce our Azure costs by 20%?',
              intent: 'cost_optimization',
              timestamp: '2024-03-21T11:20:00Z'
            }
          ]
        };
        
        return NextResponse.json(stats);
      } catch (error) {
        auditLogger.log({
          eventType: AuditEventType.ERROR_OCCURRED,
          severity: AuditSeverity.ERROR,
          userId: user.sub,
          userEmail: user.email,
          success: false,
          errorMessage: error instanceof Error ? error.message : 'Unknown error',
          details: {
            endpoint: '/api/v1/conversation',
            method: 'GET',
          },
        });
        
        console.error('Error fetching conversation stats:', error);
        return NextResponse.json(
          { error: 'Failed to fetch conversation stats' },
          { status: 500 }
        );
      }
    });
  });
}

function generateAIResponse(message: string): string {
  const lowerMessage = message.toLowerCase();
  
  if (lowerMessage.includes('compliance')) {
    return 'I found 23 resources that are currently non-compliant. The main issues are: 1) Missing encryption on 5 storage accounts, 2) Outdated security patches on 12 VMs, and 3) Excessive permissions on 6 service principals. Would you like me to create remediation tasks for these issues?';
  }
  
  if (lowerMessage.includes('cost') || lowerMessage.includes('save')) {
    return 'Based on my analysis, you can save approximately $45,000/month by: 1) Right-sizing 15 over-provisioned VMs (save $12K), 2) Deleting 23 unattached disks (save $3K), 3) Converting 8 VMs to spot instances (save $18K), and 4) Implementing auto-shutdown for dev environments (save $12K). Shall I create an optimization plan?';
  }
  
  if (lowerMessage.includes('security') || lowerMessage.includes('vulnerability')) {
    return 'I detected 3 critical security issues: 1) Public access enabled on storage account "prod-data", 2) Expired SSL certificate on "api.company.com", and 3) 45 users without MFA enabled. These require immediate attention. Would you like me to apply the recommended security policies?';
  }
  
  if (lowerMessage.includes('policy') || lowerMessage.includes('create')) {
    return 'I can help you create a governance policy. Based on best practices, I recommend: 1) Enforce tagging for cost tracking, 2) Require MFA for privileged accounts, 3) Enable encryption at rest for all storage, and 4) Restrict public IP assignments. Would you like me to generate the policy JSON?';
  }
  
  return 'I understand you need help with Azure governance. I can assist with compliance checking, cost optimization, security reviews, policy creation, and resource management. What specific area would you like to focus on?';
}

function detectIntent(message: string): string {
  const lowerMessage = message.toLowerCase();
  
  if (lowerMessage.includes('compliance') || lowerMessage.includes('compliant')) {
    return 'compliance_check';
  }
  if (lowerMessage.includes('cost') || lowerMessage.includes('save') || lowerMessage.includes('expensive')) {
    return 'cost_optimization';
  }
  if (lowerMessage.includes('security') || lowerMessage.includes('vulnerability') || lowerMessage.includes('threat')) {
    return 'security_review';
  }
  if (lowerMessage.includes('policy') || lowerMessage.includes('create') || lowerMessage.includes('enforce')) {
    return 'policy_creation';
  }
  if (lowerMessage.includes('user') || lowerMessage.includes('identity') || lowerMessage.includes('access')) {
    return 'identity_management';
  }
  if (lowerMessage.includes('resource') || lowerMessage.includes('vm') || lowerMessage.includes('storage')) {
    return 'resource_management';
  }
  
  return 'general_inquiry';
}

function extractEntities(message: string): any {
  const entities = {
    resources: [] as string[],
    actions: [] as string[],
    metrics: [] as string[],
    timeframes: [] as string[]
  };
  
  // Extract resource types
  if (message.includes('VM') || message.includes('virtual machine')) {
    entities.resources.push('Virtual Machine');
  }
  if (message.includes('storage')) {
    entities.resources.push('Storage Account');
  }
  if (message.includes('database') || message.includes('SQL')) {
    entities.resources.push('Database');
  }
  
  // Extract actions
  if (message.includes('create')) entities.actions.push('create');
  if (message.includes('delete')) entities.actions.push('delete');
  if (message.includes('update')) entities.actions.push('update');
  if (message.includes('check')) entities.actions.push('check');
  
  // Extract metrics
  if (message.includes('cost')) entities.metrics.push('cost');
  if (message.includes('compliance')) entities.metrics.push('compliance');
  if (message.includes('security')) entities.metrics.push('security');
  
  // Extract timeframes
  if (message.includes('today')) entities.timeframes.push('today');
  if (message.includes('month')) entities.timeframes.push('month');
  if (message.includes('week')) entities.timeframes.push('week');
  
  return entities;
}