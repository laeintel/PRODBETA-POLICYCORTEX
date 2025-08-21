import { z } from 'zod';

// Common validation patterns
const emailSchema = z.string().email().max(255);
const uuidSchema = z.string().uuid();
const dateSchema = z.string().datetime();
const urlSchema = z.string().url();

// Pagination schema
export const paginationSchema = z.object({
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().positive().max(100).default(20),
  sort: z.string().optional(),
  order: z.enum(['asc', 'desc']).optional(),
});

// Resource schemas
export const resourceIdSchema = z.object({
  id: uuidSchema,
});

export const resourceQuerySchema = z.object({
  type: z.enum(['vm', 'storage', 'database', 'network', 'compute']).optional(),
  status: z.enum(['running', 'stopped', 'pending', 'error']).optional(),
  resourceGroup: z.string().optional(),
  search: z.string().max(100).optional(),
  ...paginationSchema.shape,
});

export const createResourceSchema = z.object({
  name: z.string().min(1).max(255),
  type: z.enum(['vm', 'storage', 'database', 'network', 'compute']),
  resourceGroup: z.string().min(1).max(255),
  location: z.string().min(1).max(100),
  tags: z.record(z.string(), z.string()).optional(),
  configuration: z.record(z.string(), z.any()).optional(),
});

// User/Auth schemas
export const loginSchema = z.object({
  email: emailSchema,
  password: z.string().min(8).max(100),
  rememberMe: z.boolean().optional(),
});

export const userProfileSchema = z.object({
  name: z.string().min(1).max(255),
  email: emailSchema,
  phone: z.string().regex(/^\+?[1-9]\d{1,14}$/).optional(),
  department: z.string().max(100).optional(),
  role: z.enum(['admin', 'user', 'viewer', 'operator']).optional(),
});

// RBAC schemas
export const roleAssignmentSchema = z.object({
  principalId: uuidSchema,
  roleId: uuidSchema,
  scope: z.string().min(1),
  expiresAt: dateSchema.optional(),
});

export const permissionRequestSchema = z.object({
  permissionId: z.string().min(1),
  justification: z.string().min(10).max(1000),
  duration: z.number().int().positive().max(365),
});

// Compliance schemas
export const policyViolationSchema = z.object({
  resourceId: uuidSchema,
  policyId: z.string().min(1),
  severity: z.enum(['critical', 'high', 'medium', 'low']),
  description: z.string().max(1000),
  remediation: z.string().max(2000).optional(),
});

export const complianceReportSchema = z.object({
  startDate: dateSchema,
  endDate: dateSchema,
  frameworks: z.array(z.enum(['SOC2', 'ISO27001', 'HIPAA', 'PCI-DSS', 'GDPR'])),
  includeDetails: z.boolean().optional(),
});

// AI/ML schemas
export const predictionRequestSchema = z.object({
  resourceId: uuidSchema.optional(),
  timeframe: z.enum(['24h', '7d', '30d', '90d']),
  metric: z.enum(['compliance', 'cost', 'performance', 'security']),
});

export const conversationSchema = z.object({
  message: z.string().min(1).max(5000),
  context: z.array(z.string()).optional(),
  sessionId: uuidSchema.optional(),
});

// Correlation schemas
export const correlationQuerySchema = z.object({
  domain: z.enum(['security', 'compliance', 'cost', 'performance']).optional(),
  resourceIds: z.array(uuidSchema).optional(),
  timeRange: z.object({
    start: dateSchema,
    end: dateSchema,
  }).optional(),
  minCorrelation: z.number().min(0).max(1).optional(),
});

// Cost schemas
export const costAnalysisSchema = z.object({
  subscriptionId: uuidSchema,
  startDate: dateSchema,
  endDate: dateSchema,
  groupBy: z.enum(['resource', 'service', 'location', 'tag']).optional(),
  includeForecasts: z.boolean().optional(),
});

// Security schemas
export const securityAlertSchema = z.object({
  title: z.string().min(1).max(255),
  description: z.string().max(2000),
  severity: z.enum(['critical', 'high', 'medium', 'low', 'info']),
  resourceIds: z.array(uuidSchema),
  mitigationSteps: z.array(z.string()).optional(),
});

export const threatDetectionSchema = z.object({
  resourceId: uuidSchema,
  scanType: z.enum(['vulnerability', 'malware', 'intrusion', 'all']),
  deep: z.boolean().optional(),
});

// Export/Import schemas
export const exportRequestSchema = z.object({
  format: z.enum(['csv', 'json', 'pdf', 'excel']),
  data: z.any(),
  filename: z.string().max(255).optional(),
  filters: z.record(z.string(), z.any()).optional(),
});

// Settings schemas
export const settingsUpdateSchema = z.object({
  notifications: z.object({
    email: z.boolean(),
    sms: z.boolean(),
    push: z.boolean(),
    frequency: z.enum(['immediate', 'hourly', 'daily', 'weekly']),
  }).optional(),
  security: z.object({
    mfaEnabled: z.boolean(),
    sessionTimeout: z.number().int().min(5).max(1440),
    passwordPolicy: z.enum(['basic', 'medium', 'strong']),
  }).optional(),
  preferences: z.object({
    theme: z.enum(['light', 'dark', 'system']),
    language: z.string().max(10),
    timezone: z.string().max(50),
  }).optional(),
});

// Validation helper functions
export function validateRequest<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: any } {
  const result = schema.safeParse(data);
  
  if (result.success) {
    return { success: true, data: result.data };
  } else {
    return { success: false, errors: result.error };
  }
}

export function validateAndSanitize<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): T {
  const result = schema.parse(data);
  // Additional sanitization can be added here
  return result;
}