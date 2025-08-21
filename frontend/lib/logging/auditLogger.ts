import { v4 as uuidv4 } from 'uuid';

export enum AuditEventType {
  // Authentication events
  LOGIN_SUCCESS = 'AUTH_LOGIN_SUCCESS',
  LOGIN_FAILED = 'AUTH_LOGIN_FAILED',
  LOGOUT = 'AUTH_LOGOUT',
  TOKEN_REFRESH = 'AUTH_TOKEN_REFRESH',
  MFA_REQUIRED = 'AUTH_MFA_REQUIRED',
  MFA_SUCCESS = 'AUTH_MFA_SUCCESS',
  MFA_FAILED = 'AUTH_MFA_FAILED',
  
  // Authorization events
  ACCESS_GRANTED = 'AUTHZ_ACCESS_GRANTED',
  ACCESS_DENIED = 'AUTHZ_ACCESS_DENIED',
  PERMISSION_CHANGED = 'AUTHZ_PERMISSION_CHANGED',
  ROLE_ASSIGNED = 'AUTHZ_ROLE_ASSIGNED',
  ROLE_REMOVED = 'AUTHZ_ROLE_REMOVED',
  
  // Resource events
  RESOURCE_CREATED = 'RESOURCE_CREATED',
  RESOURCE_UPDATED = 'RESOURCE_UPDATED',
  RESOURCE_DELETED = 'RESOURCE_DELETED',
  RESOURCE_ACCESSED = 'RESOURCE_ACCESSED',
  RESOURCE_EXPORTED = 'RESOURCE_EXPORTED',
  
  // Compliance events
  POLICY_VIOLATION = 'COMPLIANCE_POLICY_VIOLATION',
  REMEDIATION_STARTED = 'COMPLIANCE_REMEDIATION_STARTED',
  REMEDIATION_COMPLETED = 'COMPLIANCE_REMEDIATION_COMPLETED',
  REPORT_GENERATED = 'COMPLIANCE_REPORT_GENERATED',
  
  // Security events
  SECURITY_ALERT = 'SECURITY_ALERT',
  THREAT_DETECTED = 'SECURITY_THREAT_DETECTED',
  VULNERABILITY_FOUND = 'SECURITY_VULNERABILITY_FOUND',
  SECURITY_SCAN = 'SECURITY_SCAN',
  
  // Data events
  DATA_EXPORT = 'DATA_EXPORT',
  DATA_IMPORT = 'DATA_IMPORT',
  DATA_DELETION = 'DATA_DELETION',
  BACKUP_CREATED = 'DATA_BACKUP_CREATED',
  
  // Configuration events
  SETTINGS_CHANGED = 'CONFIG_SETTINGS_CHANGED',
  FEATURE_TOGGLED = 'CONFIG_FEATURE_TOGGLED',
  INTEGRATION_ADDED = 'CONFIG_INTEGRATION_ADDED',
  INTEGRATION_REMOVED = 'CONFIG_INTEGRATION_REMOVED',
  
  // AI/ML events
  AI_PREDICTION = 'AI_PREDICTION_MADE',
  AI_TRAINING = 'AI_MODEL_TRAINING',
  AI_FEEDBACK = 'AI_FEEDBACK_PROVIDED',
  
  // Error events
  ERROR_OCCURRED = 'ERROR_OCCURRED',
  RATE_LIMIT_EXCEEDED = 'ERROR_RATE_LIMIT',
  INVALID_REQUEST = 'ERROR_INVALID_REQUEST',
}

export enum AuditSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical',
}

export interface AuditEvent {
  id: string;
  timestamp: string;
  eventType: AuditEventType;
  severity: AuditSeverity;
  userId?: string;
  userEmail?: string;
  userRoles?: string[];
  ipAddress?: string;
  userAgent?: string;
  resource?: {
    type: string;
    id: string;
    name?: string;
  };
  details?: Record<string, any>;
  metadata?: {
    traceId?: string;
    sessionId?: string;
    correlationId?: string;
    [key: string]: any;
  };
  success: boolean;
  errorMessage?: string;
}

export interface PiiScrubConfig {
  enabled: boolean;
  fields: string[];
  replacementText: string;
}

class AuditLogger {
  private static instance: AuditLogger;
  private events: AuditEvent[] = [];
  private traceId: string = uuidv4();
  private sessionId: string = uuidv4();
  private piiConfig: PiiScrubConfig = {
    enabled: true,
    fields: ['password', 'ssn', 'creditCard', 'email', 'phone', 'address'],
    replacementText: '[REDACTED]',
  };

  private constructor() {
    // Initialize trace ID for request correlation
    this.resetTraceId();
  }

  public static getInstance(): AuditLogger {
    if (!AuditLogger.instance) {
      AuditLogger.instance = new AuditLogger();
    }
    return AuditLogger.instance;
  }

  /**
   * Reset trace ID for new request
   */
  public resetTraceId(): void {
    this.traceId = uuidv4();
  }

  /**
   * Get current trace ID
   */
  public getTraceId(): string {
    return this.traceId;
  }

  /**
   * Set session ID
   */
  public setSessionId(sessionId: string): void {
    this.sessionId = sessionId;
  }

  /**
   * Scrub PII from data
   */
  private scrubPii(data: any): any {
    if (!this.piiConfig.enabled) return data;
    
    if (typeof data === 'string') {
      // Scrub common PII patterns
      return data
        .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, this.piiConfig.replacementText) // Email
        .replace(/\b\d{3}-\d{2}-\d{4}\b/g, this.piiConfig.replacementText) // SSN
        .replace(/\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g, this.piiConfig.replacementText) // Credit card
        .replace(/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, this.piiConfig.replacementText); // Phone
    }
    
    if (typeof data !== 'object' || data === null) return data;
    
    const scrubbed: any = Array.isArray(data) ? [] : {};
    
    for (const key in data) {
      const lowerKey = key.toLowerCase();
      
      // Check if field should be redacted
      if (this.piiConfig.fields.some(field => lowerKey.includes(field))) {
        scrubbed[key] = this.piiConfig.replacementText;
      } else {
        scrubbed[key] = this.scrubPii(data[key]);
      }
    }
    
    return scrubbed;
  }

  /**
   * Log an audit event
   */
  public log(event: Omit<AuditEvent, 'id' | 'timestamp'> & { metadata?: Partial<AuditEvent['metadata']> }): void {
    const auditEvent: AuditEvent = {
      ...event,
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      metadata: {
        ...event.metadata,
        traceId: this.traceId,
        sessionId: this.sessionId,
        correlationId: event.metadata?.correlationId || uuidv4(),
      },
      // Scrub PII from details
      details: this.scrubPii(event.details),
    };

    // Store locally (in production, send to backend)
    this.events.push(auditEvent);
    
    // Send to backend audit service
    this.sendToBackend(auditEvent);
    
    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.log('[AUDIT]', auditEvent);
    }
  }

  /**
   * Send audit event to backend
   */
  private async sendToBackend(event: AuditEvent): Promise<void> {
    // Skip sending in build/SSR context
    if (typeof window === 'undefined') {
      return;
    }
    
    try {
      await fetch('/api/v1/audit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Trace-Id': this.traceId,
        },
        body: JSON.stringify(event),
      });
    } catch (error) {
      console.error('Failed to send audit event:', error);
      // Store failed events for retry
      this.storeFailedEvent(event);
    }
  }

  /**
   * Store failed events for retry
   */
  private storeFailedEvent(event: AuditEvent): void {
    if (typeof window !== 'undefined') {
      const failedEvents = JSON.parse(
        localStorage.getItem('failedAuditEvents') || '[]'
      );
      failedEvents.push(event);
      
      // Keep only last 100 failed events
      if (failedEvents.length > 100) {
        failedEvents.shift();
      }
      
      localStorage.setItem('failedAuditEvents', JSON.stringify(failedEvents));
    }
  }

  /**
   * Retry failed events
   */
  public async retryFailedEvents(): Promise<void> {
    if (typeof window === 'undefined') return;
    
    const failedEvents = JSON.parse(
      localStorage.getItem('failedAuditEvents') || '[]'
    );
    
    if (failedEvents.length === 0) return;
    
    const successfulEvents: AuditEvent[] = [];
    
    for (const event of failedEvents) {
      try {
        await this.sendToBackend(event);
        successfulEvents.push(event);
      } catch (error) {
        // Keep failed events for next retry
      }
    }
    
    // Remove successful events from storage
    const remainingEvents = failedEvents.filter(
      (e: AuditEvent) => !successfulEvents.includes(e)
    );
    
    localStorage.setItem('failedAuditEvents', JSON.stringify(remainingEvents));
  }

  /**
   * Get recent audit events (for UI display)
   */
  public getRecentEvents(limit: number = 100): AuditEvent[] {
    return this.events.slice(-limit);
  }

  /**
   * Clear local event cache
   */
  public clearLocalCache(): void {
    this.events = [];
  }
}

// Export singleton instance
export const auditLogger = AuditLogger.getInstance();

// Helper functions for common audit scenarios
export function auditLogin(userId: string, email: string, success: boolean, ipAddress?: string): void {
  auditLogger.log({
    eventType: success ? AuditEventType.LOGIN_SUCCESS : AuditEventType.LOGIN_FAILED,
    severity: success ? AuditSeverity.INFO : AuditSeverity.WARNING,
    userId,
    userEmail: email,
    ipAddress,
    success,
    details: {
      timestamp: new Date().toISOString(),
      method: 'MSAL',
    },
  });
}

export function auditResourceAccess(
  userId: string,
  resourceType: string,
  resourceId: string,
  action: string,
  success: boolean
): void {
  auditLogger.log({
    eventType: AuditEventType.RESOURCE_ACCESSED,
    severity: AuditSeverity.INFO,
    userId,
    resource: {
      type: resourceType,
      id: resourceId,
    },
    success,
    details: {
      action,
      timestamp: new Date().toISOString(),
    },
  });
}

export function auditDataExport(
  userId: string,
  dataType: string,
  format: string,
  recordCount: number
): void {
  auditLogger.log({
    eventType: AuditEventType.DATA_EXPORT,
    severity: AuditSeverity.INFO,
    userId,
    success: true,
    details: {
      dataType,
      format,
      recordCount,
      timestamp: new Date().toISOString(),
    },
  });
}

export function auditSecurityAlert(
  severity: AuditSeverity,
  alertType: string,
  details: Record<string, any>
): void {
  auditLogger.log({
    eventType: AuditEventType.SECURITY_ALERT,
    severity,
    success: false,
    details: {
      alertType,
      ...details,
      timestamp: new Date().toISOString(),
    },
  });
}