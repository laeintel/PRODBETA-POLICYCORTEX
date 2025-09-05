/**
 * Health Check Endpoint - TD.MD compliance
 * Provides comprehensive health status with sub-checks for all services
 */

import { NextResponse } from 'next/server';

interface HealthCheck {
  healthy: boolean;
  message?: string;
  details?: any;
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  checks: {
    db_ok: boolean;
    provider_ok: boolean;
    ml_service_ok: boolean;
    cache_ok: boolean;
    auth_ok: boolean;
  };
  mode: {
    demo_mode: boolean;
    real_data: boolean;
  };
  details: {
    database?: HealthCheck;
    azure_provider?: HealthCheck;
    ml_service?: HealthCheck;
    cache?: HealthCheck;
    authentication?: HealthCheck;
  };
}

async function checkDatabase(): Promise<HealthCheck> {
  // In real mode, would check actual database connection
  const useRealData = process.env.USE_REAL_DATA === 'true';
  
  if (!useRealData) {
    return {
      healthy: true,
      message: 'Mock mode - database check skipped'
    };
  }
  
  // Check if database URL is configured
  const dbUrl = process.env.DATABASE_URL;
  if (!dbUrl) {
    return {
      healthy: false,
      message: 'DATABASE_URL not configured',
      details: { hint: 'Set DATABASE_URL in environment variables' }
    };
  }
  
  // In production, would attempt actual connection
  // For now, return config check only
  return {
    healthy: false,
    message: 'Database connection not implemented',
    details: { configured: true, connected: false }
  };
}

async function checkAzureProvider(): Promise<HealthCheck> {
  const useRealData = process.env.USE_REAL_DATA === 'true';
  
  if (!useRealData) {
    return {
      healthy: true,
      message: 'Mock mode - Azure check skipped'
    };
  }
  
  // Check Azure configuration
  const tenantId = process.env.NEXT_PUBLIC_AZURE_TENANT_ID;
  const clientId = process.env.NEXT_PUBLIC_AZURE_CLIENT_ID;
  const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID;
  
  if (!tenantId || !clientId) {
    return {
      healthy: false,
      message: 'Azure credentials not configured',
      details: {
        tenant_configured: !!tenantId,
        client_configured: !!clientId,
        subscription_configured: !!subscriptionId,
        hint: 'Configure Azure credentials in environment variables'
      }
    };
  }
  
  // In production, would attempt actual Azure API call
  return {
    healthy: false,
    message: 'Azure connection not implemented',
    details: { 
      configured: true, 
      connected: false,
      tenant_id: tenantId,
      client_id: clientId
    }
  };
}

async function checkMLService(): Promise<HealthCheck> {
  try {
    // Check if ML service endpoint is available
    const mlEndpoint = process.env.ML_SERVICE_URL || 'http://localhost:8082';
    
    // In real implementation, would make actual health check request
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    
    try {
      const response = await fetch(`${mlEndpoint}/health`, {
        signal: controller.signal,
        method: 'GET'
      }).catch(() => null);
      
      clearTimeout(timeoutId);
      
      if (response && response.ok) {
        return {
          healthy: true,
          message: 'ML service responding'
        };
      }
    } catch (error) {
      clearTimeout(timeoutId);
    }
    
    return {
      healthy: false,
      message: 'ML service not responding',
      details: { endpoint: mlEndpoint }
    };
  } catch (error) {
    return {
      healthy: false,
      message: 'ML service check failed',
      details: { error: error instanceof Error ? error.message : 'Unknown error' }
    };
  }
}

async function checkCache(): Promise<HealthCheck> {
  const useRealData = process.env.USE_REAL_DATA === 'true';
  
  if (!useRealData) {
    return {
      healthy: true,
      message: 'Mock mode - cache check skipped'
    };
  }
  
  // Check Redis/DragonflyDB configuration
  const redisUrl = process.env.REDIS_URL;
  
  if (!redisUrl) {
    return {
      healthy: false,
      message: 'Cache not configured',
      details: { hint: 'Set REDIS_URL in environment variables' }
    };
  }
  
  // In production, would attempt actual Redis ping
  return {
    healthy: false,
    message: 'Cache connection not implemented',
    details: { configured: true, connected: false }
  };
}

async function checkAuthentication(): Promise<HealthCheck> {
  const demoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  
  if (demoMode) {
    return {
      healthy: true,
      message: 'Demo mode - authentication bypassed'
    };
  }
  
  // Check MSAL configuration
  const tenantId = process.env.NEXT_PUBLIC_AZURE_TENANT_ID;
  const clientId = process.env.NEXT_PUBLIC_AZURE_CLIENT_ID;
  
  if (!tenantId || !clientId) {
    return {
      healthy: false,
      message: 'Authentication not configured',
      details: {
        tenant_configured: !!tenantId,
        client_configured: !!clientId,
        hint: 'Configure Azure AD credentials'
      }
    };
  }
  
  return {
    healthy: true,
    message: 'Authentication configured',
    details: {
      provider: 'Azure AD',
      tenant_id: tenantId
    }
  };
}

export async function GET() {
  const startTime = Date.now();
  
  // Run all health checks in parallel
  const [database, azureProvider, mlService, cache, authentication] = await Promise.all([
    checkDatabase(),
    checkAzureProvider(),
    checkMLService(),
    checkCache(),
    checkAuthentication()
  ]);
  
  // Determine overall status
  const allHealthy = [database, azureProvider, mlService, cache, authentication]
    .every(check => check.healthy);
  
  const anyUnhealthy = [database, azureProvider, mlService, cache, authentication]
    .some(check => !check.healthy);
  
  const status: HealthStatus['status'] = 
    allHealthy ? 'healthy' : 
    anyUnhealthy ? 'degraded' : 
    'healthy';
  
  const healthStatus: HealthStatus = {
    status,
    timestamp: new Date().toISOString(),
    checks: {
      db_ok: database.healthy,
      provider_ok: azureProvider.healthy,
      ml_service_ok: mlService.healthy,
      cache_ok: cache.healthy,
      auth_ok: authentication.healthy
    },
    mode: {
      demo_mode: process.env.NEXT_PUBLIC_DEMO_MODE === 'true',
      real_data: process.env.USE_REAL_DATA === 'true'
    },
    details: {
      database,
      azure_provider: azureProvider,
      ml_service: mlService,
      cache,
      authentication
    }
  };
  
  // Add response time
  const responseTime = Date.now() - startTime;
  (healthStatus as any).response_time_ms = responseTime;
  
  // Return appropriate status code based on health
  const statusCode = status === 'healthy' ? 200 : status === 'degraded' ? 503 : 500;
  
  return NextResponse.json(healthStatus, { status: statusCode });
}