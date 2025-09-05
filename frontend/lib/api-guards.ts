/**
 * API Guards - TD.MD Fail-Fast Implementation
 * Prevents mock data leakage in production by enforcing strict real-data checks
 */

import { NextResponse } from 'next/server';

export interface GuardResult {
  allowed: boolean;
  response?: NextResponse;
}

export interface ConfigurationError {
  error: string;
  message: string;
  hint: string;
  required_config?: string[];
  documentation?: string;
}

/**
 * Check if the application is in demo mode
 */
export function isDemoMode(): boolean {
  return process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
}

/**
 * Check if the application should use real data
 */
export function useRealData(): boolean {
  return process.env.USE_REAL_DATA === 'true';
}

/**
 * Guard against serving mock data in production
 * Returns 503 with configuration hints when real mode is enabled but service is unavailable
 */
export function failFastGuard(serviceName: string): GuardResult {
  const isDemo = isDemoMode();
  const realData = useRealData();
  
  // In demo mode, allow mock data
  if (isDemo && !realData) {
    return { allowed: true };
  }
  
  // In real data mode, check for proper configuration
  if (realData) {
    const missingConfig = checkRequiredConfiguration(serviceName);
    
    if (missingConfig.length > 0) {
      const error: ConfigurationError = {
        error: 'Service Unavailable',
        message: `${serviceName} is not configured for real data mode`,
        hint: `Configure the following environment variables: ${missingConfig.join(', ')}`,
        required_config: missingConfig,
        documentation: '/docs/REVAMP/REAL_MODE_SETUP.md'
      };
      
      return {
        allowed: false,
        response: NextResponse.json(error, { status: 503 })
      };
    }
    
    // Even with configuration, we need actual implementation
    // This is where real service connections would be checked
    const error: ConfigurationError = {
      error: 'Service Unavailable',
      message: `${serviceName} real data connection not implemented`,
      hint: 'Real data mode is enabled but service integration is pending',
      documentation: '/docs/REVAMP/REAL_MODE_SETUP.md'
    };
    
    return {
      allowed: false,
      response: NextResponse.json(error, { status: 503 })
    };
  }
  
  // Default to demo mode if not explicitly configured
  return { allowed: true };
}

/**
 * Check required configuration for a service
 */
function checkRequiredConfiguration(serviceName: string): string[] {
  const missing: string[] = [];
  
  switch (serviceName.toLowerCase()) {
    case 'azure':
    case 'resources':
      if (!process.env.AZURE_SUBSCRIPTION_ID) missing.push('AZURE_SUBSCRIPTION_ID');
      if (!process.env.NEXT_PUBLIC_AZURE_TENANT_ID) missing.push('NEXT_PUBLIC_AZURE_TENANT_ID');
      if (!process.env.NEXT_PUBLIC_AZURE_CLIENT_ID) missing.push('NEXT_PUBLIC_AZURE_CLIENT_ID');
      break;
      
    case 'database':
    case 'policies':
      if (!process.env.DATABASE_URL) missing.push('DATABASE_URL');
      break;
      
    case 'ml':
    case 'predictions':
      if (!process.env.ML_SERVICE_URL) missing.push('ML_SERVICE_URL');
      break;
      
    case 'cache':
      if (!process.env.REDIS_URL) missing.push('REDIS_URL');
      break;
      
    case 'graphql':
      if (!process.env.GRAPHQL_ENDPOINT) missing.push('GRAPHQL_ENDPOINT');
      break;
  }
  
  return missing;
}

/**
 * Create a standard 503 response for unconfigured services
 */
export function createServiceUnavailableResponse(
  serviceName: string,
  additionalInfo?: Record<string, any>
): NextResponse {
  const error: ConfigurationError = {
    error: 'Service Unavailable',
    message: `${serviceName} service is not available`,
    hint: useRealData() 
      ? `Configure ${serviceName} for real data mode. See documentation.`
      : 'Service temporarily unavailable',
    documentation: '/docs/REVAMP/REAL_MODE_SETUP.md',
    ...additionalInfo
  };
  
  return NextResponse.json(error, { status: 503 });
}

/**
 * Validate API request and apply fail-fast guard
 * Use this at the beginning of all API route handlers
 */
export async function validateApiRequest(
  serviceName: string,
  request?: Request
): Promise<GuardResult> {
  // Apply fail-fast guard
  const guard = failFastGuard(serviceName);
  
  if (!guard.allowed) {
    return guard;
  }
  
  // Additional request validation can be added here
  // e.g., rate limiting, authentication checks, etc.
  
  return { allowed: true };
}

/**
 * Wrap an API handler with fail-fast protection
 */
export function withFailFastProtection<T extends any[], R>(
  serviceName: string,
  handler: (...args: T) => Promise<R>
) {
  return async (...args: T): Promise<R | NextResponse> => {
    const guard = failFastGuard(serviceName);
    
    if (!guard.allowed && guard.response) {
      return guard.response as any;
    }
    
    try {
      return await handler(...args);
    } catch (error) {
      console.error(`API Error in ${serviceName}:`, error);
      
      // Return configuration hint if it's a connection error
      if (error instanceof Error && 
          (error.message.includes('ECONNREFUSED') || 
           error.message.includes('ETIMEDOUT') ||
           error.message.includes('ENOTFOUND'))) {
        return createServiceUnavailableResponse(serviceName, {
          error_type: 'connection_failed',
          details: useRealData() 
            ? 'Service connection failed. Check if the service is running and properly configured.'
            : undefined
        }) as any;
      }
      
      // Generic error response
      return NextResponse.json(
        { 
          error: 'Internal Server Error',
          message: `An error occurred in ${serviceName} service`
        },
        { status: 500 }
      ) as any;
    }
  };
}