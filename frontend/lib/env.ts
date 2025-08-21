import { z } from 'zod';

/**
 * Environment variable schema for runtime validation
 * This ensures all required environment variables are present and valid
 */
const envSchema = z.object({
  // Node environment
  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),
  
  // API URLs
  NEXT_PUBLIC_API_URL: z.string().url().optional(),
  NEXT_PUBLIC_GRAPHQL_URL: z.string().url().optional(),
  NEXT_PUBLIC_WS_URL: z.string().url().optional(),
  
  // Azure Configuration
  AZURE_SUBSCRIPTION_ID: z.string().uuid().optional(),
  AZURE_TENANT_ID: z.string().uuid().optional(),
  AZURE_CLIENT_ID: z.string().uuid().optional(),
  AZURE_CLIENT_SECRET: z.string().min(1).optional(),
  
  // Authentication
  JWT_SECRET: z.string().min(32).optional(),
  NEXTAUTH_SECRET: z.string().min(32).optional(),
  NEXTAUTH_URL: z.string().url().optional(),
  
  // Database
  DATABASE_URL: z.string().url().optional(),
  REDIS_URL: z.string().url().optional(),
  
  // Security
  DISABLE_CSP: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  
  // Feature Flags
  NEXT_PUBLIC_DEMO_MODE: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  USE_REAL_DATA: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  NEXT_PUBLIC_USE_WS: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  
  // Docker/Container
  IN_DOCKER: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  DOCKER: z.enum(['true', 'false']).transform(val => val === 'true').optional(),
  BACKEND_SERVICE_NAME: z.string().optional(),
  BACKEND_SERVICE_PORT: z.string().regex(/^\d+$/).optional(),
  
  // Monitoring
  SENTRY_DSN: z.string().url().optional(),
  
  // OpenTelemetry
  OTEL_EXPORTER_OTLP_ENDPOINT: z.string().url().optional(),
  OTEL_SERVICE_NAME: z.string().optional(),
  
  // Package Info
  npm_package_version: z.string().optional(),
});

/**
 * Type for validated environment variables
 */
export type Env = z.infer<typeof envSchema>;

/**
 * Validated environment variables
 * This will throw an error if required variables are missing or invalid
 */
let env: Env;

/**
 * Validate environment variables
 * Call this function early in application startup
 */
export function validateEnv(): Env {
  try {
    env = envSchema.parse(process.env);
    
    // Strict production validation
    if (env.NODE_ENV === 'production') {
      const errors: string[] = [];
      
      // Security requirements
      if (!env.JWT_SECRET && !env.NEXTAUTH_SECRET) {
        errors.push('JWT_SECRET or NEXTAUTH_SECRET is required in production');
      }
      
      if (env.DISABLE_CSP === true) {
        errors.push('CSP cannot be disabled in production (DISABLE_CSP must be false or unset)');
      }
      
      // Azure requirements when using real data
      if (env.USE_REAL_DATA) {
        if (!env.AZURE_SUBSCRIPTION_ID) {
          errors.push('AZURE_SUBSCRIPTION_ID is required when USE_REAL_DATA is true');
        }
        if (!env.AZURE_TENANT_ID) {
          errors.push('AZURE_TENANT_ID is required when USE_REAL_DATA is true');
        }
        if (!env.AZURE_CLIENT_ID) {
          errors.push('AZURE_CLIENT_ID is required when USE_REAL_DATA is true');
        }
      }
      
      // Database requirements
      if (!env.DATABASE_URL) {
        errors.push('DATABASE_URL is required in production');
      }
      
      // Monitoring requirements
      if (!env.SENTRY_DSN && !env.OTEL_EXPORTER_OTLP_ENDPOINT) {
        console.warn('⚠️ Warning: No error tracking configured (SENTRY_DSN or OTEL_EXPORTER_OTLP_ENDPOINT)');
      }
      
      // Fail fast if any errors
      if (errors.length > 0) {
        throw new Error(`Production environment validation failed:\n${errors.map(e => `  - ${e}`).join('\n')}`);
      }
    }
    
    // Log successful validation (without exposing secrets)
    console.log('✅ Environment variables validated successfully');
    console.log(`   NODE_ENV: ${env.NODE_ENV}`);
    console.log(`   DEMO_MODE: ${env.NEXT_PUBLIC_DEMO_MODE}`);
    console.log(`   USE_REAL_DATA: ${env.USE_REAL_DATA}`);
    console.log(`   IN_DOCKER: ${env.IN_DOCKER}`);
    
    return env;
  } catch (error) {
    if (error instanceof z.ZodError) {
      console.error('❌ Environment validation failed:');
      error.errors.forEach(err => {
        console.error(`   ${err.path.join('.')}: ${err.message}`);
      });
      
      // In development, provide helpful suggestions
      if (process.env.NODE_ENV !== 'production') {
        console.log('\n💡 Suggestions:');
        console.log('   1. Copy .env.example to .env.local');
        console.log('   2. Fill in the required values');
        console.log('   3. Restart the application');
      }
    } else {
      console.error('❌ Environment validation error:', error);
    }
    
    // Exit the process with error code
    process.exit(1);
  }
}

/**
 * Get validated environment variables
 * Must call validateEnv() first
 */
export function getEnv(): Env {
  if (!env) {
    throw new Error('Environment not validated. Call validateEnv() first.');
  }
  return env;
}

/**
 * Check if running in production
 */
export function isProduction(): boolean {
  return getEnv().NODE_ENV === 'production';
}

/**
 * Check if running in development
 */
export function isDevelopment(): boolean {
  return getEnv().NODE_ENV === 'development';
}

/**
 * Check if running in Docker container
 */
export function isDocker(): boolean {
  const envVars = getEnv();
  return envVars.IN_DOCKER === true || envVars.DOCKER === true;
}

/**
 * Check if demo mode is enabled
 */
export function isDemoMode(): boolean {
  return getEnv().NEXT_PUBLIC_DEMO_MODE === true;
}

/**
 * Get API base URL
 */
export function getApiUrl(): string {
  const envVars = getEnv();
  if (envVars.NEXT_PUBLIC_API_URL) {
    return envVars.NEXT_PUBLIC_API_URL;
  }
  
  if (isDocker()) {
    const service = envVars.BACKEND_SERVICE_NAME || 'core';
    const port = envVars.BACKEND_SERVICE_PORT || '8080';
    return `http://${service}:${port}`;
  }
  
  return 'http://localhost:8080';
}

/**
 * Safe environment variable access
 * Returns undefined instead of throwing for missing optional vars
 */
export function safeEnv<K extends keyof Env>(key: K): Env[K] | undefined {
  try {
    return getEnv()[key];
  } catch {
    return undefined;
  }
}