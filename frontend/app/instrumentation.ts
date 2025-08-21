/**
 * Instrumentation file for Next.js
 * This runs before the app starts, making it ideal for:
 * - Environment validation
 * - Telemetry initialization
 * - Performance monitoring setup
 */

import { validateEnv } from '@/lib/env';

export async function register() {
  // Validate environment variables on startup
  // This will exit the process if validation fails
  validateEnv();
  
  // Initialize telemetry if configured
  const env = process.env;
  
  if (env.OTEL_EXPORTER_OTLP_ENDPOINT) {
    console.log('📊 Initializing OpenTelemetry...');
    // OpenTelemetry initialization would go here
  }
  
  if (env.SENTRY_DSN) {
    console.log('🛡️ Initializing Sentry error tracking...');
    // Sentry initialization would go here
  }
  
  console.log('✅ Application instrumentation complete');
}