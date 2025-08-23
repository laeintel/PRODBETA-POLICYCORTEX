import { headers } from 'next/headers';

/**
 * Generate a cryptographically secure nonce for CSP
 */
export function generateNonce(): string {
  // Use Web Crypto API for Edge Runtime compatibility
  if (typeof globalThis.crypto !== 'undefined' && globalThis.crypto.getRandomValues) {
    const array = new Uint8Array(16);
    globalThis.crypto.getRandomValues(array);
    return Buffer.from(array).toString('base64');
  }
  
  // Fallback for Node.js environment
  const crypto = require('crypto');
  return crypto.randomBytes(16).toString('base64');
}

/**
 * Get the CSP nonce from request headers (set by middleware)
 */
export function getNonce(): string {
  const headersList = headers();
  return headersList.get('x-nonce') || '';
}

/**
 * Generate CSP header with nonce
 * @param nonce - The nonce for this request
 * @param reportOnly - Whether to use Content-Security-Policy-Report-Only header
 */
export function generateCSP(nonce: string, reportOnly: boolean = false): string {
  const isProd = process.env.NODE_ENV === 'production';
  const useWs = (process.env.NEXT_PUBLIC_USE_WS || '').toLowerCase() === 'true';
  
  const allowedConnect = new Set(["'self'", 'https:']);
  if (!isProd) {
    allowedConnect.add('http://localhost:8080');
    allowedConnect.add('http://localhost:4000');
    allowedConnect.add('http://localhost:3000');
    allowedConnect.add('http://localhost:3001');
    allowedConnect.add('http://localhost:3002');
  }
  if (process.env.NEXT_PUBLIC_API_URL) allowedConnect.add(process.env.NEXT_PUBLIC_API_URL);
  if (process.env.NEXT_PUBLIC_WS_URL) allowedConnect.add(process.env.NEXT_PUBLIC_WS_URL);
  if (useWs) { 
    allowedConnect.add('wss:'); 
    if (!isProd) allowedConnect.add('ws:');
  }
  
  // Add Sentry endpoint only if configured
  if (process.env.NEXT_PUBLIC_SENTRY_DSN) {
    // Extract the domain from the DSN
    try {
      const sentryUrl = new URL(process.env.NEXT_PUBLIC_SENTRY_DSN);
      const sentryIngest = `https://${sentryUrl.hostname.replace('.ingest.', '.ingest.us.')}`;
      allowedConnect.add(sentryIngest);
    } catch (e) {
      // Invalid Sentry DSN, skip
    }
  }
  
  // In development, be more permissive with scripts to avoid breaking HMR
  // React development mode requires unsafe-eval for hot reload functionality
  const scriptSrc = !isProd 
    ? `script-src 'nonce-${nonce}' 'strict-dynamic' 'unsafe-eval' 'sha256-drs6v8sKWnmmrrD9KTCfeZyk9sh/EMNvzKJUm8rdVwo=' 'sha256-YoiTZbP35ftJSuqcXHIQKR0GkOgvwuSrIESq73qEh+4='`
    : `script-src 'nonce-${nonce}' 'strict-dynamic'`;
  
  // Style needs unsafe-inline in development for hot reload to work properly
  const styleSrc = !isProd
    ? `style-src 'self' 'unsafe-inline'`
    : `style-src 'self' 'nonce-${nonce}'`;
  
  const cspDirectives = [
    "default-src 'self'",
    scriptSrc,
    styleSrc,
    "img-src 'self' data: blob: https:",
    "font-src 'self' data:",
    `connect-src ${Array.from(allowedConnect).join(' ')}`,
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ];
  
  // Only add upgrade-insecure-requests in production
  if (isProd) {
    cspDirectives.push("upgrade-insecure-requests");
  }
  
  // Add report-uri for CSP violation monitoring only in production
  if (isProd) {
    cspDirectives.push("report-uri /api/v1/csp-report");
    cspDirectives.push("report-to csp-endpoint");
  }
  
  return cspDirectives.join('; ');
}