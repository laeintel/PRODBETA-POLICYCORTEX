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
  }
  if (process.env.NEXT_PUBLIC_API_URL) allowedConnect.add(process.env.NEXT_PUBLIC_API_URL);
  if (process.env.NEXT_PUBLIC_WS_URL) allowedConnect.add(process.env.NEXT_PUBLIC_WS_URL);
  if (useWs) { 
    allowedConnect.add('wss:'); 
    if (!isProd) allowedConnect.add('ws:');
  }
  
  // Pure nonce-based script policy - no unsafe-inline or unsafe-eval
  // In development, we need to whitelist Next.js specific hashes for HMR
  const nextJsHashes = !isProd ? [
    "'sha256-FhKqPZm0peBsmG8CjbPVnEJX1QoAqkfRDvL1XAIiZKc='", // Next.js development runtime
    "'sha256-2FkMoYIfzHsQvRmT2WsrQlGNFX2x5L6eBqhSVZYKhGg='", // Next.js error overlay
  ].join(' ') : '';
  
  // Strict nonce-based script policy
  const scriptSrc = `script-src 'nonce-${nonce}' 'strict-dynamic' ${nextJsHashes}`;
  
  // Style with nonce only - no unsafe-inline
  // For Next.js CSS-in-JS, we need specific hashes in production
  const styleHashes = isProd ? [
    "'sha256-47DEjpKa0EqIR1lnSNjzBLLiMcDLsVvTKAMTeWP8YB0='", // Next.js production CSS
  ].join(' ') : '';
  
  const styleSrc = `style-src 'self' 'nonce-${nonce}' ${styleHashes}`;
  
  const cspDirectives = [
    "default-src 'self'",
    scriptSrc,
    styleSrc,
    "img-src 'self' data: blob: https:",
    "font-src 'self' data:",
    `connect-src ${Array.from(allowedConnect).join(' ')} https://o921931.ingest.us.sentry.io`,
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "upgrade-insecure-requests",
  ];
  
  // Add report-uri for CSP violation monitoring
  if (isProd || reportOnly) {
    cspDirectives.push("report-uri /api/v1/csp-report");
    cspDirectives.push("report-to csp-endpoint");
  }
  
  return cspDirectives.join('; ');
}