import crypto from 'crypto';
import { headers } from 'next/headers';

/**
 * Generate a cryptographically secure nonce for CSP
 */
export function generateNonce(): string {
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
 */
export function generateCSP(nonce: string): string {
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
  
  // Nonce-based script policy - much more secure than unsafe-inline
  const scriptSrc = `script-src 'self' 'nonce-${nonce}' 'strict-dynamic' https: ${!isProd ? "'unsafe-eval'" : ""}`;
  
  // Style with nonce for inline styles
  const styleSrc = `style-src 'self' 'nonce-${nonce}'`;
  
  return [
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
  ].join('; ');
}