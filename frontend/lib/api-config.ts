/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

// API configuration with environment-based URL resolution
export function getApiUrl(path: string): string {
  const envBase = process.env.NEXT_PUBLIC_API_URL
  if (envBase) return `${envBase}${path}`

  if (typeof window !== 'undefined') {
    // If running frontend on 3005 (docker), backend is at 8085 on host
    if (window.location.port === '3005') return `http://localhost:8085${path}`
    // Default: let Next.js middleware proxy relative paths
    return path
  }
  // Server-side rendering: keep relative, middleware rewrites
  return path
}

export function getHealthUrl(): string {
  const envBase = process.env.NEXT_PUBLIC_API_URL
  if (envBase) return `${envBase}/health`
  if (typeof window !== 'undefined' && window.location.port === '3005') {
    return 'http://localhost:8085/health'
  }
  return '/health'
}

export const AZURE_OPENAI = {
  endpoint: process.env.NEXT_PUBLIC_AOAI_ENDPOINT || '',
  apiKey: process.env.NEXT_PUBLIC_AOAI_API_KEY || '',
  apiVersion: process.env.NEXT_PUBLIC_AOAI_API_VERSION || '2024-08-01-preview',
  deployment: process.env.NEXT_PUBLIC_AOAI_CHAT_DEPLOYMENT || 'chat-dev',
} as const