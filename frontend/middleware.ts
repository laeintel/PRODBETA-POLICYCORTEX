/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { generateNonce, generateCSP } from '@/lib/security/nonce'

// Check for demo mode from environment
const BYPASS_ROUTE_AUTH = process.env.NEXT_PUBLIC_DEMO_MODE === 'true'

// Protected routes that require authentication
// Actually, we'll protect everything except login
const protectedRoutes = [
  '/', // Landing page is also protected
  '/dashboard',
  '/tactical',
  '/ai-expert',
  '/chat',
  '/policies',
  '/rbac',
  // '/costs', // old design removed
  '/network',
  '/resources',
  // '/settings', // old design removed
  '/security',
  '/training',
  '/anomalies',
  '/roadmap',
  '/api/v1' // Protect all API routes
]

// Public routes that don't require authentication
// ONLY login and auth endpoints should be public
const publicRoutes = [
  '/login',
  '/api/auth'
]

export function middleware(request: NextRequest) {
  // Generate nonce for CSP
  const nonce = generateNonce()
  
  // Clone request headers to add nonce
  const requestHeaders = new Headers(request.headers)
  requestHeaders.set('x-nonce', nonce)
  
  // Handle authentication bypass for demo mode
  if (BYPASS_ROUTE_AUTH) {
    const response = NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    })
    
    // Apply security headers even in demo mode
    applySecurityHeaders(response, nonce)
    return response
  }
  
  const { pathname } = request.nextUrl

  // Only trust server-issued auth token presence (httpOnly). Do NOT trust client-set flags.
  const authToken = request.cookies.get('auth-token')
  const isAuthenticated = !!authToken

  // Only allow root (login) and auth endpoints without authentication
  if (pathname === '/' || pathname === '/login' || pathname.startsWith('/api/auth')) {
    // If already authenticated and trying to access login pages, redirect to tactical
    if (isAuthenticated && (pathname === '/' || pathname === '/login')) {
      const response = NextResponse.redirect(new URL('/tactical', request.url))
      applySecurityHeaders(response, nonce)
      return response
    }
    const response = NextResponse.next({
      request: {
        headers: requestHeaders,
      },
    })
    applySecurityHeaders(response, nonce)
    return response
  }

  // For all other routes, require authentication
  if (!isAuthenticated) {
    // For API routes, return 401 instead of redirecting
    if (pathname.startsWith('/api/')) {
      const response = NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      )
      applySecurityHeaders(response, nonce)
      return response
    }
    
    // For other routes, redirect to root (login page)
    const response = NextResponse.redirect(new URL('/', request.url))
    applySecurityHeaders(response, nonce)
    return response
  }

  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  })
  applySecurityHeaders(response, nonce)
  return response
}

function applySecurityHeaders(response: NextResponse, nonce: string) {
  const isProd = process.env.NODE_ENV === 'production'
  
  // Apply nonce-based CSP unless disabled
  if (process.env.DISABLE_CSP !== 'true') {
    response.headers.set('Content-Security-Policy', generateCSP(nonce))
  }
  
  // Additional security headers
  response.headers.set('X-Frame-Options', 'DENY')
  response.headers.set('X-Content-Type-Options', 'nosniff')
  response.headers.set('X-XSS-Protection', '1; mode=block')
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin')
  response.headers.set('Permissions-Policy', 
    'camera=(), microphone=(), geolocation=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()'
  )
  
  // HSTS in production
  if (isProd) {
    response.headers.set('Strict-Transport-Security', 'max-age=31536000; includeSubDomains; preload')
  }
  
  // Add nonce to response for client-side access
  response.headers.set('X-Nonce', nonce)
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files with extensions
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\..*|api/health).*)',
  ],
}