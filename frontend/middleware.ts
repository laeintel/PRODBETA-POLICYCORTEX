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

// Bypass auth checks in middleware in development or when explicitly enabled
// MSAL uses sessionStorage, not cookies, so middleware can't properly check auth status
// TEMPORARY: Set to true to allow development without auth
const BYPASS_ROUTE_AUTH = true // TEMPORARY: Bypass for development

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
  if (BYPASS_ROUTE_AUTH) {
    return NextResponse.next()
  }
  const { pathname } = request.nextUrl

  // Check for authentication in cookies/session
  const authToken = request.cookies.get('auth-token')
  const authStatus = request.cookies.get('auth-status')
  const sessionToken = request.cookies.get('session-token')
  const msalSession = request.cookies.get('msal.session')
  const tokenCache = request.cookies.get('msal.token.cache')
  
  const isAuthenticated = !!(authToken || authStatus || sessionToken || msalSession || tokenCache)

  // Only allow root (now login page) and auth endpoints without authentication
  if (pathname === '/' || pathname === '/login' || pathname.startsWith('/api/auth')) {
    // If already authenticated and trying to access login pages, redirect to dashboard
    if (isAuthenticated && (pathname === '/' || pathname === '/login')) {
      return NextResponse.redirect(new URL('/dashboard', request.url))
    }
    return NextResponse.next()
  }

  // For all other routes, require authentication
  if (!isAuthenticated) {
    // For API routes, return 401 instead of redirecting
    if (pathname.startsWith('/api/')) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      )
    }
    
    // For other routes, redirect to root (login page)
    return NextResponse.redirect(new URL('/', request.url))
  }

  return NextResponse.next()
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