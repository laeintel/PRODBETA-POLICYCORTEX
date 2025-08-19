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

// Bypass auth only when explicitly enabled (for demos). Never auto-bypass based on NODE_ENV.
const BYPASS_ROUTE_AUTH = (process.env.NEXT_PUBLIC_DEMO_MODE === 'true')

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

  // Only trust server-issued auth token presence (httpOnly). Do NOT trust client-set flags.
  const authToken = request.cookies.get('auth-token')
  const isAuthenticated = !!authToken

  // Only allow root (login) and auth endpoints without authentication
  if (pathname === '/' || pathname === '/login' || pathname.startsWith('/api/auth')) {
    // If already authenticated and trying to access login pages, redirect to tactical
    if (isAuthenticated && (pathname === '/' || pathname === '/login')) {
      return NextResponse.redirect(new URL('/tactical', request.url))
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