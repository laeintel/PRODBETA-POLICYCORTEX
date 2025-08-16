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

// Bypass auth checks in middleware when explicitly enabled (e.g., local dev)
const BYPASS_ROUTE_AUTH = process.env.BYPASS_ROUTE_AUTH === 'true' || process.env.NODE_ENV !== 'production'

// Protected routes that require authentication
const protectedRoutes = [
  '/dashboard',
  '/ai-expert',
  '/chat',
  '/policies',
  '/rbac',
  '/costs',
  '/network',
  '/resources',
  '/settings',
  '/security',
  '/training',
  '/anomalies',
  '/roadmap'
]

// Public routes that don't require authentication
const publicRoutes = [
  '/',
  '/login',
  '/features',
  '/about',
  '/api/health'
]

export function middleware(request: NextRequest) {
  if (BYPASS_ROUTE_AUTH) {
    return NextResponse.next()
  }
  const { pathname } = request.nextUrl

  // Allow public routes
  if (publicRoutes.some(route => pathname === route || pathname.startsWith('/api/health'))) {
    return NextResponse.next()
  }

  // Check if the route is protected
  const isProtectedRoute = protectedRoutes.some(route => pathname.startsWith(route))

  if (isProtectedRoute) {
    // Check for authentication in cookies/session
    const sessionCookie = request.cookies.get('msal.session')
    const tokenCache = request.cookies.get('msal.token.cache')
    
    // Also check sessionStorage keys for MSAL authentication
    // Note: We can't directly access sessionStorage from middleware
    // but we can check for the presence of MSAL cookies or headers
    
    // If no authentication evidence found, redirect to login
    if (!sessionCookie && !tokenCache) {
      // Store the original URL to redirect back after login
      const loginUrl = new URL('/login', request.url)
      loginUrl.searchParams.set('returnUrl', pathname)
      
      return NextResponse.redirect(loginUrl)
    }
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