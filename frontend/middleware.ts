import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const url = request.nextUrl.clone()
  
  // Dev hint: ensure MSAL redirect origin matches current origin to avoid AADSTS9002326
  try {
    const configured = process.env.NEXT_PUBLIC_MSAL_REDIRECT_URI
    if (configured && process.env.NODE_ENV !== 'production') {
      const cfg = new URL(configured)
      if (cfg.origin !== url.origin) {
        console.warn(`MSAL redirect origin mismatch. Current: ${url.origin} Configured: ${cfg.origin}`)
      }
    }
  } catch {}
  
  // Proxy API requests to backend (prefer explicit env; fall back smartly for local dev)
  let backendBase = process.env.NEXT_PUBLIC_API_URL || 'http://backend:8080'
  // If we're on localhost and no env provided, prefer localhost:8080 instead of docker hostname
  if (!process.env.NEXT_PUBLIC_API_URL && (url.hostname === 'localhost' || url.hostname === '127.0.0.1')) {
    backendBase = 'http://localhost:8080'
  }
  if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
    url.href = `${backendBase}${url.pathname}${url.search}`
    return NextResponse.rewrite(url)
  }
  
  // Proxy actions to backend
  if (url.pathname.startsWith('/actions/')) {
    const newPath = url.pathname.replace('/actions/', '/api/v1/actions/')
    url.href = `${backendBase}${newPath}${url.search}`
    return NextResponse.rewrite(url)
  }
  
  // Proxy GraphQL requests
  if (url.pathname === '/graphql') {
    url.href = `http://graphql:4000${url.pathname}${url.search}`
    return NextResponse.rewrite(url)
  }
}

export const config = {
  matcher: ['/api/:path*', '/health', '/actions/:path*', '/graphql']
}