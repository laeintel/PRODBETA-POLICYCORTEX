import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const url = request.nextUrl.clone()
  // Enforce read-only UI in simulated mode by adding header flag
  if (process.env.NEXT_PUBLIC_USE_REAL_DATA !== 'true') {
    const res = NextResponse.next({ request: { headers: request.headers } })
    res.headers.set('x-data-mode', 'simulated')
    return res
  }
  
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

  // Let next.config.js handle /api and /actions rewrites to avoid env coupling.
  // Only handle /health here for local development convenience.
  if (url.pathname === '/health') {
    const healthBase = (url.hostname === 'localhost' || url.hostname === '127.0.0.1')
      ? 'http://localhost:8080'
      : (process.env.NEXT_PUBLIC_API_URL || 'http://backend:8080')
    url.href = `${healthBase}${url.pathname}${url.search}`
    return NextResponse.rewrite(url)
  }
}

export const config = {
  matcher: ['/health']
}