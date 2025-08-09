import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const url = request.nextUrl.clone()
  
  // Proxy API requests to backend
  if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
    url.href = `http://backend:8080${url.pathname}${url.search}`
    return NextResponse.rewrite(url)
  }
  
  // Proxy actions to backend
  if (url.pathname.startsWith('/actions/')) {
    const newPath = url.pathname.replace('/actions/', '/api/v1/actions/')
    url.href = `http://backend:8080${newPath}${url.search}`
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