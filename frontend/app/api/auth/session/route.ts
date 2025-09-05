import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  // Check if in demo mode
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  
  if (isDemoMode) {
    // Return demo session
    return NextResponse.json({
      user: {
        id: 'demo-user',
        email: 'demo@policycortex.ai',
        name: 'Demo User',
        roles: ['admin'],
        tenantId: 'demo-tenant',
        authenticated: true
      },
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString() // 24 hours
    });
  }
  
  // Check for auth cookies
  const authToken = request.cookies.get('auth-token');
  const sessionToken = request.cookies.get('pcx-session');
  
  if (authToken || sessionToken) {
    // Return authenticated session
    return NextResponse.json({
      user: {
        id: 'user-1',
        email: 'user@policycortex.ai',
        name: 'Authenticated User',
        roles: ['admin'],
        authenticated: true
      },
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
    });
  }
  
  // No session
  return NextResponse.json({
    user: null,
    expires: null
  }, { status: 401 });
}

export async function POST(request: NextRequest) {
  // Handle session creation (login)
  try {
    const body = await request.json();
    
    // In demo mode, always succeed
    if (process.env.NEXT_PUBLIC_DEMO_MODE === 'true') {
      const response = NextResponse.json({
        user: {
          id: 'demo-user',
          email: 'demo@policycortex.ai',
          name: 'Demo User',
          roles: ['admin'],
          tenantId: 'demo-tenant',
          authenticated: true
        }
      });
      
      // Set auth cookies
      response.cookies.set('auth-token', 'demo-token', {
        httpOnly: true,
        secure: false,
        sameSite: 'lax',
        path: '/',
        maxAge: 86400 // 24 hours
      });
      
      response.cookies.set('pcx-session', 'demo-session', {
        httpOnly: true,
        secure: false,
        sameSite: 'lax',
        path: '/',
        maxAge: 86400
      });
      
      return response;
    }
    
    // Real authentication would happen here
    return NextResponse.json({
      error: 'Authentication not implemented'
    }, { status: 501 });
    
  } catch (error) {
    return NextResponse.json({
      error: 'Invalid request'
    }, { status: 400 });
  }
}