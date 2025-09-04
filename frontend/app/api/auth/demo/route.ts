import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

// POST - Create demo session
export async function POST(request: NextRequest) {
  // Only allow demo sessions in development with demo mode enabled
  if (process.env.NEXT_PUBLIC_DEMO_MODE !== 'true') {
    return NextResponse.json(
      { error: 'Demo mode is not enabled' },
      { status: 403 }
    );
  }

  try {
    // Set all necessary cookies for demo mode
    cookies().set('auth-token', 'demo-token', {
      httpOnly: true,
      secure: false,
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24, // 24 hours
    });

    cookies().set('pcx-session', 'demo-session', {
      httpOnly: true,
      secure: false,
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24, // 24 hours
    });

    cookies().set('demo-mode', 'true', {
      httpOnly: false,
      secure: false,
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24, // 24 hours
    });

    return NextResponse.json({
      success: true,
      demoMode: true,
      user: {
        userId: 'demo-user',
        email: 'demo@policycortex.local',
        name: 'Demo User',
        roles: ['admin'],
        tenantId: 'demo-tenant',
      },
    });
  } catch (error) {
    console.error('Failed to create demo session:', error);
    return NextResponse.json(
      { error: 'Failed to create demo session' },
      { status: 500 }
    );
  }
}

// GET - Check demo mode status
export async function GET() {
  const isDemoMode = process.env.NEXT_PUBLIC_DEMO_MODE === 'true';
  const hasAuthToken = cookies().get('auth-token');
  const hasSession = cookies().get('pcx-session');
  
  return NextResponse.json({
    demoModeEnabled: isDemoMode,
    authenticated: !!(hasAuthToken || hasSession),
  });
}