import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { token, user } = body;

    if (!token) {
      return NextResponse.json(
        { error: 'Token required' },
        { status: 400 }
      );
    }

    // Create response
    const response = NextResponse.json({ success: true });

    // Set authentication cookie (httpOnly for security)
    response.cookies.set('auth-token', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24, // 24 hours
      path: '/'
    });

    // Do not set client-visible auth status flags (avoid spoofing)

    if (user) {
      response.cookies.set('user-info', JSON.stringify(user), {
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 60 * 60 * 24, // 24 hours
        path: '/'
      });
    }

    return response;
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to set authentication cookie' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  const response = NextResponse.json({ success: true });
  
  // Clear all auth cookies
  response.cookies.delete('auth-token');
  // Remove any legacy flags if present
  response.cookies.delete('auth-status');
  response.cookies.delete('user-info');
  
  return response;
}