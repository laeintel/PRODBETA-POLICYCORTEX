import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production';
const SESSION_COOKIE = 'pcx-session';
const REFRESH_COOKIE = 'pcx-refresh';

export interface SessionData {
  userId: string;
  email: string;
  name: string;
  roles: string[];
  tenantId: string;
}

// GET - Check current session
export async function GET() {
  try {
    const sessionToken = cookies().get(SESSION_COOKIE);
    
    if (!sessionToken) {
      return NextResponse.json({ authenticated: false });
    }

    try {
      const decoded = jwt.verify(sessionToken.value, JWT_SECRET) as SessionData & { exp: number };
      
      // Check if token is about to expire (within 5 minutes)
      const now = Date.now() / 1000;
      const timeUntilExpiry = decoded.exp - now;
      
      if (timeUntilExpiry < 300) { // Less than 5 minutes
        // Attempt to refresh the token
        const refreshToken = cookies().get(REFRESH_COOKIE);
        if (refreshToken) {
          // Generate new session token
          const newSessionToken = jwt.sign(
            {
              userId: decoded.userId,
              email: decoded.email,
              name: decoded.name,
              roles: decoded.roles,
              tenantId: decoded.tenantId,
            },
            JWT_SECRET,
            { expiresIn: '15m' }
          );
          
          // Set new session cookie
          cookies().set(SESSION_COOKIE, newSessionToken, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            sameSite: 'strict',
            path: '/',
            maxAge: 60 * 15, // 15 minutes
          });
        }
      }
      
      return NextResponse.json({
        authenticated: true,
        user: {
          userId: decoded.userId,
          email: decoded.email,
          name: decoded.name,
          roles: decoded.roles,
          tenantId: decoded.tenantId,
        },
      });
    } catch (error) {
      // Token is invalid or expired
      cookies().delete(SESSION_COOKIE);
      cookies().delete(REFRESH_COOKIE);
      return NextResponse.json({ authenticated: false });
    }
  } catch (error) {
    console.error('Session check failed:', error);
    return NextResponse.json(
      { error: 'Failed to check session' },
      { status: 500 }
    );
  }
}

// POST - Create new session
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { accessToken, idToken, user } = body;
    
    if (!user) {
      return NextResponse.json(
        { error: 'User data is required' },
        { status: 400 }
      );
    }
    
    // Create session data
    const sessionData: SessionData = {
      userId: user.homeAccountId || user.localAccountId,
      email: user.username,
      name: user.name || user.username,
      roles: user.roles || [],
      tenantId: user.tenantId,
    };
    
    // Generate JWT tokens
    const sessionToken = jwt.sign(sessionData, JWT_SECRET, { expiresIn: '15m' });
    const refreshToken = jwt.sign(
      { userId: sessionData.userId, type: 'refresh' },
      JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    // Set httpOnly cookies
    cookies().set(SESSION_COOKIE, sessionToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      maxAge: 60 * 15, // 15 minutes
    });
    
    cookies().set(REFRESH_COOKIE, refreshToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    });
    
    // Store the actual Azure access token separately if needed
    if (accessToken) {
      cookies().set('pcx-azure-token', accessToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        path: '/',
        maxAge: 60 * 60, // 1 hour (typical Azure token lifetime)
      });
    }
    
    return NextResponse.json({
      success: true,
      user: sessionData,
    });
  } catch (error) {
    console.error('Failed to create session:', error);
    return NextResponse.json(
      { error: 'Failed to create session' },
      { status: 500 }
    );
  }
}

// DELETE - Destroy session
export async function DELETE() {
  try {
    cookies().delete(SESSION_COOKIE);
    cookies().delete(REFRESH_COOKIE);
    cookies().delete('pcx-azure-token');
    cookies().delete('csrf-token');
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Failed to destroy session:', error);
    return NextResponse.json(
      { error: 'Failed to destroy session' },
      { status: 500 }
    );
  }
}