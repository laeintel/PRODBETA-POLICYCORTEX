import { NextResponse } from 'next/server';
import crypto from 'crypto';
import { cookies } from 'next/headers';

export async function GET() {
  try {
    // Generate a secure CSRF token
    const csrfToken = crypto.randomBytes(32).toString('hex');
    
    // Store the CSRF token in an httpOnly cookie
    cookies().set('csrf-token', csrfToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      path: '/',
      maxAge: 60 * 60 * 24, // 24 hours
    });

    // Also return it to the client for inclusion in requests
    return NextResponse.json({ csrfToken });
  } catch (error) {
    console.error('Failed to generate CSRF token:', error);
    return NextResponse.json(
      { error: 'Failed to generate CSRF token' },
      { status: 500 }
    );
  }
}