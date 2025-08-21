import { NextRequest, NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';

export interface DecodedToken {
  sub: string;
  email?: string;
  name?: string;
  roles?: string[];
  permissions?: string[];
  exp?: number;
  iat?: number;
  scope?: string;
}

export interface AuthValidationResult {
  valid: boolean;
  user?: DecodedToken;
  error?: string;
}

/**
 * Validate JWT token from Authorization header
 */
export async function validateToken(token: string): Promise<AuthValidationResult> {
  try {
    // In production, use Azure AD public key or JWKS endpoint
    const secret = process.env.JWT_SECRET || 'development-secret';
    
    const decoded = jwt.verify(token, secret) as DecodedToken;
    
    // Check token expiration
    if (decoded.exp && decoded.exp * 1000 < Date.now()) {
      return { valid: false, error: 'Token expired' };
    }
    
    return { valid: true, user: decoded };
  } catch (error) {
    return { valid: false, error: 'Invalid token' };
  }
}

/**
 * Extract bearer token from Authorization header
 */
export function extractBearerToken(authHeader: string | null): string | null {
  if (!authHeader) return null;
  
  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0].toLowerCase() !== 'bearer') {
    return null;
  }
  
  return parts[1];
}

/**
 * Middleware to validate API authentication
 */
export async function withAuth(
  request: NextRequest,
  handler: (request: NextRequest, user: DecodedToken) => Promise<NextResponse>
): Promise<NextResponse> {
  const authHeader = request.headers.get('authorization');
  const token = extractBearerToken(authHeader);
  
  if (!token) {
    return NextResponse.json(
      { error: 'No authorization token provided' },
      { status: 401 }
    );
  }
  
  const validation = await validateToken(token);
  
  if (!validation.valid) {
    return NextResponse.json(
      { error: validation.error || 'Invalid token' },
      { status: 401 }
    );
  }
  
  // Add user to request headers for downstream use
  const modifiedRequest = request.clone();
  modifiedRequest.headers.set('x-user-id', validation.user!.sub);
  modifiedRequest.headers.set('x-user-email', validation.user!.email || '');
  modifiedRequest.headers.set('x-user-roles', JSON.stringify(validation.user!.roles || []));
  
  return handler(modifiedRequest, validation.user!);
}

/**
 * Check if user has required role
 */
export function hasRole(user: DecodedToken, role: string): boolean {
  return user.roles?.includes(role) || false;
}

/**
 * Check if user has required permission
 */
export function hasPermission(user: DecodedToken, permission: string): boolean {
  return user.permissions?.includes(permission) || false;
}

/**
 * Check if user has any of the required roles
 */
export function hasAnyRole(user: DecodedToken, roles: string[]): boolean {
  return roles.some(role => hasRole(user, role));
}

/**
 * Check if user has all required roles
 */
export function hasAllRoles(user: DecodedToken, roles: string[]): boolean {
  return roles.every(role => hasRole(user, role));
}

/**
 * Middleware to require specific role
 */
export async function withRole(
  request: NextRequest,
  role: string,
  handler: (request: NextRequest, user: DecodedToken) => Promise<NextResponse>
): Promise<NextResponse> {
  return withAuth(request, async (req, user) => {
    if (!hasRole(user, role)) {
      return NextResponse.json(
        { error: 'Insufficient permissions' },
        { status: 403 }
      );
    }
    return handler(req, user);
  });
}

/**
 * Middleware to require specific permission
 */
export async function withPermission(
  request: NextRequest,
  permission: string,
  handler: (request: NextRequest, user: DecodedToken) => Promise<NextResponse>
): Promise<NextResponse> {
  return withAuth(request, async (req, user) => {
    if (!hasPermission(user, permission)) {
      return NextResponse.json(
        { error: 'Insufficient permissions' },
        { status: 403 }
      );
    }
    return handler(req, user);
  });
}