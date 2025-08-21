import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';

interface CSRFOptions {
  tokenLength?: number;
  cookieName?: string;
  headerName?: string;
  sameSite?: 'strict' | 'lax' | 'none';
  secure?: boolean;
  httpOnly?: boolean;
  maxAge?: number;
  excludePaths?: string[];
}

class CSRFProtection {
  private options: Required<CSRFOptions>;
  private tokens: Map<string, { token: string; expires: number }> = new Map();

  constructor(options: CSRFOptions = {}) {
    this.options = {
      tokenLength: options.tokenLength || 32,
      cookieName: options.cookieName || 'csrf-token',
      headerName: options.headerName || 'x-csrf-token',
      sameSite: options.sameSite || 'strict',
      secure: options.secure !== false,
      httpOnly: options.httpOnly !== false,
      maxAge: options.maxAge || 86400, // 24 hours
      excludePaths: options.excludePaths || [],
    };

    // Clean up expired tokens every hour
    setInterval(() => this.cleanup(), 3600000);
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, value] of this.tokens.entries()) {
      if (value.expires < now) {
        this.tokens.delete(key);
      }
    }
  }

  private generateToken(): string {
    return crypto.randomBytes(this.options.tokenLength).toString('hex');
  }

  private getSessionId(request: NextRequest): string {
    // Use session ID from cookie or generate a new one
    const sessionCookie = request.cookies.get('session-id');
    return sessionCookie?.value || crypto.randomBytes(16).toString('hex');
  }

  private isExcluded(pathname: string): boolean {
    return this.options.excludePaths.some(path => 
      pathname.startsWith(path)
    );
  }

  public async middleware(
    request: NextRequest,
    handler: (request: NextRequest) => Promise<NextResponse>
  ): Promise<NextResponse> {
    const { pathname } = new URL(request.url);
    
    // Skip CSRF check for excluded paths
    if (this.isExcluded(pathname)) {
      return handler(request);
    }

    const method = request.method.toUpperCase();
    const sessionId = this.getSessionId(request);

    // For GET requests, generate and set CSRF token
    if (method === 'GET' || method === 'HEAD' || method === 'OPTIONS') {
      const token = this.generateToken();
      const expires = Date.now() + (this.options.maxAge * 1000);
      
      this.tokens.set(sessionId, { token, expires });
      
      const response = await handler(request);
      
      // Set CSRF token in cookie
      response.cookies.set({
        name: this.options.cookieName,
        value: token,
        httpOnly: this.options.httpOnly,
        secure: this.options.secure,
        sameSite: this.options.sameSite,
        maxAge: this.options.maxAge,
        path: '/',
      });
      
      // Set session ID cookie if new
      if (!request.cookies.get('session-id')) {
        response.cookies.set({
          name: 'session-id',
          value: sessionId,
          httpOnly: true,
          secure: this.options.secure,
          sameSite: this.options.sameSite,
          maxAge: this.options.maxAge,
          path: '/',
        });
      }
      
      return response;
    }

    // For state-changing requests, validate CSRF token
    const cookieToken = request.cookies.get(this.options.cookieName)?.value;
    const headerToken = request.headers.get(this.options.headerName);
    const bodyToken = await this.extractBodyToken(request);
    
    const providedToken = headerToken || bodyToken;
    const storedData = this.tokens.get(sessionId);
    
    // Validate token
    if (
      !providedToken ||
      !cookieToken ||
      !storedData ||
      storedData.token !== providedToken ||
      storedData.token !== cookieToken ||
      storedData.expires < Date.now()
    ) {
      return NextResponse.json(
        { error: 'Invalid or missing CSRF token' },
        { status: 403 }
      );
    }

    // Token is valid, process request
    return handler(request);
  }

  private async extractBodyToken(request: NextRequest): Promise<string | null> {
    try {
      // Clone request to read body without consuming it
      const contentType = request.headers.get('content-type');
      
      if (contentType?.includes('application/json')) {
        const clonedRequest = request.clone();
        const body = await clonedRequest.json();
        return body._csrf || body.csrfToken || null;
      }
      
      if (contentType?.includes('application/x-www-form-urlencoded')) {
        const clonedRequest = request.clone();
        const text = await clonedRequest.text();
        const params = new URLSearchParams(text);
        return params.get('_csrf') || params.get('csrfToken');
      }
    } catch {
      // Body parsing failed, ignore
    }
    
    return null;
  }

  public getToken(request: NextRequest): string | null {
    const sessionId = this.getSessionId(request);
    const storedData = this.tokens.get(sessionId);
    
    if (storedData && storedData.expires > Date.now()) {
      return storedData.token;
    }
    
    return null;
  }
}

// Pre-configured CSRF protection instances
export const csrfProtection = new CSRFProtection({
  secure: process.env.NODE_ENV === 'production',
  excludePaths: ['/api/health', '/api/metrics', '/api/public'],
});

export const strictCSRFProtection = new CSRFProtection({
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'strict',
  maxAge: 3600, // 1 hour
  excludePaths: [],
});

// Helper function to add CSRF token to fetch requests
export function addCSRFToken(headers: HeadersInit = {}): HeadersInit {
  if (typeof window !== 'undefined') {
    const token = document.cookie
      .split('; ')
      .find(row => row.startsWith('csrf-token='))
      ?.split('=')[1];
    
    if (token) {
      return {
        ...headers,
        'x-csrf-token': token,
      };
    }
  }
  
  return headers;
}

export default CSRFProtection;