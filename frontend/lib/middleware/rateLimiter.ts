import { NextRequest, NextResponse } from 'next/server';

interface RateLimitOptions {
  windowMs?: number;
  max?: number;
  message?: string;
  standardHeaders?: boolean;
  legacyHeaders?: boolean;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: NextRequest) => string;
}

interface RateLimitStore {
  [key: string]: {
    hits: number;
    resetTime: number;
  };
}

class RateLimiter {
  private store: RateLimitStore = {};
  private options: Required<RateLimitOptions>;

  constructor(options: RateLimitOptions = {}) {
    this.options = {
      windowMs: options.windowMs || 60 * 1000, // 1 minute default
      max: options.max || 100, // 100 requests per window default
      message: options.message || 'Too many requests, please try again later.',
      standardHeaders: options.standardHeaders !== false,
      legacyHeaders: options.legacyHeaders !== false,
      skipSuccessfulRequests: options.skipSuccessfulRequests || false,
      skipFailedRequests: options.skipFailedRequests || false,
      keyGenerator: options.keyGenerator || this.defaultKeyGenerator,
    };

    // Clean up expired entries every minute
    setInterval(() => this.cleanup(), 60000);
  }

  private defaultKeyGenerator(req: NextRequest): string {
    // Use IP address as default key
    const forwarded = req.headers.get('x-forwarded-for');
    const ip = forwarded ? forwarded.split(',')[0] : req.ip || 'unknown';
    return ip;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const key in this.store) {
      if (this.store[key].resetTime < now) {
        delete this.store[key];
      }
    }
  }

  public async middleware(
    request: NextRequest,
    handler: (request: NextRequest) => Promise<NextResponse>
  ): Promise<NextResponse> {
    const key = this.options.keyGenerator(request);
    const now = Date.now();

    // Get or create rate limit entry
    if (!this.store[key] || this.store[key].resetTime < now) {
      this.store[key] = {
        hits: 0,
        resetTime: now + this.options.windowMs,
      };
    }

    const entry = this.store[key];
    entry.hits++;

    // Calculate rate limit info
    const remaining = Math.max(0, this.options.max - entry.hits);
    const resetTime = new Date(entry.resetTime).toISOString();

    // Check if rate limit exceeded
    if (entry.hits > this.options.max) {
      const response = NextResponse.json(
        { error: this.options.message },
        { status: 429 }
      );

      // Add rate limit headers
      if (this.options.standardHeaders) {
        response.headers.set('RateLimit-Limit', String(this.options.max));
        response.headers.set('RateLimit-Remaining', String(remaining));
        response.headers.set('RateLimit-Reset', resetTime);
      }

      if (this.options.legacyHeaders) {
        response.headers.set('x-ratelimit-limit', String(this.options.max));
        response.headers.set('x-ratelimit-remaining', String(remaining));
        response.headers.set('x-ratelimit-reset', resetTime);
      }

      response.headers.set('Retry-After', String(Math.ceil((entry.resetTime - now) / 1000)));

      return response;
    }

    // Process request
    const response = await handler(request);

    // Add rate limit headers to successful responses
    if (this.options.standardHeaders) {
      response.headers.set('RateLimit-Limit', String(this.options.max));
      response.headers.set('RateLimit-Remaining', String(remaining));
      response.headers.set('RateLimit-Reset', resetTime);
    }

    if (this.options.legacyHeaders) {
      response.headers.set('x-ratelimit-limit', String(this.options.max));
      response.headers.set('x-ratelimit-remaining', String(remaining));
      response.headers.set('x-ratelimit-reset', resetTime);
    }

    // Handle skip options
    if (
      (this.options.skipSuccessfulRequests && response.status < 400) ||
      (this.options.skipFailedRequests && response.status >= 400)
    ) {
      entry.hits--;
    }

    return response;
  }
}

// Pre-configured rate limiters for different use cases
export const generalRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
});

export const strictRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 20, // 20 requests per minute
  message: 'Rate limit exceeded. Please wait before making more requests.',
});

export const authRateLimiter = new RateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 attempts per 15 minutes
  message: 'Too many authentication attempts. Please try again later.',
  skipSuccessfulRequests: true, // Don't count successful logins
});

export const apiRateLimiter = new RateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: process.env.NODE_ENV === 'test' ? 1000 : 60, // Higher limit for tests
  keyGenerator: (req) => {
    // Use both IP and user ID for API rate limiting
    const forwarded = req.headers.get('x-forwarded-for');
    const ip = forwarded ? forwarded.split(',')[0] : req.ip || 'unknown';
    const userId = req.headers.get('x-user-id') || 'anonymous';
    return `${ip}:${userId}`;
  },
});

export default RateLimiter;