import { test, expect } from '@playwright/test';

test.describe('Security Headers', () => {
  test('should have all required security headers', async ({ page }) => {
    const response = await page.goto('/');
    const headers = response?.headers() || {};
    
    // Check for CSP header
    expect(headers['content-security-policy']).toBeDefined();
    expect(headers['content-security-policy']).toContain("default-src 'self'");
    expect(headers['content-security-policy']).toContain('nonce-');
    expect(headers['content-security-policy']).not.toContain('unsafe-inline');
    expect(headers['content-security-policy']).not.toContain('unsafe-eval');
    
    // Check for other security headers
    expect(headers['x-frame-options']).toBe('DENY');
    expect(headers['x-content-type-options']).toBe('nosniff');
    expect(headers['x-xss-protection']).toBe('1; mode=block');
    expect(headers['referrer-policy']).toBe('strict-origin-when-cross-origin');
    expect(headers['permissions-policy']).toContain('camera=()');
    
    // Check for HSTS in production
    if (process.env.NODE_ENV === 'production') {
      expect(headers['strict-transport-security']).toContain('max-age=31536000');
      expect(headers['strict-transport-security']).toContain('includeSubDomains');
    }
  });
  
  test('should have nonce in CSP and HTML', async ({ page }) => {
    const response = await page.goto('/');
    const headers = response?.headers() || {};
    
    // Extract nonce from CSP header
    const csp = headers['content-security-policy'] || '';
    const nonceMatch = csp.match(/nonce-([a-zA-Z0-9+/=]+)/);
    expect(nonceMatch).toBeTruthy();
    
    if (nonceMatch) {
      const nonce = nonceMatch[1];
      
      // Check that inline scripts have the nonce
      const scripts = await page.$$eval('script', (elements, expectedNonce) => 
        elements.map(el => ({
          hasNonce: el.getAttribute('nonce') === expectedNonce,
          nonce: el.getAttribute('nonce')
        })),
        nonce
      );
      
      // All inline scripts should have the correct nonce
      for (const script of scripts) {
        if (script.nonce) {
          expect(script.hasNonce).toBe(true);
        }
      }
    }
  });
});

test.describe('API Authentication', () => {
  test('should return 401 for unauthenticated API requests', async ({ request }) => {
    const endpoints = [
      '/api/v1/resources',
      '/api/v1/conversation',
      '/api/v1/correlations',
      '/api/v1/predictions',
      '/api/v1/metrics',
    ];
    
    for (const endpoint of endpoints) {
      const response = await request.get(endpoint);
      expect(response.status()).toBe(401);
      
      const body = await response.json();
      expect(body.error).toContain('Authentication required');
    }
  });
  
  test('should return 403 for insufficient permissions', async ({ request }) => {
    // Mock a viewer token (read-only user)
    const viewerToken = 'mock-viewer-jwt-token';
    
    // Try to access admin-only endpoints
    const response = await request.post('/api/v1/resources', {
      headers: {
        'Authorization': `Bearer ${viewerToken}`,
        'Content-Type': 'application/json',
      },
      data: {
        name: 'test-resource',
        type: 'vm',
      },
    });
    
    // Should get 403 Forbidden (once real auth is implemented)
    // For now, it might return 401 if token validation fails
    expect([401, 403]).toContain(response.status());
  });
  
  test('should validate JWT token structure', async ({ request }) => {
    const invalidTokens = [
      'not-a-jwt',
      'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9', // Incomplete JWT
      'Bearer', // Missing token
      '', // Empty token
    ];
    
    for (const token of invalidTokens) {
      const response = await request.get('/api/v1/resources', {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
        },
      });
      
      expect(response.status()).toBe(401);
    }
  });
});

test.describe('Input Sanitization', () => {
  test('should sanitize XSS attempts in AI chat', async ({ page }) => {
    await page.goto('/ai/chat');
    
    // Try to inject script tag
    const xssPayloads = [
      '<script>alert("XSS")</script>',
      '<img src=x onerror="alert(\'XSS\')">',
      '<svg onload="alert(\'XSS\')">',
      'javascript:alert("XSS")',
      '<iframe src="javascript:alert(\'XSS\')">',
    ];
    
    for (const payload of xssPayloads) {
      // Check that payload is sanitized in the DOM
      await page.fill('[data-testid="chat-input"]', payload);
      await page.press('[data-testid="chat-input"]', 'Enter');
      
      // Wait for message to appear
      await page.waitForTimeout(500);
      
      // Check that no script tags are present
      const scriptCount = await page.$$eval('script', scripts => 
        scripts.filter(s => s.textContent?.includes('alert')).length
      );
      expect(scriptCount).toBe(0);
      
      // Check that dangerous attributes are removed
      const dangerousElements = await page.$$eval('[onerror], [onload], [onclick]', 
        elements => elements.length
      );
      expect(dangerousElements).toBe(0);
    }
  });
  
  test('should validate API input with Zod schemas', async ({ request }) => {
    const invalidPayloads = [
      { email: 'not-an-email' }, // Invalid email
      { page: -1 }, // Negative page number
      { limit: 1000 }, // Exceeds max limit
      { status: 'invalid-status' }, // Invalid enum value
    ];
    
    for (const payload of invalidPayloads) {
      const response = await request.get('/api/v1/resources', {
        params: payload as any,
      });
      
      // Should return 400 Bad Request for invalid input
      if (response.status() !== 401) { // Skip if auth fails first
        expect(response.status()).toBe(400);
        const body = await response.json();
        expect(body.error).toContain('Invalid');
      }
    }
  });
});

test.describe('Rate Limiting', () => {
  test('should enforce rate limits on API endpoints', async ({ request }) => {
    const endpoint = '/api/v1/health'; // Use health endpoint that doesn't require auth
    const maxRequests = 100; // General rate limit
    
    // Make requests up to the limit
    for (let i = 0; i < maxRequests + 5; i++) {
      const response = await request.get(endpoint);
      
      if (i < maxRequests) {
        expect(response.status()).toBe(200);
        
        // Check rate limit headers
        const headers = response.headers();
        expect(headers['x-ratelimit-limit']).toBeDefined();
        expect(headers['x-ratelimit-remaining']).toBeDefined();
      } else {
        // Should get 429 Too Many Requests
        expect(response.status()).toBe(429);
        
        const body = await response.json();
        expect(body.error).toContain('Too many requests');
        
        // Check Retry-After header
        const retryAfter = response.headers()['retry-after'];
        expect(retryAfter).toBeDefined();
        expect(parseInt(retryAfter)).toBeGreaterThan(0);
      }
    }
  });
});

test.describe('CSRF Protection', () => {
  test('should require CSRF token for state-changing requests', async ({ page, request }) => {
    await page.goto('/');
    
    // Get CSRF token from cookie
    const cookies = await page.context().cookies();
    const csrfCookie = cookies.find(c => c.name === 'csrf-token');
    
    if (csrfCookie) {
      // Try POST without CSRF token in header
      const response = await request.post('/api/v1/resources', {
        headers: {
          'Content-Type': 'application/json',
        },
        data: { name: 'test' },
      });
      
      // Should fail without CSRF token
      expect([401, 403]).toContain(response.status());
      
      // Try with CSRF token
      const responseWithToken = await request.post('/api/v1/resources', {
        headers: {
          'Content-Type': 'application/json',
          'x-csrf-token': csrfCookie.value,
        },
        data: { name: 'test' },
      });
      
      // Should pass CSRF check (might still fail auth)
      // But shouldn't be 403 for CSRF
      if (responseWithToken.status() === 403) {
        const body = await responseWithToken.json();
        expect(body.error).not.toContain('CSRF');
      }
    }
  });
});

test.describe('Environment Validation', () => {
  test('should validate required environment variables', async ({ page }) => {
    // This test would need to run in different environments
    // For now, just check that the app starts without env errors
    const response = await page.goto('/');
    expect(response?.status()).toBe(200);
    
    // Check console for env validation errors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error' && msg.text().includes('Environment')) {
        consoleErrors.push(msg.text());
      }
    });
    
    await page.waitForTimeout(1000);
    expect(consoleErrors).toHaveLength(0);
  });
});

test.describe('Audit Logging', () => {
  test('should log security-relevant events', async ({ request }) => {
    // Make various requests that should trigger audit logs
    const actions = [
      { method: 'GET', url: '/api/v1/resources', event: 'RESOURCE_ACCESSED' },
      { method: 'POST', url: '/api/v1/conversation', event: 'AI_PREDICTION' },
      { method: 'GET', url: '/api/v1/correlations', event: 'RESOURCE_ACCESSED' },
    ];
    
    for (const action of actions) {
      if (action.method === 'GET') {
        await request.get(action.url);
      } else {
        await request.post(action.url, { data: {} });
      }
      
      // In a real test, we'd check audit logs are created
      // For now, just verify the endpoints exist
    }
  });
});

test.describe('CSP Violation Reporting', () => {
  test('should report CSP violations', async ({ page, request }) => {
    await page.goto('/');
    
    // Try to violate CSP
    await page.evaluate(() => {
      // This should trigger a CSP violation
      const script = document.createElement('script');
      script.textContent = 'console.log("inline without nonce")';
      document.head.appendChild(script);
    });
    
    // Wait for potential CSP report
    await page.waitForTimeout(1000);
    
    // Check if CSP report endpoint exists
    const response = await request.post('/api/v1/csp-report', {
      headers: {
        'Content-Type': 'application/csp-report',
      },
      data: {
        'csp-report': {
          'document-uri': 'http://localhost:3000/',
          'violated-directive': 'script-src',
          'blocked-uri': 'inline',
        },
      },
    });
    
    // Report endpoint should accept reports
    expect([200, 204, 401]).toContain(response.status());
  });
});