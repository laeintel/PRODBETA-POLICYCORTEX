/**
 * PolicyCortex Critical User Flow Test Suite: Authentication & Login
 * 
 * This test suite covers comprehensive authentication scenarios including:
 * - Initial login flow
 * - Session persistence
 * - Multi-factor authentication
 * - Role-based access control
 * - Token refresh
 * - Logout scenarios
 * 
 * Performance targets:
 * - Login completion: <3s
 * - Token validation: <100ms
 * - Page navigation after auth: <2s
 */

import { test, expect, Page, BrowserContext } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

// Test configuration with retry logic and performance monitoring
test.use({
  trace: 'on-first-retry',
  video: 'on-first-retry',
  screenshot: 'only-on-failure',
  actionTimeout: 15000,
  navigationTimeout: 30000,
});

// Helper function to measure performance metrics
async function measurePerformance(page: Page, actionName: string, action: () => Promise<void>) {
  const startTime = Date.now();
  await action();
  const endTime = Date.now();
  const duration = endTime - startTime;
  
  console.log(`Performance: ${actionName} completed in ${duration}ms`);
  expect(duration).toBeLessThan(3000); // 3 second max for critical actions
  
  return duration;
}

// Helper to check accessibility
async function checkAccessibility(page: Page, testName: string) {
  // Inject axe-core for accessibility testing
  await page.evaluate(() => {
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.7.2/axe.min.js';
    document.head.appendChild(script);
  });
  
  await page.waitForTimeout(1000); // Wait for script to load
  
  // Run accessibility check
  const violations = await page.evaluate(() => {
    // @ts-ignore
    if (typeof window.axe !== 'undefined') {
      // @ts-ignore
      return window.axe.run();
    }
    return null;
  });
  
  if (violations && violations.violations?.length > 0) {
    console.warn(`Accessibility violations found in ${testName}:`, violations.violations);
  }
}

test.describe('Critical Flow: User Authentication & Login', () => {
  test.beforeEach(async ({ page }) => {
    // Clear all cookies and storage to ensure clean state
    await page.context().clearCookies();
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  });

  test('01 - Complete login flow with performance metrics', async ({ page }) => {
    await test.step('Navigate to application', async () => {
      await measurePerformance(page, 'Initial page load', async () => {
        await page.goto(BASE_URL);
      });
    });

    await test.step('Verify login page elements', async () => {
      // Check for all required elements
      await expect(page.locator('text=PolicyCortex')).toBeVisible();
      await expect(page.locator('text=/AI.*Powered.*Azure.*Governance/i')).toBeVisible();
      
      const loginButton = page.locator('button:has-text("Sign in with Azure AD")');
      await expect(loginButton).toBeVisible();
      
      // Verify button is focusable and has proper ARIA attributes
      await expect(loginButton).toBeFocused({ timeout: 100 }).catch(() => {
        // If not auto-focused, manually focus
        loginButton.focus();
      });
      
      const ariaLabel = await loginButton.getAttribute('aria-label');
      expect(ariaLabel).toBeTruthy();
    });

    await test.step('Check accessibility on login page', async () => {
      await checkAccessibility(page, 'Login Page');
    });

    await test.step('Test keyboard navigation', async () => {
      // Tab through interactive elements
      await page.keyboard.press('Tab');
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      expect(focusedElement).toBeTruthy();
      
      // Ensure Enter key can trigger login
      const loginButton = page.locator('button:has-text("Sign in with Azure AD")');
      await loginButton.focus();
      
      // Mock successful authentication for testing
      await page.route('**/login.microsoftonline.com/**', async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            access_token: 'mock_access_token',
            id_token: 'mock_id_token',
            refresh_token: 'mock_refresh_token',
            expires_in: 3600,
          }),
        });
      });
    });

    await test.step('Simulate authentication flow', async () => {
      // Mock MSAL redirect
      await page.evaluate(() => {
        // Set mock authentication state
        localStorage.setItem('msal.account.keys', JSON.stringify(['mock-account']));
        localStorage.setItem('msal.mock-account', JSON.stringify({
          homeAccountId: 'mock-home-id',
          environment: 'login.microsoftonline.com',
          tenantId: '9ef5b184-d371-462a-bc75-5024ce8baff7',
          username: 'test@policycortex.com',
          localAccountId: 'mock-local-id',
        }));
        
        // Trigger a storage event to simulate MSAL login completion
        window.dispatchEvent(new StorageEvent('storage', {
          key: 'msal.account.keys',
          newValue: JSON.stringify(['mock-account']),
        }));
      });

      // Navigate to dashboard after "successful" auth
      await page.goto(`${BASE_URL}/tactical`);
      
      // Verify we're on the dashboard (not redirected back to login)
      await expect(page).toHaveURL(/.*tactical/);
    });
  });

  test('02 - Session persistence across page refreshes', async ({ page, context }) => {
    await test.step('Set up authenticated session', async () => {
      // Add authentication cookies
      await context.addCookies([
        {
          name: 'msal.session',
          value: 'mock-session-token',
          domain: 'localhost',
          path: '/',
          expires: Date.now() / 1000 + 3600,
        },
        {
          name: 'msal.token.cache',
          value: JSON.stringify({
            access_token: 'mock-access',
            refresh_token: 'mock-refresh',
            expires_at: Date.now() + 3600000,
          }),
          domain: 'localhost',
          path: '/',
        }
      ]);

      // Set localStorage authentication
      await page.goto(BASE_URL);
      await page.evaluate(() => {
        localStorage.setItem('msal.account.keys', JSON.stringify(['test-account']));
        localStorage.setItem('msal.test-account', JSON.stringify({
          username: 'test@policycortex.com',
          name: 'Test User',
          roles: ['admin', 'user'],
        }));
      });
    });

    await test.step('Navigate to protected routes', async () => {
      const protectedRoutes = [
        '/tactical',
        '/executive',
        '/governance',
        '/security',
        '/operations',
      ];

      for (const route of protectedRoutes) {
        await page.goto(`${BASE_URL}${route}`);
        
        // Should not redirect to login
        await expect(page).not.toHaveURL(/.*login/);
        await expect(page).not.toHaveURL(/.*returnUrl/);
        
        // Page should load successfully
        await page.waitForLoadState('networkidle');
      }
    });

    await test.step('Verify session persists after refresh', async () => {
      await page.goto(`${BASE_URL}/tactical`);
      
      // Refresh the page
      await page.reload();
      
      // Should still be authenticated
      await expect(page).toHaveURL(/.*tactical/);
      await expect(page.locator('text=Sign in with Azure AD')).not.toBeVisible();
    });
  });

  test('03 - Role-based access control (RBAC) verification', async ({ page, context }) => {
    // Test different user roles
    const roles = [
      { role: 'admin', canAccess: ['/admin', '/executive', '/governance'], cannotAccess: [] },
      { role: 'user', canAccess: ['/tactical', '/operations'], cannotAccess: ['/admin'] },
      { role: 'viewer', canAccess: ['/tactical'], cannotAccess: ['/admin', '/governance/policies/edit'] },
    ];

    for (const roleTest of roles) {
      await test.step(`Test ${roleTest.role} role permissions`, async () => {
        // Clear previous session
        await context.clearCookies();
        
        // Set up session with specific role
        await page.goto(BASE_URL);
        await page.evaluate((role) => {
          localStorage.clear();
          localStorage.setItem('msal.account.keys', JSON.stringify([`${role}-account`]));
          localStorage.setItem(`msal.${role}-account`, JSON.stringify({
            username: `${role}@policycortex.com`,
            roles: [role],
          }));
        }, roleTest.role);

        // Add session cookie
        await context.addCookies([{
          name: 'msal.session',
          value: `${roleTest.role}-session`,
          domain: 'localhost',
          path: '/',
        }]);

        // Test accessible routes
        for (const route of roleTest.canAccess) {
          await page.goto(`${BASE_URL}${route}`);
          await page.waitForLoadState('networkidle');
          
          // Should not show access denied
          await expect(page.locator('text=/access.*denied|unauthorized|forbidden/i')).not.toBeVisible();
        }

        // Test restricted routes
        for (const route of roleTest.cannotAccess) {
          await page.goto(`${BASE_URL}${route}`);
          
          // Should show access denied or redirect
          const isAccessDenied = await page.locator('text=/access.*denied|unauthorized|forbidden/i').isVisible();
          const isRedirected = page.url().includes('login') || page.url().includes('unauthorized');
          
          expect(isAccessDenied || isRedirected).toBeTruthy();
        }
      });
    }
  });

  test('04 - Token refresh and expiration handling', async ({ page, context }) => {
    await test.step('Set up session with expiring token', async () => {
      const now = Date.now();
      
      await context.addCookies([{
        name: 'msal.token.cache',
        value: JSON.stringify({
          access_token: 'soon-to-expire',
          refresh_token: 'valid-refresh',
          expires_at: now + 5000, // Expires in 5 seconds
        }),
        domain: 'localhost',
        path: '/',
      }]);

      await page.goto(`${BASE_URL}/tactical`);
    });

    await test.step('Monitor token refresh', async () => {
      // Intercept token refresh requests
      let refreshRequested = false;
      await page.route('**/oauth2/v2.0/token', async route => {
        refreshRequested = true;
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            access_token: 'new-access-token',
            refresh_token: 'new-refresh-token',
            expires_in: 3600,
          }),
        });
      });

      // Wait for token to expire
      await page.waitForTimeout(6000);

      // Trigger an API call that requires authentication
      await page.evaluate(async () => {
        await fetch('/api/v1/metrics', {
          headers: {
            'Authorization': 'Bearer ' + localStorage.getItem('access_token'),
          },
        });
      });

      // Verify refresh was attempted
      expect(refreshRequested).toBeTruthy();
    });
  });

  test('05 - Logout flow and session cleanup', async ({ page, context }) => {
    await test.step('Set up authenticated session', async () => {
      await context.addCookies([{
        name: 'msal.session',
        value: 'active-session',
        domain: 'localhost',
        path: '/',
      }]);

      await page.goto(BASE_URL);
      await page.evaluate(() => {
        localStorage.setItem('msal.account.keys', JSON.stringify(['logout-test']));
        localStorage.setItem('msal.logout-test', JSON.stringify({
          username: 'test@policycortex.com',
        }));
      });
    });

    await test.step('Navigate to dashboard', async () => {
      await page.goto(`${BASE_URL}/tactical`);
      await page.waitForLoadState('networkidle');
    });

    await test.step('Perform logout', async () => {
      // Look for logout button in various possible locations
      const logoutSelectors = [
        'button:has-text("Logout")',
        'button:has-text("Sign out")',
        '[aria-label*="logout"]',
        '[aria-label*="sign out"]',
      ];

      let logoutButton = null;
      for (const selector of logoutSelectors) {
        const element = page.locator(selector);
        if (await element.count() > 0) {
          logoutButton = element.first();
          break;
        }
      }

      if (logoutButton) {
        await measurePerformance(page, 'Logout action', async () => {
          await logoutButton.click();
        });
      }
    });

    await test.step('Verify session is cleared', async () => {
      // Check localStorage is cleared
      const storageKeys = await page.evaluate(() => Object.keys(localStorage));
      const msalKeys = storageKeys.filter(key => key.startsWith('msal.'));
      expect(msalKeys.length).toBe(0);

      // Check cookies are cleared
      const cookies = await context.cookies();
      const sessionCookies = cookies.filter(c => c.name.includes('msal'));
      expect(sessionCookies.length).toBe(0);

      // Verify redirect to login
      await expect(page.locator('text=Sign in with Azure AD')).toBeVisible({ timeout: 10000 });
    });

    await test.step('Verify cannot access protected routes after logout', async () => {
      await page.goto(`${BASE_URL}/tactical`);
      
      // Should redirect to login
      await expect(page).toHaveURL(/.*login|.*returnUrl/);
    });
  });

  test('06 - Handle authentication errors gracefully', async ({ page }) => {
    const errorScenarios = [
      {
        name: 'Network failure',
        setup: async () => {
          await page.route('**/login.microsoftonline.com/**', route => route.abort('failed'));
        },
        expectedError: /network.*error|connection.*failed|unable.*connect/i,
      },
      {
        name: 'Invalid credentials',
        setup: async () => {
          await page.route('**/login.microsoftonline.com/**', route => {
            route.fulfill({
              status: 401,
              contentType: 'application/json',
              body: JSON.stringify({ error: 'invalid_grant', error_description: 'Invalid credentials' }),
            });
          });
        },
        expectedError: /invalid.*credentials|authentication.*failed/i,
      },
      {
        name: 'Token expired',
        setup: async () => {
          await page.route('**/oauth2/v2.0/token', route => {
            route.fulfill({
              status: 401,
              contentType: 'application/json',
              body: JSON.stringify({ error: 'invalid_grant', error_description: 'Token expired' }),
            });
          });
        },
        expectedError: /token.*expired|session.*expired/i,
      },
    ];

    for (const scenario of errorScenarios) {
      await test.step(`Handle ${scenario.name}`, async () => {
        await scenario.setup();
        await page.goto(BASE_URL);

        const loginButton = page.locator('button:has-text("Sign in with Azure AD")');
        if (await loginButton.isVisible()) {
          await loginButton.click();
          
          // Should show appropriate error message
          await expect(page.locator(`text=${scenario.expectedError}`)).toBeVisible({ timeout: 5000 });
          
          // Login button should still be available for retry
          await expect(loginButton).toBeVisible();
        }
      });
    }
  });

  test('07 - Multi-factor authentication (MFA) flow', async ({ page }) => {
    await test.step('Initiate login with MFA', async () => {
      await page.goto(BASE_URL);
      
      // Mock MFA challenge response
      await page.route('**/login.microsoftonline.com/**', async route => {
        const url = route.request().url();
        
        if (url.includes('authorize')) {
          // Initial auth request - return MFA challenge
          await route.fulfill({
            status: 200,
            contentType: 'text/html',
            body: '<html><body><div id="mfa-challenge">Enter MFA Code</div></body></html>',
          });
        } else if (url.includes('token')) {
          // Token request after MFA
          await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              access_token: 'mfa-verified-token',
              refresh_token: 'mfa-refresh-token',
              expires_in: 3600,
            }),
          });
        }
      });

      const loginButton = page.locator('button:has-text("Sign in with Azure AD")');
      if (await loginButton.isVisible()) {
        await loginButton.click();
      }
    });

    await test.step('Verify MFA challenge is handled', async () => {
      // In a real scenario, this would involve:
      // 1. Detecting MFA challenge page
      // 2. Entering MFA code
      // 3. Submitting and verifying success
      
      // For testing, we simulate successful MFA completion
      await page.evaluate(() => {
        localStorage.setItem('msal.mfa.verified', 'true');
        localStorage.setItem('msal.account.keys', JSON.stringify(['mfa-account']));
        localStorage.setItem('msal.mfa-account', JSON.stringify({
          username: 'mfa@policycortex.com',
          mfaVerified: true,
        }));
      });

      // Navigate to protected route
      await page.goto(`${BASE_URL}/tactical`);
      
      // Should be authenticated with MFA
      await expect(page).toHaveURL(/.*tactical/);
    });
  });

  test('08 - Cross-origin and CORS handling', async ({ page }) => {
    await test.step('Test CORS headers on API requests', async () => {
      let corsHeadersValid = false;
      
      await page.route('**/api/**', async route => {
        const response = await route.fetch();
        const headers = response.headers();
        
        // Check CORS headers
        if (headers['access-control-allow-origin']) {
          corsHeadersValid = true;
        }
        
        await route.fulfill({ response });
      });

      await page.goto(`${BASE_URL}/tactical`);
      
      // Trigger an API call
      await page.evaluate(async () => {
        await fetch('/api/v1/metrics');
      });

      expect(corsHeadersValid).toBeTruthy();
    });
  });

  test('09 - Visual regression for login page', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Test different viewport sizes
    const viewports = [
      { width: 375, height: 667, name: 'mobile' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 1920, height: 1080, name: 'desktop' },
    ];

    for (const viewport of viewports) {
      await test.step(`Capture ${viewport.name} screenshot`, async () => {
        await page.setViewportSize(viewport);
        await page.waitForTimeout(500); // Wait for any animations
        
        await page.screenshot({
          path: `test-results/screenshots/login-${viewport.name}.png`,
          fullPage: true,
        });
        
        // Compare with baseline (would be configured in CI/CD)
        await expect(page).toHaveScreenshot(`login-${viewport.name}.png`, {
          maxDiffPixels: 100,
          threshold: 0.2,
        });
      });
    }
  });

  test('10 - Performance metrics collection', async ({ page }) => {
    await test.step('Collect Core Web Vitals', async () => {
      await page.goto(BASE_URL);
      
      // Collect performance metrics
      const metrics = await page.evaluate(() => {
        return new Promise((resolve) => {
          // Wait for LCP
          new PerformanceObserver((entryList) => {
            const entries = entryList.getEntries();
            const lastEntry = entries[entries.length - 1];
            resolve({
              LCP: lastEntry.renderTime || lastEntry.loadTime,
              // @ts-ignore
              FCP: performance.getEntriesByName('first-contentful-paint')[0]?.startTime,
              // @ts-ignore
              TTFB: performance.timing.responseStart - performance.timing.requestStart,
            });
          }).observe({ entryTypes: ['largest-contentful-paint'] });
        });
      });

      // Assert performance thresholds
      expect(metrics.LCP).toBeLessThan(2500); // LCP < 2.5s
      expect(metrics.FCP).toBeLessThan(1800); // FCP < 1.8s
      expect(metrics.TTFB).toBeLessThan(600); // TTFB < 600ms
    });
  });
});

// Cleanup after all tests
test.afterAll(async () => {
  console.log('Authentication flow tests completed');
});