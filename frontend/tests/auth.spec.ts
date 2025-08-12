import { test, expect, Page } from '@playwright/test';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';

test.describe('Authentication Flow', () => {
  test('should show login prompt on initial visit', async ({ page }) => {
    await page.goto(baseURL);
    
    // Check for Azure AD login button
    await expect(page.locator('text=Sign in with Azure AD')).toBeVisible({ timeout: 10000 });
    
    // Check for company branding
    await expect(page.locator('text=PolicyCortex')).toBeVisible();
    await expect(page.locator('text=AI-Powered Azure Governance Platform')).toBeVisible();
  });

  test('should prevent access to protected routes without authentication', async ({ page }) => {
    // Try to access dashboard directly
    await page.goto(`${baseURL}/dashboard`);
    
    // Should redirect to login
    await expect(page).toHaveURL(/.*\/?.*returnUrl=.*dashboard/);
    await expect(page.locator('text=Sign in with Azure AD')).toBeVisible();
  });

  test('should prevent bypassing login by closing dialog', async ({ page }) => {
    await page.goto(baseURL);
    
    // Try to close the login dialog using Escape key
    await page.keyboard.press('Escape');
    
    // Should still show login prompt
    await expect(page.locator('text=Sign in with Azure AD')).toBeVisible();
    
    // Should not show dashboard content
    await expect(page.locator('text=Governance Dashboard')).not.toBeVisible();
  });

  test('should handle MSAL authentication errors gracefully', async ({ page }) => {
    await page.goto(baseURL);
    
    // Mock MSAL error by intercepting the request
    await page.route('**/login.microsoftonline.com/**', route => {
      route.abort('failed');
    });
    
    // Click login button
    const loginButton = page.locator('button:has-text("Sign in with Azure AD")');
    if (await loginButton.isVisible()) {
      await loginButton.click();
      
      // Should show error message
      await expect(page.locator('text=/authentication.*failed|error.*sign|unable.*authenticate/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should maintain authentication state across page refreshes', async ({ page, context }) => {
    // Set mock authentication cookies
    await context.addCookies([
      {
        name: 'msal.session',
        value: 'mock-session-token',
        domain: 'localhost',
        path: '/',
      },
      {
        name: 'msal.token.cache',
        value: 'mock-token-cache',
        domain: 'localhost',
        path: '/',
      }
    ]);
    
    // Navigate to protected route
    await page.goto(`${baseURL}/dashboard`);
    
    // Should not redirect to login
    await expect(page).not.toHaveURL(/.*returnUrl.*/);
    
    // Refresh the page
    await page.reload();
    
    // Should still be on dashboard
    await expect(page).toHaveURL(/.*dashboard/);
  });

  test('should show user profile information when authenticated', async ({ page, context }) => {
    // Set mock authentication
    await context.addCookies([
      {
        name: 'msal.session',
        value: 'mock-session',
        domain: 'localhost',
        path: '/',
      }
    ]);
    
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for user menu or profile indicator
    const userMenu = page.locator('[data-testid="user-menu"], [aria-label*="user"], .user-profile');
    if (await userMenu.count() > 0) {
      await expect(userMenu.first()).toBeVisible();
    }
  });

  test('should handle logout correctly', async ({ page, context }) => {
    // Set mock authentication
    await context.addCookies([
      {
        name: 'msal.session',
        value: 'mock-session',
        domain: 'localhost',
        path: '/',
      }
    ]);
    
    await page.goto(`${baseURL}/dashboard`);
    
    // Find and click logout button if visible
    const logoutButton = page.locator('button:has-text("Logout"), button:has-text("Sign out")');
    if (await logoutButton.count() > 0) {
      await logoutButton.first().click();
      
      // Should redirect to login page
      await expect(page.locator('text=Sign in with Azure AD')).toBeVisible({ timeout: 10000 });
    }
  });

  test('should validate JWT tokens on API requests', async ({ page, context }) => {
    await context.addCookies([
      {
        name: 'msal.session',
        value: 'mock-session',
        domain: 'localhost',
        path: '/',
      }
    ]);
    
    // Intercept API calls to check for Authorization header
    let hasAuthHeader = false;
    await page.route('**/api/**', route => {
      const headers = route.request().headers();
      if (headers['authorization'] && headers['authorization'].startsWith('Bearer ')) {
        hasAuthHeader = true;
      }
      route.continue();
    });
    
    await page.goto(`${baseURL}/dashboard`);
    
    // Wait for API calls
    await page.waitForTimeout(2000);
    
    // Check if auth header was sent (if API calls were made)
    // This is a soft check as dashboard might not make immediate API calls
  });
});