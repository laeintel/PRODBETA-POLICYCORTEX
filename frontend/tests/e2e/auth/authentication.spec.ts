/**
 * Authentication E2E Tests
 * Critical user flows for authentication and authorization
 */

import { test, expect } from '../../fixtures/test-fixtures';
import { testData } from '../../fixtures/test-fixtures';

test.describe('Authentication Flows', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });
  
  test('should login with valid credentials', async ({ authPage }) => {
    await authPage.goto();
    await authPage.login(testData.users.admin.email, testData.users.admin.password);
    await authPage.verifyLoggedIn();
  });
  
  test('should show error with invalid credentials', async ({ authPage }) => {
    await authPage.goto();
    await authPage.login('invalid@example.com', 'wrongpassword');
    await authPage.verifyError('Invalid email or password');
  });
  
  test('should logout successfully', async ({ authPage, authenticatedContext }) => {
    await authPage.goto();
    await authPage.login(testData.users.admin.email, testData.users.admin.password);
    await authPage.verifyLoggedIn();
    
    await authPage.logout();
    await authPage.verifyLoggedOut();
  });
  
  test('should redirect to login when accessing protected route', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page).toHaveURL('/auth/login');
  });
  
  test('should persist session across page refreshes', async ({ page, authPage }) => {
    await authPage.goto();
    await authPage.login(testData.users.admin.email, testData.users.admin.password);
    await authPage.verifyLoggedIn();
    
    // Refresh the page
    await page.reload();
    
    // Should still be logged in
    await authPage.verifyLoggedIn();
  });
  
  test('should handle Azure AD login', async ({ authPage }) => {
    test.skip(!process.env.AZURE_TEST_EMAIL, 'Azure credentials not configured');
    
    await authPage.goto();
    await authPage.loginWithAzure();
    await authPage.verifyLoggedIn();
  });
  
  test('should enforce role-based access', async ({ page, authPage }) => {
    // Login as read-only user
    await authPage.goto();
    await authPage.login(testData.users.readonly.email, testData.users.readonly.password);
    
    // Try to access admin page
    await page.goto('/admin');
    
    // Should show access denied
    await expect(page.locator('[data-testid="access-denied"]')).toBeVisible();
  });
  
  test('should handle session timeout', async ({ page, authPage, context }) => {
    await authPage.goto();
    await authPage.login(testData.users.admin.email, testData.users.admin.password);
    
    // Simulate session expiry
    await context.addCookies([{
      name: 'auth-token',
      value: 'expired-token',
      domain: 'localhost',
      path: '/',
      expires: Date.now() / 1000 - 3600 // Expired 1 hour ago
    }]);
    
    // Try to access protected resource
    await page.goto('/dashboard');
    
    // Should redirect to login
    await expect(page).toHaveURL('/auth/login');
  });
  
  test('should handle password reset flow', async ({ page, authPage }) => {
    await authPage.requestPasswordReset(testData.users.admin.email);
    
    // Verify success message
    await expect(page.locator('[data-testid="reset-success"]')).toBeVisible();
  });
  
  test('should register new user', async ({ page, authPage }) => {
    const newEmail = `test-${Date.now()}@example.com`;
    const password = 'TestPassword123!';
    
    await authPage.register(newEmail, password, password);
    
    // Verify registration success
    await expect(page.locator('[data-testid="registration-success"]')).toBeVisible();
    
    // Login with new account
    await authPage.login(newEmail, password);
    await authPage.verifyLoggedIn();
  });
  
  test('should validate password requirements', async ({ page }) => {
    await page.goto('/auth/register');
    
    // Try weak password
    await page.fill('[data-testid="register-password"]', 'weak');
    
    // Should show password requirements
    await expect(page.locator('[data-testid="password-requirements"]')).toBeVisible();
    await expect(page.locator('[data-testid="password-requirements"]')).toContainText('at least 8 characters');
  });
  
  test('should handle MFA setup', async ({ page, authPage }) => {
    await authPage.goto();
    await authPage.login(testData.users.admin.email, testData.users.admin.password);
    
    // Navigate to security settings
    await page.goto('/settings/security');
    
    // Enable MFA
    const mfaToggle = page.locator('[data-testid="enable-mfa"]');
    await mfaToggle.click();
    
    // Verify MFA setup screen
    await authPage.verifyMFASetup();
  });
  
  test('should handle MFA verification', async ({ authPage }) => {
    // Assuming MFA is already set up for this user
    await authPage.goto();
    await authPage.fillEmail(testData.users.admin.email);
    await authPage.fillPassword(testData.users.admin.password);
    await authPage.clickLogin();
    
    // Should prompt for MFA code
    const mfaPrompt = authPage.page.locator('[data-testid="mfa-prompt"]');
    if (await mfaPrompt.isVisible({ timeout: 5000 }).catch(() => false)) {
      await authPage.enterMFACode('123456'); // Mock code for testing
      await authPage.verifyLoggedIn();
    }
  });
  
  test('should handle concurrent login attempts', async ({ browser }) => {
    // Create multiple contexts
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();
    
    // Attempt login from both contexts
    await Promise.all([
      page1.goto('/auth/login'),
      page2.goto('/auth/login')
    ]);
    
    // Login from first context
    await page1.fill('input[type="email"]', testData.users.admin.email);
    await page1.fill('input[type="password"]', testData.users.admin.password);
    await page1.click('button[type="submit"]');
    
    // Login from second context
    await page2.fill('input[type="email"]', testData.users.admin.email);
    await page2.fill('input[type="password"]', testData.users.admin.password);
    await page2.click('button[type="submit"]');
    
    // Both should be successful
    await expect(page1).toHaveURL(/dashboard/);
    await expect(page2).toHaveURL(/dashboard/);
    
    // Clean up
    await context1.close();
    await context2.close();
  });
  
  test('should measure login performance', async ({ authPage }) => {
    const loginTime = await authPage.measureLoginPerformance();
    
    // Login should complete within 3 seconds
    expect(loginTime).toBeLessThan(3000);
  });
  
  test('should handle CSRF protection', async ({ page }) => {
    await page.goto('/auth/login');
    
    // Try to submit form without CSRF token
    const response = await page.request.post('/api/auth/login', {
      data: {
        email: testData.users.admin.email,
        password: testData.users.admin.password
      },
      headers: {
        'Content-Type': 'application/json'
        // Omitting CSRF token
      }
    });
    
    // Should be rejected
    expect(response.status()).toBe(403);
  });
});

test.describe('Authorization and Permissions', () => {
  test.beforeEach(async ({ authenticatedContext }) => {
    // Use authenticated context
  });
  
  test('should enforce resource-level permissions', async ({ page }) => {
    // Try to access a resource without permission
    await page.goto('/resources/restricted-resource-123');
    
    // Should show permission denied
    await expect(page.locator('[data-testid="permission-denied"]')).toBeVisible();
  });
  
  test('should handle API authorization', async ({ page }) => {
    const response = await page.request.get('/api/v1/admin/users');
    
    // Non-admin should get 403
    if (!page.url().includes('admin')) {
      expect(response.status()).toBe(403);
    } else {
      expect(response.status()).toBe(200);
    }
  });
  
  test('should respect feature flags', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Check if beta features are hidden for non-beta users
    const betaFeature = page.locator('[data-testid="beta-feature"]');
    if (!testData.users.admin.role.includes('beta')) {
      await expect(betaFeature).toBeHidden();
    }
  });
});