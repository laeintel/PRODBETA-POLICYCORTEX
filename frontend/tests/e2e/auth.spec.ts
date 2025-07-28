import { test, expect, Page } from '@playwright/test';

// Test data
const testUser = {
  email: 'test@policycortex.com',
  password: 'TestPassword123!',
  name: 'Test User'
};

// Helper functions
async function mockAzureADLogin(page: Page) {
  // Intercept Azure AD login and mock successful response
  await page.route('**/login.microsoftonline.com/**', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        access_token: 'mock-access-token',
        id_token: 'mock-id-token',
        expires_in: 3600,
        account: {
          homeAccountId: 'test-account-id',
          username: testUser.email,
          name: testUser.name
        }
      })
    });
  });
}

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display login page for unauthenticated users', async ({ page }) => {
    // Check for login page elements
    await expect(page.locator('h1')).toContainText('Welcome to PolicyCortex');
    await expect(page.locator('button:has-text("Sign in with Microsoft")')).toBeVisible();
    await expect(page.locator('text=AI-Powered Azure Governance')).toBeVisible();
  });

  test('should complete Azure AD login flow', async ({ page }) => {
    // Mock Azure AD login
    await mockAzureADLogin(page);

    // Click login button
    await page.click('button:has-text("Sign in with Microsoft")');

    // Wait for redirect to dashboard
    await page.waitForURL('/dashboard', { timeout: 10000 });

    // Verify user is logged in
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
    await expect(page.locator('[data-testid="user-name"]')).toContainText(testUser.name);
  });

  test('should handle login errors gracefully', async ({ page }) => {
    // Mock failed login
    await page.route('**/login.microsoftonline.com/**', async route => {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'invalid_request',
          error_description: 'Invalid credentials'
        })
      });
    });

    // Try to login
    await page.click('button:has-text("Sign in with Microsoft")');

    // Check error message
    await expect(page.locator('.error-message')).toContainText('Login failed');
    await expect(page.locator('button:has-text("Sign in with Microsoft")')).toBeVisible();
  });

  test('should logout successfully', async ({ page }) => {
    // First login
    await mockAzureADLogin(page);
    await page.click('button:has-text("Sign in with Microsoft")');
    await page.waitForURL('/dashboard');

    // Open user menu and logout
    await page.click('[data-testid="user-menu"]');
    await page.click('button:has-text("Sign out")');

    // Should redirect to login page
    await page.waitForURL('/');
    await expect(page.locator('button:has-text("Sign in with Microsoft")')).toBeVisible();
  });

  test('should persist authentication across page refreshes', async ({ page }) => {
    // Login
    await mockAzureADLogin(page);
    await page.click('button:has-text("Sign in with Microsoft")');
    await page.waitForURL('/dashboard');

    // Refresh page
    await page.reload();

    // Should still be on dashboard
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  });

  test('should redirect to requested page after login', async ({ page }) => {
    // Try to access protected route
    await page.goto('/policies');

    // Should redirect to login
    await expect(page).toHaveURL('/');

    // Login
    await mockAzureADLogin(page);
    await page.click('button:has-text("Sign in with Microsoft")');

    // Should redirect to originally requested page
    await page.waitForURL('/policies');
    await expect(page.locator('h1')).toContainText('Policies');
  });
});

test.describe('Authorization', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto('/');
    await mockAzureADLogin(page);
    await page.click('button:has-text("Sign in with Microsoft")');
    await page.waitForURL('/dashboard');
  });

  test('should show/hide features based on user role', async ({ page }) => {
    // Mock user with limited permissions
    await page.evaluate(() => {
      window.localStorage.setItem('user_role', 'viewer');
    });

    await page.reload();

    // Check that edit buttons are hidden
    await expect(page.locator('button:has-text("Create Policy")')).not.toBeVisible();
    await expect(page.locator('button:has-text("Edit")')).not.toBeVisible();
    
    // But view actions should be visible
    await expect(page.locator('button:has-text("View Details")')).toBeVisible();
  });

  test('should handle unauthorized API calls', async ({ page }) => {
    // Mock 403 response
    await page.route('**/api/v1/policies', route => {
      route.fulfill({
        status: 403,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'Forbidden',
          message: 'You do not have permission to perform this action'
        })
      });
    });

    // Try to access policies
    await page.goto('/policies');

    // Should show permission error
    await expect(page.locator('.permission-error')).toContainText('do not have permission');
  });
});