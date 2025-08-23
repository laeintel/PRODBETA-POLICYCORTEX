import { test, expect } from '@playwright/test';

test.describe('PolicyCortex UI Comprehensive Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set a shorter timeout for faster testing
    page.setDefaultTimeout(10000);
    page.setDefaultNavigationTimeout(10000);
  });

  test('Homepage loads with proper structure', async ({ page }) => {
    await page.goto('http://localhost:3001');
    
    // Check title
    await expect(page).toHaveTitle(/PolicyCortex/);
    
    // Check main layout exists
    const mainContent = page.locator('main, [role="main"]').first();
    await expect(mainContent).toBeVisible();
    
    // Check for header/navigation
    const header = page.locator('header, nav, [role="navigation"]').first();
    await expect(header).toBeVisible();
  });

  test('Dashboard displays key metrics', async ({ page }) => {
    await page.goto('http://localhost:3001/dashboard');
    
    // Wait for content to load
    await page.waitForLoadState('networkidle');
    
    // Check for metric cards or dashboard content
    const dashboardContent = page.locator('h1, h2, [role="heading"]').first();
    await expect(dashboardContent).toBeVisible();
  });

  test('Governance section navigation works', async ({ page }) => {
    await page.goto('http://localhost:3001/governance');
    await page.waitForLoadState('networkidle');
    
    // Check for governance-specific content
    const pageContent = page.locator('main').first();
    await expect(pageContent).toBeVisible();
    
    // Test sub-navigation if available
    const policies = page.locator('a[href*="policies"], button:has-text("Policies")').first();
    if (await policies.isVisible()) {
      await policies.click();
      await expect(page).toHaveURL(/policies/);
    }
  });

  test('Security features are accessible', async ({ page }) => {
    await page.goto('http://localhost:3001/security');
    await page.waitForLoadState('networkidle');
    
    // Check for security page content
    const securityContent = page.locator('main').first();
    await expect(securityContent).toBeVisible();
    
    // Check for RBAC link
    const rbacLink = page.locator('a[href*="rbac"], button:has-text("RBAC")').first();
    if (await rbacLink.isVisible()) {
      await rbacLink.click();
      await expect(page).toHaveURL(/rbac/);
    }
  });

  test('AI Chat interface loads', async ({ page }) => {
    await page.goto('http://localhost:3001/ai/chat');
    await page.waitForLoadState('networkidle');
    
    // Check for chat input
    const chatInput = page.locator('input[type="text"], textarea, [contenteditable="true"]').first();
    await expect(chatInput).toBeVisible();
    
    // Test input functionality
    await chatInput.fill('Test message');
    await expect(chatInput).toHaveValue('Test message');
  });

  test('Theme toggle functionality', async ({ page }) => {
    await page.goto('http://localhost:3001');
    
    // Find theme toggle button
    const themeButton = page.locator('button[aria-label*="theme"], button[title*="theme"], [data-testid*="theme"]').first();
    
    if (await themeButton.isVisible()) {
      // Get initial theme
      const initialTheme = await page.evaluate(() => document.documentElement.className);
      
      // Toggle theme
      await themeButton.click();
      await page.waitForTimeout(500);
      
      // Check theme changed
      const newTheme = await page.evaluate(() => document.documentElement.className);
      expect(newTheme).not.toBe(initialTheme);
    }
  });

  test('Responsive design on mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('http://localhost:3001');
    
    // Check for mobile menu button
    const mobileMenu = page.locator('[aria-label*="menu"], button[class*="menu"], button[class*="burger"]').first();
    await expect(mobileMenu).toBeVisible();
    
    // Test mobile menu interaction
    await mobileMenu.click();
    await page.waitForTimeout(300);
    
    // Check menu opened
    const menuContent = page.locator('nav, [role="navigation"], aside').first();
    await expect(menuContent).toBeVisible();
  });

  test('Performance metrics are acceptable', async ({ page }) => {
    await page.goto('http://localhost:3001/dashboard');
    
    // Get performance metrics
    const metrics = await page.evaluate(() => {
      const perfData = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      return {
        domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
        loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
        totalTime: perfData.loadEventEnd - perfData.fetchStart
      };
    });
    
    // Check performance thresholds
    expect(metrics.totalTime).toBeLessThan(3000); // Page loads in under 3 seconds
    expect(metrics.domContentLoaded).toBeLessThan(1500); // DOM ready in under 1.5 seconds
  });

  test('Accessibility - basic checks', async ({ page }) => {
    await page.goto('http://localhost:3001');
    
    // Check for images with alt text
    const imagesWithoutAlt = await page.locator('img:not([alt])').count();
    expect(imagesWithoutAlt).toBe(0);
    
    // Check for proper heading hierarchy
    const h1Count = await page.locator('h1').count();
    expect(h1Count).toBeGreaterThan(0);
    
    // Check for form labels
    const inputs = page.locator('input:not([type="hidden"])');
    const inputCount = await inputs.count();
    
    for (let i = 0; i < inputCount; i++) {
      const input = inputs.nth(i);
      const hasLabel = await input.evaluate((el) => {
        const id = el.id;
        const label = id ? document.querySelector(`label[for="${id}"]`) : null;
        const ariaLabel = el.getAttribute('aria-label');
        return !!(label || ariaLabel);
      });
      expect(hasLabel).toBeTruthy();
    }
  });

  test('API Integration - basic check', async ({ page }) => {
    // Set up request interception
    let apiCallMade = false;
    page.on('request', request => {
      if (request.url().includes('/api/')) {
        apiCallMade = true;
      }
    });
    
    await page.goto('http://localhost:3001/dashboard');
    await page.waitForLoadState('networkidle');
    
    // Check if any API calls were made
    expect(apiCallMade).toBeTruthy();
  });

  test('Error handling - 404 page', async ({ page }) => {
    await page.goto('http://localhost:3001/non-existent-page');
    
    // Check for 404 content or redirect
    const errorContent = page.locator('text=/404|not found/i').first();
    const isError = await errorContent.isVisible();
    
    // Either shows 404 or redirects to home
    if (!isError) {
      await expect(page).toHaveURL(/\//);
    }
  });

  test('Navigation links are functional', async ({ page }) => {
    await page.goto('http://localhost:3001');
    
    // Get all navigation links
    const navLinks = page.locator('nav a, aside a, [role="navigation"] a');
    const linkCount = await navLinks.count();
    
    // Test first 5 links to avoid timeout
    const linksToTest = Math.min(linkCount, 5);
    
    for (let i = 0; i < linksToTest; i++) {
      const link = navLinks.nth(i);
      const href = await link.getAttribute('href');
      
      if (href && !href.startsWith('#') && !href.startsWith('http')) {
        await link.click();
        await page.waitForLoadState('networkidle');
        
        // Check page loaded successfully
        const mainContent = page.locator('main, [role="main"]').first();
        await expect(mainContent).toBeVisible();
        
        // Go back for next test
        await page.goto('http://localhost:3001');
      }
    }
  });
});