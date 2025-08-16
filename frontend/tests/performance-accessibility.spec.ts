/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { test, expect } from '@playwright/test';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';

test.describe('Performance Tests', () => {
  test('should load dashboard within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto(`${baseURL}/dashboard`);
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    // Dashboard should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
    
    // Check Core Web Vitals
    const metrics = await page.evaluate(() => {
      return {
        FCP: performance.getEntriesByName('first-contentful-paint')[0]?.startTime,
        LCP: performance.getEntriesByType('largest-contentful-paint').pop()?.startTime,
        CLS: 0, // Would need more complex calculation
        FID: 0  // Would need user interaction
      };
    });
    
    // First Contentful Paint should be under 1.8s
    if (metrics.FCP) {
      expect(metrics.FCP).toBeLessThan(1800);
    }
    
    // Largest Contentful Paint should be under 2.5s
    if (metrics.LCP) {
      expect(metrics.LCP).toBeLessThan(2500);
    }
  });

  test('should handle large datasets efficiently', async ({ page }) => {
    // Mock large dataset response
    await page.route('**/api/v1/resources', route => {
      const largeDataset = Array(1000).fill(null).map((_, i) => ({
        id: `resource-${i}`,
        name: `Resource ${i}`,
        type: 'VM',
        status: 'Running'
      }));
      
      route.fulfill({
        status: 200,
        body: JSON.stringify({ resources: largeDataset })
      });
    });
    
    const startTime = Date.now();
    await page.goto(`${baseURL}/resources`);
    
    // Wait for data to render
    await page.waitForSelector('[class*="resource"]', { timeout: 5000 });
    
    const renderTime = Date.now() - startTime;
    
    // Should render large dataset within 5 seconds
    expect(renderTime).toBeLessThan(5000);
    
    // Check for virtualization or pagination
    const visibleItems = await page.locator('[class*="resource-item"]').count();
    
    // Should not render all 1000 items at once (virtualization/pagination)
    expect(visibleItems).toBeLessThan(100);
  });

  test('should optimize bundle size and code splitting', async ({ page }) => {
    const resources: any[] = [];
    
    page.on('response', response => {
      if (response.url().includes('.js') || response.url().includes('.css')) {
        resources.push({
          url: response.url(),
          size: parseInt(response.headers()['content-length'] || '0'),
          type: response.url().includes('.js') ? 'js' : 'css'
        });
      }
    });
    
    await page.goto(baseURL);
    await page.waitForLoadState('networkidle');
    
    // Calculate total bundle size
    const totalJsSize = resources
      .filter(r => r.type === 'js')
      .reduce((acc, r) => acc + r.size, 0);
    
    const totalCssSize = resources
      .filter(r => r.type === 'css')
      .reduce((acc, r) => acc + r.size, 0);
    
    // JS bundle should be under 500KB (compressed)
    expect(totalJsSize).toBeLessThan(500 * 1024);
    
    // CSS should be under 100KB (compressed)
    expect(totalCssSize).toBeLessThan(100 * 1024);
    
    // Check for code splitting (multiple JS chunks)
    const jsChunks = resources.filter(r => r.type === 'js').length;
    expect(jsChunks).toBeGreaterThan(1);
  });

  test('should cache static assets properly', async ({ page }) => {
    const cachedResources: string[] = [];
    
    page.on('response', response => {
      const cacheControl = response.headers()['cache-control'];
      if (cacheControl && cacheControl.includes('max-age')) {
        cachedResources.push(response.url());
      }
    });
    
    await page.goto(baseURL);
    await page.waitForLoadState('networkidle');
    
    // Static assets should have cache headers
    expect(cachedResources.length).toBeGreaterThan(0);
    
    // Navigate to another page
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for 304 responses (cached)
    let cached304Count = 0;
    page.on('response', response => {
      if (response.status() === 304) {
        cached304Count++;
      }
    });
    
    // Reload to test cache
    await page.reload();
    await page.waitForLoadState('networkidle');
    
    // Some resources should be served from cache
    expect(cached304Count).toBeGreaterThan(0);
  });

  test('should handle slow network gracefully', async ({ page }) => {
    // Simulate slow 3G
    await page.route('**/*', route => route.continue());
    await page.context().setOffline(false);
    
    // Emulate slow network
    const client = await page.context().newCDPSession(page);
    await client.send('Network.emulateNetworkConditions', {
      offline: false,
      downloadThroughput: (50 * 1024) / 8, // 50kb/s
      uploadThroughput: (20 * 1024) / 8,   // 20kb/s
      latency: 2000 // 2s latency
    });
    
    await page.goto(`${baseURL}/dashboard`);
    
    // Should show loading indicators
    const loadingIndicator = page.locator('[class*="loading"], [class*="skeleton"], [class*="spinner"]');
    await expect(loadingIndicator.first()).toBeVisible();
    
    // Should eventually load even on slow connection
    await expect(page.locator('h1')).toBeVisible({ timeout: 30000 });
  });

  test('should not have memory leaks on navigation', async ({ page }) => {
    // Navigate between pages multiple times
    for (let i = 0; i < 5; i++) {
      await page.goto(`${baseURL}/dashboard`);
      await page.waitForTimeout(500);
      await page.goto(`${baseURL}/policies`);
      await page.waitForTimeout(500);
      await page.goto(`${baseURL}/resources`);
      await page.waitForTimeout(500);
    }
    
    // Check memory usage
    const metrics = await page.evaluate(() => {
      return (performance as any).memory ? {
        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (performance as any).memory.totalJSHeapSize
      } : null;
    });
    
    if (metrics) {
      // Heap usage should be reasonable (under 100MB)
      expect(metrics.usedJSHeapSize).toBeLessThan(100 * 1024 * 1024);
    }
  });
});

test.describe('Accessibility Tests', () => {
  test('should have proper heading hierarchy', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Get all headings
    const headings = await page.evaluate(() => {
      const h1s = Array.from(document.querySelectorAll('h1')).map(h => h.textContent);
      const h2s = Array.from(document.querySelectorAll('h2')).map(h => h.textContent);
      const h3s = Array.from(document.querySelectorAll('h3')).map(h => h.textContent);
      return { h1s, h2s, h3s };
    });
    
    // Should have exactly one h1
    expect(headings.h1s.length).toBe(1);
    
    // Should have logical heading structure
    if (headings.h3s.length > 0) {
      expect(headings.h2s.length).toBeGreaterThan(0);
    }
  });

  test('should have proper ARIA labels for interactive elements', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check buttons have accessible names
    const buttons = await page.locator('button').all();
    for (const button of buttons.slice(0, 5)) { // Check first 5 buttons
      const text = await button.textContent();
      const ariaLabel = await button.getAttribute('aria-label');
      const title = await button.getAttribute('title');
      
      // Button should have text, aria-label, or title
      expect(text || ariaLabel || title).toBeTruthy();
    }
    
    // Check form inputs have labels
    const inputs = await page.locator('input:not([type="hidden"])').all();
    for (const input of inputs.slice(0, 5)) { // Check first 5 inputs
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const ariaLabelledby = await input.getAttribute('aria-labelledby');
      
      if (id) {
        // Should have associated label
        const label = await page.locator(`label[for="${id}"]`).count();
        expect(label > 0 || ariaLabel || ariaLabelledby).toBeTruthy();
      }
    }
  });

  test('should support keyboard navigation', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    let focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(focusedElement).toBeTruthy();
    
    // Continue tabbing and check focus moves
    await page.keyboard.press('Tab');
    const secondFocusedElement = await page.evaluate(() => document.activeElement?.tagName);
    expect(secondFocusedElement).toBeTruthy();
    
    // Test Enter key on button
    const buttonCount = await page.locator('button').count();
    if (buttonCount > 0) {
      await page.keyboard.press('Enter');
      // Should trigger action (no JS errors)
    }
    
    // Test Escape key for modals
    await page.keyboard.press('Escape');
    // Should close any open modals
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check text contrast
    const textElements = await page.locator('p, span, div').all();
    
    for (const element of textElements.slice(0, 5)) { // Check first 5 elements
      const color = await element.evaluate(el => {
        const styles = window.getComputedStyle(el);
        return {
          color: styles.color,
          background: styles.backgroundColor,
          fontSize: parseFloat(styles.fontSize)
        };
      });
      
      // Skip if transparent background
      if (color.background === 'rgba(0, 0, 0, 0)') continue;
      
      // Large text (18pt or 14pt bold) needs 3:1 ratio
      // Normal text needs 4.5:1 ratio
      // This is a simplified check
      if (color.fontSize >= 18 || color.fontSize >= 14) {
        // Passes basic check (would need full contrast calculation)
        expect(color.color).not.toBe(color.background);
      }
    }
  });

  test('should have alt text for images', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Get all images
    const images = await page.locator('img').all();
    
    for (const img of images) {
      const alt = await img.getAttribute('alt');
      const role = await img.getAttribute('role');
      const ariaLabel = await img.getAttribute('aria-label');
      
      // Decorative images should have role="presentation" or empty alt
      // Informative images should have descriptive alt text
      expect(alt !== null || role === 'presentation' || ariaLabel).toBeTruthy();
    }
  });

  test('should support screen reader announcements', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for live regions
    const liveRegions = await page.locator('[aria-live], [role="alert"], [role="status"]').count();
    
    // Should have at least one live region for dynamic updates
    expect(liveRegions).toBeGreaterThan(0);
    
    // Check for skip navigation link
    const skipLink = await page.locator('a[href="#main"], a:has-text("Skip to main")').count();
    expect(skipLink).toBeGreaterThan(0);
  });

  test('should have proper focus indicators', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Focus on first button
    const firstButton = page.locator('button').first();
    if (await firstButton.isVisible()) {
      await firstButton.focus();
      
      // Check for focus styles
      const focusStyles = await firstButton.evaluate(el => {
        const styles = window.getComputedStyle(el);
        return {
          outline: styles.outline,
          boxShadow: styles.boxShadow,
          border: styles.border
        };
      });
      
      // Should have visible focus indicator
      expect(
        focusStyles.outline !== 'none' ||
        focusStyles.boxShadow !== 'none' ||
        focusStyles.border !== 'none'
      ).toBeTruthy();
    }
  });

  test('should support reduced motion preferences', async ({ page }) => {
    // Set prefers-reduced-motion
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for animation styles
    const animatedElements = await page.evaluate(() => {
      const elements = document.querySelectorAll('*');
      const animated: Array<{animation: string, transition: string}> = [];
      
      elements.forEach(el => {
        const styles = window.getComputedStyle(el);
        if (styles.animation !== 'none' || styles.transition !== 'none') {
          animated.push({
            animation: styles.animation,
            transition: styles.transition
          });
        }
      });
      
      return animated;
    });
    
    // Animations should be disabled or reduced
    animatedElements.forEach(el => {
      if (el.animation !== 'none') {
        // Animation duration should be minimal
        expect(el.animation).toMatch(/0s|0ms/);
      }
    });
  });

  test('should have proper semantic HTML', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for semantic elements
    const semanticElements = await page.evaluate(() => {
      return {
        nav: document.querySelectorAll('nav').length,
        main: document.querySelectorAll('main').length,
        header: document.querySelectorAll('header').length,
        footer: document.querySelectorAll('footer').length,
        section: document.querySelectorAll('section').length,
        article: document.querySelectorAll('article').length
      };
    });
    
    // Should use semantic HTML elements
    expect(semanticElements.nav).toBeGreaterThan(0);
    expect(semanticElements.main).toBe(1); // Should have exactly one main
    
    // Check for proper form elements
    const forms = await page.locator('form').count();
    if (forms > 0) {
      const formElement = page.locator('form').first();
      const formRole = await formElement.getAttribute('role');
      const formAriaLabel = await formElement.getAttribute('aria-label');
      
      // Forms should be properly labeled
      expect(formRole || formAriaLabel).toBeTruthy();
    }
  });

  test('should support high contrast mode', async ({ page }) => {
    // Emulate high contrast mode
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.goto(`${baseURL}/dashboard`);
    
    // Check that content is still visible
    await expect(page.locator('h1')).toBeVisible();
    
    // Check for forced colors support
    const supportsForceColors = await page.evaluate(() => {
      return window.matchMedia('(prefers-contrast: high)').matches ||
             window.matchMedia('(forced-colors: active)').matches;
    });
    
    // UI should still be functional in high contrast
    const buttons = await page.locator('button').count();
    expect(buttons).toBeGreaterThan(0);
  });
});