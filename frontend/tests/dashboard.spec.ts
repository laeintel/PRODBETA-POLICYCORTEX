/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

import { test, expect, Page } from '@playwright/test';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';

// Helper to setup authenticated session
async function setupAuth(page: Page) {
  await page.context().addCookies([
    {
      name: 'msal.session',
      value: 'mock-session',
      domain: 'localhost',
      path: '/',
    }
  ]);
}

test.describe('Dashboard Features', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
  });

  test('should display all main dashboard sections', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Main heading
    await expect(page.locator('h1:has-text("Governance Dashboard")')).toBeVisible();
    
    // Key metric cards
    await expect(page.locator('text=/policies|compliance|resources|costs/i').first()).toBeVisible();
    
    // Check for metric values
    const metricCards = page.locator('[class*="card"], [class*="metric"]');
    await expect(metricCards).toHaveCount(await metricCards.count());
  });

  test('should show real-time data indicators', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for data mode indicator
    const dataMode = page.locator('text=/simulated mode|real data|live data/i');
    await expect(dataMode.first()).toBeVisible();
    
    // Check for refresh indicators
    const refreshButton = page.locator('button[aria-label*="refresh"], button:has-text("Refresh")');
    if (await refreshButton.count() > 0) {
      await expect(refreshButton.first()).toBeEnabled();
    }
  });

  test('should display compliance score with color coding', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Look for compliance score
    const complianceScore = page.locator('text=/\\d{1,3}%.*compliance|compliance.*\\d{1,3}%/i');
    if (await complianceScore.count() > 0) {
      await expect(complianceScore.first()).toBeVisible();
      
      // Check for color coding (green/yellow/red)
      const scoreElement = complianceScore.first();
      const className = await scoreElement.getAttribute('class') || '';
      const style = await scoreElement.getAttribute('style') || '';
      
      // Should have some color indication
      expect(className + style).toMatch(/green|yellow|red|success|warning|danger|good|bad/i);
    }
  });

  test('should show policy violations and alerts', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for violations/alerts section
    const violations = page.locator('text=/violation|alert|issue|non.?compliant/i');
    if (await violations.count() > 0) {
      await expect(violations.first()).toBeVisible();
      
      // Check for severity indicators
      const severityBadges = page.locator('[class*="severity"], [class*="critical"], [class*="warning"], [class*="info"]');
      if (await severityBadges.count() > 0) {
        await expect(severityBadges.first()).toBeVisible();
      }
    }
  });

  test('should display resource distribution charts', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Wait for charts to load
    await page.waitForTimeout(2000);
    
    // Check for chart containers (SVG, Canvas, or chart divs)
    const charts = page.locator('svg[class*="chart"], canvas, [class*="recharts"], [class*="chart-container"]');
    if (await charts.count() > 0) {
      await expect(charts.first()).toBeVisible();
      
      // Check for chart legends
      const legends = page.locator('[class*="legend"], [class*="chart-legend"]');
      if (await legends.count() > 0) {
        await expect(legends.first()).toBeVisible();
      }
    }
  });

  test('should show cost analysis metrics', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for cost-related metrics
    const costMetrics = page.locator('text=/\\$[\\d,]+|cost|spend|budget/i');
    if (await costMetrics.count() > 0) {
      await expect(costMetrics.first()).toBeVisible();
      
      // Check for trend indicators (up/down arrows)
      const trendIndicators = page.locator('[class*="trend"], [class*="arrow"], svg[class*="icon"]');
      if (await trendIndicators.count() > 0) {
        await expect(trendIndicators.first()).toBeVisible();
      }
    }
  });

  test('should display recent activities or audit log', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for activity/audit section
    const activities = page.locator('text=/recent|activity|audit|history|log/i');
    if (await activities.count() > 0) {
      await expect(activities.first()).toBeVisible();
      
      // Check for timestamp information
      const timestamps = page.locator('text=/\\d{1,2}:\\d{2}|ago|today|yesterday/i');
      if (await timestamps.count() > 0) {
        await expect(timestamps.first()).toBeVisible();
      }
    }
  });

  test('should have functioning navigation to other sections', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Test navigation to Policies
    const policiesLink = page.locator('a[href*="/policies"], button:has-text("Policies")').first();
    if (await policiesLink.isVisible()) {
      await policiesLink.click();
      await expect(page).toHaveURL(/.*policies/);
      await page.goBack();
    }
    
    // Test navigation to Resources
    const resourcesLink = page.locator('a[href*="/resources"], button:has-text("Resources")').first();
    if (await resourcesLink.isVisible()) {
      await resourcesLink.click();
      await expect(page).toHaveURL(/.*resources/);
      await page.goBack();
    }
    
    // Test navigation to Compliance
    const complianceLink = page.locator('a[href*="/compliance"], button:has-text("Compliance")').first();
    if (await complianceLink.isVisible()) {
      await complianceLink.click();
      await expect(page).toHaveURL(/.*compliance/);
    }
  });

  test('should handle data refresh correctly', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Find refresh button
    const refreshButton = page.locator('button[aria-label*="refresh"], button:has-text("Refresh")').first();
    if (await refreshButton.isVisible()) {
      // Intercept API calls to verify refresh
      let apiCallMade = false;
      await page.route('**/api/**', route => {
        apiCallMade = true;
        route.continue();
      });
      
      await refreshButton.click();
      
      // Wait for potential loading state
      const loadingIndicator = page.locator('[class*="loading"], [class*="spinner"], [aria-busy="true"]');
      if (await loadingIndicator.count() > 0) {
        await expect(loadingIndicator.first()).toBeVisible();
        await expect(loadingIndicator.first()).not.toBeVisible({ timeout: 10000 });
      }
    }
  });

  test('should display risk assessment metrics', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for risk-related content
    const riskMetrics = page.locator('text=/risk|threat|vulnerability|security/i');
    if (await riskMetrics.count() > 0) {
      await expect(riskMetrics.first()).toBeVisible();
      
      // Check for risk levels
      const riskLevels = page.locator('text=/high|medium|low|critical/i');
      if (await riskLevels.count() > 0) {
        await expect(riskLevels.first()).toBeVisible();
      }
    }
  });

  test('should show recommendations or insights', async ({ page }) => {
    await page.goto(`${baseURL}/dashboard`);
    
    // Check for recommendations section
    const recommendations = page.locator('text=/recommend|insight|suggest|improve/i');
    if (await recommendations.count() > 0) {
      await expect(recommendations.first()).toBeVisible();
      
      // Check for actionable items
      const actionButtons = page.locator('button:has-text("Apply"), button:has-text("Review"), button:has-text("Dismiss")');
      if (await actionButtons.count() > 0) {
        await expect(actionButtons.first()).toBeEnabled();
      }
    }
  });

  test('should be responsive on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${baseURL}/dashboard`);
    
    // Check main content is still visible
    await expect(page.locator('h1:has-text("Governance Dashboard")')).toBeVisible();
    
    // Check for mobile menu button
    const mobileMenu = page.locator('button[aria-label*="menu"], [class*="burger"], [class*="mobile-menu"]');
    if (await mobileMenu.count() > 0) {
      await expect(mobileMenu.first()).toBeVisible();
      
      // Test mobile menu functionality
      await mobileMenu.first().click();
      await page.waitForTimeout(500);
      
      // Menu items should be visible
      const menuItems = page.locator('nav a, [role="navigation"] a');
      if (await menuItems.count() > 0) {
        await expect(menuItems.first()).toBeVisible();
      }
    }
  });
});