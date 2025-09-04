/**
 * Dashboard Interaction E2E Tests
 * Tests for dashboard features, visualizations, and data interactions
 */

import { test, expect } from '../../fixtures/test-fixtures';
import { performanceThresholds } from '../../fixtures/test-fixtures';

test.describe('Dashboard Interactions', () => {
  test.beforeEach(async ({ authenticatedContext, dashboardPage }) => {
    await dashboardPage.goto();
  });
  
  test('should load dashboard with all sections', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Verify all main sections are present
    const sections = ['operations', 'security', 'compliance', 'cost', 'performance'];
    for (const section of sections) {
      const sectionElement = dashboardPage.page.locator(`[data-testid="section-${section}"]`);
      await expect(sectionElement).toBeVisible();
    }
  });
  
  test('should toggle between card and visualization views', async ({ dashboardPage }) => {
    // Start in card view
    await dashboardPage.toggleView('card');
    await dashboardPage.verifyCardLayout();
    
    // Switch to visualization view
    await dashboardPage.toggleView('visualization');
    await dashboardPage.verifyVisualizationLayout();
    
    // Switch back to card view
    await dashboardPage.toggleView('card');
    await dashboardPage.verifyCardLayout();
  });
  
  test('should display real-time metrics', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Get initial metric values
    const initialValue = await dashboardPage.getMetricValue('Total Resources');
    
    // Refresh data
    await dashboardPage.refreshData();
    
    // Verify data is fresh
    await dashboardPage.verifyDataFreshness();
  });
  
  test('should export data in multiple formats', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Export as CSV
    const csvDownload = await dashboardPage.exportData('csv');
    expect(csvDownload.suggestedFilename()).toContain('.csv');
    
    // Export as JSON
    const jsonDownload = await dashboardPage.exportData('json');
    expect(jsonDownload.suggestedFilename()).toContain('.json');
  });
  
  test('should open charts in fullscreen mode', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    await dashboardPage.toggleView('visualization');
    
    // Open chart in fullscreen
    await dashboardPage.openChartFullscreen('Resource Distribution');
    
    // Verify fullscreen mode
    const fullscreenModal = dashboardPage.page.locator('[data-testid="fullscreen-modal"]');
    await expect(fullscreenModal).toBeVisible();
    
    // Close fullscreen
    await dashboardPage.closeFullscreen();
    await expect(fullscreenModal).toBeHidden();
  });
  
  test('should handle drill-in functionality', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    await dashboardPage.toggleView('visualization');
    
    // Drill into a chart
    await dashboardPage.drillIntoChart('Cost by Service', 'compute');
    
    // Verify drill-in view shows detailed data
    const drillInView = dashboardPage.page.locator('[data-testid="drill-in-view"]');
    await expect(drillInView).toBeVisible();
    await expect(drillInView).toContainText('Compute Services Detail');
  });
  
  test('should apply and persist filters', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Apply filters
    await dashboardPage.interactWithFilter('region', 'us-east-1');
    await dashboardPage.interactWithFilter('environment', 'production');
    
    // Verify filtered data
    const filteredData = await page.locator('[data-testid="filtered-indicator"]').textContent();
    expect(filteredData).toContain('Filtered');
    
    // Navigate away and back
    await page.goto('/settings');
    await dashboardPage.goto();
    
    // Filters should persist
    const regionFilter = page.locator('[data-testid="filter-region"]');
    await expect(regionFilter).toHaveValue('us-east-1');
  });
  
  test('should handle search functionality', async ({ dashboardPage }) => {
    await dashboardPage.searchDashboard('virtual machines');
    
    // Verify search results
    const results = dashboardPage.page.locator('[data-testid="search-results"]');
    await expect(results).toBeVisible();
    await expect(results).toContainText('virtual machines');
  });
  
  test('should update metrics in real-time', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Listen for WebSocket updates
    page.on('websocket', ws => {
      ws.on('framereceived', frame => {
        const data = JSON.parse(frame.payload?.toString() || '{}');
        if (data.type === 'metric-update') {
          expect(data).toHaveProperty('metric');
          expect(data).toHaveProperty('value');
        }
      });
    });
    
    // Wait for at least one update
    await page.waitForTimeout(5000);
    
    // Verify metrics have updated
    await dashboardPage.verifyDataFreshness();
  });
  
  test('should handle responsive layout', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Test different viewport sizes
    const viewports = [
      { width: 1920, height: 1080 }, // Desktop
      { width: 768, height: 1024 },  // Tablet
      { width: 375, height: 667 }    // Mobile
    ];
    
    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.waitForTimeout(500); // Wait for layout adjustment
      
      // Verify layout adapts
      const container = page.locator('[data-testid="dashboard-container"]');
      const containerBox = await container.boundingBox();
      
      expect(containerBox?.width).toBeLessThanOrEqual(viewport.width);
      
      // On mobile, verify cards stack vertically
      if (viewport.width < 768) {
        const cards = await page.locator('[data-testid="metric-card"]').all();
        if (cards.length > 1) {
          const firstCardBox = await cards[0].boundingBox();
          const secondCardBox = await cards[1].boundingBox();
          
          // Second card should be below first card
          expect(secondCardBox?.y).toBeGreaterThan(firstCardBox?.y || 0);
        }
      }
    }
  });
  
  test('should display trend indicators correctly', async ({ dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Verify trend indicators
    await dashboardPage.verifyMetricTrend('Total Cost', 'down');
    await dashboardPage.verifyMetricTrend('Compliance Score', 'up');
    await dashboardPage.verifyMetricTrend('Active Incidents', 'stable');
  });
  
  test('should handle error states gracefully', async ({ page, dashboardPage }) => {
    // Intercept API calls and force error
    await page.route('/api/v1/metrics', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal Server Error' })
      });
    });
    
    await dashboardPage.goto();
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Unable to load data');
    
    // Should show retry button
    const retryButton = page.locator('[data-testid="retry-button"]');
    await expect(retryButton).toBeVisible();
  });
  
  test('should meet performance thresholds', async ({ dashboardPage }) => {
    const loadTime = await dashboardPage.measureDashboardLoadTime();
    
    // Dashboard should load within 3 seconds
    expect(loadTime).toBeLessThan(performanceThresholds.pageLoad);
  });
  
  test('should handle keyboard navigation', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Tab through interactive elements
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    await page.keyboard.press('Tab');
    
    // Verify focus is visible
    const focusedElement = await page.evaluate(() => {
      const el = document.activeElement;
      return {
        tagName: el?.tagName,
        hasVisibleFocus: window.getComputedStyle(el as Element).outline !== 'none'
      };
    });
    
    expect(focusedElement.tagName).toBeTruthy();
    
    // Test keyboard shortcuts
    await page.keyboard.press('Control+r'); // Refresh
    await dashboardPage.waitForDataLoad();
    
    await page.keyboard.press('Control+e'); // Export
    await expect(page.locator('[data-testid="export-menu"]')).toBeVisible();
  });
  
  test('should display tooltips on hover', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Hover over metric card
    const metricCard = page.locator('[data-testid="metric-card"]').first();
    await metricCard.hover();
    
    // Tooltip should appear
    const tooltip = page.locator('[role="tooltip"]');
    await expect(tooltip).toBeVisible();
    
    // Move away to hide tooltip
    await page.mouse.move(0, 0);
    await expect(tooltip).toBeHidden();
  });
  
  test('should handle theme switching', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Get initial theme
    const initialTheme = await page.evaluate(() => 
      document.documentElement.getAttribute('data-theme')
    );
    
    // Toggle theme
    const themeToggle = page.locator('[data-testid="theme-toggle"]');
    await themeToggle.click();
    
    // Verify theme changed
    const newTheme = await page.evaluate(() => 
      document.documentElement.getAttribute('data-theme')
    );
    
    expect(newTheme).not.toBe(initialTheme);
    
    // Verify theme persists after refresh
    await page.reload();
    
    const persistedTheme = await page.evaluate(() => 
      document.documentElement.getAttribute('data-theme')
    );
    
    expect(persistedTheme).toBe(newTheme);
  });
  
  test('should handle concurrent data updates', async ({ page, dashboardPage }) => {
    await dashboardPage.waitForDataLoad();
    
    // Trigger multiple updates simultaneously
    await Promise.all([
      dashboardPage.refreshData(),
      dashboardPage.interactWithFilter('region', 'us-west-2'),
      dashboardPage.searchDashboard('storage')
    ]);
    
    // Dashboard should remain stable
    const errorMessage = page.locator('[data-testid="error-message"]');
    await expect(errorMessage).toBeHidden();
    
    // Data should be consistent
    await dashboardPage.verifyDataFreshness();
  });
});

test.describe('Dashboard Navigation', () => {
  test.beforeEach(async ({ authenticatedContext }) => {
    // Authenticated context
  });
  
  test('should navigate between dashboard sections', async ({ page, dashboardPage }) => {
    await dashboardPage.goto();
    
    const sections = [
      { name: 'Operations', path: '/operations' },
      { name: 'Security', path: '/security' },
      { name: 'Compliance', path: '/governance/compliance' },
      { name: 'DevOps', path: '/devops' },
      { name: 'AI', path: '/ai' }
    ];
    
    for (const section of sections) {
      await dashboardPage.navigateToSection(section.name);
      await expect(page).toHaveURL(new RegExp(section.path));
      
      // Verify section loaded
      await page.waitForLoadState('networkidle');
      const sectionHeader = page.locator('h1, [data-testid="page-title"]');
      await expect(sectionHeader).toContainText(section.name);
    }
  });
  
  test('should maintain navigation state', async ({ page }) => {
    // Navigate to a specific section
    await page.goto('/operations/monitoring');
    
    // Expand sidebar menu item
    const menuItem = page.locator('[data-testid="menu-operations"]');
    await menuItem.click();
    
    // Navigate to sub-item
    const subItem = page.locator('[data-testid="menu-operations-alerts"]');
    await subItem.click();
    
    // Refresh page
    await page.reload();
    
    // Menu should remain expanded
    await expect(menuItem).toHaveAttribute('aria-expanded', 'true');
  });
  
  test('should show breadcrumb navigation', async ({ page }) => {
    await page.goto('/operations/monitoring/alerts');
    
    const breadcrumb = page.locator('[data-testid="breadcrumb"]');
    await expect(breadcrumb).toBeVisible();
    
    // Verify breadcrumb items
    await expect(breadcrumb).toContainText('Home');
    await expect(breadcrumb).toContainText('Operations');
    await expect(breadcrumb).toContainText('Monitoring');
    await expect(breadcrumb).toContainText('Alerts');
    
    // Click breadcrumb to navigate back
    const operationsLink = breadcrumb.locator('a:has-text("Operations")');
    await operationsLink.click();
    
    await expect(page).toHaveURL('/operations');
  });
});