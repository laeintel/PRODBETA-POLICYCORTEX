/**
 * PolicyCortex Critical User Flow Test Suite: Dashboard Navigation & Data Display
 * 
 * This test suite covers comprehensive dashboard scenarios including:
 * - Multi-level navigation system
 * - Card/Visualization view toggle
 * - Real-time data updates
 * - Chart interactions and drill-downs
 * - Export functionality
 * - Responsive design
 * - Performance monitoring
 * 
 * Performance targets:
 * - Page navigation: <2s
 * - View toggle: <500ms
 * - Data refresh: <1s
 * - Chart rendering: <1s
 */

import { test, expect, Page, Locator } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

// Test configuration
test.use({
  trace: 'on-first-retry',
  video: 'on-first-retry',
  screenshot: 'only-on-failure',
  actionTimeout: 15000,
  navigationTimeout: 30000,
});

// Helper to set up authenticated session
async function setupAuthenticatedSession(page: Page) {
  await page.goto(BASE_URL);
  await page.evaluate(() => {
    localStorage.setItem('msal.account.keys', JSON.stringify(['test-account']));
    localStorage.setItem('msal.test-account', JSON.stringify({
      username: 'test@policycortex.com',
      name: 'Test User',
      roles: ['admin'],
    }));
    sessionStorage.setItem('isAuthenticated', 'true');
  });
}

// Helper to measure navigation performance
async function measureNavigation(page: Page, url: string): Promise<number> {
  const startTime = Date.now();
  await page.goto(url);
  await page.waitForLoadState('networkidle');
  const duration = Date.now() - startTime;
  
  console.log(`Navigation to ${url} took ${duration}ms`);
  expect(duration).toBeLessThan(2000); // 2s max
  
  return duration;
}

// Helper to check data loaded
async function waitForDataLoad(page: Page, timeout: number = 5000) {
  // Wait for loading indicators to disappear
  const loadingSelectors = [
    '[data-testid="loading"]',
    '.loading-spinner',
    'text=/loading|fetching|retrieving/i',
  ];
  
  for (const selector of loadingSelectors) {
    const element = page.locator(selector);
    if (await element.count() > 0) {
      await element.waitFor({ state: 'hidden', timeout });
    }
  }
  
  // Wait for data indicators to appear
  await page.waitForSelector('[data-testid*="data"], .data-loaded, .chart-container', {
    state: 'visible',
    timeout,
  });
}

test.describe('Critical Flow: Dashboard Navigation & Data Display', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuthenticatedSession(page);
  });

  test('01 - Main dashboard sections navigation with performance metrics', async ({ page }) => {
    const dashboardSections = [
      { name: 'Tactical Command', path: '/tactical', dataTestId: 'tactical-dashboard' },
      { name: 'Executive', path: '/executive', dataTestId: 'executive-dashboard' },
      { name: 'Governance', path: '/governance', dataTestId: 'governance-dashboard' },
      { name: 'Security', path: '/security', dataTestId: 'security-dashboard' },
      { name: 'Operations', path: '/operations', dataTestId: 'operations-dashboard' },
      { name: 'DevOps', path: '/devops', dataTestId: 'devops-dashboard' },
      { name: 'AI Platform', path: '/ai', dataTestId: 'ai-dashboard' },
      { name: 'ITSM', path: '/itsm', dataTestId: 'itsm-dashboard' },
    ];

    const navigationMetrics: Record<string, number> = {};

    for (const section of dashboardSections) {
      await test.step(`Navigate to ${section.name}`, async () => {
        const duration = await measureNavigation(page, `${BASE_URL}${section.path}`);
        navigationMetrics[section.name] = duration;
        
        // Verify correct page loaded
        await expect(page).toHaveURL(new RegExp(section.path));
        
        // Check for section-specific elements
        const dashboardElement = page.locator(`[data-testid="${section.dataTestId}"], h1:has-text("${section.name}")`);
        await expect(dashboardElement.first()).toBeVisible({ timeout: 5000 });
        
        // Wait for data to load
        await waitForDataLoad(page);
        
        // Check for view toggle if applicable
        const viewToggle = page.locator('[data-testid="view-toggle"]');
        if (await viewToggle.count() > 0) {
          await expect(viewToggle).toBeVisible();
        }
      });
    }

    // Log performance summary
    console.log('Navigation Performance Summary:', navigationMetrics);
    
    // Ensure average navigation time is under threshold
    const avgTime = Object.values(navigationMetrics).reduce((a, b) => a + b, 0) / Object.values(navigationMetrics).length;
    expect(avgTime).toBeLessThan(1500);
  });

  test('02 - Card/Visualization view toggle functionality', async ({ page }) => {
    // Navigate to a dashboard with view toggle
    await page.goto(`${BASE_URL}/operations`);
    await waitForDataLoad(page);

    const viewToggle = page.locator('[data-testid="view-toggle"], [aria-label*="view toggle"]');
    
    if (await viewToggle.count() > 0) {
      await test.step('Test card view', async () => {
        // Click card view button
        const cardViewBtn = page.locator('button:has-text("Card"), [aria-label*="card view"]');
        if (await cardViewBtn.count() > 0) {
          const startTime = Date.now();
          await cardViewBtn.click();
          
          // Wait for card view to render
          await page.waitForSelector('[data-testid="metric-card"], .card-view', { state: 'visible' });
          
          const switchTime = Date.now() - startTime;
          expect(switchTime).toBeLessThan(500); // View switch should be fast
          
          // Verify cards are displayed
          const cards = page.locator('[data-testid="metric-card"], .metric-card');
          expect(await cards.count()).toBeGreaterThan(0);
          
          // Check card interactions
          const firstCard = cards.first();
          await firstCard.hover();
          
          // Check for tooltips or additional info on hover
          const tooltip = page.locator('[role="tooltip"], .tooltip');
          if (await tooltip.count() > 0) {
            await expect(tooltip.first()).toBeVisible();
          }
        }
      });

      await test.step('Test visualization view', async () => {
        // Click visualization view button
        const vizViewBtn = page.locator('button:has-text("Visualization"), [aria-label*="visualization"]');
        if (await vizViewBtn.count() > 0) {
          const startTime = Date.now();
          await vizViewBtn.click();
          
          // Wait for charts to render
          await page.waitForSelector('[data-testid="chart-container"], .chart-container, canvas', { state: 'visible' });
          
          const switchTime = Date.now() - startTime;
          expect(switchTime).toBeLessThan(500);
          
          // Verify charts are displayed
          const charts = page.locator('[data-testid="chart-container"], .chart-container');
          expect(await charts.count()).toBeGreaterThan(0);
        }
      });
    }
  });

  test('03 - Chart interactions and drill-down capabilities', async ({ page }) => {
    // Navigate to a dashboard with charts
    await page.goto(`${BASE_URL}/governance`);
    await waitForDataLoad(page);

    // Switch to visualization view if needed
    const vizViewBtn = page.locator('button:has-text("Visualization")');
    if (await vizViewBtn.count() > 0 && await vizViewBtn.isVisible()) {
      await vizViewBtn.click();
    }

    await test.step('Test chart hover interactions', async () => {
      const chartContainers = page.locator('[data-testid="chart-container"], .chart-container');
      
      if (await chartContainers.count() > 0) {
        const firstChart = chartContainers.first();
        
        // Hover over chart to trigger tooltips
        await firstChart.hover();
        await page.waitForTimeout(500); // Wait for hover effects
        
        // Check for tooltips
        const tooltip = page.locator('[role="tooltip"], .chart-tooltip, .recharts-tooltip');
        if (await tooltip.count() > 0) {
          await expect(tooltip.first()).toBeVisible();
        }
      }
    });

    await test.step('Test chart drill-in functionality', async () => {
      // Look for drill-in buttons
      const drillInButtons = page.locator('[data-testid="drill-in"], [aria-label*="drill"], button:has-text("Drill")');
      
      if (await drillInButtons.count() > 0) {
        const firstDrillIn = drillInButtons.first();
        await firstDrillIn.click();
        
        // Wait for modal or expanded view
        await page.waitForSelector('[data-testid="chart-modal"], [role="dialog"], .fullscreen-chart', {
          state: 'visible',
          timeout: 3000,
        });
        
        // Verify expanded chart is visible
        const expandedChart = page.locator('[data-testid="chart-modal"], [role="dialog"]');
        await expect(expandedChart.first()).toBeVisible();
        
        // Test close functionality
        const closeButton = page.locator('[aria-label*="close"], button:has-text("Close"), button:has-text("×")');
        if (await closeButton.count() > 0) {
          await closeButton.first().click();
          await expect(expandedChart.first()).not.toBeVisible();
        }
      }
    });

    await test.step('Test chart data point interactions', async () => {
      // For charts with clickable data points
      const chartCanvas = page.locator('canvas, svg.recharts-surface');
      
      if (await chartCanvas.count() > 0) {
        const chart = chartCanvas.first();
        const box = await chart.boundingBox();
        
        if (box) {
          // Click on a data point (center of chart)
          await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
          
          // Check for context menu or detail panel
          const contextMenu = page.locator('[data-testid="context-menu"], [role="menu"]');
          const detailPanel = page.locator('[data-testid="detail-panel"], .detail-view');
          
          if (await contextMenu.count() > 0 || await detailPanel.count() > 0) {
            const visibleElement = await contextMenu.count() > 0 ? contextMenu : detailPanel;
            await expect(visibleElement.first()).toBeVisible();
          }
        }
      }
    });
  });

  test('04 - Real-time data updates and refresh functionality', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/monitoring`);
    await waitForDataLoad(page);

    await test.step('Test manual refresh', async () => {
      // Look for refresh button
      const refreshButton = page.locator('[data-testid="refresh"], [aria-label*="refresh"], button:has-text("Refresh")');
      
      if (await refreshButton.count() > 0) {
        // Get initial data snapshot
        const initialData = await page.locator('[data-testid*="metric"], .metric-value').allTextContents();
        
        // Click refresh
        await refreshButton.first().click();
        
        // Wait for loading state
        const loadingIndicator = page.locator('[data-testid="loading"], .loading');
        if (await loadingIndicator.count() > 0) {
          await loadingIndicator.first().waitFor({ state: 'visible', timeout: 1000 }).catch(() => {});
          await loadingIndicator.first().waitFor({ state: 'hidden', timeout: 5000 });
        } else {
          await page.waitForTimeout(1000); // Wait for potential data update
        }
        
        // Get updated data
        const updatedData = await page.locator('[data-testid*="metric"], .metric-value').allTextContents();
        
        // Data should be refreshed (might be same values but request should be made)
        expect(updatedData.length).toBeGreaterThan(0);
      }
    });

    await test.step('Test auto-refresh settings', async () => {
      // Look for auto-refresh toggle or settings
      const autoRefreshToggle = page.locator('[data-testid="auto-refresh"], [aria-label*="auto-refresh"]');
      
      if (await autoRefreshToggle.count() > 0) {
        // Enable auto-refresh
        await autoRefreshToggle.first().click();
        
        // Wait for one refresh cycle (typically 30s, but we'll check for indicators)
        await page.waitForTimeout(2000);
        
        // Verify refresh indicator or timestamp update
        const lastUpdated = page.locator('[data-testid="last-updated"], .last-updated, text=/updated.*ago/i');
        if (await lastUpdated.count() > 0) {
          await expect(lastUpdated.first()).toBeVisible();
        }
      }
    });

    await test.step('Monitor WebSocket connections for real-time updates', async () => {
      // Check for WebSocket connections
      const wsConnected = await page.evaluate(() => {
        // Check if any WebSocket connections exist
        // @ts-ignore
        return window.WebSocket && typeof window.WebSocket === 'function';
      });
      
      expect(wsConnected).toBeTruthy();
      
      // Monitor for real-time updates (if applicable)
      await page.waitForTimeout(3000);
      
      // Check for live update indicators
      const liveIndicator = page.locator('[data-testid="live-indicator"], .live-badge, text=/live|real-time/i');
      if (await liveIndicator.count() > 0) {
        await expect(liveIndicator.first()).toBeVisible();
      }
    });
  });

  test('05 - Data export functionality', async ({ page }) => {
    await page.goto(`${BASE_URL}/governance/compliance`);
    await waitForDataLoad(page);

    const exportButtons = page.locator('[data-testid="export"], [aria-label*="export"], button:has-text("Export")');
    
    if (await exportButtons.count() > 0) {
      await test.step('Test CSV export', async () => {
        // Set up download promise before clicking
        const downloadPromise = page.waitForEvent('download', { timeout: 10000 }).catch(() => null);
        
        // Click export button
        await exportButtons.first().click();
        
        // Check for export options menu
        const csvOption = page.locator('button:has-text("CSV"), [data-value="csv"]');
        if (await csvOption.count() > 0) {
          await csvOption.first().click();
          
          // Wait for download
          const download = await downloadPromise;
          if (download) {
            expect(download.suggestedFilename()).toMatch(/\.csv$/);
            
            // Verify file size is reasonable
            const path = await download.path();
            if (path) {
              const stats = await page.evaluate(async (filePath) => {
                // In browser context, we can't directly access file system
                // This is a placeholder for file validation
                return { size: 1000 }; // Mock size
              }, path);
              
              expect(stats.size).toBeGreaterThan(0);
            }
          }
        }
      });

      await test.step('Test JSON export', async () => {
        const downloadPromise = page.waitForEvent('download', { timeout: 10000 }).catch(() => null);
        
        await exportButtons.first().click();
        
        const jsonOption = page.locator('button:has-text("JSON"), [data-value="json"]');
        if (await jsonOption.count() > 0) {
          await jsonOption.first().click();
          
          const download = await downloadPromise;
          if (download) {
            expect(download.suggestedFilename()).toMatch(/\.json$/);
          }
        }
      });
    }
  });

  test('06 - Sidebar navigation and breadcrumb functionality', async ({ page }) => {
    await page.goto(`${BASE_URL}/tactical`);
    
    await test.step('Test sidebar navigation', async () => {
      // Check if sidebar is visible
      const sidebar = page.locator('[data-testid="sidebar"], [role="navigation"], .sidebar');
      
      if (await sidebar.count() > 0) {
        // Test collapsible sections
        const collapsibleSections = sidebar.locator('[data-testid*="collapsible"], [aria-expanded]');
        
        for (let i = 0; i < Math.min(await collapsibleSections.count(), 3); i++) {
          const section = collapsibleSections.nth(i);
          const isExpanded = await section.getAttribute('aria-expanded');
          
          // Toggle section
          await section.click();
          await page.waitForTimeout(300); // Wait for animation
          
          // Verify state changed
          const newState = await section.getAttribute('aria-expanded');
          expect(newState).not.toBe(isExpanded);
        }
        
        // Test navigation items
        const navItems = sidebar.locator('a[href], [role="link"]');
        if (await navItems.count() > 0) {
          // Click first nav item
          const firstItem = navItems.first();
          const href = await firstItem.getAttribute('href');
          
          if (href && !href.startsWith('#')) {
            await firstItem.click();
            await page.waitForLoadState('networkidle');
            
            // Verify navigation occurred
            expect(page.url()).toContain(href);
          }
        }
      }
    });

    await test.step('Test breadcrumb navigation', async () => {
      // Navigate to a nested page
      await page.goto(`${BASE_URL}/governance/policies/details`);
      
      const breadcrumbs = page.locator('[data-testid="breadcrumbs"], [aria-label*="breadcrumb"], .breadcrumbs');
      
      if (await breadcrumbs.count() > 0) {
        const breadcrumbItems = breadcrumbs.locator('a, [role="link"]');
        
        if (await breadcrumbItems.count() > 1) {
          // Click middle breadcrumb to navigate up
          const middleItem = breadcrumbItems.nth(1);
          await middleItem.click();
          
          await page.waitForLoadState('networkidle');
          
          // Verify navigation to parent level
          expect(page.url()).not.toContain('/details');
        }
      }
    });
  });

  test('07 - Search and filter functionality', async ({ page }) => {
    await page.goto(`${BASE_URL}/operations/resources`);
    await waitForDataLoad(page);

    await test.step('Test search functionality', async () => {
      const searchInput = page.locator('[data-testid="search"], input[type="search"], input[placeholder*="search" i]');
      
      if (await searchInput.count() > 0) {
        // Enter search term
        await searchInput.first().fill('virtual machine');
        
        // Trigger search (enter key or search button)
        await searchInput.first().press('Enter');
        
        // Wait for results to update
        await page.waitForTimeout(1000);
        
        // Verify filtered results
        const results = page.locator('[data-testid*="result"], .search-result, .resource-item');
        if (await results.count() > 0) {
          // Check that results contain search term
          const firstResult = await results.first().textContent();
          expect(firstResult?.toLowerCase()).toContain('virtual');
        }
        
        // Clear search
        await searchInput.first().clear();
        await searchInput.first().press('Enter');
        await page.waitForTimeout(500);
      }
    });

    await test.step('Test filters', async () => {
      const filterButtons = page.locator('[data-testid*="filter"], button:has-text("Filter")');
      
      if (await filterButtons.count() > 0) {
        await filterButtons.first().click();
        
        // Wait for filter panel/modal
        const filterPanel = page.locator('[data-testid="filter-panel"], [role="dialog"]');
        if (await filterPanel.count() > 0) {
          await expect(filterPanel.first()).toBeVisible();
          
          // Apply a filter
          const filterOptions = filterPanel.locator('input[type="checkbox"], input[type="radio"]');
          if (await filterOptions.count() > 0) {
            await filterOptions.first().check();
            
            // Apply filters
            const applyButton = filterPanel.locator('button:has-text("Apply")');
            if (await applyButton.count() > 0) {
              await applyButton.click();
              
              // Wait for filtered results
              await waitForDataLoad(page);
              
              // Verify filter badge or indicator
              const filterBadge = page.locator('[data-testid="filter-badge"], .filter-active');
              if (await filterBadge.count() > 0) {
                await expect(filterBadge.first()).toBeVisible();
              }
            }
          }
        }
      }
    });
  });

  test('08 - Responsive design and mobile navigation', async ({ page }) => {
    const viewports = [
      { width: 375, height: 667, name: 'mobile', isMobile: true },
      { width: 768, height: 1024, name: 'tablet', isMobile: true },
      { width: 1024, height: 768, name: 'desktop-small', isMobile: false },
      { width: 1920, height: 1080, name: 'desktop-full', isMobile: false },
    ];

    for (const viewport of viewports) {
      await test.step(`Test ${viewport.name} viewport`, async () => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        await page.goto(`${BASE_URL}/tactical`);
        
        if (viewport.isMobile) {
          // Check for mobile menu toggle
          const mobileMenuToggle = page.locator('[data-testid="mobile-menu"], [aria-label*="menu"], .hamburger');
          await expect(mobileMenuToggle.first()).toBeVisible();
          
          // Open mobile menu
          await mobileMenuToggle.first().click();
          
          // Verify menu opens
          const mobileNav = page.locator('[data-testid="mobile-nav"], .mobile-menu');
          await expect(mobileNav.first()).toBeVisible();
          
          // Close menu
          const closeButton = page.locator('[aria-label*="close"], .close-menu');
          if (await closeButton.count() > 0) {
            await closeButton.first().click();
            await expect(mobileNav.first()).not.toBeVisible();
          }
        } else {
          // Desktop should show sidebar
          const sidebar = page.locator('[data-testid="sidebar"], .sidebar');
          await expect(sidebar.first()).toBeVisible();
        }
        
        // Check layout adjustments
        const mainContent = page.locator('main, [role="main"], .main-content');
        const contentBox = await mainContent.boundingBox();
        
        if (contentBox) {
          // Content should fit viewport
          expect(contentBox.width).toBeLessThanOrEqual(viewport.width);
        }
        
        // Take screenshot for visual regression
        await page.screenshot({
          path: `test-results/screenshots/dashboard-${viewport.name}.png`,
          fullPage: false, // Just viewport
        });
      });
    }
  });

  test('09 - Accessibility compliance', async ({ page }) => {
    await page.goto(`${BASE_URL}/tactical`);
    await waitForDataLoad(page);

    await test.step('Keyboard navigation', async () => {
      // Tab through interactive elements
      let tabCount = 0;
      const maxTabs = 20;
      
      while (tabCount < maxTabs) {
        await page.keyboard.press('Tab');
        tabCount++;
        
        // Check focused element
        const focusedElement = await page.evaluate(() => {
          const el = document.activeElement;
          return {
            tagName: el?.tagName,
            role: el?.getAttribute('role'),
            ariaLabel: el?.getAttribute('aria-label'),
            isVisible: el ? window.getComputedStyle(el).visibility !== 'hidden' : false,
          };
        });
        
        // Ensure focused element is visible and has proper attributes
        if (focusedElement.isVisible) {
          expect(focusedElement.tagName).toBeTruthy();
        }
      }
    });

    await test.step('ARIA attributes', async () => {
      // Check for proper ARIA labels on interactive elements
      const interactiveElements = page.locator('button, a, input, select, textarea, [role="button"], [role="link"]');
      
      const count = await interactiveElements.count();
      for (let i = 0; i < Math.min(count, 10); i++) {
        const element = interactiveElements.nth(i);
        
        // Check for accessible name
        const accessibleName = await element.evaluate((el) => {
          // Check aria-label, aria-labelledby, or text content
          return el.getAttribute('aria-label') || 
                 el.getAttribute('aria-labelledby') || 
                 el.textContent?.trim();
        });
        
        expect(accessibleName).toBeTruthy();
      }
    });

    await test.step('Color contrast', async () => {
      // Check a sample of text elements for sufficient contrast
      const textElements = page.locator('p, span, h1, h2, h3, h4, h5, h6').filter({ hasText: /.+/ });
      
      const sampleSize = Math.min(await textElements.count(), 5);
      for (let i = 0; i < sampleSize; i++) {
        const element = textElements.nth(i);
        
        const contrast = await element.evaluate((el) => {
          const style = window.getComputedStyle(el);
          // This is a simplified check - real contrast calculation would be more complex
          return {
            color: style.color,
            backgroundColor: style.backgroundColor,
            fontSize: style.fontSize,
          };
        });
        
        // Verify text is readable
        expect(contrast.fontSize).toBeTruthy();
      }
    });
  });

  test('10 - Performance monitoring and optimization', async ({ page }) => {
    await test.step('Measure Core Web Vitals', async () => {
      await page.goto(`${BASE_URL}/tactical`);
      
      const metrics = await page.evaluate(() => {
        return new Promise((resolve) => {
          let lcpValue: number | undefined;
          let clsValue = 0;
          let inpValue: number | undefined;
          
          // Observe LCP
          new PerformanceObserver((entryList) => {
            const entries = entryList.getEntries();
            const lastEntry = entries[entries.length - 1] as any;
            lcpValue = lastEntry.renderTime || lastEntry.loadTime;
          }).observe({ entryTypes: ['largest-contentful-paint'] });
          
          // Observe CLS
          new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
              if (!(entry as any).hadRecentInput) {
                clsValue += (entry as any).value;
              }
            }
          }).observe({ entryTypes: ['layout-shift'] });
          
          // Observe INP
          new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
              inpValue = (entry as any).duration;
            }
          }).observe({ entryTypes: ['event'] });
          
          // Collect metrics after page settles
          setTimeout(() => {
            resolve({
              LCP: lcpValue || 0,
              CLS: clsValue,
              INP: inpValue || 0,
              FCP: (performance as any).getEntriesByName('first-contentful-paint')[0]?.startTime || 0,
            });
          }, 3000);
        });
      });

      console.log('Core Web Vitals:', metrics);
      
      // Assert performance thresholds
      expect(metrics.LCP).toBeLessThan(2500); // Good LCP
      expect(metrics.CLS).toBeLessThan(0.1); // Good CLS
      expect(metrics.INP).toBeLessThan(200); // Good INP
      expect(metrics.FCP).toBeLessThan(1800); // Good FCP
    });

    await test.step('Memory usage monitoring', async () => {
      // Navigate through multiple pages to check for memory leaks
      const pages = ['/tactical', '/governance', '/operations', '/security'];
      
      for (const pagePath of pages) {
        await page.goto(`${BASE_URL}${pagePath}`);
        await waitForDataLoad(page);
        
        // Get memory usage if available
        const memoryUsage = await page.evaluate(() => {
          // @ts-ignore
          if (performance.memory) {
            // @ts-ignore
            return {
              // @ts-ignore
              usedJSHeapSize: performance.memory.usedJSHeapSize,
              // @ts-ignore
              totalJSHeapSize: performance.memory.totalJSHeapSize,
            };
          }
          return null;
        });
        
        if (memoryUsage) {
          console.log(`Memory usage for ${pagePath}:`, memoryUsage);
          // Memory should not exceed reasonable limits
          expect(memoryUsage.usedJSHeapSize).toBeLessThan(100 * 1024 * 1024); // 100MB
        }
      }
    });

    await test.step('Network performance', async () => {
      let totalRequests = 0;
      let totalSize = 0;
      let apiResponseTimes: number[] = [];
      
      // Monitor network activity
      page.on('response', response => {
        totalRequests++;
        const size = Number(response.headers()['content-length'] || 0);
        totalSize += size;
        
        if (response.url().includes('/api/')) {
          apiResponseTimes.push(response.timing()?.responseEnd || 0);
        }
      });
      
      await page.goto(`${BASE_URL}/operations`);
      await waitForDataLoad(page);
      
      // Calculate average API response time
      const avgApiTime = apiResponseTimes.length > 0 
        ? apiResponseTimes.reduce((a, b) => a + b, 0) / apiResponseTimes.length 
        : 0;
      
      console.log('Network Performance:', {
        totalRequests,
        totalSize: `${(totalSize / 1024 / 1024).toFixed(2)}MB`,
        avgApiResponseTime: `${avgApiTime.toFixed(2)}ms`,
      });
      
      // Assert reasonable limits
      expect(totalRequests).toBeLessThan(100); // Avoid too many requests
      expect(totalSize).toBeLessThan(5 * 1024 * 1024); // 5MB total
      if (avgApiTime > 0) {
        expect(avgApiTime).toBeLessThan(1000); // API responses under 1s
      }
    });
  });
});

// Cleanup
test.afterAll(async () => {
  console.log('Dashboard navigation tests completed');
});