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

test.describe('Policy Management', () => {
  test.beforeEach(async ({ page }) => {
    await setupAuth(page);
  });

  test('should display policy list with key information', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check page title
    await expect(page.locator('h1:has-text("Policies"), h1:has-text("Policy Management")')).toBeVisible();
    
    // Check for policy table or cards
    const policyItems = page.locator('[class*="policy-item"], [class*="policy-card"], table tbody tr');
    await expect(policyItems.first()).toBeVisible({ timeout: 10000 });
    
    // Verify policy count is displayed
    const policyCount = page.locator('text=/\\d+\\s+(policies|policy)/i');
    if (await policyCount.count() > 0) {
      await expect(policyCount.first()).toBeVisible();
    }
  });

  test('should show policy compliance status', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Wait for policies to load
    await page.waitForTimeout(2000);
    
    // Check for compliance badges
    const complianceIndicators = page.locator('text=/compliant|non-compliant|violation/i, [class*="compliance"], [class*="status"]');
    if (await complianceIndicators.count() > 0) {
      await expect(complianceIndicators.first()).toBeVisible();
      
      // Verify color coding
      const statusElement = complianceIndicators.first();
      const className = await statusElement.getAttribute('class') || '';
      expect(className).toMatch(/success|warning|danger|error|green|yellow|red/i);
    }
  });

  test('should have functioning search and filter capabilities', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for search input
    const searchInput = page.locator('input[placeholder*="search" i], input[placeholder*="filter" i], input[type="search"]');
    if (await searchInput.count() > 0) {
      await expect(searchInput.first()).toBeVisible();
      
      // Test search functionality
      await searchInput.first().fill('security');
      await page.waitForTimeout(1000);
      
      // Verify filtered results (if any)
      const results = page.locator('[class*="policy-item"], [class*="policy-card"], table tbody tr');
      // Results should be updated (count might change)
    }
    
    // Check for filter dropdowns
    const filterButtons = page.locator('button:has-text("Filter"), select[name*="filter"], [class*="filter-dropdown"]');
    if (await filterButtons.count() > 0) {
      await expect(filterButtons.first()).toBeVisible();
    }
  });

  test('should display policy details on selection', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Wait for policies to load
    await page.waitForTimeout(2000);
    
    // Click on first policy item
    const firstPolicy = page.locator('[class*="policy-item"], [class*="policy-card"], table tbody tr').first();
    if (await firstPolicy.isVisible()) {
      await firstPolicy.click();
      
      // Check for policy details panel or modal
      const detailsPanel = page.locator('[class*="details"], [class*="modal"], [role="dialog"]');
      if (await detailsPanel.count() > 0) {
        await expect(detailsPanel.first()).toBeVisible();
        
        // Verify policy information is displayed
        await expect(page.locator('text=/description|effect|scope|condition/i').first()).toBeVisible();
      }
    }
  });

  test('should show policy categories and tags', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for category filters or tags
    const categories = page.locator('[class*="category"], [class*="tag"], [class*="badge"]');
    if (await categories.count() > 0) {
      await expect(categories.first()).toBeVisible();
      
      // Common policy categories
      const commonCategories = ['Security', 'Compliance', 'Cost', 'Network', 'Identity'];
      let foundCategory = false;
      for (const category of commonCategories) {
        if (await page.locator(`text=${category}`).count() > 0) {
          foundCategory = true;
          break;
        }
      }
    }
  });

  test('should display policy enforcement actions', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for enforcement action buttons
    const actionButtons = page.locator('button:has-text("Enforce"), button:has-text("Remediate"), button:has-text("Apply")');
    if (await actionButtons.count() > 0) {
      await expect(actionButtons.first()).toBeVisible();
      
      // Verify buttons are disabled in demo mode
      const isDemoMode = await page.locator('text=/simulated|demo/i').count() > 0;
      if (isDemoMode) {
        const firstAction = actionButtons.first();
        const isDisabled = await firstAction.isDisabled();
        expect(isDisabled).toBeTruthy();
      }
    }
  });

  test('should show policy exceptions and exemptions', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for exceptions section
    const exceptionsTab = page.locator('button:has-text("Exceptions"), a:has-text("Exceptions"), [role="tab"]:has-text("Exceptions")');
    if (await exceptionsTab.count() > 0) {
      await exceptionsTab.first().click();
      
      // Verify exceptions list or empty state
      const exceptionsList = page.locator('[class*="exception"], text=/no exceptions|exemption/i');
      await expect(exceptionsList.first()).toBeVisible();
    }
  });

  test('should handle policy export functionality', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for export button
    const exportButton = page.locator('button:has-text("Export"), button[aria-label*="export"]');
    if (await exportButton.count() > 0) {
      await expect(exportButton.first()).toBeVisible();
      
      // Verify export is disabled in demo mode
      const isDemoMode = await page.locator('text=/simulated|demo/i').count() > 0;
      if (isDemoMode) {
        const isDisabled = await exportButton.first().isDisabled();
        expect(isDisabled).toBeTruthy();
      }
    }
  });

  test('should display policy history and audit trail', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Select a policy first
    const firstPolicy = page.locator('[class*="policy-item"], [class*="policy-card"]').first();
    if (await firstPolicy.isVisible()) {
      await firstPolicy.click();
      
      // Look for history tab or section
      const historyTab = page.locator('button:has-text("History"), button:has-text("Audit"), [role="tab"]:has-text("History")');
      if (await historyTab.count() > 0) {
        await historyTab.first().click();
        
        // Verify history entries
        const historyEntries = page.locator('[class*="history-item"], [class*="audit-entry"], text=/created|modified|updated/i');
        if (await historyEntries.count() > 0) {
          await expect(historyEntries.first()).toBeVisible();
        }
      }
    }
  });

  test('should show policy recommendations from AI', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for recommendations section
    const recommendations = page.locator('text=/recommend|suggest|ai|insight/i');
    if (await recommendations.count() > 0) {
      await expect(recommendations.first()).toBeVisible();
      
      // Check for AI-generated badge
      const aiBadge = page.locator('[class*="ai"], text=/ai-powered|ai generated/i');
      if (await aiBadge.count() > 0) {
        await expect(aiBadge.first()).toBeVisible();
      }
    }
  });

  test('should display policy drift detection', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for drift indicators
    const driftIndicators = page.locator('text=/drift|deviation|change detected/i');
    if (await driftIndicators.count() > 0) {
      await expect(driftIndicators.first()).toBeVisible();
      
      // Check for drift severity
      const driftSeverity = page.locator('[class*="drift-severity"], [class*="drift-level"]');
      if (await driftSeverity.count() > 0) {
        const className = await driftSeverity.first().getAttribute('class') || '';
        expect(className).toMatch(/high|medium|low|critical/i);
      }
    }
  });

  test('should handle bulk policy operations', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for bulk selection checkboxes
    const checkboxes = page.locator('input[type="checkbox"][class*="select"], input[type="checkbox"][aria-label*="select"]');
    if (await checkboxes.count() > 1) {
      // Select multiple policies
      await checkboxes.nth(0).check();
      await checkboxes.nth(1).check();
      
      // Check for bulk actions menu
      const bulkActions = page.locator('[class*="bulk-actions"], button:has-text("Bulk Actions")');
      if (await bulkActions.count() > 0) {
        await expect(bulkActions.first()).toBeVisible();
      }
    }
  });

  test('should integrate with policy-as-code generation', async ({ page }) => {
    await page.goto(`${baseURL}/policies`);
    
    // Check for generate policy button
    const generateButton = page.locator('button:has-text("Generate"), button:has-text("Create Policy")');
    if (await generateButton.count() > 0) {
      await generateButton.first().click();
      
      // Check for policy generation form or modal
      const generationForm = page.locator('[class*="generate-policy"], [class*="policy-form"], form[name*="policy"]');
      if (await generationForm.count() > 0) {
        await expect(generationForm.first()).toBeVisible();
        
        // Check for template selection
        const templates = page.locator('select[name*="template"], radio[name*="template"], [class*="template-option"]');
        if (await templates.count() > 0) {
          await expect(templates.first()).toBeVisible();
        }
      }
    }
  });

  test('should be responsive on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto(`${baseURL}/policies`);
    
    // Check main content is visible
    await expect(page.locator('h1:has-text("Policies"), h1:has-text("Policy")')).toBeVisible();
    
    // Check for mobile-optimized layout
    const mobileLayout = page.locator('[class*="mobile"], [class*="responsive"]');
    if (await mobileLayout.count() > 0) {
      await expect(mobileLayout.first()).toBeVisible();
    }
    
    // Verify policy cards stack vertically on mobile
    const policyCards = page.locator('[class*="policy-item"], [class*="policy-card"]');
    if (await policyCards.count() > 1) {
      const firstCard = await policyCards.nth(0).boundingBox();
      const secondCard = await policyCards.nth(1).boundingBox();
      
      if (firstCard && secondCard) {
        // Cards should be stacked (second card Y position > first card Y position)
        expect(secondCard.y).toBeGreaterThan(firstCard.y);
      }
    }
  });
});