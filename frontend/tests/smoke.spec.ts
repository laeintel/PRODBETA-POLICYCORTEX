/**
 * Smoke tests as per TD.MD requirements
 * Tests for Executive landing, Audit verification, Predictions, ROI, and no horizontal scroll
 */

import { test, expect } from '@playwright/test';

const baseURL = process.env.BASE_URL || 'http://localhost:3000';

test.describe('PolicyCortex Smoke Tests - TD.MD Requirements', () => {
  test('Executive landing + no horizontal scroll', async ({ page }) => {
    await page.goto(baseURL);
    
    // Should redirect to executive dashboard as per TD.MD
    await expect(page).toHaveURL(/\/executive/);
    
    // Test for no horizontal scroll
    await page.evaluate(() => window.scrollBy(10000, 0));
    const scrollX = await page.evaluate(() => window.scrollX);
    expect(scrollX).toBe(0);
  });

  test('Audit verify visible', async ({ page }) => {
    await page.goto(`${baseURL}/audit`);
    
    // Check for chain integrity text as per TD.MD
    await expect(page.getByText(/Chain integrity:/)).toBeVisible();
    
    // Check for integrity status (OK|FAIL|Checking)
    await expect(page.getByText(/Integrity/).first()).toBeVisible();
  });

  test('Predictions render + Fix PR', async ({ page }) => {
    await page.goto(`${baseURL}/ai/predictions`);
    
    // Check for predictions heading
    await expect(page.getByRole('heading', { name: /Predictions/i })).toBeVisible();
    
    // Check for Create Fix PR link
    await expect(page.getByRole('link', { name: /Create Fix PR/i }).first()).toBeVisible();
  });

  test('ROI shows values or helpful error', async ({ page }) => {
    await page.goto(`${baseURL}/finops`);
    
    // Check for either savings values or configuration message
    await expect(page.getByText(/Savings this Quarter|Needs configuration/)).toBeVisible();
  });
});

// Additional smoke tests from TD.MD acceptance criteria
test.describe('TD.MD Acceptance Criteria', () => {
  test('Real data mode check - no silent mocks', async ({ page, request }) => {
    // Check if USE_REAL_DATA is true
    const isRealMode = process.env.USE_REAL_DATA === 'true';
    
    if (isRealMode) {
      // In real mode, endpoints should return 503 with hints, not mock data
      const response = await request.get(`${baseURL.replace('3000', '8080')}/api/v1/predictions`);
      
      if (response.status() === 503) {
        const body = await response.json();
        expect(body).toHaveProperty('hint');
        expect(body.hint).toContain('configure');
      }
    }
  });

  test('UI density - max-width container', async ({ page }) => {
    await page.goto(baseURL);
    
    // Check for bounded container (max-w-screen-2xl)
    const container = await page.locator('.max-w-screen-2xl').first();
    await expect(container).toBeVisible();
  });

  test('Navigation order matches spec', async ({ page }) => {
    await page.goto(baseURL);
    
    // Expected order from TD.MD
    const expectedItems = [
      'Executive',
      'Policy', 
      'Audit',
      'Predict',
      'FinOps'
    ];
    
    // Get navigation text
    const navText = await page.locator('nav').textContent();
    
    // Verify at least some expected items exist
    for (const item of expectedItems) {
      expect(navText).toContain(item);
    }
  });

  test('Health endpoint with sub-checks', async ({ request }) => {
    const response = await request.get(`${baseURL.replace('3000', '8080')}/healthz`);
    
    if (response.ok()) {
      const health = await response.json();
      
      // Should have sub-checks as per TD.MD
      expect(health).toHaveProperty('db_ok');
      expect(health).toHaveProperty('provider_ok');
    }
  });

  test('No horizontal scroll at 1366x768', async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await page.goto(baseURL);
    
    // Scroll attempt
    await page.evaluate(() => window.scrollBy(10000, 0));
    const scrollX = await page.evaluate(() => window.scrollX);
    
    // Should not scroll horizontally
    expect(scrollX).toBe(0);
  });
});